# Prefill 向上选图，真实长度保留在 wave metadata

默认 host-planned candidate 不再把尾巴按二进制拆成多个 forward。
70 token 选择容量128的图一次执行；不是64+4+2。
超过最大 chunk 的请求仍分成最大 chunk 和一个向上选桶的尾巴。
此改变仅在 owned prototype，不涉及已发布 DSV4 Worker。

## 哪些量是容量，哪些量是有效工作

- `scheduler.py` 的 graph key / ids shape 是容量；`query_tokens` 和
  ingress `device_q_lengths` 是本 wave 的真实 query 数。
- `root.py` 的有效 positions 是 `cursor + offset`，padding positions 为0，
  防止近 max-context 尾巴越界读取 RoPE 或 block table。
- padding slot mapping 为-1，不写 KV；cursor/phase/预算按真实长度推进。
- 最后有效 hidden row 在图内通过 tensor index_select 取出；不能采样桶尾。
- FIA 只写有效输出前缀；图内 masked_fill 清零 padding attention 输出，
  不让未初始化内容进入后续投影和 MoE router。这是每层一个额外的图内算子，
  不是 host task update；其开销包含在整模型测量内。
- 其余 dense/MoE 部分仍计算桶容量，不声称 padding 没有算力成本。
  同一 token 的 GEMM 行数变化也可能改变 BF16 舍入/最终生成路由。

## 原生 FIA planner 的真实边界

`fia-ceiling-leaf1` 尝试保持 TND query descriptor T=16 而 actualQ=13，
原生 planner 返回561002：
`queryT(16) must be equal to the last element of actualSequenceLengthQ(13)`。
这是 host descriptor 合同，不是设备 graph 必须重新 capture 的证明。

修正后的 `plan_native_queries` 接收独立 query offsets：分配仍为 bucket容量，
host planner 描述符只是同地址上的连续有效前缀视图（Q/out T=actualQ）。
原生 planner 的实际长度计划写入原有 banked GM tiling；固定图仍绑定同地址，
不替换 task、不重新 capture、不在图外执行数值 attention。
新 C symbol 防止旧 ctypes 调用者把新增指针参数误当 stream。

FD eligibility 的 capture catalog 仍按桶上界枚举；当前 power-of-two catalog
与 q<=16 的 FD 边界一致。多 row decode offset保持1,2,3,4，未引入变长多请求
prefill packing。不要把单请求尾巴 padding 称为已经实现 mixed/prefill batching。
旧 `OWNED_HOST_FIA=0` 或 `OWNED_STATIC_FIA=0` 对照路径保留原 exact-floor调度，
不把可变 query 元数据送给只预制过固定 query 的旧 plan。

## 验证与复用

- `fia-ceiling-leaf2`：120检查，decode4row和prefill容量16/32/64/128/1024；
  query包含13/9/16、30/17/32、41/33/64、70/65/127、513/777/1023。
  双bank复用、换页面、FD/non-FD及上下文至32K；对原生未padding query oracle
  所有有效输出 max_error0，exit0/release。未检查 padding 输出数值合同。
- `serving/probe_prefill_state.py`：在CPU上执行实际 prefill 方法，用数值模型double
  检查整份 KV sentinel、padding slot=-1、近最大上下文地址、cursor和最后有效采样。
  这不是神经网络数值等价证据。运行需 CANN环境，PYTHONPATH 加 retained
  LiveInference/src、owned-wave、joint-wave及serving；先按 native Worker导入顺序初始化。
- `serving/test_scheduler.py`：尾巴向上取桶、实际 ids/padding、N+2 projected cursor
  与回执一致，以及旧generation drain；常规 serving CPU suite 共18项通过。
- native C++ admission/lifecycle CPU tests通过。
- `swe-ceiling-dummy1`：两dummy层 Qwen TP2/EP2、20graphs、两轮30/26waves，
  各8calls/60outputs，TP quorum、前缀缓存、N+2退休通过，exit0/release。
  此短上下文 fixture 没有覆盖整模型长上下文 FD，不能替代下面的真实权重运行。

启动复测仍用 `serving/run.sh`，只跑 candidate，复用既有 native 结果。
新的 round receipt 保留 `prefill_policy/prefill_waves/prefill_tokens/padding_tokens`，
便于直接核对真实请求工作和 padding 投资；不从总 wave 数倒推 prefill 组成。

## 真实30B candidate-only复测

`swe-ceiling-candidate1`：本机device4/5，Qwen3-30B-A3B真实48层，TP2/EP2、DP1，
原四会话44calls/12,111输出预算；冷一轮、热两轮。两rank PASS、exit0/release。
原 config/trace 与历史native逐项一致；APC命中与上版owned也一致。
没有重跑baseline，也没有为此计时运行采profile。

| 指标 | 旧 exact-floor host计划 | 新 ceiling host计划 |
|---|---:|---:|
| 冷轮秒 | 100.820 | 95.138 |
| 热轮1秒 | 89.496 | 86.224 |
| 热轮2秒 | 89.439 | 86.222 |
| 热轮均值秒 | 89.468 | **86.223** |
| 热轮 TTFT p50 | 172.1–172.5ms | **63.1–63.3ms** |
| 热轮 TTFT p95 | 306.3–306.6ms | **91.6–92.4ms** |
| 冷 prefill waves | 278 | 103 |
| 热 prefill waves | 154 | **44** |
| 热全部 waves | 3843 | 3703 |

热轮耗时缩短3.63%，冷轮缩短5.64%。热轮每轮仍2698个实际prefill token，
额外计算687个padding row；冷轮实际83210、padding8758。累计APC命中仍为
冷545152、热625664，没有通过少跑真实输入或少生成输出来改善。
旧 prefill 数由精确分块算法与保存的逐请求 remaining 长度重建，新值直接计数。

热轮少140个总wave，其中110个是prefill；decode波次数也从3689降到3659，
因为请求更早ready改变了批次组合。不能把所有收益都归为110个孤立forward，
也不保证跨方案相同token/MoE路由。这里仅是固定输出预算的执行协议比较。

复用历史native热均值87.561s，新值约短1.53%；不是同场A/B或稳健普遍胜幅。
冷轮95.138s仍略慢于历史native94.413s。native与owned的TTFT前端观测边界
不同，本表TTFT仅比较同一owned前端的两个版本，不宣称HTTP端到端延迟。

36graph目录不变，activation90.957s排除计时。每rank36templates/1728绑定，
3456次Python attention调用均在初始化，运行时没有model/runner Python forward；
11,178个wave对应11,178次host计划，两rank一致。新图内padding清零的成本已计入。
这次没有新profile，因此不把旧四步profile的算子分布当成新调度的实测分布。

证据：capsule内 `tail-comparison.json`、`reused-baseline-comparison.json`、
`engine/candidate-rank{0,1}.json` 与 frozen `source/`。
