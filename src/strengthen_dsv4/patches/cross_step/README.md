# Cross-step：让 device 继续推进，不让 CPU acceptance 记账挡住下一轮

这个补丁面向 **请求集合稳定的 K5 投机解码波次**。上一轮 target 验证结束后，
device 已知道接受结果，可以据此准备下一轮的精确位置和 KV 访问；但原生 host
路径仍会在提交下一轮 target 前，等待结果回传并修正 CPU 账本。

我们把其中可后移的 CPU 工作放到 **本轮 target 提交之后**，让记账与设备计算
有机会重叠。不是增加一个调度器，也不是猜测 acceptance 后跳过验证。

**device 消费 acceptance 的能力来自 donor，补丁解除的是它前面的 host 依赖。**
不新增设备进度状态，不改接受算法、KV 映射或页面回收规则，不引入双槽 / N+2。

## 1. 原路径：device 已知道答案，host 仍要先追上它

K5 每请求安排 6 个 target queries，包含 5 个 draft slots。实际有效推进量
由验证结果决定，不能把每一轮都当成全部接受。

原生 DSV4 的 `_prepare_inputs` 已通过
`update_num_computed_tokens_for_batch_change` 消费上一轮
`valid_sampled_token_count_gpu`，在 device 上修正进度，再生成精确的 positions、
sequence lengths 和 slot mappings。CPU 则先有“乐观接受”的进度估计。
有效 token 计数通过异步 D2H 回传后，CPU 才能完成自己的精确修正。

两处 host 消费者会在 target 提交之前要求这份回传结果：

1. `_correct_optimistic_seq_lens_cpu` 等待 `valid_sampled_token_count_event`，
   用回传计数把 CPU 乐观长度改成精确长度。
2. `execute_model` 在 compressed-attention 路径提前调用 deferred CPU 状态
   修正回调，再准备 `_dsa_positions_cpu_buf`。已核对的 DSACP 路径使用 device
   positions 生成 RoPE / compressor 输入，不消费这个 CPU positions 缓冲。

相关源码：

- [`model_runner_v1.py`](../../../../upstream/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)：
  `NPUModelRunner.execute_model`、`_prepare_inputs`、
  `_correct_optimistic_seq_lens_cpu`、`_copy_valid_sampled_token_count`。
- [`gpu_model_runner.py`](../../../../upstream/vllm/vllm/v1/worker/gpu_model_runner.py)：
  基类 `_update_states`，是 deferred CPU 状态修正回调的来源；Ascend 的
  `_update_states` 完成自己的处理后调用它。

原生事件可能已经完成，所以“有等待调用”不等于每次都长时间阻塞。这里要去掉的
是提交链上的依赖；它能省多少时间，要看同工作量实测，而不是累计 synchronize
函数的调用次数。

## 2. 新顺序：先提交本轮 target，再完成 CPU 记账

```text
原路径（只画与本补丁有关的依赖）：
  准备输入 → 等上一轮结果并修正 CPU 长度 / 状态 → 构建 metadata → 提交 target

补丁后的准入路径：
  退役上一轮输入 DMA（原生保护保留）
    → 准备 CPU 长度上界，提交 device 精确状态 / metadata 更新
    → 提交本轮 target
    → 等本轮输入 DMA 完成
    → 执行上一轮结果对应的原 CPU 记账回调，一次且仅一次
    → 返回原 runner，继续 sampling / draft
```

这里 `_model_forward` 返回通常意味着设备工作已提交，而不是设备已完成。
因此后面的 CPU 等待 / 记账可以与已排队的 target 工作重叠。回调仍在同一次
模型调用内完成，在 sampling 和下一次请求状态更新之前，不跨请求代际拖延。

## 3. 五处 hook：分别拦在哪里、负责什么

[`Worker.compile_or_warm_up_model`](../../worker.py) 在原生 warmup 后调用
`cross_step.install(self)`。它创建一个 `CrossStepBounds`，挂到当前 runner 的
`_cross_step_bounds`；构造函数保存原 bound methods，再替换当前 runner 的
以下五个方法。不是全局修改所有 runner 类，也不自动安装其他 patch。

| 原 runner 方法 | 本模块接管方法 | 作用 |
|---|---|---|
| `_update_states` | `update_states` | 先执行原方法；若返回回调且波次满足准入，返回一个仅把原回调存入 `pending` 的包装。原调用点仍被执行，但 CPU 修正暂不发生。 |
| `_prepare_inputs` | `prepare_inputs` | 根据当前请求集合和 token 数设置本波 `admitted`，随后照常调用原输入准备；device 进度修正并未被替换。 |
| `_correct_optimistic_seq_lens_cpu` | `correct_bounds` | 准入且上界合法时保留 CPU 乐观长度，跳过这一处同步精确修正；否则执行原方法。 |
| `_build_attention_metadata` | `build_metadata` | 保存调用参数供 shadow 验收重建参考元数据，随后原样调用原 builder；它不是另一套 metadata 算法。 |
| `_model_forward` | `model_forward` | 先调用原 forward 提交设备工作；随后等输入 DMA 退役，再执行并清空 `pending` 回调。 |

源码集中在 [`__init__.py`](./__init__.py)。建议先读 `stable_verification`，
再读 `correct_bounds`，最后联看 `update_states` 与 `model_forward`；这三部分
分别回答“允许谁”“少等什么”“剩下的工作何时完成”。

## 4. 为什么 CPU 上界可以代替精确长度？

对已验证的 DSACP decode 路径，必须区分：

| 信息 | 责任 |
|---|---|
| CPU 长度最大值 | 为 SAS / QLI 生成保守的 tiling / 工作划分范围 |
| device 精确进度、长度和 positions | 决定有效计算范围、RoPE、压缩和 KV 实际寻址 |

例如某轮 CPU 乐观长度为 106，而 device 精确长度为 103，按 106 规划可以有
余量，但实际访问仍按 device 的 103 执行。不是把额外三行当成有效 tokens。
**上界不能低估实际长度；也不能把上界当成精确寻址信息。**

`correct_bounds` 会检查上界为正且不超过 `max_model_len`，不满足就回退原修正。
生产中不逐波 D2H 比较上界和精确值；上界不低估的契约通过受限准入及 shadow
验收保障。这不能推广成“任何 backend 都能使用过期 CPU 状态”。

与 [`qli_cpu`](../qli_cpu/README.md) 的关系是：这里允许 CPU 使用规划上界，
那里直接复用 CPU 副本以避免从 device 取两个最大值。源码没有相互 import，
但组合正确性依赖同一条已验证的 CPU 规划 / device 实际访问契约。

## 5. 不能省掉的等待：CPU H2D 源缓冲的所有权

`model_forward` 提交 target 后，仍执行：

```python
event = self.runner.prepare_inputs_event
event.synchronize()
callback()
```

原因不是 target 必须算完，而是 **本轮输入 DMA 必须读完 CPU 源缓冲**。
原回调可能修改刚用于异步 H2D 的 CPU 状态 tensors；提前修改会让设备读到
混杂的新旧内容。设备 stream 上排一个 wait，并不能阻止 CPU 提前覆写源内存。

因此我们保留原生输入准备保护，也保留这次 host event 等待，只把回调放在
forward 提交之后。Graph 输入的设备更新和 replay 仍沿原有 stream 顺序执行，
不新增 copy stream，不提前释放 KV，也不宣称整个 host 路径已无等待。

## 6. 准入范围与回退

每波必须满足 `stable_verification`：

- 1–4 个请求，请求 ID 和排列与上一轮一致，没有新请求。
- 每请求本波 6 个 target queries，安排了 5 个 draft slots。
- prompt 已处理完，调度请求集合与当前 batch 一致。
- 已有上一轮 device 有效 token 计数及 draft 数据。

安装还限定 DSV4、async K5、DSACP builders、无独立 DCP，并排除当前未经验证的
多模态输入、prompt embeds 等配置。不是仅凭“使用 compressed KV”就能放行。
Prefill、mixed、请求增减 / 换序等不满足准入的波次保留原有处理。
`all_modes=True` 明确拒绝，不能用它重新开启已舍弃的 all-mode N+2 实验。

## 7. 效益怎么体现，正确性怎么验？

应比较 **相同活跃请求数、相同 K5 query 工作量的波次间隔**，而不是把输出轨迹
不同的整批完成时间差，全部归因于这个补丁。收益应主要出现在跨步提交间隔，
不是把神经网络算子本身说成更快了。

历史 run031 使用真权重、TP8、普通 HCCL，在同一已加载引擎上交替 OFF / ON，
两组都保留 target / draft FULL、ordered replay 和 CPU QLI。测量前已预热
全部 1–4 请求 draft banks。四请求 K5 波次的 rank-0 中位数如下：

| 对照轮次 | OFF 周期 | ON 周期 | OFF draft 结束到下次 target | ON 对应间隔 |
|---|---:|---:|---:|---:|
| 1 | 51.810 ms | 46.522 ms | 6.452 ms | 1.574 ms |
| 2 | 52.322 ms | 46.231 ms | 6.963 ms | 1.298 ms |
| 3 | 51.672 ms | 46.145 ms | 6.386 ms | 1.288 ms |

匹配波次周期缩短 **10.2–11.6%**。这些跨步间隔包含实际 metadata / copy 等工作，
不能全部当作设备空闲。后移的等待与 device 执行重叠，也不能把 host 等待时长
直接加到设备周期上。短 cohort 的波次数和投机轨迹仍有变化，这不是生产吞吐或
高并发 SLA 的证明，也不是本次文档修改重新跑出的结果。

正确性参考不是“使用同一份新 metadata 连跑两遍”：shadow 会恢复原生同步得到的
精确 CPU 长度，检查其与 device 长度相等且不超过候选上界，再用原 builder
重建参考 metadata，从相同 KV / MTP 状态比较输出和状态。run029 真权重验收中，
各 rank 通过 17 次精确 metadata shadow，观察到 CPU 上界余量 0–5 tokens，
输出 / MTP 差异为零，所比较的唯一 KV backing 字节一致。

- 详细配置、数值及限制：[`CROSS_STEP.md`](../../../../prototypes/full-mixed/CROSS_STEP.md)。
- CPU 合约测试：[`test_cross_step.py`](../../../../tests/test_cross_step.py)，覆盖
  稳态准入、回退、上界 shadow 和 `forward → DMA 退役 → callback` 顺序。
- `reference_begin` / `reference_end` 是 shadow 专用辅助方法，不由生产 hook
  自动调用；`rows`、计数器和 `receipt()` 是诊断信息，不是另一套设备状态。

## 安装边界

同一 wheel 内的独立模块，不是新的 pip 包。import 不安装 hook；worker 显式
安装并统一检查配置。安装时全 device synchronize 一次，不是逐波次同步。
源码 / hook 独立不代表任意组合都已验收；更换组合应停服重启，不在活跃 graph
上热卸载。这里没有重新授权旧实验文档中的历史启动选项。
