# 原生 host FD 计划进入 wave attention metadata

实现入口：`host_metadata.cpp`、`../serving/host_attention.py`。
保留已安装 CANN9.0.1 的 FIA host planner 与数值 kernel，不移植新版 upstream
heuristic、不安装/替换算子、不恢复逐层 graph task update。

## 协议

每个 wave/rank 根据本次精确 query/KV 长度，调用一次原生
`aclnnFusedInferAttentionScoreGetWorkspaceSize` + execute host 路径。
隔离 preload adapter 截获并**抑制**该次目标 numerical launch，只取得原生
kernel variant、block 数和 opaque tiling；不运行额外一遍 attention。
直接调用 ACLNN，绕过 Torch task queue；不需要 drain 之前的 compute stream。

native planner 决定 FD/non-FD、原始分段边界、core/split/归约 scratch 布局。
每个 wave 只导出一份2528B计划，和 ingress 一起 H2D；所有48层共享。
GM tiling 地址替代原来的 inline relocation。长度仍由 graph 内 State 写到GM。
不同 FD/non-FD 预捕获 entry、不同奇偶 bank，各自持有稳定元数据地址。
N+2 reactor 的 old.done reader fence 保护 bank覆盖，Ticket 保留 pinned源直到退休。

### 固定 launch 的小适配

原生 FD 的活跃核数会变化（实测16、19、22、23、24等）。不为每种核数复制
整模型图：capture 固定24 blocks，只给新增的空核写 `startBIdx=1/endBIdx=0`，
使其主循环不执行。**原来的有效核分段、splitInfo、归约数值工作不变。**
原生 CombineScale 用实际 launch block数作归约遍历步长，所有核参加 SyncAll。
这不是完全相同的原生 block launch 数；额外空核仍有初始化/同步开销。

适配严格限制到已验证 BF16/head128/paged128/causal、910B2、原生两个精确
500…/510… binary符号。安装包 coreInfo 的起点200、26元素数组、payload2528，
以及24核 base scratch 四段的准确容量均检查；不能拿较新源码结构体直接覆盖。
FD split scratch 加入容量检查，共享128MiB只适用于串行 compute stream。

### capture 与实际请求不是同一次执行

FD entry 的 warmup 使用原生短序列 non-FD plan，符合空 arena 的状态；
长序列 FD kernel/GM plan 仅绑定到 capture，实际第一次 replay 前必须填本 wave
计划。Capture 不提交长请求，也不能通过图内写死8192长度来“修复”warmup。
LiveModule 原有 State snapshot/restore、runner fencing 不变。

### N+2 终止 drain 是 metadata 的真实边界

已发出的下一 wave 可能在前一 wave终止后成为无写入 drain。
此时 graph 的 inactive row 有效 KV长度是1；host planner 也必须看到1，
不能继续使用已完成请求的长上下文长度。固定 ignore_eos/output budget 下，
scheduler 的 projected cursor 足以提前准确识别它，不需要 sampled token回传。
CPU数值状态 double 检查每个 decode row 的预计长度与实际 active/drain 状态。
这不自动扩展到 EOS/投机接受等尚未支持的动态停止条件。

## 有界叶子证据

- `fia-host-metadata1`：原生核数不改，两bank变化 GM tiling，4row decode至32K，
  FD/non-FD切换/混合长度/页面置换；20检查，max_error0，exit0/release。
- `fia-host-metadata2`：同样20检查，启用空核 padding固定24，max_error0，exit0。
- `fia-host-prefill1`：单row q1/2/4/8/16/128，各20检查max_error0；多进程轮换
  的后续阶段被 occupancy watchdog中止（driver PID无容器PID），整个capsule
  **不是PASS**，不能把已有单项正确性当完整成功。没有杀外部进程。
- `fia-host-final-leaf1`：新guard版本、单进程持有lease，decode4row/q8/q1024，
  共60检查、双bank、短长混合/32K、所有max_error0；exit0/release。
- `swe-host-dummy1`：两dummy层 TP2/EP2，32graphs、41waves，8calls/60outputs，
  protocol/TP一致、runner calls空，host planner41次；此短fixture没有FD请求，
  不能作为整模型FD执行的证据。

全模型性能与短profile另行记录；叶子数值相同不等于完整模型生成序列相同。

## 开关

`serving/run.sh` 默认 `OWNED_STATIC_FIA=1 OWNED_HOST_FIA=1`。
`OWNED_HOST_FIA=0` 回到上一个静态non-FD candidate；
`OWNED_STATIC_FIA=0` 回到更早的逐层host-update owned路径。
仅 owned prototype，不改已发布DSV4 Worker/PyPI。

## Capture pool 的资源边界

`swe-host-candidate1` 在36-entry整模型 capture 的 decode阶段失败：
`aclrtReserveMemAddress` /207001，未进入计时。新增FD变体的独立 expandable
allocation pools使地址预留扩大；不把此失败伪装成模型性能或HBM峰值证据。

修复复用未修改 LiveInference 的 `graph_pool_key` 合同：同bank、同形状的
FD/non-FD entry共用一个临时pool（仍24个pool），不跨bank/形状合并。
跨调用值留在 State/帧元数据/ingress，临时张量不跨wave存活。
各轮 reactor 复用同一 compute stream，满足 runtime 的串行pool校验；
H2D/D2H bank reader/retirement fences不变。没有关闭 expandable allocator
或更改安装环境来绕过失败。

## 完整30B candidate-only结果

`swe-host-candidate2` 两rank PASS、exit0/release，真实48层 Qwen3-30B-A3B，
TP2/EP2、DP1、device4/5、原四会话44calls/12,111输出token，配置/trace与历史baseline
完全一致。没有重跑native。默认host metadata和serial variant pools都已启用。

| 指标 | 上版 frozen non-FD | 新host计划 | 历史native |
|---|---:|---:|---:|
| 冷轮秒 | 106.805 | 100.820 | 94.413 |
| 热轮1秒 | 95.839 | 89.496 | — |
| 热轮2秒 | 95.814 | 89.439 | — |
| 热轮均值秒 | 95.827 | **89.468** | 87.561 |

相对上版热轮缩短6.64%；仍比历史native慢2.18%，**不是已超过baseline**。
历史旧host-update owned热轮89.809s，与本版89.468s相近；不能把0.38%差异当稳健优势。
历史对照不是同时A/B；各轮token/MoE路由可能不同，不宣称质量等价。
新冷轮3996waves/APC545152；热轮各3843waves/APC625664。

36graphs/24共享变体pool，activation94.831s另计。每rank36templates、1728绑定，
3456个Python attention launch全部在warmup/capture内；runner calls为空。
3轮合计11,682waves，对应**11,682份原生host计划**：6,939non-FD，4,743FD；
原生FD核数覆盖14、17–24，由wrapper补空核为固定24。两rank计数一致。
变长prefill、FD/non-FD切换、终止drain、原始多会话轨迹与热APC在整体协议中通过。
原始receipt和同config/trace/完整请求计数/TP一致性检查：
`swe-host-candidate2/reused-baseline-comparison.json`。

## 四步 TraceLoom 审核

`swe-host-matched-profile1`，与上一版相同原始首次prompt、同40步预热后只采
step40–43；输出预算16仅用于有界结束。36-entry完整图目录，非缩减形状替代。
两rank的官方provider DB均已离线解析，经TraceLoom37323af分析/导出。

每rank **4次replay、192个图内FD FIA、4个图内ArgMax**；
`aclmdlRICaptureTaskUpdateBegin/End`均0。FIA host execute/GetWorkspaceSize/tiling
各4次，恰好每wave一次，**不再宣称host FIA API调用为0**：这些是被抑制数值
launch的host计划提取。所有实际FIA device任务都在graph中；
两rank整个窗口的**图外compute task总数均为0**，不只是attention为0。

两rank平均每步：

| 项 | frozen non-FD | 新host计划 |
|---|---:|---:|
| 48层attention kernel合计 | 3.3852ms | **2.9977ms** |
| attention每层 | 70.52us | **62.45us** |
| 整个device计算体跨度 | 20.8898ms | 20.4244ms |
| expert GMM合计 | 4.9355ms | 4.7664ms |
| dense matmul合计 | 2.0601ms | 2.0711ms |
| collective envelope并集 | 4.1147ms | 4.1985ms |
| compute/collective未覆盖时间 | 2.9636ms | 2.9600ms |

此窗口attention设备时间降约11.4%，但仍高于历史旧host-update owned的
2.8100ms/58.54us每层；没有宣称达到相同kernel时延。原生FD有效核23，本版
launch24（含一个空核），GM tiling载入/空核同步等差异尚未做独立消融。
四步窗口也不量化全部长轨迹6.64%改善的因果份额，尤其生成路由并非完全相同。

原生host计划每wave的GetWorkspaceSize平均约191.7us、execute外壳68.5us；
tiling85.6us是嵌套部分，不能相加重复计费。这是profile期间的host API时间，
不是未profile serving的独立开销或未被compute覆盖的墙钟时间。

可复查产物（均在`runs/owned-wave/swe-host-matched-profile1/traceloom/`）：
- `static-fia-audit.json`：保留旧工具输出名，`protocol=native-host-wave`，执行新计数门禁。
- `static-vs-host-rank{0,1}.json`：对比上版静态non-FD的同四步。
- `matched-rank{0,1}.json`：对比历史逐层host-update owned的同四步。
- `owned-rank{0,1}.db`、`owned-rank{0,1}.perfetto.json.gz`。

复用原有native基线，没有再次采baseline。未覆盖时间不是所有引擎空闲；
compute求和与collective可能重叠，不能相加当墙钟。TraceLoom精确Ascend replay
partition仍不支持，沿用provider model-ID连续体/每体48FIA+1sampler断言。

后续向上选桶、真实 query metadata 与新 C ABI 见
[`CEILING-PREFILL.zh-CN.md`](CEILING-PREFILL.zh-CN.md)。上面性能表保留旧 exact-tail
协议的历史结果，不应当当作新 ceiling candidate 的测量。
