# 持续 H2D / D2H：算子几乎不慢，首层 metadata 就绪明显推迟

## 实验边界

2026-09-15 本机物理卡 6/7（逻辑 rank 0/1），Qwen3-30B-A3B 真 BF16，
DP2/TP1/EP2，每 rank 一条 4096-token prefill，48 层 native FULL。
复用同状态模型输出和全部 KV backing-byte oracle；48 项测量/rank 全过，
输出全部 exact。不是在线 scheduler 或新的 serving 实现。

`DMA_SUSTAINED=host`：两卡同时分别测 H2D、D2H，每卡 4/8 次 4GiB 连续
拷贝，共 16/32GiB 流量，使用同一块 pinned host buffer 和稳定 device
backing。没有同时测双向传输，也没有 RDMA/peer D2D。每配置三个交替顺序
trials，包含 compute-only、copy-only、serial、overlap。

选定卡的完整租约、admission、source、日志、release 都在：
`runs/qwen-dma-local-20260915/host-sustained/`。没有使用 hw3 或改动其他任务。
卡号与上一轮 D2D 不同；本轮控制组和实验组同卡，不作跨卡绝对性能归因。

## 无 profiler 的实际延迟

|方向|每卡流量|baseline compute|overlap compute|compute 增长|
|---|---:|---:|---:|---:|
|H2D|16GiB|286.83ms|313.55ms|9.32%|
|H2D|32GiB|286.70ms|312.39ms|8.96%|
|D2H|16GiB|287.15ms|312.92ms|8.97%|
|D2H|32GiB|286.75ms|312.97ms|9.15%|

是每 trial 先取两 rank 较长 event span 再取三次中位数，不是跨 rank
绝对时间跨度。32GiB copy-only 较慢 rank 约 1541/1694ms（H2D/D2H），
远长于 compute，不能把最终 compute+copy 的 1594/1679ms 当作 prefill
时延，也不能声称这 32GiB 全部被 313ms prefill 隐藏。

## 单独 profile 的同算子对照

额外采集无 DMA、32GiB H2D、32GiB D2H 三个 forward。脚本核对 op、
shape、dtype、stream、task type 和次数完全一致；没有重复 native task ID。
下表是 rank0 的 48 层累计算子时长（ms），不是独占 critical-path 时间。

|算子|baseline|H2D|D2H|
|---|---:|---:|---:|
|FusedInferAttentionScore|51.742|51.767|51.817|
|GroupedMatmul|46.039|46.169|46.127|
|MatMulV3（QKV/O）|25.785|25.870|25.868|
|SwiGlu|13.064|13.246|13.179|
|MoeInitRoutingV3|13.834|13.784|13.767|
|AddRmsNormBias|8.872|8.890|8.895|
|MoeTokenUnpermute|5.475|5.518|5.512|
|ReduceScatter|53.094|53.140|53.168|
|AllGather|46.341|45.886|47.127|

rank1 计算核趋势相同。例外是 H2D 下 rank1 AllGather 累计
46.028→50.478ms，多 4.45ms；collective task 包含等待，不能解释成
AllGather 数据搬运本身慢了 9.7%。小于 1% 的变化只有一对 profile，
不作稳定加速/回归声明。

## 约 26ms 去哪里了

rank0 模型 task 的首尾跨度 291.06→316.41/315.56ms，区间并集却只有
285.98→286.01/287.19ms。未被这些模型 task 覆盖的空隙从
5.08→30.40/28.37ms。DMA 本身仍可能在这些空隙运行，不能叫整机空闲。

最大单段空隙都是首层 ScatterPaKvCache → FusedInferAttentionScore：

- 无 DMA：0.272ms；首个 metadata stream41 copy 在 scope +1.748ms。
- H2D：25.664ms（+2.379 到 +28.043ms）；该 copy 在 +28.023ms。
- D2H：23.650ms（+2.380 到 +26.030ms）；该 copy 在 +26.006ms。

这些 metadata copy 自身仍约 3us。rank1 同样出现 20.83/23.84ms 的
首层空隙，其余部分有 collective 等待补齐。8 次显式 aclrtMemcpyAsync
主机调用每次约 0.08–0.12ms，不是 25ms 的单次 host API 阻塞。

native FULL 依赖 attention parameter-update / event 发布，这是已知
协议。观察支持“首个 metadata 就绪延迟”，但不唯一确定延迟出在运行时
发射、DMA 准备、共享传输队列还是其他资源；未抓 event 生产消费句柄或
硬件 counters。不要把邻接当作因果，更不要把 9% 全算成 HBM 带宽争用。

后台 stream36 有 512 个 memcpy task（32GiB / 64MiB 的分片数量），持续
跨越后续模型执行。它们在首层等待末尾附近才开始被 profiler 记录为
实际执行，因此这不是“从第一微秒就完全覆盖 compute”的声明；足以观察
后续绝大部分 forward 与持续 DMA 的竞争。

## 产物与复现

- `host-sustained-result.json`：三轮未 profile 的完整数据。
- `h2d_sustained-operators.json`、`d2h_sustained-operators.json`：逐算子和形状。
- `operator_interference.py --phase h2d_sustained`（或 d2h_sustained）：CPU-only 分析。
- 原始 profile / native SQLite 和 TraceLoom derived DB 在 capsule 的 measurements。
- 两 rank 对齐 timeline：`measurements/analysis/qwen-dma-dp2-end-aligned.json.gz`，
  920237 bytes、26554 events。TraceLoom37323af，唯一 collective 身份匹配，
  display-only affine fit holdout P95 0.693us，drift 0.100ppm；非物理时钟证明。

旧 parser 写死逻辑卡等于物理卡，在本轮解析完成后触发检查失败。
复核 RANK_DEVICE_MAP 分别为 (0,6)/(1,7)、SQLite quick_check=ok 后直接使用
已完成 DB；没有重新采样、修改 provider 数据或隐藏采集失败。

结论：持续主存 DMA 下大算子基本不受影响，主要代价是启动处的协议等待。
这是支持异步 KV 预热的证据，不是把整个主存容量都免费搬进来，也尚未
证明通过更好编排就一定能消掉这段等待。
