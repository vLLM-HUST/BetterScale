# 持续 local D2D 把哪些算子拖慢了

对象是 `sustained/measurements/analysis/rank{0,1}.db` 中独立采集的一对
完整 forward：无额外 DMA，以及两卡分别 64×4GiB 的同设备 D2D。
Qwen3-30B-A3B 真权重，DP2/TP1/EP2，每 rank 4096 tokens，48 层。
不是 HCCS peer copy，也不是 H2D；重复地址压力不代表 256GiB 独立 KV。

`operator_interference.py` 按 op、输入形状、dtype、stream、task type 对齐，
要求两阶段各组次数相同，且 native task ID 不重复。输出在
`operator-interference.json`。每 rank 一对 profile，用来归因；正式耗时仍用
三轮无 profiler 的 `sustained-result.json`，不能混用。

## 结果

以下是 rank0 全部 48 层的累计 task 耗时，毫秒。rank1 趋势相同。

| 算子 | 原来 | 持续 DMA | 倍数 |
|---|---:|---:|---:|
| FusedInferAttentionScore | 51.72 | 125.54 | 2.43 |
| GroupedMatmul | 46.07 | 113.96 | 2.47 |
| MatMulV3 | 25.75 | 51.51 | 2.00 |
| hcom_reduceScatter_ | 53.13 | 74.43 | 1.40 |
| SwiGlu | 13.13 | 23.74 | 1.81 |
| MoeInitRoutingV3 | 13.63 | 21.99 | 1.61 |
| AddRmsNormBias | 8.75 | 14.00 | 1.60 |
| MoeTokenUnpermute | 5.56 | 9.59 | 1.72 |
| hcom_allGather_ | 46.70 | 47.17 | 1.01 |

大 GEMM 进一步拆开（按形状解释模型位置）：

- QKV 投影 `[4096,2048] × [5120,2048]`：13.95→28.53ms，2.05×。
- O 投影 `[4096,4096] × [2048,4096]`：11.80→22.99ms，1.95×。
- Expert gate/up，权重 `[64,2048,1536]`：30.47→72.95ms，2.39×。
- Expert down，权重 `[64,768,2048]`：15.60→41.01ms，2.63×。

Attention + expert GEMM + 两个投影的累计增长约 167.47ms。
不能把这些累加值当作严格独占的关键路径贡献：存在并行任务，HCCL task
也可能包含同步等待，并不等于有效通信时间。

## 是否只是 host 没喂饱？

排除额外 memcpy，仅取 COMPUTE_TASK_INFO 对应的模型 tasks（含 HCCL）：
rank0 的首尾跨度 290.99→512.25ms；任务区间并集 286.12→504.96ms。
并集之外的空隙仅 4.87→7.29ms。rank1 的空隙 4.89→7.39ms。
因此约 221ms 的跨度增长主要出现在既有 device task 区间，而不是多出
200ms 的无任务 host 气泡。并集仍包含 task 内部等待，不能用于证明计算
单元一直在忙。

几乎不受影响的有融合 split-QKV/norm/RoPE（6.09→6.18ms）、
MoeGatingTopK（7.79→7.84ms）和 Index（2.98→3.05ms）。
不能简单归结为“所有 vector 算子慢、所有 GEMM 都能免费藏 DMA”。

## 解释边界

观察支持持续 D2D 与模型访存争用共享资源，且影响主要落在 attention、
expert GEMM 和投影；尚无 PMU 证据能区分 HBM 通道、片上互联、缓存或
其他执行资源。AllGather 没明显变慢不代表通信无资源竞争，更不能外推到
其他消息大小和 H2D。这个极端带宽压力试验否定的是“整段 prefill 带宽
都免费”，不否定限量/限速、有 deadline 的 KV 预热。

复现（CPU-only）：

```sh
python3 prototypes/graph-dma/qwen-dp2/operator_interference.py \
  runs/qwen-dma-local-20260914/sustained/measurements/analysis \
  --output prototypes/graph-dma/qwen-dp2/operator-interference.json
```
