# Qwen prefill 1024 → 1536：新增成本归因

2026-09-19，产品98f5071；hw3 6/7，原step矩阵同一candidate配置：TP2、no-MTP、
APC/AIV on、6GiB KV/rank、8seats、2048budget、原26张双bank图。未修改产品或原生库。

## 采样契约

独立capsule `runs/qwen38-tp2-serving/prefill-delta1`。复用step-matrix1冻结产品与
原prompt种子，warmup1024/2048后各四次无profiler计时。1536是2048prompt的第一段，
不是换成一个独立1536prompt；两种实际query均冷prefix0。分别保留prepare/forward/
after-forward。无profiler rank0 period238.908→337.457ms，forward232.740→331.216ms；
rank1 period238.730→337.432ms。与此前完整矩阵的约98.33ms增量相符。

随后只采六个step，每次请求前清cache，输出预算1。请求顺序1024/2048/2048/1024；
两rank实际dispatch均为 `[1024],[1536],[512],[1536],[512],[1024]`，FULL。512是续段，
不进入冷prefill差额。每rank两个1024和两个1536body，ABBA形状顺序；不是另一次服务
benchmark或新的native对照。两rank每body的304MatMul、16FIA、128内部communication
语义检查通过。TraceLoom由复制的native raw PROF export构建，不使用丢失graph信息的
简化导出。硬件admission exit0，卡已回收。

## 图内增量（rank0，两个body均值；rank1结论一致）

| 类别 | 1024 (ms) | 1536 (ms) | 增量 (ms) |
|---|---:|---:|---:|
| MatMulV2 + V3 | 93.591 | 138.508 | 44.917 |
| AIV AllReduce | 74.473 | 111.328 | 36.855 |
| GDN核心算子 | 39.277 | 50.803 | 11.526 |
| FIA | 1.593 | 2.239 | 0.646 |
| 布局算子（跨模型） | 9.863 | 10.533 | 0.670 |
| 其他计算 | 11.610 | 15.269 | 3.659 |
| 上述计算未覆盖区间 | 4.153 | 4.301 | 0.148 |
| **完整body** | **234.560** | **332.981** | **98.422** |

类别为device区间union。本次类别间无实质重叠，以上可加闭合；不是任意profile都能
把kernel duration相加。AllReduce耗时包含peer等待，不等于纯传输或可回收时间；
未覆盖区间也不能自动称为idle。GDN区域（conv开始到output-projection开始）另为
48.921→61.016ms，其中包含部分布局，**不能再加到上表**。

MatMul与AllReduce解释约83%的增量。GDN H状态传播kernel为5.385→7.566ms，+2.181ms；
不能用它解释整段约98ms。GDN内部较大增量还有WY recompute+3.466ms、triangle merge
+1.884ms。FIA在这个短输入范围不是主要增长源，不据此推广到任意长上下文。

## 值得下一步隔离的具体落点

64层MLP gate/up projection：`[N,5120] @ [17408,5120]^T`，BF16，ND/ND。
1024调用`aclnnMatmul_MatMulCommon_MatMulV2`；1536调用
`aclnnMatmul_MatMulV3Common_MatMulV3`；均blockNum24/mixBlockNum0，transpose_x2=1。
两个尺寸的完整64次调用分别42.045→69.680ms，**+27.635ms（1.657x）**，是MatMul
增量的约61.5%。不能分别拿MatMulV3的+86.9ms当增量，因为一组V2工作换了V3名称。

按固定数学FLOPs/观测kernel时间计算，gate/up约277.85→251.48TFLOP/s/rank，下降9.5%；
不是硬件利用率计数器，也没有宣称达到峰值。若仅以1024的1.5倍时间作参照，1536多出
6.612ms/64层；这只是候选优化量级，不是已验证可回收收益。其余主要MatMul组约1.31–
1.36倍，并非所有矩阵乘都等比例变慢。

源路径：Qwen3_5DecoderLayer选择Qwen3NextMLP（Qwen2MoeMLP别名），gate_up_proj是
MergedColumnParallelLinear；AscendUnquantizedLinearMethod.apply进入
`torch.ops.vllm.unquantized_gemm`，再到`torch.nn.functional.linear`。BetterScale没有
给这组GEMM加专用替代。profile确认CANN路径变化，但**尚未证明V3选路或某个tiling是
性能下降的唯一原因**。下一步适合做该单一GEMM的形状邻域／替代分块隔离，而不是
直接改GDN recurrence或宣称prefill算法复杂度不佳。

128个内部AllReduce的逻辑payload每次10→15MiB，耗时约1.495倍。按逻辑payload除以
kernel时间约16.8GiB/s两边接近；这不是物理链路实测带宽，不能据此认定HCCS已饱和。
大消息协议或与计算重叠仍是另一独立候选，不能沿用小消息AIV收益推断。

## 复用

- `step_probe.py` 的 `STEP_PREFILL_DELTA=1`：只改变诊断workload，旧默认路径不变。
- `prefill_delta_compare.py`：一轮candidate加载；原子subset admission由capsule launcher持有。
- `profile_steps.py`：native msprof → TraceLoom →六步语义验收与Perfetto。
- `prefill_delta_analysis.py`：按实际dispatch配对、类别union、48个GDN区域与MatMul形状账。
- 完整逐step结果：capsule下`delta-analysis.json`；紧凑结果：
  `docs/evidence/qwen-prefill-delta.json`。
- 两rank可读timeline：capsule的`candidate/traceloom/candidate-rank{0,1}.perfetto.json.gz`。

这是成本归因与下一处候选定位，没有发布运行时优化，也没有网站数字更新。
