# TP／DP 的状态预取候选窗口

这是已有 timeline 的离线计账，不是新增 DMA 集成验收。
**能看到计算时间，不等于已经证明它是免费传输时间。**

## 口径

读取 TraceLoom 保留的原生 TASK / COMPUTE_TASK_INFO，全部8 ranks 独立计算。
不使用显示用跨 rank 时钟拟合，不把别的 rank 时间累加到本 rank。
按模型ID与真实GEMM形状区分 target prefill/decode；每个 target 主体包含
43个MoE层、86对HcPre/HcPost，作为分层边界。排除embedding、LM head、
draft、host间隙及首层启动搬运。数值先取每rank波次中位数，再取rank中位数。

- **较干净窗口**：普通、非Grouped GEMM区间的并集，减去所有其他计算任务、
  现有设备hcom任务与MEMCPY_ASYNC区间。不是PMU认证的独占资源，仍是候选。
- **扩大候选窗口**：包括Grouped/fused MoE GEMM的时间并集，减去已有hcom
  设备任务与MEMCPY_ASYNC。可能与Vector核重叠；尤其MoE竞争尚未实测。
- 不相加重叠核；不把collective等待或host气泡算为可用窗口。
- 带宽折算使用前一单卡实验的约22GB/s（十进制），即20.98MiB/ms。
  **本机单卡带宽 × 历史hw3窗口是规划估计，不是同机8卡带宽验收。**

## 每rank结果

| 路线 | target主体 ms | 较干净窗口 ms | 扩大候选 ms | 较干净窗口折算 MiB/波 | 平均 MiB/层：较干净／扩大 |
|---|---:|---:|---:|---:|---:|
| TP prefill |195.7|25.45|61.28|534|12.4 / 29.9|
| DP prefill |416.8|36.32|126.34|762|17.7 / 61.6|
| TP decode |48.47|7.73|14.30|162|3.8 / 7.0|
| DP decode |51.75|9.69|21.91|203|4.7 / 10.7|

平均每层字节数只用于稳态的初步预算。不同层的状态规模不同，实际必须
按层deadline与buffer复用约束排程；不能把某层的过量需求自动摊给全模型。
较干净列也不是保证值：小DMA的固定延迟、分片、依赖事件与内存竞争未扣除。

如果按整个target主体平均，较干净列分别相当于约2.86、1.92、3.51、
4.12GB/s的附加数据服务量，而不是始终有22GB/s空闲。扩大候选对应约
6.89、6.67、6.49、9.31GB/s；它们需要更多竞争实验证明。

### 波次范围不是相同吞吐配置

- TP prefill / decode：`runs/hw3-split-046/analysis/rank{0..7}.db`。
  prefill捕获桶4128，Q-A/Q-B每rank516行，3波；decode target24行，
  四个K5席位，5波。分别model49/48，均不含draft。
- DP prefill：`/workspace/strengthen-dsv4-dp-full/runs/hw3-dp8-065/engine/profileskew/analysis/`。
  TP1/DP8/EP8，固定1026行/rank，8波倾斜输入；含padding和既有负载不均。
  **这是旧固定桶的运行时间，不能把浪费的计算永久当作免费带宽资产。**
- DP decode：`runs/hw3-dp8-052/profiledecode/analysis/`。
  native保留窗口，选择model48、每rank6个target query（一席K5），50波。
  其他12行桶、eager prefill与draft均排除；不是最新发布版性能声明。

这些窗口不是TP vs DP的公平吞吐竞赛，只回答各自已有执行轨迹的预取空间。
DP prefill尤其不均：各rank扩大候选窗口中位数约118–216ms，不能仅看最宽
的rank或把rank间时间相加。较干净窗口在35.7–37.0ms，稳定得多。

## 连续窗口比累计时间更苛刻

每层最长的“普通GEMM且不与已有传输重叠”片段，中位数：
TP prefill约139us；DP prefill约1.02ms；TP decode约50us；DP decode约69us。
这一口径仍允许Vector并行，不是上表的“较干净”过滤。
按22GB/s，分别约2.9、21.4、1.1、1.4MiB；没有扣小DMA启动延迟。

因此上表534MiB/波不等于可以在某处发一笔534MiB DMA：要么做分块预取，
要么允许DMA穿过其他算子并实测干扰。事件不会自动暂停已经提交的大搬运。
跨层lookahead能扩大deadline、吸收不均衡，但不能创造持续带宽。

## 结论与下一步选择

1. Prefill值得验证分层双缓冲。先把每rank每层十余MiB作为候选预算量级，
   而不是把整个forward几百ms当作空闲DMA时间；真正可接纳量按层实测。
2. Decode并非无空间，但应优先稀疏命中和小块预取。当前16/64MiB实验不证明
   50–70us窗口内能达到相同带宽，小传输与小GEMM组合需要另测。
3. MoE窗口能显著扩张预算，但不能直接沿用普通GEMM的低干扰结论。
4. H2D、D2H的单独成绩不能相加为全双工保证；8卡共同读主存、NUMA与HCCS
   共存也未验证。RDMA尚未进入这份预算。
5. 新增当前层KV需等生产完成；提前跨层搬运的对象首先应是已发布的旧前缀。

复现：`python3 prototypes/graph-dma/timeline_windows.py`。
完整rank/波次/层分位数与源路径：`timeline-windows.json`。只读取DB，不改源
数据库，不重跑NPU或导入大体积timeline JSON。
