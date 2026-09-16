> 本文分析的是58f6c0c的 exact-floor candidate。其尾巴问题已由后续
> [ceiling-prefill实现与复测](CEILING-PREFILL.zh-CN.md)接续；旧profile不是新调度的profile。

# FD 修复后的热点与剩余差距

2026-09-16，复用 `swe-host-matched-profile1/traceloom/owned-rank{0,1}.db`；
没有新跑模型或baseline。当前热轮89.468s，历史native87.561s，差2.18%。
本次把**绝对热点**与**比baseline多做的工作**分开，不把四个冷启动后decode
step解释成整条热轨迹的精确因果分解。

复算：`python3 prototypes/owned-wave/serving/inspect_hotspots.py
runs/owned-wave/swe-host-matched-profile1/traceloom`，产物`hotspots.json`。
读取的是TraceLoom augmented DB保留的官方provider表；有序graph body/API关联
不是尚未支持的Ascend exact replay partition。每体断言48FIA+1ArgMax。

## Host暂时不是这个窗口的供给瓶颈

后3个step的replay API结束均领先device body开始15.58–16.34ms。
其host FIA计划API区间全部落在前一个device body内（同rank provider时标）。
首step受profile启动前drain影响，不拿它代表稳态。
去掉第一个边界，图间compute body间隙约62–63us；四步全窗口平均约88us。
这支持N+2供给充足，而不是“还有0.26ms host planner就一定该优先设备化”。
只是此窗口观察，不证明整条多会话请求都无host starvation。

## 绝对热点：GMM、attention，以及通信边界

两rank平均，每个约20.424ms的device计算体：

| 项 | 次数/step | kernel合计或envelope |
|---|---:|---:|
| Expert GroupedMatmul | 96 | 4.766ms |
| FIA | 48 | 2.998ms |
| Dense MatMul（含lm_head） | 145 | 2.071ms |
| AllReduce | 97 | 与下行一起：通信envelope并集4.199ms |
| AllGather | 1 | 上项包含 |
| compute/通信未覆盖 | — | 2.960ms |

kernel求和与通信可能重叠，不能将此表相加当critical path，也不能把通信
API envelope视为纯链路传输。GMM是最大数值kernel热点，但不是已证明的新增回退。

### 约80%的未覆盖时间集中在两类层间通信衔接

| compute边界 | 次数/step | 两compute间总gap | 扣除通信envelope后 |
|---|---:|---:|---:|
| O投影MatMul → AddRmsNormBias | 48 | 2.985ms | 1.203ms |
| MoeTokenUnpermute → AddRmsNormBias | 48 | 3.468ms | 1.176ms |

两处剩余合计2.379ms，占2.960ms的80.4%。它们映射到TP输出all-reduce及MoE输出
归约的跨stream交接，并非重新出现的逐层Python launch。

一个rank0原始区间以MatMul结束为0：下一RMSNorm在61.44us开始；
通信op envelope为14.02–50.00us，但其AI_CORE控制任务覆盖12.62–51.16us。
主stream120还有CAPTURE_RECORD/WAIT，通信stream119也有CAPTURE_RECORD/WAIT，
其中两边可见约5us的CAPTURE_WAIT。SDMA/Write Value/Notify Wait在通信内部。
因此“envelope以外25.46us”**不是25.46us空闲或可无损删除的开销**。
没有event handles/完整dependency因果边，暂不把某条WAIT绑定到具体生产者。

同一窗口的历史旧owned两边界未覆盖合计2.496ms；本版反而降到2.379ms。
所以它是值得研究的共同成本，不是解释本版2.18%总差距的已证据化新增项。
下一步若优化它，先区分小消息collective本体、graph控制任务、跨stream交接；
不能因看到缝就重新去删host attention更新——该更新已经不存在。

## 更直接的工作量差异：热APC尾巴被拆得过碎

完整未profile `swe-host-candidate2` 两个热轮，44条request receipt均给出
prompt_tokens/hit_tokens。与scheduler精确二次幂chunk策略交叉计算：

- 剩余prompt总计2698tokens，每请求1–128tokens；APC命中625664。
- 实际需要154个prefill waves，另外3689个decode waves，总计3843。
- 每个prefill仍执行全部48层；70tokens会变成64+4+2的三次整模型执行。
- 首四个请求尾巴70/13/41/30，共154tokens，却分13个prefill waves。
- 若支持一个有真实有效query长度的prefill bucket，每请求一次仅需44次；
  154−44=110次是该改变可消除的**结构性执行次数**，不是native实测wave数。
  mixed packing还会改变decode插队/批组成，不能简单用110×decode时延预测收益。

热轮TTFT中位数约172ms、P95约307ms；历史native中位约88–95ms、P95约114–121ms
（保留原来不同frontend边界的限制）。首四个请求在热轮的命中token完全相同，
但candidate首token约189/220/251/318ms，历史native一轮约51/107/107/106ms。
这是调度组织值得优先处理的证据，不是严格同输入数值轨迹的延迟归因。

冷轮还有另一个独立差异：native首四请求APC命中0/3712/3712/3712，candidate
同时admit时均0，导致两边初始profile的prefill工作不同。禁止直接用native前64步
与candidate四步kernel总量算“算子回退”。本次没有把native长窗口当等工作对照。

## 两个次优先项与投资顺序

1. **优先调查真实query长度的prefill bucket／短尾合批**：减少154次整模型执行，
   同时保留KV slot屏蔽、最后有效token采样、APC与N+2终止/复用合同。
   目前host planner的query offsets仍按固定bucket构造；变长Q还没被资格验证，
   不可只pad ids后继续把padding当有效token写KV。
2. **小成本FIA消融**：当前62.45us/layer，历史旧owned58.54us/layer，差约0.188ms/step。
   同一native split计划做23原生blocks vs24空核padding、inline vsGM tiling对照，
   才能区分空核同步/metadata载入/测量差异。当前只有机制猜测，没有因果结论。
3. **通信边界的共同成本**：有2.38ms的局部研究面，但需先补足runtime交接证据；
   GMM4.77ms是绝对热点，不因此立即重写专家kernel。

Owner控制前后有36+33个小kernel，合计span约0.16ms（后段包含sample dtype转换）。
可考虑融合，但不是毫秒级主项；其中baseline也有自己的GPU输入准备，不能把
candidate全部控制成本都记作净新增。此轮只做分析，未改变执行路径或默认配置。
