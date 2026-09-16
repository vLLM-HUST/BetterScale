# Ceiling candidate：少跑step，不等于单个step的缝更小

2026-09-16，`swe-ceiling-profile1`：真实 Qwen3-30B-A3B TP2/EP2、36-entry完整
目录，本机device4/5，先26wave预热、只采26–29四个decode wave，两rank PASS、
exit0/release。原始四会话首次prompt，output预算16仅用于结束诊断；共39waves。
这不是新一轮完整 SWE 计时，也不是热APC全轨迹profile。没有重跑native。

上次 `swe-host-matched-profile1` 采40–43；ceiling改变了前面的prefill数量，
不能机械沿用40。新window四行KV长度从[4557,4626,4397,4388]递增到
[4560,4629,4400,4391]；旧/新均为四row、约4.5K、FD decode，不是逐token完全
一致的上下文/路由，不能把微小kernel波动解释成代码收益。

## 步间的两种口径

先定义 body 为从本次首个 graph compute task 到最后一个 graph compute task。
这是可核验的provider边界，**不是完整graph执行含控制尾部的边界**。

| 量 | 历史native | 上版host-plan owned | 新ceiling owned |
|---|---:|---:|---:|
| body之间的原始间隔 | rank中位数711–735us | 稳态62.0–63.0us | 全6个间隔61.9–62.9us |
| 扣除compute/communication覆盖后 | rank中位数208–233us | 62.0–63.0us | 61.9–62.9us |

native取原64wave profile中model41的全部56个decode-to-decode间隔，包含其
outliers，不只选择最快四步。该图ID连续复用，不能直接用groupby(modelId)
划step：`inspect_step_gaps.py`用每个model的重复首compute task三元组
(modelId,streamId,taskId)作锚，每体断言48 FIA，并校验body数==replay API数。
这是有界有序关联，不是TraceLoom尚未支持的Ascend exact replay partition。

native的原始间隔包含约452us图外compute、约48us通信：logits投影、ArgMax、
cast、下一轮slot mapping/输入准备等。不应把整个0.7ms叫idle，也不能把
0.7ms到0.062ms全部当作消除了host starvation。前后版本profile并非同时A/B。
旧owned第一个间隔136–140us处于profile启动边界；不拿它代替稳态62–63us。

**结论：相对native，步间拆分/供给成本已显著压缩；相对上版owned，单步缝隙
没有继续缩小。** ceiling主要减少尾巴forward及改变批次组合，不能把其3.63%
完整热轨迹收益归到图间间隔。新稳态body仍约20.4–20.6ms，62us约占0.3%。

## 这62us主要也不是空白：graph控制尾部

新profile两rank、三个边界完全一致地出现390个`MEM_WRITE_VALUE`任务，仍归属
旧graph的modelId/stream。以rank0 step27→28为例：

- 最后compute结束后0.02us，旧graph先做一次0.64us MEMCPY。
- 0.68–58.50us是390个MEM_WRITE_VALUE，任务duration合计49.12us；
  57.82us跨度包含任务间隙，不能混为相同量。
- 之后是NOTIFY_RECORD、执行stream的EVENT_RECORD/WAIT、下一MODEL_EXECUTE。
- D2H stream的MEMCPY在59.50–61.78us；下一graph的起始MEMCPY在62.06–62.88us，
  下一compute在62.90us。不能仅凭时间邻近说D2H因果性阻塞了下一graph。

这把剩余边界指向graph控制尾部，而不是host没提交。**并不知道390次写的地址/
数值或完整依赖关系，不宣称已证明它们都是冗余清零、可安全合并或删除。**
长跨step的EVENT_WAIT/NOTIFY_WAIT仍不能当成整个芯片空闲。
用 `inspect_step_gaps.py ... --controls`复查边界内开始的任务分组；它明确不是
所有已跨入该区间的任务全集，duration裁剪在边界内，不能相加推导多流critical path。

## 调度轨迹：N+2没有供给断档

本次仅profiling路径保存host submit、receive、TP quorum时间与wave元数据。
每条timeline的四次CANN replay API均落在唯一的同sequence submit括号内；
wall/monotonic offset全程漂移范围分别1.63/1.60us。验证通过后才叠加host轨道，
没有静默拟合，也没有跨rank device时钟对齐。

后3个replay在device body开始前15.19–16.14ms已经提交。每wave host FIA计划
约0.21–0.32ms，全部位于上一个body内。Host的receive等待约15ms以及quorum
约1.8–2.0ms，不等于GPU同样空等：下一step已经排队，图间仍只有62us。

初始profile启动和末尾drain单独标记。rank0的step28 quorum约24.95ms发生在
profile stop之后；rank1的profiler stop比rank0晚约22.52ms，随后双方才到quorum。
这是诊断收尾阶段，不能拿来断言正常serving每步有25ms CPU屏障。
录制全程39wave的host调度；device只录四step，不把未录制的warmup画成GPU idle。

## 主体热点仍在

新四步两rank平均：GMM约4.981ms、FIA约2.952ms、dense MatMul约2.052ms。
这些是kernel时间之和，可与通信/其他引擎重叠，不是可加的墙钟分解。
96个层间all-reduce边界附近，compute/comm未覆盖时间仍约2.400ms/step，
上版约2.379ms；没有显示改善，也不能称它全部可移除。它比62us步间尾部更大，
若继续查性能，先厘清这些层内交接的必要依赖更值得，不能只盯host调度。

审计每rank：4replay、192图内FIA、4图内ArgMax、0图外compute、0task update，
原生host FIA execute/GetWorkspaceSize/tiling各4次，仍为每wave一次。

## 复查与下载

所有新产物位于 `runs/owned-wave/swe-ceiling-profile1/traceloom/`：

- `owned-rank{0,1}.perfetto.json.gz`：原样TraceLoom37323af导出。
- `owned-rank{0,1}.with-schedule.perfetto.json.gz`：另存的调度叠加版，不覆盖原版。
- `schedule-rank{0,1}.perfetto.json.gz`：完整39wave host轨迹，四step device envelope。
- `step-gaps-rank{0,1}.json`：compute边界、覆盖差与控制任务；`hotspots.json`。
- `schedule-audit.json`：时间括号/clock稳定性门禁和真实wave长度；`static-fia-audit.json`。

先官方 `parse_profiles.py`，再 `export_traceloom.py`，然后
`inspect_step_gaps.py DB --controls --output FILE`、`inspect_hotspots.py DIRECTORY`、
`export_schedule.py CAPSULE`。后者只消费现有profile，不启动模型。
四份可读调度/timeline另打包为capsule内 `timeline-and-schedule.tar`，可一次下载。
