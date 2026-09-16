# SWE trace 多会话/APC 执行路径对照（2026-09-16）

## 结论

这是**保留旧逐层host attention更新的历史接管路径**，当时没有性能优势。两个执行顺序均复现：冷缓存整轮慢6.6–7.5%，
保留缓存后慢2.2–3.0%。FULL execute+sampler 确实已在图内，但这不自动等于
更好的批组织、首 token 延迟或吞吐。以下是本机有界实验，不是生产服务评测。

## 实验口径

- 完整48层 BF16 Qwen3-30B-A3B，TP2/EP2、DP1，本机910B2卡4、5。
- 两边4个并发会话、同一6GiB/rank KV容量、APC、FULL图、1024 prefill上限。
  原生 async scheduling 默认保持不变，HCCL_DETERMINISTIC=strict 两边相同。
- NVIDIA Open-SWE-Traces 的4条完整轨迹，共44次调用、12,111输出token/轮。
  每轮 prompt 总量相同，原始历史不替换成生成结果，无截断、工具等待为零。
- 两个新进程分别 native-first / owned-first；每个臂先冷缓存，再保留缓存两轮。
  模型加载、native初始化、owned图激活不计入整轮时间；未开 profiler。
- native计时涵盖客户端LLMEngine交互，owned从worker内部开始计时，输入预装。
  **不是同一前端/HTTP入口比较**。TTFT也分别在客户端输出/worker全rank回执处观测。
- 按Fletcher决定，跨臂token差异保留但不阻断计时；工作量、TP一致性、生命周期
  仍检查。输出质量等价未建立；不能把native自身不确定性当成差异无害的证明。
- 本机与ssh hw2是同一宿主。没有把它们当两台机器并行跑；hw0未部署已知模型/环境。

## 未采样计时

每格为整轮秒数。括号为owned相对native耗时增加。

| 顺序 | 缓存起点 | 原生 | owned |
|---|---|---:|---:|
| native-first | 冷 | 94.521 | 100.780（6.6%） |
| native-first | 保留1 | 87.400 | 89.316（2.2%） |
| native-first | 保留2 | 87.689 | 89.877（2.5%） |
| owned-first | 冷 | 94.305 | 101.367（7.5%） |
| owned-first | 保留1 | 87.515 | 89.758（2.6%） |
| owned-first | 保留2 | 87.642 | 90.284（3.0%） |

聚合每种缓存起点的等长轮次（不是统计置信区间）：

| 缓存起点 | 原生平均秒 / 输出tok/s | owned平均秒 / 输出tok/s |
|---|---:|---:|
| 冷（2轮） | 94.413 / 128.28 | 101.073 / 119.82 |
| 保留（4轮） | 87.561 / 138.31 | 89.809 / 134.85 |

保留轮两边命中625,664 prompt tokens；冷轮native556,416、owned545,152。
相同输入/输出预算不等于相同实际prefill算量或MoE路由。owned冷轮3996waves，
保留轮3843waves。所有候选rank零runner执行调用，24预建图；激活时48次数值
Python调用，replay中没有新增数值Python forward。

各保留轮TTFT p50：native87.7–95.3ms、owned169.6–171.2ms；p95分别
113.6–121.0ms、305.8–308.8ms。观测边界不同，不能冒充相同API的SLA，但
已足以说明尚未取得首token优势。TPOT p50反而略低，不能据此掩盖总时长退步。

当前代码的明确结构差异：owned每个prefill wave只处理一个resident的精确二次幂
chunk，prefill与decode轮流，而非合并mixed batch；decode固定4行，包括inactive。
这使批组织/首token衔接值得先调查；尚未通过消融证明它解释了全部退步。

内存JSON保留PyTorch allocated/reserved峰值，但不是驱动总HBM。两臂同进程，
owned激活后资源留存影响后跑native（观测峰值约40.89GiB，先跑native约34.50GiB）；
因此不能用本实验声称独立部署内存差。owned图激活约51–65s，单独记录，不混入吞吐。

## CANN/msprof 采集及初步观察

单独的swe-profile1完整执行44调用，两个rank、两个臂分别采前64waves。
torch-npu Level1 CPU+NPU采集，保留原始PROF目录；worker不解析，释放卡后使用
官方parser逐rank独立进程导出SQLite与Chrome trace。四个DB均quick_check=ok，
身份rank0→device4、rank1→device5；未制造跨rank时钟对齐。

原生provider中，每个rank：

- native的63个ArgMaxV2 sampler task具有非图modelId=4294967295；owned的64个
  ArgMax task均落在实际图modelId中。这是sampler进入图的设备侧证据。
- 两臂均64次aclmdlRIExecuteAsync，且均3072次FIA调用/任务更新（64×48层）。
  模型体和采样进图**没有消除逐层host attention task-update接口**。
- owned每rank62次aclrtSynchronizeEvent；native rank0有127次Event同步，两个rank
  各64次Stream同步。同步时间包含排队中的有效计算，不等于可删除CPU开销。

前64waves的prompt长度/批组成不同，不能把profile中kernel总时间差当成算子
退步或加速率。当前证据支持“图内采样成立、host metadata接缝仍在”，不支持
把总时长回退精确归因给单一接口。后续应围绕相同工作窗口的批组织和衔接做消融，
而不是据这两个非等工作窗口改写GMM内核。

## 可重用证据

仓库根下（大文件忽略跟踪）：

- `runs/owned-wave/swe-perf-native-first2/comparison.json`
- `runs/owned-wave/swe-perf-owned-first1/comparison.json`
- `runs/owned-wave/swe-profile1/profile-exports.json`：四个原生DB/压缩时间线完整路径。
- `runs/owned-wave/swe-profile1/profile-summary.json`：可由summarize_profiles.py重建的
  原生API、sampler图身份和coverage观察，含解释边界。
- 各capsule保留冻结源、配置、输入、逐请求输出、双rank回执、启动日志及释放记录。
  两组计时及profile均exit0；没有selected-card外来占用。

不计成绩：swe-real1-hw2曾与同宿主profile启动重叠；swe-perf-native-first1在
启动时检测到0、1卡外来占用，监督器只终止自有组。均保留排除记录，没有择优混入。
