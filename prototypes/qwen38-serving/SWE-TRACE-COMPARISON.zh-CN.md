# Qwen27 SWE 多轮 HTTP 回放与短 timeline（2026-09-17）

## 对照结果

候选为已集成的 `betterscale.qwen_worker.MixedWorker`，不是旧的30B owned-wave
执行器。原生对照为固定 donor 的 NPUWorker。两边相同：TP2、无MTP/APC、
8席、2048 token调度预算、8K上下文、6GiB KV。候选使用raw ACL要求的queue0，
原生queue1；两边都挂同一个未激活的诊断观察器，计时期间不开profiler。

hw3同机两对卡同时跑，第一轮baseline2/3、candidate6/7，第二轮交换。
每轮分别以C4/C8回放相同的8条完整会话，每会话最多一个在途请求。
C4表示最多四个会话并行，并非只选四条轨迹；两边每个cohort都是78次调用、
20,648输出token。预热、加载、编译、profile不计入性能统计。

| 最大并发会话 | native tok/s | candidate tok/s | 吞吐变化 | 平均TTFT native→candidate |
|---|---:|---:|---:|---:|
|4|70.316|71.196|+1.25%|1363→1239ms|
|8|97.556|99.860|+2.36%|1526→1395ms|

保留轮次，不只给合并均值：

- C4：native70.23/70.41，candidate71.49/70.90。
- C8：native97.12/98.00，candidate100.15/99.58。
- C4平均HTTP TPOT47.83→47.38ms；C8 62.24→60.40ms。
  TPOT是请求完成减首个内容事件、除以剩余输出token数，不是SSE事件间隔。

两次交换卡对重复不是统计置信保证，也不能称为普遍SWE吞吐增益。
并行同机服务共享CPU/内存带宽；没有CPU隔离的因果声明。
完整轮次/配置在 `docs/evidence/qwen-swe-traces.json`。

## 数据与选择偏差

本机固定 NVIDIA Open-SWE-Traces，revision
`fb0c0dccc7a5cce79b3f6de891848acdede36685`，CC-BY-4.0。
选用 `data/minisweagent/qwen38_27b`，用27B自身tokenizer/template重新渲染。
扫描15,525行取得8条整轨迹，7–11次调用/轨迹，prompt1472–7874tokens。
选择条件是2–12次assistant调用、每次prompt+完整assistant预算<=8192、
整轨迹输出预算<=8192。**这是很窄的短轨迹样本，不代表完整SWE分布。**

历史OpenHands fixture的长会话超过当前8K服务边界；没有截断来凑测试，也没有
偷偷放宽Worker的资格限制。Qwen模板需要时将tool-call arguments的JSON字符串
解析为映射；不执行任何工具。后续请求总是用原始记录历史，不接回生成文本。
输出预算来自完整assistant序列化后缀独立BPE计数，ignore_eos固定工作量。
没有真实到达时间；会话闭环、工具等待为零。这不是SWE任务正确率测试。

`prepare_trace.py`新增显式model/source/subset参数，旧30B默认行为保留。
CPU准备需pyarrow、transformers5.14.1、jinja2，可用隔离uv环境，不改donor。
合格fixture在证据根 `swe-qwen27-v5/trace.json`；保留早期拒绝清单，勿重复扫描。

## 同形调度的六步profile

两边完成计时后，在正在decode的真实SWE请求旁加入另一个原始SWE prompt。
这是有意安排的两请求诊断，不是C4/C8整轮服务profile或记录到达时间回放。
双方实际调度完全相同：`[1]、[1]、[1,1472]、[1,1]、[1,1]、[1,1]`。
原生mixed走NONE，候选走FULL，物理token容量1536。两rank均校验六次初始norm/
sampling、每模型body304个MatMulV2/V3、16个FIA、128个通信区间。

用raw PROF经native msprof导出DB，再用冻结TraceLoom37323af生成augmented DB
及Perfetto。每rank独立时钟。只有三个重复decode2被重建为exact_direct graph；
不能把唯一mixed形状未重建成exact label误读为没有FULL。调度、replay API和
模型task区间保留了对应证据。没有额外长profile或因导出问题重跑硬件。

### 已发挥作用的部分

mixed模型body约507.11→346.40ms（两rank接近）；body内CANN API计数约
23.6k→310/303。rank0未被compute/communication覆盖的部分153.02→4.89ms；
rank1通信等待234.29→112.07ms。compute union反而224.5→229.0ms，包含padding
与不同GDN执行策略。支持FULL消掉host供给空洞/peer等待，不是矩阵算子变快。
未覆盖时间也可能含memory/control，不等于全部设备空闲。

### 优先小鱼：GDN metadata发布把decode接缝串起来了

稳态decode body结束→下一body开始，三处/rank：

- native约1.52–1.63ms；candidate约4.74–4.96ms。
- 两边sampling尾部约1.1ms，不能把它也当metadata损失。
- candidate在gap内有三次`aclrtMemcpy`，合计约0.14ms；不是大数据带宽问题。
- 以rank0的step3→4为例：候选第一处stream wait到body结束后1.323ms才返回，
  接着H2D1.325–1.373ms；第二组H2D2.406–2.452ms；第三组3.403–3.448ms；
  下一次replay API到4.645ms才发出。原生在1.388ms已发出replay。
- 与 `qwen_gdn/metadata.py:Metadata.update` 相符：三套metadata逐套执行pageable
  blocking cu拷贝、cast/copy、computed-token的两次Sub和一次Greater等派生操作。
  gap里看到18次aclnnInplaceCopy、6次Sub、3次GtScalar和36次kernel launch。
  三组提交之间host执行串行，原生相关工作能更早排队。

第一处SynchronizeStream跨越上一轮模型，**不能把它整个23ms等待算新增开销**。
这里只比较模型之间的实际区间。API嵌套求和也不是CPU独占时间。
profiler可能放大host提交成本；约3.2ms的差值不是未profile服务可直接回收的承诺。
但它是明确的逐step热点，优先于分散的几十微秒通信缝隙。

优先原型方向：保留各组状态槽位所有权，合并/前移metadata发布，把派生计算
移入图；若改为异步pinned slab，必须有生命周期和复用fence，不能只加non_blocking。
当前研究没有改动生产实现，也没有声称已实现N+2通信重叠。

### 第二条线索：空chunk屏蔽访存，却仍执行三角求解

48层`solve_tril_16x16`的算子耗时和在native约3.007ms，候选约10.522ms。
候选capacity1536的1216粒度任务表固定为9项；`[1,1472]`只需要3项，余下6项
指向空sentinel。固定donor的 `solve_tril_16x16_kernel` 在T=0时仍跑固定的矩阵
递推，只将load/store mask掉，并无空任务的计算跳过。这是源码事实。

额外时间不能全部归因于六个空任务：native把decode前缀交给recurrent算子，
候选统一走chunk，且物理padding/卡对不同。值得做device-side空任务guard的
小原型与完整状态回归，不需要重新枚举请求partition。
同时候选decode里的48次conv权重Transpose确实消失，rank0约0.95ms/step；
模型body约35.6→34.4ms，却被更长host接缝抵消一部分。

## 资源失败与复现资产

`swe-elastic1`第一轮完成，但交换到0/1时native启动内存检查拒绝candidate：
free48.31GiB < requested56.08GiB。随后1号卡有明显HBM/AICore活动却没有列出进程。
不能确定未列出的使用者身份。该campaign不用于headline，也没有降低内存门槛。
`swe-elastic2`改用2/3和6/7，所有server重载前都在已有租约内重新检查空闲；
CPU负例验证有不明HBM时不会到达launch marker。最终任务退出、所用卡回收。

运行源b05739f；生产源a34462a。远端root：
`/workspace/my-ascend-workspace/runs/qwen27-partition-serving/swe-elastic2`。
本机root：`/workspace/strengthen-dsv4/runs/qwen38-tp2-serving/hw3-swe-elastic2`。

- `swe_compare.py` / `swe_service.py`：两并行HTTP服务、换卡、whole-session闭环。
- `swe_launch.sh`：冻结capsule及显式卡对，subset admission/监控/回收。
- `summarize_swe.py`：校验全部(session,turn,prompt长度,output预算)相同并保留两轮。
- `analyze_swe_profiles.py`：完整body守卫、调度、API与decode gap证据。
- 四份可读timeline+调度+汇总包：证据根 `qwen-swe-traceloom-timelines.tar.gz`。

没有修改已安装donor、生产Worker或PyPI版本。无需重新跑已得到的profile；
继续优化时先从上述metadata发布和空任务guard进入，不要从大段日志重建本次发现。
