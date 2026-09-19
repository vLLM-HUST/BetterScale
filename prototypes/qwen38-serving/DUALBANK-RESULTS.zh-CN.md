# 双 bank metadata 与空任务三角求解：完整验收

2026-09-17；运行源7115858；入口仍为独立的
`betterscale.qwen_worker.MixedWorker`。没有改DSV4 Worker、旧Qwen Worker、
已安装donor或PyPI。不是把整个LiveInference executor重新塞进native runner。

## 做了什么

- 每个token capacity两张图交替，26张图；不枚举请求数量/长度partition。
  FIA captured task资源同样按capacity+bank隔离，不可更新错另一张图。
- 每wave把所有GDN组的metadata打包为一份pinned slab，独立ingress stream
  一次H2D。slot来自CPU blocktable，cold flag按seq_lens-query_lens计算，
  各dtype、cu和chunk表在host一次准备；取消逐组阻塞复制及GPU Sub/Gt/cast。
- uploaded保护host源复用，consumed保护device bank旧读者，compute只做
  device-side wait。只支持当前noMTP/mamba_cache_mode=none合同。
  分离input-reader与源生命周期边界，沿用LiveInference N→N+2复用思想。
- fork固定donor的16x16三角求解，在`base_t < T`分支内才做重计算。
  空sentinel不再只mask访存却照算整套递推；有效任务算术与merge不变。

native input_ids/positions、普通attention metadata、sampling/D2H及KV退休
仍由native协议管理。没有宣称全N+2调度接管、D2H已隐藏或消灭所有FIA host更新。

## 正确性与容量成本

hw3 `dualbank-core1`：7组triangle对照active输出逐位相同；12组完整GDN
FULL/NONE的output、conv和整池state均max_abs0，独立initial-H warm/cold检查通过。
`dualbank-service2`：10种prompt长度1..2051、C4/C8；每rank22步，合计5676项
whole-model有效输出/状态检查max_abs0。双bank均被观测，shadow另逐字段核对
所有GDN组的host/device元数据。shadow本身同步且仅1GiB KV，不用其时延做性能结论。

26张图的capture占用约5.25GiB/卡；旧13张图约2.9GiB。增加的graph/scratch
内存是真实成本；正式SWE仍使用与baseline相同的6GiB KV，未缩KV换性能。
`dualbank-service1`已完成capture，但诊断RPC404；修复诊断launcher缺少
VLLM_SERVER_DEV_MODE后新capsule通过。不是数值失败，也不把失败capsule算PASS。

## 同卡端到端 ABBA

hw3仅6/7，两臂顺序native→candidate→candidate→native，每次reload前重新
admit空闲。TP2、8seats、2048budget、context8192、6GiB KV、无MTP/APC。
沿用`swe-elastic2/trace.json`：8条完整NVIDIA Open-SWE原始轨迹、78calls、
每cohort20648输出token。两并发档各两轮，profile不在计时中。

| 并发 | native第一/第二轮 tok/s | candidate第一/第二轮 | pooled增益 |
|---|---:|---:|---:|
| C4 |70.03 /70.02|74.94 /75.25|**+7.24%**|
| C8 |96.70 /96.85|104.26 /104.56|**+7.89%**|

pooled mean TTFT：C4 1388.64→1205.32ms，C8 1533.93→1348.96ms。
mean TPOT：C4 48.011→44.956ms，C8 63.047→58.033ms。
这些不是SSE事件间隔，也不是模型质量结论。原始history回放、固定记录output预算、
工具不执行/零工具等待、只选<=8K轨迹，不能代表整个SWE人群。
两轮不是置信区间。旧C1/2048回归未复测，不能据此说已消失。

## 六步 TraceLoom：缺口确实被压下去了

两臂两rank调度都为 `[1],[1],[1,1472],[1,1],[1,1],[1,1]`。
新candidate bank为1,0,1,0,1,0。每body均通过304 MatMul、16 FIA、128通信守卫。
所有timeline经native msprof DB→TraceLoom37323af→Perfetto，不用Torch的缺损DB。

rank0稳态model-end→next-model-start：

- 历史single-bank candidate：4.955 /4.793 /4.742ms。
- 本次dual-bank candidate：**1.496 /1.487 /1.478ms**。
- 本次native：1.478 /1.535 /1.554ms。

旧每step三次阻塞metadata发布、18个InplaceCopy及Sub/Gt图外提交不再落在
step gap。仍存在native sampling/replay fence，约1.1ms sampling尾巴没有改造；
跨越前一个model的长host wait不能全算新增device idle。

provider通过connectionId关联到stream40每rank6条MemcpyAsync task；首条在
首body前，其后五条完整落在前一model envelope内，每条约2.24–2.76us。
源码与这个独立六拷贝stream吻合metadata ingress；provider未记Python buffer
owner，stream归属为结合源码的推断。envelope内不等于每周期同时有compute核。

mixed body：fresh native491.843ms，candidate343.201ms；native仍NONE而candidate
FULL，不能把整段差额算给这次两个优化。历史candidate为346.398ms。
48层16x16 solve耗时和：历史candidate10.522ms→本次4.604ms（两种JIT名合计）。
独立`dualbank-solve2`的图内20次solve×30replays ABBA验证whole-solve含merge：
[1,1472] 378.85→253.86us，约−33.0%；7种长度的有效输出均逐位相同。
profiler可放大host差异；端到端收益以上面的无profile ABBA为准。

## 产物

本机证据根 `/workspace/strengthen-dsv4/runs/qwen38-tp2-serving/dualbank-swe1`；
远端同名capsule在 `/workspace/my-ascend-workspace/runs/qwen27-partition-serving`。
`summary.json`、`profile-analysis.json`、原始receipts和每rank增强DB均保留。
同级 `qwen-dualbank-traceloom-timelines.tar.gz` 收齐四份可读timeline、两rank调度
与汇总。卡已回收，admission exit0，6/7健康且无进程；未使用其他卡跑本轮服务。
