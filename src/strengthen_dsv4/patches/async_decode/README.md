# 让稳定 decode 的下一步，不再等 CPU 重做输入与 attention metadata

这个补丁支持两条独立准入的 K5 路线：**TP1 × DP8/EP8、每 rank 两个活跃请求**，
以及 **TP8/EP/DSACP、四个全局活跃请求**。
原生 scheduler 仍逐步发任务，原生 sampler 仍验证输出；它不是另一个调度器，
也不把未知的 acceptance 当成已知。prefill、请求入场/离场、槽位重排及不完整
K5 查询仍走原生准备路径。TP 保留原来的 split-draft，不把大 prefill 填进 draft 图。

## 原来的空档在哪里

一次 target 校验后，device 已经得到实际接受数量、采样 token 和下一组 draft。
但下一次 target 之前，CPU 仍会准备位置、槽位映射和 attention 的派生 metadata。
如果这些操作等 CPU 收据，或逐个从 host 发射，就会让已有数值反馈的 device
等待 host。仅仅复制两份 target graph，并不能消除这些生产者依赖。

这里把两类信息分开：

- **host 授权**：这波服务哪些请求，已经分配哪些 KV 页，允许的资源边界。
- **device 数值状态**：到底接受几个 token，实际当前位置，以及下一组 token ID。

host 只发布它已经知道的部分；exact positions、lengths、slot mapping 在
compute stream 上读取上一波真正的反馈后生成。随后重放 native attention
builder 的 device 程序，而不是每一步重新让 Python 发射那一串小任务。
CPU sequence length 只作保守的 tiling 上界，不能代替 device 实际访问长度。

## 嵌入原调用链的位置

初始化时，`Worker` 先安装对应的 `target_full` 分支，再调用
`install_capture()`。它包装原生 `ACLGraphWrapper.__call__`，为小 target
bucket 各捕获两份拥有私有输入 packet 的 graph。大 prefill 和 draft 不经过
这个双槽适配器；两份 target graph 复用原生 graph pool，不复制权重或 KV。

TP 还先安装 `ordered_replay` 的调用入口，再让双槽包装它；warmup 后只准入
其 stream，不覆盖已装好的双槽入口。这样大 prefill 继续走原有 ordered replay。

原生 warmup 完成后，`Worker` 先装 `cross_step`（TP 也保留 split-draft、CPU QLI），
再调用这里的 `install(worker)`：

1. `_host.py` 接管 `runner.synchronize_input_prep`：native CPU 输入源变成两槽，
   Python 进入下一槽之前，只等待**该槽上次 H2D 读取结束**，不是等待相邻整步。
2. `_producer.py` 接入 `cross_step.prepare`。稳定 K5 时，把 host 授权快照放入
   自有 pinned 槽，在 ingress stream 上 H2D；compute 等 ready，然后读取原生
   device feedback，replay 准备程序。未通过 admission 时调用原生 `_prepare_inputs`。
3. `_metadata.py` 包装 `_build_attention_metadata`。稳定路径重放捕获的 native
   device builder；拒绝把 H2D、D2H 或 device scalar readback 偷带进 capture。
4. `_target.py` 从原生持久 device 输入向当前 target packet 发布数据；这些 D2D
   copy 已记录在 target graph 内，不在热路径逐层遍历 metadata 或逐项发射 copy。
5. `cross_step` 在 target 已提交后处理原来的 CPU 收据回调。只有授权预算已经
   快照进 producer 自有存储时，才跳过当前输入 DMA fence；**旧收据等待仍保留**。

这些模块按不同的存储生命周期组织，不是多层 runnable 包装框架。
`cross_step` 是明确的安装依赖；worker 负责组合，patch 不替用户安装另一份 patch。

## 输入输出都必须有自己的所有权

双槽的目的不是增加 graph 数量，而是让一边写、一边读时不互相踩坏：

| 存储 | 谁写／谁读 | 何时能重用 |
|---|---|---|
| native pinned 输入两槽 | host／native H2D | 该槽 preparation event 完成 |
| producer pinned 授权槽 | host／ingress H2D | 上次该槽 ready 完成 |
| producer device 授权槽 | ingress／device preparation | ingress 等该槽 consumed |
| target 输入与 metadata 两槽 | graph 内 D2D／target | 同一 compute stream 串行发布与消费 |
| target hidden / draft 中间结果 | target／sampler、draft | 原生 compute 顺序保证读完后再重放覆盖 |
| sampler 最终输出 | 原生 sampler／异步 D2H | 原生逐 invocation 分配并由 AsyncGPUModelRunnerOutput 持有至 copy 完成 |
| cursor、KV、draft feedback | 有序 device 程序 | 单份 State，不复制成两个独立进度 |

**当前 donor 的最终 sampler 输出不是捕获图里的固定双槽。** 它仍是原生
uncaptured sampler 每次分配的独立 Tensor，copy stream 等 compute，异步输出对象
持有源 Tensor，读结果前等 copy event。这已经隔离相邻 invocation 的输出，
不能为了形式上凑“双槽”再添一遍无用拷贝。

以后若把 sampler/commit 也捕进完整 wave graph，输出地址会固定，就必须明确
改成输出两槽，并在 graph 下次写该槽之前等待其 D2H 完成。仅持有 Tensor 引用
不能阻止 graph 覆盖它。**本补丁没有宣称整个 donor wave 已捕成 LiveModule，
也没有新增 all-mode N+2 scheduler。**

## 成本、回退与证据边界

辅助 preparation 与 metadata graph 各共享串行 scratch pool；host sources、
返回的 live tensors 与 target packets 分别保留所有权。Worker 在 READY 之前，
通过 `_warmup.prepare()` 遍历实际已捕获的 small target descriptors、允许的
请求数与两份 CPU carrier，完成全部 producer／metadata 图准备。只运行输入和
metadata 程序，不运行 target forward，也不复制或改写模型 KV；结束后恢复原生
输入样本。启动耗时属于服务启动，不能转嫁到第一批请求再用热身测量掩盖。

READY 之后，这两条路径只查询已准备的 entries；缺失形状明确走原生回退，
不在线创建 Slot 或 capture。DP 某 rank 本地是 K5、全局却被其他 rank 的大
prefill 补齐时，也走原生大桶回退。shape key 仍包含 native CPU carrier 身份，
避免交替输入槽引用错误的 CPU metadata。prefill、turnover 以及 DP dummy drain
不继承上一波的稳定 decode admission。

**当前分支的启动期改造仍在硬件验收，以下旧结果不自动覆盖它。** 这里只描述
async_decode 两种辅助图；split_draft 的历史首用捕获是另一个待收口的入口，
不能据此声称整套 Worker 已无任何在线捕获。

2026-09-13 原型同机 run120：DP8、16 个全局请求、K5，每 rank 两个请求与
12 个实际 target queries，FULL target、原生 eager draft。两次对照中，
仅双槽端点 → 双槽加 producer/metadata 的匹配 cycle 为 **59.89→50.38 ms、
59.09→50.00 ms**；draft→下一 target 的 device-event 区间为
**9.42→1.24 ms、9.49→1.23 ms**。这一区间包含实际工作，不全是硬件 idle。

在 `HCCL_DETERMINISTIC=strict` 下，原型 run121 的八 rank 同状态检验全部通过，每 rank 48 次 target/整份 KV
对照及 12 次精确 preparation 对照；run120 的 32 道原始长输入 OpenCompass
LongBench retrieval 题全部通过。这是有限质量集，不是完整 OpenCompass。

**包内接入已单独验收。** run122 同样在 strict HCCL 下，通过外部 oracle 子类完成同样的 384 次
 target/整份 KV 与 96 次 preparation 对照；run124 从 wheel 加载实际
`strengthen_dsv4.worker.Worker`，无需激活 RPC，32/32 检索题通过。
该轮两次 matched cycle 为 **50.74 / 49.88 ms**，draft→target 为
**1.24 / 1.23 ms**。包含长 prefill 质量题后的总预留为 **58.66–58.70 GiB/rank**，
测量峰值 allocated **58.02 GiB/rank**，其中用户指定的 KV 预算为 8 GiB/rank。
这些不是相对于未打补丁单图路径的净增量内存。

保留不那么漂亮的观察：前一轮 run123 已完成的 cycle 为 **50.08 / 52.16 ms**，
区间 **1.25 / 1.94 ms**；之后因观察器漏返回 quality 所需容量字段而停止，
不能计为质量通过。run124 仅修复测试观察器，wheel 的产品代码未改。
明细见仓库 `evidence/dp-continuation-20260913.json`。
整组生成轨迹、波次数和总吞吐可能变化，不能把上述 cycle 改善当成已经证明的
在线吞吐提升。旧 worker-retirement 扩展没有稳定增量收益，未纳入这个补丁。
