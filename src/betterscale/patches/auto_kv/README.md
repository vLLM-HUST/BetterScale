# 按实际执行程序定容 KV，而不是按显存百分比猜预算

这个补丁解决的是启动时的容量账：模型权重放好后，多少显存必须留给执行，
剩余多少才可以交给原生 KV 管理器。它不改变 SWA、C4/C128 的格式、页面分配、
请求调度或 TP/DP 的 KV 分布，也不会创建第二个 KV 池。

## 原来在哪里估算，为什么不够

原生 `NPUWorker` 先做内存 profile、决定 KV 大小，随后才分配 KV 并捕获 graph。
百分比上限与此时尚未完整驻留的 graph/draft 程序不是同一个量。只量 target
更不够：TP 的 draft 还有独立输入元数据和有限的图目录，但应当与 target
共享同一个 graph 内存池，而不是额外留一整份激活 arena。

`Worker` 将 `PhysicalMemoryMixin` 放在原生 `NPUWorker` 前面，拦截
`_init_device` 和 `determine_available_memory`。用户仍使用原生
`vllm serve ... --worker-cls betterscale.worker.Worker`，不需要另一个启动器。

## 启动顺序

1. 自动模式只临时绕过原生初始化的百分比准入检查，恢复配置后，以启动时真正
   空闲的显存为基数。显式 `--kv-cache-memory-bytes` 保留原生手工预算路径。
2. TP 在任何捕获之前预热现有 HCCL 通信组的缓冲，再做原生 profile。
   构造 communicator 不等于已经物化通信缓冲；预热产生的常驻量必须进入账本。
3. 用小份试验 KV 捕获 target；TP 再安装并预热完整的有限 draft 目录。
   所有图使用最终运行的共享 graph 池，试捕获不是第二个池。
4. 同步后退休试验 State、元数据与目录，恢复原生 draft 绑定。保留旧 graph
   **句柄**维护已观察到的 HCCL 资源生命周期，但永远不再 replay 旧地址。
5. 按下式计算 KV 字节预算，再交回原生 KV 分组/分配逻辑：

   `启动空闲 − 模型 − eager峰值 − 非Torch常驻 − 完整graph驻留 − 退休后残留 − 1GiB安全量`

   这里采用完整试捕获量，不拿复用池之后很小的“最终捕获增量”冒充总成本。
   退休后残留超过256MiB或预算非正时明确失败，不静默掩盖一个未释放的试验池。
6. 分配正式 KV、完成原生捕获和补丁安装；TP 再预热最终 draft，清除启动时
   写入的 scratch KV 内容而不更换地址，然后才报告 READY。
   清理按当前 KV 专属 backing 去重，以连续 byte view 清零整份 storage，包含
   page padding；不逐个清理带间隙的类型视图，避免 NPU strided zero 的临时量。
   这依赖当前 pinned 原生分配器的 KV 独占 backing 契约，不适用于任意张量。

DP 沿用 target 试捕获和其已验收的辅助元数据预热；不会安装 TP draft。
保守 eager 峰值仍保留，不能因为 target 是 FULL graph 就将其直接扣掉。

## 模块边界与代价

- `auto_kv` 只拥有内存账、试验 State 的退休和最终内容清理。
- `split_draft/_warmup.py` 拥有 native DSpark 的有限形状准备。
- `worker.py` 组合这两个模块；补丁之间不互相导入、安装。
- 没有 dummy loader、开发 RPC、文件收据目录或通信环境变量设置。

自动模式会增加一次试捕获的启动成本，换取针对实际执行程序的字节预算。
它适用于独占所选设备的服务实例，不是多个不协调进程的动态显存仲裁器。
并发进程在定容后抢占显存仍可能使服务失败。

## 验收口径

DP 的原型真实权重质量与64K执行见仓库 run170/171 证据。
TP 的 run188/189 覆盖512K配置、448Ki输入、近满池96.84%占用和后续回收；
这些是 dummy 机制证据，不是语义质量或真权重吞吐成绩。
此前3GiB诊断预算下的抢占恢复失败尚未修复；近满池验收没有发生抢占，不能
据此宣称修复。真实权重与打包集成结果必须单独记录，不能借用旧版本分数。

原生日志里的“KV tokens”是依赖模型长度的混合缓存等效指标，不是固定字节/token
除法，也不代表所有席位都能同时达到 `max-model-len`。容量应同时报告物理KV字节、
上下文上限、实际同时驻留请求数以及观测占用，不能只贴一个最大的 token 数。

启动清理的 TP8 dummy 整机对照见 `prototypes/peak-memory/FIXES.md`：
消除清理临时峰值本身不等于增加已定容的 KV，须将 READY 预留与 KV 预算分开报告。
