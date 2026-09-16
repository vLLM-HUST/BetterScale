# N+2 的异步搬运：源代码依赖与实际 DMA 分开看

2026-09-16，复用 `swe-ceiling-profile1`，没有再次加载模型/采profile。
对照本地 LiveInference `05ac15419c0e73650e687ceb9daffeb7874865f0` 的
`src/livemodule/arch/ascend/request_parallel/dsv4/wave_executor.py`、
`wave_resources.py`；BetterScale 的对应实现是 `serving/reactor.py`。

## 核心异步协议已经一致，不必再次移植一份 executor

两边都使用独立 ingress / compute / copy streams，两bank、最多两个未退休wave：

1. ingress 等同bank上一代 graph_done，再non-blocking复制pinned输入，shadow replay，
   记录ingress_ready；pinned源一直保留到对应invocation退休。
2. compute 等本wave ready，以及**同bank上一代**copy_done，然后replay。
3. copy stream 等本wave graph_done，non-blocking D2H到pinned目标，记录copy_done。
4. CPU回收等copy_done后才能读取结果、退休invocation；所有rank同意才推进host ledger。

旧wave N 的下一wave N+1使用另一个bank；它不等待N的D2H，只等待N-1的copy。
不能把安全的N→N+2输出buffer复用fence误认为逐step串行化。
CPU `copied.synchronize()` 阻塞的是回收线程，不是device-wide synchronize；
LiveInference的blocking receive也使用它，并另有`poll_oldest()`供响应式host loop。
当前closed-loop worker填满两个槽之后已无独立请求事件可处理，改为忙poll没有已知收益。
LiveInference还有generation-owned预制events/resources；本prototype每次创建events，
但这属于资源生命周期/host成本，不是其异步语义缺失，当前提交有15–16ms提前量。

## 实测，而不只看non_blocking参数

`inspect_transport.py CAPSULE`以host submit括号匹配CANN memcpy API，再以
provider connectionId精确连接到对应device MEMCPY_ASYNC任务；每wave验证三条ingress、
一条egress。方向/字节数来自这个closed reactor的源码合同，stream与时间来自provider，
没有假称profile记录了拷贝指针或字节数。

本四row decode的H2D内容为2528B tiling +56B command +32B generations，共2616B；
D2H为4×8个INT64回执，共256B。

| 时间关系 | rank0 | rank1 |
|---|---:|---:|
| 后三wave的H2D DMA时间/step | 8.66 / 8.70 / 10.94us | 8.40 / 7.74 / 7.60us |
| 这些DMA被前一body envelope覆盖 | 100% | 100% |
| 其中与实际compute kernel重叠的总时间 | 19.48us /28.30us | 16.30us /23.74us |
| D2H DMA/step | 2.04–2.28us | 2.08–2.14us |
| D2H与profile内compute主体重叠 | 0 | 0 |

H2D有部分落在前一步的通信/控制间隙，所以“全部在前一body内”不等于每个周期
都有Cube kernel同时运行。第一sample之前profile已drain，H2D自然不与旧body重叠；
它不是稳态反例。

D2H确实在body间隙，不能宣称这次已与下一step主干重叠。但它也是异步的；
以rank0 step27为例，在本body compute结束后59.50–61.78us执行，下一compute
62.90us开始。这个时间邻近本身没有证明D2H是下一graph的必要依赖。
前面的390个MEM_WRITE_VALUE仍是旧graph收尾；见 `STEP-GAPS.zh-CN.md`。

两次device-wide synchronize均在显式profile末尾drain及profiler stop阶段。
正常已采step只有event等待，没有逐wave设备全局同步。较长CPU receive等待发生时，
下一wave已排队；不要把它的15ms等待当成15ms device bubble。

## 真正更早回收需要改变什么

若要D2H早于整个graph_done，必须提供**device egress已发布**事件，而不只是换
copy API。结果publication、graph完成、D2H完成和State/KV lease退休需要分别证明；
当前copy_done能隐含graph_done，这个推理将不再成立。不能为了让timeline视觉上
重叠，延迟D2H或提前释放还被graph读取的资源。

当前证据支持保留已移植的三流协议，不支持为了每step约2us小回执进行一次
更复杂的提前publication改造。H2D已隐藏；现有62us边界最多约占20ms step的0.3%，
其主要可见内容是graph控制尾部，而不是数据搬运。大回执、RPC和真实异步到达的
场景可能改变投资判断，但不是本次四row固定预算workload的已观测缺口。

产物：`traceloom/transport-audit.json`。`export_schedule.py`检测到该审计后，在另存的
with-schedule timeline中增加H2D/D2H专用轨道；不改原始TraceLoom导出。
