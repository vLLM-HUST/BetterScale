# Ordered replay：省去主模型 FULL replay 前的一次 host 同步等待

这个补丁的目的，是让已捕获的 DSV4 主模型 FULL graph **直接排入设备 stream**，
不再要求 host 先等待当前 stream 的前序工作全部完成，再提交 replay。
它不修改模型算子，不引入双槽 metadata 或 N+2 调度器。

准确地说，删去的是符合准入条件的每次 replay 前的一次
`torch.npu.current_stream().synchronize()`，**不是全 device 同步**
`torch.npu.synchronize()`。当前 stream 若依赖其他 streams 的 events，等待
也会包含这些依赖，但不能据此说它等待了设备上所有无关任务。

## 原调用路径与同步位置

正常服务波次先由 `NPUModelRunner.execute_model` 准备输入及 attention
metadata，再调用主模型的 `ACLGraphWrapper.__call__`。匹配到已捕获的 graph
后，上游通常在主模型 FULL 路径上执行：

```python
torch.npu.current_stream().synchronize()
entry.aclgraph.replay()
return entry.output
```

这里的 synchronize 阻塞的是 host：设备可以继续完成已排队的任务，但 host
必须等到当前 stream 的前序工作完成，才能沿这条路径提交下一次 replay。
因此它在“前序工作完成”和“下一个 replay 提交”之间增加了一次 host 往返。
若队列无法由其他工作填充，这个提交间隔就可能成为设备气泡；多 rank 场景中，
迟到的 rank 还可能让其他 rank 等待后续 collective。

即使模型计算已经 FULL graph 化，这段 wrapper 的 Python 提交逻辑也仍在
每次调用时执行，所以 FULL graph 本身不会自动消除这道栅栏。

## 上游为什么要等，我们为什么可以不等？

上游 wrapper 服务多种 backend。其注释说明，在异步调度或多线程情形下，
逐轮 graph-task 参数更新可能与上一轮 replay 发生时序冲突，因此保留同步保护。
ENPU 和 merged EAGLE draft 已有原生例外；本补丁不改变这些例外。

我们已核对的 DSV4 compressed-attention 路径，被 runner 排除在那条逐轮
host graph-task 参数更新路径之外。这里仍需更新输入和 metadata tensor 的
**内容**，但不是用 host 重写已捕获任务的参数。输入 / metadata 生产与 replay
在已验证的同一 stream 上有序执行：

```text
原路径：host 提交输入更新 → host 等 stream 完成 → host 提交 replay
补丁后：host 提交输入更新 → host 直接提交 replay

设备顺序仍然是：[前序工作] → [本波输入 / metadata 更新] → [本波 replay]
```

设备 stream 已保证“先写、后读”，host 无需再等一次。这个修改保留 graph
内部 shared-expert 等并行任务的 event 依赖，没有把数据生产和消费变成无序执行。
`entry.output` 是已有输出 tensor 的引用；返回它不表示设备已经执行完毕，
后续消费者仍需遵守原有 stream / event 依赖。

## 适用条件与保留的保护

- 只给已验证的 DSV4 compressed 主模型 wrapper 标记准入 stream。
- 只有匹配模式、已经捕获好的 FULL entry，且当前不处于 capture，才能省去栅栏。
  eager、首次 capture、非准入 wrapper 均保留原生逻辑。
- 准入 replay 会断言当前 stream 与安装时登记的 stream 相同；不一致时直接报错。
  这只是对已核对调用路径的保护，不是自动证明任意跨 stream 数据生产都安全。
- 调试模式保留输入地址检查，capture 的 offloader 同步和 workspace 生命周期不变。
- 安装在原生 warmup 后执行。`install(worker)` 中的
  `torch.npu.synchronize()` 是**一次性的全 device 安装同步**，仍然保留；
  不要把它与本补丁省去的逐次 current-stream 同步混为一谈。

这个补丁减少了一处不必要的 host 等待，**不等于实现完整连续调度，也不保证
每波都有可见收益**。其他 metadata、采样、调度或设备依赖仍可能限制提交。
历史 `ordered_replay + qli_cpu` 对照没有稳定的大幅独立加速，不能把 draft FULL
等组合优化的收益全部归给此处。实验边界见
[`DECODE.md`](../../../../prototypes/full-mixed/DECODE.md)。

## 源码怎么读

`__init__.py` 的 `_ordered_call` 直接展开了 pinned vLLM-Ascend
`compilation/acl_graph.py` 中 `ACLGraphWrapper.__call__` 的主体，不再经过
`call(original, wrapper, ...)` 转发，也不隐藏调用旧 `__call__`。从上到下是：

1. 非匹配 graph 模式：调用原 runnable，保留 eager / 嵌套 wrapper 分派。
2. 新 bucket：创建 entry，执行原生 capture，保留 offloader 同步、异常处理、
   workspace / output 弱引用和 capture 计数。
3. 已捕获 bucket：检查调试输入地址，进入带中文注释的 replay 同步分支。
   准入的 DSV4 FULL 同 stream 调用省去 host fence；其余调用仍按原生
   ENPU / EAGLE / FULL 条件决定是否 synchronize，随后 replay。

`native.*` 仅引用上游已有的辅助函数、类型及模块状态，不转发整个调用。尤其
`_graph_params` 等工作区状态必须在使用时从上游模块读取，不能复制 import
时的值，否则后续重新绑定会让本地副本过期。上游许可和来源保留在文件头；
升级 pin 时需要复核这一份展开代码，而不是假定它会自动跟随上游。

测试同时检查展开的 dispatch / capture 主体与 pin 的 AST 一致，以及准入
replay 不同步、未准入 replay 保留同步、错误 stream 拒绝执行和原生例外。


## 安装边界

这是同一 wheel 内独立的 Python 模块，不是新的 pip 发行包。没有自动注册，
import 本模块不改 donor；worker 显式调用 `install`。版本/组合检查由 worker
统一执行。这里的独立是源码与 hook 所有权独立，不保证任意组合都已有性能/质量验收。
停掉服务再更换组合，不在活跃 graph 上热卸载。
