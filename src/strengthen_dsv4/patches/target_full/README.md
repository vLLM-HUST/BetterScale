# 让主模型的 prefill / mixed forward 支持 FULL graph

**这个补丁修复主模型输入准备中的几个捕获障碍，让现有 vLLM-Ascend 引擎能够对
prefill 和 mixed 波次使用 FULL graph，而不只是在规则的 decode 波次中使用它。**
它不重写主模型，不实现新的 attention 算子，也不另建一套 graph 执行器。
捕获、选择 graph 桶和 replay，仍然由原生引擎完成。

这里的 **target** 是负责最终 token 验证的主模型，与产生投机候选的 **draft**
模型相对。三个波次名称分别指：

- **prefill**：处理请求新送入的一段输入，可能一次有很多 token；
- **decode**：推进正在生成的请求；在本项目的 K5 配置中，还包含候选 token 验证；
- **mixed**：同一波里既有 prefill 工作，也有 decode 工作，各请求的 query 长度不一致。

本补丁只处理 target。Draft 的图执行由相邻的 `split_draft` 模块负责。

## 1. 为什么原有 decode graph 不能直接覆盖 prefill / mixed？

FULL graph 的价值是把一次模型 forward 的设备工作捕获下来，后续用 replay 提交，
减少 host 逐个发射算子的开销。但它不是“把任何一次 Python 调用录下来就能反复用”。
对同一个捕获桶，图所依赖的输入地址、张量形状和容量参数需要保持一致；
**每波数据可以变化，图所依赖的存储和程序边界不能随意变化。**

本包固定的 vLLM-Ascend 版本已经支持 target FULL decode，但 DSACP attention
backend 只声明支持 `UNIFORM_BATCH`，即它所支持的均匀 query 批次，而非任意 mixed
批次。DSACP 是这个版本的 DSV4 attention/metadata 路径；本补丁不改变其并行算法。

沿 prefill / mixed 路径继续检查，可以看到三个实际问题：

1. **prefill 的 RoPE 结果可能是临时 tensor。** 新一波准备出了新地址，旧图却仍然
   绑定捕获时的地址；仅重新赋值 Python 变量不会更新图的输入。
2. **实际请求数变小时，metadata 的请求容量可能跟着缩小。** 图原来按四个请求槽
   捕获，后来只按两个槽准备，图和 metadata 就不再遵循同一个容量约定。
3. **一些 Python 标量随实际 token/request 数变化。** 它们会影响 forward 中的
   切片或工作区范围；不能一边复用旧图，一边让这些边界任意改变。

所以，只把支持声明改成 `ALWAYS` 是不够的。那只是告诉引擎“可以尝试捕获”，
还必须把以上输入与容量问题一起修好。

## 2. 补丁在整个执行流程的什么位置？

用户继续使用原生 `vllm serve`，只指定我们的 worker 类：

```text
vllm 创建 strengthen_dsv4.worker.Worker
  ├─ 检查 donor 版本、源码及配置
  ├─ target_full.install()：替换四处原生方法/函数
  └─ 原生 worker 初始化、模型加载、warmup / graph capture

每次 target 调用：
  原生调度器给出这一波的请求和 token
    → 原生 runner 准备输入、选择 graph 桶
    → 请求 offsets 补齐 + attention metadata 构造  ← 本补丁在这里生效
    → 原生引擎执行 target forward / FULL graph replay
    → 原生采样、候选验证及后续 draft 工作
```

`install()` 必须在 runner 初始化和 capture 之前执行，否则旧图可能已经按未修复的
输入约定捕获。安装后，调用方仍然调用原来的方法名称，只是这些名称指向本模块的
实现。没有修改磁盘上的 donor 源文件，也没有额外的启动服务或后台线程。

**这个模块没有创建 `NPUGraph`。** 它修好原生引擎使用 graph 的条件，而不是接管
执行循环。Host 的调度和 metadata 准备也没有因此全部进入图中。

## 3. 四处修改如何配合？

### A. 放开 backend 的 FULL 捕获声明

`install()` 将 `AscendDSACPMetadataBuilder.get_cudagraph_support` 从原生的
`UNIFORM_BATCH` 改为 `ALWAYS`，让引擎允许这条 attention 路径参与更广的 FULL 捕获。

这里的 `ALWAYS` 是 upstream 枚举名称，**不是本包承诺支持任意模型、形状或并行布局**。
用户仍需选择原生 FULL 配置，worker 仍会检查当前验收范围。

### B. 把当前 RoPE 值写进持久缓冲，而不是把临时地址交给图

RoPE 是旋转位置编码；attention 需要与当前 token positions 对应的 cos/sin 数据。
原生 builder 在含 prefill 的路径使用 `use_cache=False`，返回本次索引产生的 tensor。

本补丁用 `_build_target()` 标记“当前正在构造 target metadata”，再让该作用域内的
`_stable_rope()` 以 `use_cache=True` 调用原函数。原函数仍按**当前 positions**
取得 cos/sin，但把它们复制到已有的 runtime buffers，返回这些持久缓冲的视图。

```text
原来：本波 positions → 本波临时 cos/sin tensor → metadata
现在：本波 positions → 更新持久 cos/sin buffer → metadata 引用固定缓冲
```

缓存的是存储位置，不是把上一波的数值冻结不动。`ContextVar` 只负责限定这个修改的
作用域，并在正常返回或异常时恢复。显式 drafting builder 不经过这段 target `build`
包装，不能理解成“全系统 RoPE 都强制共用 target 的缓存”。

### C. 用零长度空槽维持请求容量，不额外虚构请求

原生 runner 的通用 `_pad_query_start_loc_for_fia` 为满足通用 attention 的布局要求，
有时会添加一个 dummy request，让 query offsets 覆盖 padding 后的 token 行数。
但这里的 DSACP 路径已经用显式 query 范围描述真实工作，需要保持 graph descriptor
给定的请求槽容量，而不是随实际请求数收缩或临时增加请求。

`_pad_dsa_capacity()` 在 FULL 执行时采用 descriptor 的容量，并把空槽的 offset
全部填成最后一个真实请求的结束位置。例如，一个四请求槽的图，本波只有两个请求：

```text
两个真实请求各有 64 个 query token

query_start_loc = [0, 64, 128, 128, 128]
对应请求长度    = [   64,  64,   0,   0]
```

下一波请求长度可以改变，例如 offsets 变为 `[0, 7, 71, 71, 71]`，但仍有四个请求槽。
相邻 offsets 的差给出每个请求的真实 query 长度；重复尾值表示空槽，不是新增请求。
非 FULL 调用仍走原生 padding 方法。

这也不是“padding 完全没有成本”的保证：固定桶可能仍让部分计算按容量执行。
补丁解决的是布局一致性，不是消除所有 padding 工作。

### D. 把 forward 所需的标量范围绑定到当前图桶容量

`_build_fixed_capacity()` 先调用原生 builder，保留它构造的真实请求信息和 tensor，
再在 FULL 配置下调整返回 metadata 中的几个标量：

| 字段 | 本路径采用的值 | 作用 |
| --- | --- | --- |
| `num_actual_tokens` | `num_input_tokens` | forward 的 token 范围按当前输入桶容量组织 |
| `req.num_reqs_actual` | `query_start_loc.shape[0] - 1` | 请求维度按已准备的请求槽容量组织 |
| `req.num_compressed_tokens` | 对压缩层取 `min(T, T // ratio + R)` | 提供与 token/request 容量一致的压缩行数范围 |

最后一行的 `T` 是输入 token 容量，`R` 是请求槽容量，`ratio` 是该层的压缩率。
这些容量是**相对于当前捕获桶**而言的，不是所有 graph 永远使用一个全局最大值。

这里沿用了 donor 中带有 `actual` 的字段名，但这些返回字段在此 FULL 路径中被用作
容量边界；不要再把它们全部解释成真实请求计数。真实 query/sequence 范围仍在 device
metadata 中，原生 builder 也先完成真实请求和空槽状态映射的处理。补丁没有把 scheduler
里的两个请求改成四个请求，也没有改用户请求的内容。

## 4. 看源码时，按这个顺序读

实现都在同目录的 [`__init__.py`](__init__.py)，不用先遍历整个 donor：

1. `install()`：四处替换的清单，以及重复安装保护。
2. `_pad_dsa_capacity()`：请求槽容量怎么保持不变，空槽怎么表示。
3. `_build_fixed_capacity()`：哪些返回字段按容量组织。
4. `_build_target()` 和 `_stable_rope()`：如何限定作用域并复用 RoPE 缓冲。

其中 `_fixed_build = _build_target` 只是一个函数别名，没有额外执行阶段。
原函数保存在 `_original_*` 变量中，用于保留原生构造过程或非 FULL fallback。
实际安装时机可以在 [`worker.py`](../../worker.py) 的 `Worker.__init__` 中看到。

## 5. 哪些事情不属于这个补丁？

- **不负责 draft graph。** 那是 `split_draft` 的工作。
- **不负责 K5/TP 桶对齐。** `compat_lcm` 独立处理它；某些 K5/TP8+SP 配置需要此修复
  才能启动，但它不是本模块内部隐藏的安装动作。
- **不负责去掉 replay 前的 host fence。** 那是 `ordered_replay` 的工作。
- **不改变调度、KV 存储布局、attention 并行方式或神经网络算子。**
- **不保证 prefill 普遍更快。** 收益取决于原来的发射开销与新桶的 padding 代价；
  小请求落进大桶也可能不划算。FULL 支持不等于所有负载的吞吐都提升。

## 6. 支持范围与验证依据

当前交付固定在 vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1 的已核对源码，以及 DSV4 Flash
W8A8、TP8/EP、DSACP、K5、四席位等组合；完整限制见
[运行手册](../../../../docs/RUNBOOK.zh-CN.md)和
[配置检查](../../config.py)。本模块不 import 其他 patch，但独立源码组织不等于
任意组合都已验收。Import 不安装 hook，安装与配置检查由 worker 显式完成。

这套机制的原型曾完成八卡真权重同状态对照。保留的 run012 检查了每 rank 42 步，
包含实际 mixed 波次和 4112 个有效 token；有效输出、MTP 状态及 KV backing 比较通过。
这些是在明确的 strict-HCCL/shadow 配置下得到的正确性证据，不是性能测量，
也不是最近源码整理后重新跑过的硬件结果。原记录见
[原型验收说明](../../../../prototypes/full-mixed/README.md)。

当前 CPU 测试分别检查请求缩减时的 offsets、空槽、容量溢出、非 FULL fallback，
以及 import 无安装副作用、hook 所有权和重复安装：
[`test_target.py`](../../../../tests/test_target.py)、
[`test_patch_installation.py`](../../../../tests/test_patch_installation.py)。
没有把这些测试描述成任意 graph shape、backend 或长程服务质量的保证。
