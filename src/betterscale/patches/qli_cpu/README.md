# QLI 元数据：复用 CPU 已知长度，避免为两个整数等待 NPU

这个补丁优化的是 **主模型 forward 之前的 QLI 元数据准备**，不是把 QLI
计算搬到 CPU。QLI（Quant Lightning Indexer）负责稀疏 attention 的打分与
TopK；这些计算仍由原来的 NPU 算子执行，算法、权重和 KV 布局都不变。

原生代码为了取得两个长度最大值，从 NPU tensor 调用 `.item()`，让 host
等待设备结果。其实同一条路径已经准备好了 CPU 长度副本；本补丁直接使用
这些副本，去掉这一处不必要的设备到 host 依赖。

## 1. 原调用路径：等待发生在 forward 提交之前

在本仓库固定的 vLLM-Ascend 版本中，正常服务波次沿以下路径准备元数据：

```text
NPUModelRunner.execute_model(scheduler_output)
  └─ _build_attention_metadata(...)
       └─ 按 attention / KV cache group 调用 builder.build(...)
            └─ AscendDSACPMetadataBuilder.build(...)
                 └─ build_req_metadata(...)
                      ├─ 准备本 rank 的 device 长度与 query offsets
                      ├─ 准备并缓存本 rank 的 CPU 长度副本（_cpu_local）
                      ├─ _build_sas_metadata(...)
                      └─ _build_qli_metadata(...)             ← 本补丁替换这里
                           ├─ 取得 Q / KV 长度最大值
                           └─ 调用 QLI metadata 算子
  ↓ 元数据准备返回后，继续后续输入处理和模型调用
主模型 forward / 已捕获的 FULL graph replay
```

对应上游源码：

- [`model_runner_v1.py`](../../../../upstream/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)：
  `execute_model`、`_build_attention_metadata`，负责在模型调用前组织元数据。
- [`dsa_cp.py`](../../../../upstream/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py)：
  `AscendDSACPMetadataBuilder`，负责上述本地长度、副本和 QLI metadata。

**FULL graph 不等于整个 host 调用过程都被捕获。** 即使模型计算已经能 replay，
每波准备输入和元数据的 host 路径仍会执行；这里的等待依然可能推迟 replay 提交。

## 2. 为什么两个数字会干扰 device 的连续执行？

原生 `_build_qli_metadata` 在没有可复用的 `cp_qli` 元数据时执行：

```python
max_seqlen_q = max(1, int(seq_lens_q.max().item()))
max_seqlen_k = max(1, int(seq_lens.max().item()))
```

这两个输入是 device tensors。`.max()` 在设备上计算，`.item()` 则要把结果
变成 host 上可立即使用的 Python 整数。因此 host 必须等到相应的设备工作
完成、结果可读，才能继续调用 QLI metadata 算子以及后续模型 forward。

问题不是传输两个整数的带宽，而是 **host 的后续提交依赖设备完成**：

```text
原路径：host 提交 device max → 等待结果 → 准备剩余 metadata → 提交 forward
补丁后：host 从已有 CPU 副本取 max     → 准备剩余 metadata → 提交 forward
```

如果 device max 排在同一 stream 的已有工作之后，host 还要等到这些前序工作
完成。等待期间，设备已经排队的任务可以继续执行；但 host 不能沿此调用路径
提前补充后续任务。队列若因此耗尽，就会出现供给气泡。多 rank 执行中，迟到的
rank 还可能让其他 rank 在后续 collective 处等待。

这里**不是主动暂停所有 device streams**，也不意味着每次 `.item()` 都产生
同样大的空洞。可见收益取决于当时排队深度和依赖关系。它消除的是一处同步来源，
不是保证所有波次都加速，更不是消除了整个引擎的同步。

## 3. 改法：只换最大值来源，不换实际访问数据

原生 `build_req_metadata` 已通过 `_build_local_token_metadata`，从 CPU
请求信息算出本 rank 对应的长度，并存入共享字典的 `_cpu_local`：

- `qsl_cpu`：本 rank 各请求的 query 累计起点。
- `sl_cpu`：对应请求的 KV 长度信息。

[补丁实现](./__init__.py)直接复用它们，不另做一次 device 到 CPU 拷贝：

```python
qsl = cpu["qsl_cpu"]
qlens = qsl[1 : num_reqs + 1] - qsl[:num_reqs]
max_seqlen_q = max(1, int(qlens.max().item()))
max_seqlen_k = max(1, int(cpu["sl_cpu"].max().item()))
```

例如，`qsl_cpu = [0, 6, 10]` 表示两个请求分别贡献 6、4 个本地 query；
最大 query 长度为 6。若 `sl_cpu = [100, 200]`，KV 长度最大值为 200。
这里仍有 `.item()`，但 tensor 已在 CPU 上，不需要等待 NPU。

随后仍调用原来的 `npu_vllm_quant_lightning_indexer_metadata`：

| 输入 / 行为 | 补丁后的处理 |
|---|---|
| `max_seqlen_q`、`max_seqlen_k` | 从 CPU 副本取得，供 tiling / 工作划分使用 |
| `actual_seq_lengths_query`、`actual_seq_lengths_key` | 保留原 device tensors 及原来的 clone，描述实际访问 |
| heads、head dim、TopK 数、布局、压缩比 | 保持原参数 |
| QLI 打分、TopK、后续 attention | 不修改计算实现 |
| `cp_qli` 缓存与 `req_qli_metadata` 缓冲写入 | 保持原有复用和交付方式 |

`cp_qli` 在本次元数据构建的共享字典中复用，避免相关 cache groups 重复生成；
不是把第一波的长度永久缓存下来。这个补丁也没有消除上述 clone 或 metadata
算子本身，只改变两个 Python 整数的取得方式。

## 4. 与跨步执行的关系：规划上界不等于实际长度

单看这个补丁，它使用已有 CPU 副本。与 [`cross_step`](../cross_step/README.md)
组合时，CPU 长度可能是保守上界，而不再与 device 的实际进度逐项相等。

这一组合依赖已验收的 DSACP 契约：**CPU 最大值用于规划，device tensors
控制实际长度和访问。** 上界可以留有余量，但不能低估实际工作范围，也不能拿
上界替换实际寻址数据。这个结论只适用于已验证的路径，不能直接移植到其他
attention backend 或任意配置。

原生 builder 前面仍可能因缺少 CPU 长度而执行 `seq_lens.cpu()`；本补丁
没有覆盖那个位置。因此不能把“QLI 取 max 不再等待”说成“元数据准备完全无同步”。

## 5. 安装、回退与验收边界

[`Worker.compile_or_warm_up_model`](../../worker.py) 在原生 warmup 完成后
显式调用 `qli_cpu.install(self)`，将类方法 `_build_qli_metadata` 替换为
本模块的 `_cpu_qli_metadata`。安装时的 `torch.npu.synchronize()` 是一次性
安装边界，不是逐波次同步；单纯 import 不会安装 hook。

- 仅在 `compressor_ratio == 4` 时启用；非 C4 或缺 `_cpu_local` 时调用原方法。
- 不新增 KV 状态，不更改调度和 CP / TP 布局，也不修改 draft 的独立 builder 路径。
- `_verify` 是内部调试开关，安装后默认关闭。开启后会重新读取 device maxima
  检查精确相等，恢复这里要消除的等待，不能用于性能测量；保守上界模式也不应
  用这种精确相等检查作为验收契约。它不是公开的启动选项。
- [`test_decode_protocol.py`](../../../../tests/test_decode_protocol.py) 检查 CPU
  最大值传参、验证分支及关闭回退；
  [`test_patch_installation.py`](../../../../tests/test_patch_installation.py)
  检查显式安装和 hook 所有权。CPU 测试不等于 NPU 性能或完整服务质量验收。
- 历史设备实验及组合收益见
  [`DECODE.md`](../../../../prototypes/full-mixed/DECODE.md) 和
  [`CROSS_STEP.md`](../../../../prototypes/full-mixed/CROSS_STEP.md)。不要把组合优化
  的加速数字全部归功于这两个最大值的改动。

这是同一 wheel 内的独立模块，不是另一个 pip 包。worker 统一检查版本和组合；
源码 / hook 独立不代表任意组合都已验收。更换组合应停服重启，不在活跃 graph
上热卸载。
