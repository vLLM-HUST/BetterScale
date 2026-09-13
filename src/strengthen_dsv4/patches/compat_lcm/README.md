# Compat LCM：让 K5 与 TP8 的 graph 桶对齐约束正常共存

这个补丁修复 **投机解码与 sequence parallel 同时开启时的 capture 桶计算**。
原生代码用 `max(K+1, TP)` 寻找共同对齐单位，遇到 6 和 8 这样的组合就报错；
我们改用最小公倍数 `LCM(K+1, TP)`。这是启动兼容修复，不改变投机长度或模型
算法，也不单独认领推理加速。

## 为什么两个约束必须同时满足？

同一 graph 桶的 token 容量需要同时满足：

- K5 投机验证：每请求 `K+1 = 6` 个 query，桶容量需要是 6 的倍数。
- 开启 sequence parallel 的 TP8：token 维度分给 8 个 ranks，需要是 8 的倍数。

这里的 SP 是计算中的 token 维切分约束，不是历史 KV 的 context-parallel 分片。
6 和 8 并不冲突；共同可用的容量是 24、48、72……

## 上游为什么失败，补丁怎么修？

上游 [`CompilationConfig.adjust_cudagraph_sizes_for_spec_decode`](../../../../upstream/vllm/vllm/config/compilation.py)
先取较大值，再检查能否同时整除：

```text
原生：max(6, 8) = 8 → 8 不能被 6 整除 → 报错
补丁：lcm(6, 8) = 24 → 同时满足两个约束 → 继续规划 capture 桶
```

[`_adjust_joint_alignment`](./__init__.py) 只在 `enable_sp` 时计算 LCM，
随后调用保存下来的原函数，继续使用其向上取整、去重和最大容量过滤逻辑。
SP 关闭时参数原样传回；TP1 或本来就互相整除的组合不会被无谓扩大。

例如，在最大 capture 容量允许的情况下，候选桶 32 会对齐到 48；如果向上
取整超过最大容量，仍由原函数过滤。如果最后没有合法桶，仍会报错，不会绕过
配置上限。向上对齐可能增加 padding，因此不能把它解释成“扩大桶必然更快”。

传入原函数的 24 是**桶的对齐单位**。虽然原参数名叫
`uniform_decode_query_len`，这个方法仅用它调整 capture sizes；补丁没有修改
实际每请求的 6 个 query、`num_speculative_tokens=5` 或接受 / 采样算法。

## Hook 在哪里，何时生效？

```text
Worker.__init__
  → 检查 pinned runtime 和支持的配置
  → compat_lcm.install()
       替换 CompilationConfig.adjust_cudagraph_sizes_for_spec_decode
  → 原生 worker 初始化
       后续 runner 配置 graph 模式 / capture sizes
         → resolve_cudagraph_mode_and_sizes
           → _adjust_joint_alignment
             → 原 adjust_cudagraph_sizes_for_spec_decode（传入共同对齐单位）
```

原生 MRV1 在 FULL decode 且 uniform query 长度大于 1 时调用这个调整方法；
本补丁不为其他调用路径另建桶规划器。安装必须早于桶计算，因此由
[`Worker`](../../worker.py) 在原生初始化前调用 `install()`，不是 warmup 后才装。

`install` 是幂等的；import 不安装 hook。它只负责这个配置方法，不安装
`target_full`、draft graph 或 replay hook，不依赖其他补丁模块。

## 验收与使用边界

- [`test_target.py`](../../../../tests/test_target.py) 检查 K5/TP8 得到 24、
  K5/TP2 保持 6、关闭 SP 保持原 query 对齐。
- [`test_patch_installation.py`](../../../../tests/test_patch_installation.py)
  检查独立安装、重复安装及不修改其他 hook。
- 部分 pinned donor 的未优化对照也需要此修复，才能启动相同 K5/TP8+SP 配置；
  因此不能把“去掉整个补丁 worker”当成必然可运行的相同配置 baseline。
- 同一 wheel 内的独立模块，不是另一个 pip 包。更换组合应停服重启，不在
  已捕获的 graph 上热改 capture sizes。版本升级时应复核上游是否已修复此处。
