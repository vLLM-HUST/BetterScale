# 稳定 K5 波次的收据后移

拥有 runner 的 input/state/forward 五处 hook：稳定请求集合下先提交 target，
再退役输入 DMA、执行原 CPU bookkeeping。device 精确进度、位置和 KV 映射仍是原生。

- 入口：`install(worker)`，原生 warmup 后调用。
- 不 import 或安装其他 patch；本模块通过 runner 原生接口完成工作。
- 依赖已验证 DSACP、async K5 和输入 DMA 所有权等执行前提；prefill、请求变更等
  不满足准入时保留原有处理。并非所有 async backend 都能使用这一切分。
- 与 qli_cpu 共享 CPU 上界/device 精确状态语义，但并不是必须导入 qli_cpu 才工作。
  原有收益是在既有 graph 组合上隔离测得，不声称任意单模块部署也有相同收益。
- 不带双槽/N+2，不提前回收 KV。测试：`tests/test_cross_step.py`。

## 安装边界

这是同一 wheel 内独立的 Python 模块，不是新的 pip 发行包。没有自动注册，
import 本模块不改 donor；worker 显式调用 `install`。版本/组合检查由 worker
统一执行。这里的独立是源码与 hook 所有权独立，不保证任意组合都已有性能/质量验收。
停掉服务再更换组合，不在活跃 graph 上热卸载。
