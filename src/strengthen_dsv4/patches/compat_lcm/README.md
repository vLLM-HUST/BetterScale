# K5/TP 图桶对齐修复

只负责 `CompilationConfig.adjust_cudagraph_sizes_for_spec_decode`：sequence
parallel 开启时用 LCM(K+1, TP) 选择共同倍数，不改变 K 或模型算法。

- 入口：`install()`，在 runner 初始化/选择 capture buckets 之前调用。
- 不安装 target FULL、draft graph 或 replay hook；不依赖其他 patch 模块。
- 这是启动兼容修复，不单独认领推理加速。pinned donor 的部分未优化对照也需要它。
- 测试：`tests/test_target.py` 的 alignment contract、`test_patch_installation.py`。

## 安装边界

这是同一 wheel 内独立的 Python 模块，不是新的 pip 发行包。没有自动注册，
import 本模块不改 donor；worker 显式调用 `install`。版本/组合检查由 worker
统一执行。这里的独立是源码与 hook 所有权独立，不保证任意组合都已有性能/质量验收。
停掉服务再更换组合，不在活跃 graph 上热卸载。
