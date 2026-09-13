# 同 stream 的有序 replay

完整拥有 `ACLGraphWrapper.__call__` hook 和各 worker model 的 stream 准入。
已捕获的 DSV4 FULL entry 依赖设备顺序 replay；不满足准入时调用原生 wrapper。

- 入口：`install(worker)`，原生 warmup 后调用。现在不借 target_full 安装 wrapper。
- 不依赖其他 patch；但必须已有原生 FULL graph、同一输入生产/执行 stream，
  且是无需逐次 host graph-task 更新的已验证 DSV4 路径。不是任意 backend 的通用优化。
- warmup 期间原来的 wrapper 也因未准入而回退，现在直接保留原生 wrapper；
  warmup 后再装 hook 和准入。内部 graph events 保持不变。
- 不是完整连续调度，不承诺独立大幅提速。测试：`test_decode_protocol.py`、
  `test_patch_installation.py`（无需 target patch 即能安装，并保留 fallback）。

## 安装边界

这是同一 wheel 内独立的 Python 模块，不是新的 pip 发行包。没有自动注册，
import 本模块不改 donor；worker 显式调用 `install`。版本/组合检查由 worker
统一执行。这里的独立是源码与 hook 所有权独立，不保证任意组合都已有性能/质量验收。
停掉服务再更换组合，不在活跃 graph 上热卸载。
