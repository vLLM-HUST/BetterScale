# Target FULL 的捕获契约

只拥有 DSACP support/build/RoPE 和 runner FIA padding 四处 hook。固定 graph
需要的请求容量与地址，空席位采用零长度，非 FULL padding 回到原生实现。

- 入口：`install()`，在 runner 初始化/target capture 前调用。
- 不安装 LCM 对齐，也不安装 ordered replay；不 import 其他 patch。
- 对 K5/TP8+SP，worker 独立安装 `compat_lcm` 解决原生桶对齐限制；这是该部署的
  配置前提，不把修复藏在本模块里。target 已有 FULL decode 时也不等于需捕获 prefill。
- 测试：`tests/test_target.py`、`test_patch_installation.py`。原八卡证据见
  仓库 `prototypes/full-mixed/README.md`，本次移动不新增设备性能结论。

## 安装边界

这是同一 wheel 内独立的 Python 模块，不是新的 pip 发行包。没有自动注册，
import 本模块不改 donor；worker 显式调用 `install`。版本/组合检查由 worker
统一执行。这里的独立是源码与 hook 所有权独立，不保证任意组合都已有性能/质量验收。
停掉服务再更换组合，不在活跃 graph 上热卸载。
