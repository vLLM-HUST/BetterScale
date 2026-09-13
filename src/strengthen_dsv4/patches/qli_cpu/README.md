# 使用现有 CPU mirrors 生成 QLI tiling maxima

只替换 `AscendDSACPMetadataBuilder._build_qli_metadata`。C4 路径用原生 CPU
query/sequence mirrors 取最大值，避免 GPU `.item()`；GPU 实际寻址数据保持原样。

- 入口：`install(worker)`，原生 warmup 后调用；import 不再偷偷改类方法。
- 非 C4 或缺 CPU mirrors 时走原生 builder；不创建第二份状态，不依赖其他 patch。
- 如果与 cross_step 组合，CPU 值可能是保守上界而不是精确长度；正确性依赖
  已验证 DSACP「CPU 控制 tiling，device 控制实际访问」契约。不能照搬到别的 backend。
- 验证模式不在生产打开。测试：`test_decode_protocol.py`、`test_patch_installation.py`。

## 安装边界

这是同一 wheel 内独立的 Python 模块，不是新的 pip 发行包。没有自动注册，
import 本模块不改 donor；worker 显式调用 `install`。版本/组合检查由 worker
统一执行。这里的独立是源码与 hook 所有权独立，不保证任意组合都已有性能/质量验收。
停掉服务再更换组合，不在活跃 graph 上热卸载。
