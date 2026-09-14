# Draft 完整小图与 context/query 拆分

本目录是一个完整交付功能，全部辅助实现都留在内部：

- `__init__.py`：安装与分流；`DraftGraphRunner` 直接持有 decode/query 两个缓存，
  替换当前 drafter 的 `_runnable`，没有继承或嵌套的 graph-set 管理器。
- `_graph.py`：固定 metadata bank、签名准入、首次 capture/replay 与普通 K5 小图。
- query metadata 的小型整理函数直接放在 `__init__.py`，不再单独跳文件。

普通 K5 decode 用 fused context+query 小图。其他准入波次先原生写入真实长度的
context KV，再执行小 query 图；不会把大 context 一起补到 graph 上限。
保存并调用原生 runnable，保留原生神经网络、Markov 逻辑和后续拒绝验证。

- 入口：`install(worker)`，原生 warmup 后调用；首次实际 shape 触发 lazy capture。
- 不 import 其他 patch，也不安装 target、通信或 scheduler hook。
- 仍限已验证 DSpark K5、四席位、原生持久输入/同 stream 生命周期。
  普通 decode 不匹配时 fallback；转换后的 query-only 签名不匹配必须报错。
- 内部 _graph 不是独立交付包：拆分功能复用它，放在一起避免跨 patch 互相伸手。
- 测试：`test_decode_protocol.py`、`test_split_draft.py`；历史机制/质量/性能
  证据保留在 `prototypes/full-mixed/DECODE.md`、`SPLIT_DRAFT.md`，不混同新硬件验收。

## 安装边界

这是同一 wheel 内独立的 Python 模块，不是新的 pip 发行包。没有自动注册，
import 本模块不改 donor；worker 显式调用 `install`。版本/组合检查由 worker
统一执行。这里的独立是源码与 hook 所有权独立，不保证任意组合都已有性能/质量验收。
停掉服务再更换组合，不在活跃 graph 上热卸载。
