# Draft 完整小图与 context/query 拆分

## 原生已经有什么，这个补丁补什么

在本包固定的 vLLM-Ascend `9bf964cb` 中，通用 proposer 已经支持 draft FULL
graph：`spec_decode/llm_base_proposer.py` 根据 `use_cuda_graph`，将
`_run_merged_draft` 包进 `ACLGraphWrapper`。**Draft graph 本身不是本包的创新。**
但 `spec_decode/dspark_proposer.py` 的构造函数明确设置
`self.use_cuda_graph = False`，dummy/warmup 路径也据此选择 `NONE`。
因此不能从通用 proposer 的能力推断这个版本的 DSpark 已支持原生 graph。

启动命令中的 `speculative-config.enforce_eager: true` 是让原生 drafter
保持这个已验收的初始化路径，不表示补丁安装之后仍全部 eager。Worker 在原生
warmup 之后安装本目录的 `DraftGraphRunner`，接管 TP 的 `_runnable`。
本补丁的增量是这条 **DSpark 路径的图执行，以及 context 写入与 query 执行拆分**，
不是重新发明原生通用 graph wrapper。直接把配置改成 `false` 不能绕过 DSpark
构造函数的强制关闭，也不代表获得了同样的拆分行为。

这个范围限于当前 TP 安装组合。DP Worker 没有安装本目录，仍保留 eager DSpark；
DP 的 target/preparation/metadata graph 不能算作 draft model graph 覆盖。

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
