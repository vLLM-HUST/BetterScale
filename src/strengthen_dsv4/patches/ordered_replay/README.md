# 同 stream 的有序 replay

完整拥有 `ACLGraphWrapper.__call__` hook 和各 worker model 的 stream 准入。
已捕获的 DSV4 FULL entry 依赖设备顺序 replay；不满足准入时保留原生执行逻辑。

## 源码怎么读

`__init__.py` 的 `_ordered_call` 直接展开了 pinned vLLM-Ascend
`compilation/acl_graph.py` 中 `ACLGraphWrapper.__call__` 的主体，不再经过
`call(original, wrapper, ...)` 转发，也不隐藏调用旧 `__call__`。从上到下是：

1. 非匹配 graph 模式：调用原 runnable，保留 eager / 嵌套 wrapper 分派。
2. 新 bucket：创建 entry，执行原生 capture，保留 offloader 同步、异常处理、
   workspace / output 弱引用和 capture 计数。
3. 已捕获 bucket：检查调试输入地址，进入带中文注释的 replay 同步分支。
   准入的 DSV4 FULL 同 stream 调用省去 host fence；其余调用仍按原生
   ENPU / EAGLE / FULL 条件决定是否 synchronize，随后 replay。

`native.*` 仅引用上游已有的辅助函数、类型及模块状态，不转发整个调用。尤其
`_graph_params` 等工作区状态必须在使用时从上游模块读取，不能复制 import
时的值，否则后续重新绑定会让本地副本过期。上游许可和来源保留在文件头；
升级 pin 时需要复核这一份展开代码，而不是假定它会自动跟随上游。

测试同时检查展开的 dispatch / capture 主体与 pin 的 AST 一致，以及准入
replay 不同步、未准入 replay 保留同步、错误 stream 拒绝执行和原生例外。


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
