"""Exact-shape DSpark private runtime-capture banks; not general graph admission.

One observed pure-verification shape per request count (1–4) is captured.
Other shapes fall back; the bank cache cannot grow without bound.
Draft metadata has a PRIVATE stable bank, never the target global RoPE bank.
"""

# 阅读入口：这是「把 draft 计算主体变成 graph」的机制，不是双槽连续调度器。
# 已维护服务的实际调用链（相邻文件可以顺着读）：
#   用户 vllm serve --worker-cls strengthen_dsv4.worker.Worker
#   → Worker.compile_or_warm_up_model() 先完成 donor 原生 warmup
#   → 调用 split_draft/__init__.py 的 install(worker)
#   → drafter._runnable 被替换成 DraftGraphRunner
#   → 管理器直接选择 decode/query 缓存里的 ExactDraftGraph
#   → 非普通 decode 由外层管理器先写真实长度的 context，再借本类 capture query。
#
# 拦截位置是 donor 已准备好输入和 forward context 之后的 _runnable 调用。
# 原生入口见 pinned vllm_ascend/spec_decode/llm_base_proposer.py：
# _runnable 原本指向 _run_merged_draft；DSpark 继承它并关闭原生 graph。
# 我们保留该函数的数学计算，只把其设备工作记录下来，再按固定地址 replay。
# metadata 的 Python 构造和本文件 refresh 仍在图外；没有捕获整个 worker。
#
# 生效范围由 config.validate_worker_config 再收窄为：DSV4 Flash W8A8、
# TP8/DP1/PP1+EP、DSACP、K5、最多四席位、原生 scheduler 等已验收配置。
# 本文件不改 target、权重、KV 分布、拒绝验证或请求调度；不选此 worker 就不安装它。
# 普通 decode 每个请求数只有一个 exact-shape graph，不是每个桶双 graph 交替。
# 代价是各 graph 的 metadata 副本、输出/临时存储及首次 capture；不是零成本。
# run026 同等四请求 K5 周期约65→52ms，证据见 prototypes/full-mixed/DECODE.md；
# 不能把该周期收益当成稳定端到端吞吐，或转移给后来未采用的连续提交方案。
import copy
import dataclasses
from enum import Enum
import torch
from vllm.forward_context import get_forward_context
from vllm_ascend.attention.context_parallel.dsa_cp import RopeDataProxy


# 捕获时 Python 分支/标量也可能被固定，不能只比较 tensor 地址。
# GPU tensor 这里只记 device/dtype/shape，内容允许逐波刷新，不读回 GPU。
# CPU tensor 的值和普通标量进入签名；变化时不能假装还是同一个计算程序。
# 固定的原生状态输入另用 persistent_layout 检查地址、shape 和 stride。
# 这不是任意 tensor 别名/布局的通用证明，只覆盖 pinned donor 的已验收组织。
def signature(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            str(value.device),
            str(value.dtype),
            tuple(value.shape),
            tuple(value.flatten().tolist()) if value.device.type == "cpu" else None,
        )
    if isinstance(value, RopeDataProxy):
        return ("rope", value.idx, signature(value._data))
    if dataclasses.is_dataclass(value):
        return (
            type(value).__name__,
            tuple(
                (f.name, signature(getattr(value, f.name)))
                for f in dataclasses.fields(value)
            ),
        )
    if isinstance(value, dict):
        return tuple((k, signature(v)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(map(signature, value)))
    if isinstance(value, Enum):
        return (type(value).__name__, value.name)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unreviewed graph metadata type: {type(value)}")


def persistent_layout(value):
    if isinstance(value, torch.Tensor):
        return (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
        )
    if isinstance(value, (tuple, list)):
        return tuple(map(persistent_layout, value))
    if value is None:
        return None
    raise TypeError(f"Unreviewed persistent draft input: {type(value)}")


def key_changes(old, new, path=()):
    if old == new:
        return []
    if isinstance(old, tuple) and isinstance(new, tuple) and len(old) == len(new):
        result = []
        for i, (a, b) in enumerate(zip(old, new)):
            result.extend(key_changes(a, b, path + (i,)))
            if len(result) >= 32:
                break
        return result[:32]
    return [dict(path=path, old=repr(old)[:500], new=repr(new)[:500])]


# 复制的是 kwargs/attention metadata 树，不是整个 drafter 或 KV cache。
# 尤其 RopeDataProxy 要复制其 _data，不能让 target 下一次 metadata 准备覆盖
# draft graph 所绑定的 RoPE。模型权重、KV backing、原生输入/反馈缓冲仍由 donor 持有。
# 这里是朴素的递归 clone，没有保留重复引用的 DAG 共享；后来的去重实验未合入。
def bank(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, RopeDataProxy):
        result = copy.copy(value)
        result._data = bank(value._data)
        return result
    if dataclasses.is_dataclass(value):
        result = copy.copy(value)
        for f in dataclasses.fields(value):
            setattr(result, f.name, bank(getattr(value, f.name)))
        return result
    if isinstance(value, dict):
        return {k: bank(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bank(v) for v in value]
    if isinstance(value, tuple):
        return tuple(bank(v) for v in value)
    return value


# 只复制内容，不把捕获时的目标 tensor 换成新对象。replay 仍读取旧的固定地址。
# non_blocking=True 不代表自动创建传输 stream 或保证消除等待；本路径依赖原生
# 提交顺序，当前副本刷新与 replay 有序执行。它不是 LiveInfer 的双槽 H2D 协议。
def refresh(dst, src):
    if isinstance(src, torch.Tensor):
        dst.copy_(src, non_blocking=True)
    elif isinstance(src, RopeDataProxy):
        refresh(dst._data, src._data)
    elif dataclasses.is_dataclass(src):
        for f in dataclasses.fields(src):
            refresh(getattr(dst, f.name), getattr(src, f.name))
    elif isinstance(src, dict):
        for k, v in src.items():
            refresh(dst[k], v)
    elif isinstance(src, (list, tuple)):
        for d, s in zip(dst, src):
            refresh(d, s)


# 一个 entry 对应一个捕获程序及其私有 metadata，不是一个请求的 KV 生命周期。
# 同形状的后续请求可以复用 entry，前提是所有运行时数据都按原生协议刷新。
class ExactDraftGraph:
    def __init__(
        self,
        worker,
        original=None,
        request_count=4,
        context_capacity=None,
        reference=None,
    ):
        self.worker = worker
        self.drafter = worker.model_runner.drafter
        assert type(self.drafter).__name__ == "AscendDSparkProposer"
        assert not self.drafter.use_cuda_graph
        # 保存安装前的 bound callable。capture 和 fallback 调用同一原生实现，
        # 不能从已经被我们替换的 drafter._runnable 再调用，否则会递归进入自身。
        self.original = original or self.drafter._runnable
        self.request_count = request_count
        # 6 是此验收配置的 K+1，不是可以任意沿用到其他 K 的常量。
        self.context_capacity = context_capacity or 6 * request_count
        self.reference = reference or self.original
        self.enabled = True
        self.key = None
        self.graph = None
        self.replays = 0
        self.fallbacks = 0
        self.checks = 0
        self.capture_checked = False
        self.signed_zero_words = 0

    def __call__(self, **kwargs):
        if not self.enabled:
            return self.original(**kwargs)
        ctx = get_forward_context()
        d = self.drafter
        # 两个使用者共用本类：普通 decode 捕获 context+query；split 路线已经
        # 在外面完成 context 写入，临时屏蔽原生 context hook 后只捕获 query。
        query_only = getattr(self, "query_only", False)
        eligible = (
            kwargs["batch_size"] == self.request_count
            and (query_only or d._dflash_num_context == self.context_capacity)
            and not kwargs.get("is_prefill", False)
            and not ctx.capturing
        )
        if not eligible:
            self.fallbacks += 1
            return self.original(**kwargs)
        current = (kwargs, ctx.attn_metadata)
        # 这里检查原生持久输入/反馈缓冲的布局，不重绑这些属性或改变其所有权。
        # query-only 图不读取大 context 的三个输入，所以不把它们纳入其地址契约。
        names = (
            "input_ids",
            "positions",
            "_dspark_seed_buffer",
            "_dspark_draft_buffer",
        )
        if not query_only:
            names += (
                "_dflash_hidden_states",
                "_context_positions_buffer",
                "_context_slot_mapping_buffers",
            )
        state_inputs = tuple(persistent_layout(getattr(d, name)) for name in names)
        key = (
            None if query_only else d._dflash_num_context,
            signature(current),
            state_inputs,
        )
        # 普通 decode 对不匹配的签名回退；split entry 设置 strict_signature，
        # 因为其输入和 context hook 已转换，不能静默改走另一种程序，故显式报错。
        if self.key is not None and key != self.key:
            if getattr(self, "strict_signature", False):
                raise AssertionError(
                    f"FULL draft bank signature changed: {key_changes(self.key,key)}"
                )
            self.fallbacks += 1
            return self.original(**kwargs)
        # 第一次遇到这个已准入形状才捕获，不在每波重新 capture。
        # 这里的全设备 synchronize 是首次初始化成本，不是稳态每次 replay 的 fence。
        if self.key is None:
            self.key = key
            self.buffers = bank(current)

            def capture():
                # 同时换入 kwargs 与 forward context 两处 metadata，确保原生
                # 算子看见的都是 graph 私有地址；finally 恢复图外 context。
                old = ctx.attn_metadata
                ctx.attn_metadata = self.buffers[1]
                graph = torch.npu.NPUGraph()
                torch.npu.synchronize()
                was_capturing = ctx.capturing
                ctx.capturing = True
                try:
                    with torch.npu.graph(graph):
                        self.output = self.original(**self.buffers[0])
                finally:
                    ctx.attn_metadata = old
                    ctx.capturing = was_capturing
                self.graph = graph
                # Runtime capture is initialization, not a committed serving
                # invocation. Execute explicitly before exposing output/KV state.
                graph.replay()
                return self.output

            capture()
            return self.output
        # 稳态热路径：检查契约 → 刷新固定 metadata → replay。
        # output 是 graph 保留的缓冲，会被后续 replay 复用；这里没有复制输出、
        # 并发调用保护或跨 stream 消费协议，仍须遵守 donor 的原生消费/复用顺序。
        refresh(self.buffers, current)
        self.graph.replay()
        self.replays += 1
        return self.output
