"""Native context ingestion outside a bounded query-only graph; no large padding.

Keep the already-qualified fused small-context K5 route for normal decoding.
The split route uses the current stream for context writes and query reads.
"""

import copy
from contextlib import contextmanager
import torch
from torch.profiler import record_function
from vllm.forward_context import get_forward_context
from ._graph import ExactDraftGraph


def graph_metadata(value):
    if isinstance(value, dict):
        return {k: graph_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [graph_metadata(v) for v in value]
    if type(value).__name__ == "AscendDSAMetadata":
        result = copy.copy(value)
        # In the pinned DSACP forward, only num_prefills > 0 is consumed.
        # Exact counts were used by the builder already; tensor offsets and
        # req.num_reqs_actual remain untouched. Do not specialize graph bodies
        # on logging-only decode counts or the number of prefill requests.
        result.num_prefills = int(value.num_prefills > 0)
        result.num_decodes = result.num_decode_tokens = 0
        return result
    return value


@contextmanager
def query_body(drafter):
    """Only the native merged-call context hook is skipped; restore on failure."""
    original = drafter.build_model_inputs_first_pass
    drafter.build_model_inputs_first_pass = lambda *args, **kwargs: None
    try:
        yield
    finally:
        drafter.build_model_inputs_first_pass = original


# 在 donor 源码中的位置（文件均位于 pinned vllm_ascend）：
#   worker/model_runner_v1.py: sample_tokens()
#     → 当前 target logits 的原生采样/拒绝验证
#     → propose_draft_token_ids() → drafter._propose(...)
#   spec_decode/llm_base_proposer.py: _propose()
#     → 准备 draft 输入、attention metadata，进入 Ascend forward context
#     → self._runnable(**model_inputs)  ← 本类接在这里
#
# 安装点是下面的 install(worker)：原生 warmup 完成后，保存原 _runnable
# （即 donor 的 _run_merged_draft），再将该属性绑定到 DraftGraphRunner 实例。
# 调用方不变；Python 调用实例时进入这里的 __call__，不是另起一套 proposer。
#
# 在一波推理中的位置：
#   target forward → 原生采样/验证 → [本类执行 draft 计算主体]
#     → 返回 K5 draft token IDs 给原生 _propose / model runner
#     → runner 保存 _draft_token_ids，供后续 target 波次验证。
# 因此本类不是 target forward 的 wrapper，也不负责整个 worker 的调度。
# draft 的 Python 输入/metadata 准备仍在图外，候选接受与否仍由原生验证决定。
#
# 本类只选择如何执行同一份原生 draft 主体：
#   普通 K5：context KV 更新 + query/Markov 计算一起进小图；
#   其他准入波次：先按真实行数更新 context KV，再运行 query 小图。
# 两种 graph 缓存直接由这里持有；单图固定地址、capture/replay 交给 ExactDraftGraph。
# 没有第二层 graph-set 转发，也没有双槽/N+2 调度协议。
class DraftGraphRunner:
    def __init__(self, worker):
        self.worker = worker
        self.drafter = worker.model_runner.drafter
        self.original = self.drafter._runnable
        assert self.drafter.num_speculative_tokens == 5, "Only K5 is qualified here"
        assert worker.model_runner.vllm_config.scheduler_config.max_num_seqs == 4
        assert self.drafter.parallel_drafting
        self.enabled = True
        self.decode_graphs = {}
        self.query_graphs = {}
        self.context_calls = self.context_rows = 0
        self.context_max = 0

    def __call__(self, **kwargs):
        d, ctx = self.drafter, get_forward_context()
        assert not ctx.capturing
        count, actual = kwargs["batch_size"], d._dflash_num_context
        assert 1 <= count <= 4 and actual > 0
        if not self.enabled:
            return self.original(**kwargs)
        prefill = any(bool(m.num_prefills) for m in ctx.attn_metadata.values())
        # 普通 K5：每请求 K+1=6 行小 context，完整 context+query 一起 replay。
        # 这里并非所有声称 decode 的调用都能进图：内部还检查固定输入/metadata 签名。
        if actual == 6 * count and not prefill and not kwargs.get("is_prefill", False):
            # 外层已检查请求数、context 行数和模式；每个请求数只留一个 entry。
            if count not in self.decode_graphs:
                self.decode_graphs[count] = ExactDraftGraph(
                    self.worker, self.original, count
                )
            return self.decode_graphs[count](**kwargs)

        # Exactly the unpadded native operation, before ANY query graph capture.
        # No H2D round-trip or host fence is required between these same-stream ops.
        # 其他波次仍有很多 target hidden 行需要写进 draft 的 context KV；
        # 只调用 donor 原生写入，不把这些行补到大桶后塞进小 query 图。
        with record_function("strengthen::draft_context_ingest"):
            d.build_model_inputs_first_pass(
                kwargs["num_input_tokens"], d._context_slot_mapping_buffers
            )
        self.context_calls += 1
        self.context_rows += actual
        self.context_max = max(self.context_max, actual)
        original_metadata = ctx.attn_metadata
        ctx.attn_metadata = graph_metadata(original_metadata)
        kwargs = dict(kwargs, target_positions=None, is_prefill=False)
        assert kwargs["inputs_embeds"] is None
        if "multi_steps_attn_metadata" in kwargs:
            kwargs["multi_steps_attn_metadata"] = graph_metadata(
                kwargs["multi_steps_attn_metadata"]
            )
        modes = {
            (m.attn_state.name, bool(m.num_prefills))
            for m in ctx.attn_metadata.values()
        }
        assert len(modes) == 1
        mode, prefill = modes.pop()
        key = (count, mode, prefill)
        try:
            # 原生 _run_merged_draft 内仍会调用 context hook：在此作用域临时
            # 将其变成 no-op，避免写两遍；query_body 的 finally 保证异常也恢复。
            # 数学 query/Markov 逻辑仍由原生 callable 执行，不是自写第二个 drafter。
            with query_body(d), record_function("strengthen::draft_query_graph"):
                if key not in self.query_graphs:
                    entry = ExactDraftGraph(self.worker, self.original, count)
                    entry.query_only = True
                    entry.strict_signature = True
                    entry.reference_kind = "native-query-after-native-unpadded-context"
                    self.query_graphs[key] = entry
                output = self.query_graphs[key](**kwargs)
                assert self.query_graphs[key].fallbacks == 0
                return output
        finally:
            ctx.attn_metadata = original_metadata


# worker.compile_or_warm_up_model() 调用这里。
# 保存原 _runnable 后再替换实例属性；原生上游调用点不变，动态派发到管理器。
# 影响的是本 worker 的 DSpark proposal 主体，调度、target 与拒绝验证仍走原生链路。
def install(worker):
    torch.npu.synchronize()
    assert not hasattr(worker, "_exact_draft_graph")
    runner = DraftGraphRunner(worker)
    worker.model_runner.drafter._runnable = runner
    worker._exact_draft_graph = runner
    return dict(rank=worker.rank, policy="native-context-plus-query-graph")
