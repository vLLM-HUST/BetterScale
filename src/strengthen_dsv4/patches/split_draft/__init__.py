"""Native context ingestion outside a bounded query-only graph; no large padding.

Keep the already-qualified fused small-context K5 route for normal decoding.
The split route uses the current stream for context writes and query reads.
"""

from contextlib import contextmanager
import torch
from torch.profiler import record_function
from vllm.forward_context import get_forward_context
from ._graph import ExactDraftGraph
from ._metadata import graph_metadata


@contextmanager
def query_body(drafter):
    """Only the native merged-call context hook is skipped; restore on failure."""
    original = drafter.build_model_inputs_first_pass
    drafter.build_model_inputs_first_pass = lambda *args, **kwargs: None
    try:
        yield
    finally:
        drafter.build_model_inputs_first_pass = original


# 一个管理器直接拥有两种 graph 缓存：普通 decode 和拆分后的 query。
# 两条路径共用 ExactDraftGraph，但不再嵌套另一层 graph-set 管理器。
class SplitDraftGraphSet:
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
    manager = SplitDraftGraphSet(worker)
    worker.model_runner.drafter._runnable = manager
    worker._exact_draft_graph = manager
    return dict(rank=worker.rank, policy="native-context-plus-query-graph")
