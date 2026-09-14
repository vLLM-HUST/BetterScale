# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned DSV4: device-stream ordering instead of a host replay fence.

The __call__ body is copied from pinned vLLM-Ascend's compilation/acl_graph.py.
Eager dispatch and capture stay native; the marked replay fence is the change.
This is not a two-step-ahead scheduler; graph-internal events remain unchanged.
"""

from contextlib import ExitStack
from unittest.mock import patch

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm_ascend.compilation import acl_graph as native
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

_installed = False


def install(worker):
    """Own both the wrapper hook and this worker's same-stream admission."""
    global _installed
    runner = worker.model_runner
    assert isinstance(runner.model, ACLGraphWrapper)
    assert runner.use_compress, "Only the pinned DSV4 compressed-attention path"
    assert runner.vllm_config.model_config.hf_config.model_type == "deepseek_v4"
    torch.npu.synchronize()  # explicit phase transition, never per replay
    runner.model._ordered_replay_stream = torch.npu.current_stream().npu_stream
    if not _installed:
        ACLGraphWrapper.__call__ = _ordered_call
        _installed = True
    return dict(rank=worker.rank, ordered_replay=True)


def _ordered_call(self, *args, **kwargs):
    forward_context = get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

    if (
        aclgraph_runtime_mode == CUDAGraphMode.NONE
        or aclgraph_runtime_mode != self.runtime_mode
    ):
        # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
        # running without aclgraphs.
        # We do not trigger capture/replay if the runtime mode is not
        # matches. This enables properly dispatching to the correct
        # CUDAGraphWrapper when nesting multiple instances with different
        # runtime modes.
        return self.runnable(*args, **kwargs)

    if batch_descriptor not in self.concrete_aclgraph_entries:
        # create a new entry for this batch descriptor
        self.concrete_aclgraph_entries[batch_descriptor] = native.ACLGraphEntry(
            batch_descriptor=batch_descriptor
        )

    entry = self.concrete_aclgraph_entries[batch_descriptor]

    if entry.aclgraph is None:
        if self.aclgraph_options.debug_log_enable:
            # Since we capture aclgraph for many different shapes and
            # capturing is fast, we don't need to log it for every
            # shape. E.g. we only log it for the first subgraph in
            # piecewise mode.
            native.logger.debug(
                "Capturing a aclgraph on (%s,%s)",
                self.runtime_mode.name,
                entry.batch_descriptor,
            )
        # validate that aclgraph capturing is legal at this point.
        native.validate_cudagraph_capturing_enabled()

        input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        entry.input_addresses = input_addresses
        aclgraph = torch.npu.NPUGraph()

        with ExitStack() as stack:
            if self.aclgraph_options.gc_disable:
                # during every model forward for piecewise aclgraph
                # mode, we will capture many pieces of aclgraphs
                # (roughly one per layer). running gc again and again
                # across layers will make the aclgraph capture very slow.
                # therefore, we only run gc for the first graph,
                # and disable gc for the rest of the graphs.
                stack.enter_context(patch("gc.collect", lambda: None))
                stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

            # mind-exploding: carefully manage the reference and memory.

            # Sync offloader's copy stream before capture.
            # Ensure any pre-capture prefetches from offloader are complete.
            from vllm.model_executor.offloader.base import get_offloader

            get_offloader().sync_prev_onload()
            forward_context.capturing = True
            try:
                with torch.npu.graph(aclgraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's aclgraph pool
                    output = self.runnable(*args, **kwargs)
                    # Join offloader's copy stream after forward to avoid
                    # unjoined stream error. The last layer's start_prefetch
                    # forks copy_stream, but wait_prefetch only happens in
                    # the next forward pass.
                    get_offloader().join_after_forward()
                    if self.aclgraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise aclgraph mode, because
                        # the output of the last graph will not be used by
                        # any other acl graph.
                        output = native.weak_ref_tensors(output)
            except RuntimeError as exc:
                if native._is_old_hdk_capture_error(exc):
                    raise RuntimeError(
                        "ACL graph capture failed with an old Ascend HDK/CANN stack "
                        "signature (`Alloc sq cq fail`). Please upgrade Ascend HDK to "
                        "25.5.1 or later and use the matching CANN stack.\n"
                        f"Original error:\n{exc}"
                    ) from exc
                elif native._is_stream_resource_capture_error(exc):
                    raise RuntimeError(
                        "ACL graph capture failed with a known stream-resource exhaustion "
                        "signature. Consider reducing cudagraph_capture_sizes, lowering "
                        "max_cudagraph_capture_size, preferring FULL or FULL_DECODE_ONLY for "
                        "mostly uniform decode workloads, or temporarily disabling graph mode "
                        "to confirm the failure is capture-related.\n"
                        f"Original error:\n{exc}"
                    ) from exc
                raise

        # here we always use weak ref for the workspaces
        # to save memory
        native.weak_ref_workspaces(native._graph_params)
        native.weak_ref_workspaces(native._draft_graph_params)
        native.weak_ref_workspaces(native._draft_graph_prefill_params)

        # here we always use weak ref for the output
        # to save memory
        entry.output = native.weak_ref_tensors(output)
        entry.aclgraph = aclgraph

        native.compilation_counter.num_cudagraph_captured += 1

        # important: we need to return the output, rather than
        # the weak ref of the output, so that pytorch can correctly
        # manage the memory during acl graph capture
        return output

    if self.is_debugging_mode:
        # check if the input addresses are the same
        new_input_addresses = [
            x.data_ptr() for x in args if isinstance(x, torch.Tensor)
        ]
        assert new_input_addresses == entry.input_addresses, (
            f"Input addresses for aclgraphs are different "
            f"during replay. Expected {entry.input_addresses}, "
            f"got {new_input_addresses}"
        )

    # 唯一的执行策略差异：原生这里在 FULL replay 前等待当前 stream 清空。
    # DSV4 compressed 路径不走逐轮 host graph-task 参数更新；输入/metadata
    # 生产与 replay 已由同一 stream 排序，因此准入的主模型不必让 host 再等。
    # 未准入的 wrapper、非 FULL、capture 中的调用仍保留原生同步条件。
    stream = self.__dict__.get("_ordered_replay_stream")
    ordered = (
        stream is not None
        and self.runtime_mode == CUDAGraphMode.FULL
        and not forward_context.capturing
    )
    if ordered:
        assert (
            torch.npu.current_stream().npu_stream == stream
        ), "Unverified producer/replay stream"
    else:
        native.logger.info_once("Replaying aclgraph")
        # 原生保护：host graph-task 参数更新不能与上一轮 replay 发生冲突。
        # ENPU 与 merged EAGLE draft 原本就有各自的同步例外，保持不变。
        is_draft_eagle = native._EXTRA_CTX.is_draft_model and self.use_eagle
        need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
        if not self.enable_enpu and need_sync:
            torch.npu.current_stream().synchronize()
    entry.aclgraph.replay()
    return entry.output
