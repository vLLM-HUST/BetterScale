# BetterScale-owned execution and State closure

This package owns the common LiveModule/StateTensor runtime under
`betterscale.live`. It does **not** import, wrap, alias, install or locate the
external `livemodule` package. Consumers import BetterScale directly. Torch is
a host-runtime prerequisite, as for the existing BetterScale integration.

## Source and scope

One-time source transfer from LiveInference commit
`05ac15419c0e73650e687ceb9daffeb7874865f0`, `src/livemodule/`, Apache-2.0.
The original SPDX notices are preserved; the license is distributed in
`licenses/Apache-2.0.txt`. The common runtime paths preserve donor relative paths. This is a disconnected intake, not an automatic synchronization process.

The closure starts at `core/live_module.py`, `core/state_tensor.py`,
`runtime/live_runtime.py`, `runtime/state_backend.py` and
`runtime/grouped_state.py`. Following all imports, including local/lazy and
type-checking imports, yields22 implementation modules. Retaining complete
modules preserves generation activation/rollback, graph and invocation lifetime,
metadata/shadow, capacity fitting, host-State and pipeline protocol dependencies.
For that common closure only stdlib and Torch remain external. The architecture binding is a capability
interface; no architecture implementation, model, serving stack, native extension
or operator bundle is imported from LiveInference.

Changes from the source are namespace substitution (`livemodule` →
`betterscale.live`) and relocation notices. The root exports the common API but
omits the old lazy `ACLGraphBackend` compatibility export: that architecture port
is not part of this closure. No `sys.modules` compatibility alias is installed.
Large source modules are deliberately faithful transfers, not freshly designed
BetterScale abstractions; review their intake separately from the Qwen prototype.
This transfer alone neither selects the runtime in the production Worker nor
qualifies the unexercised graph/host/pipeline facilities for BetterScale serving.

## Checks

`tests/test_live_state.py` and `tests/test_live_grouped_state.py` transfer the
corresponding donor SIMD State and grouped-backend contracts. The only helper
needed from donor tests is the small construction scope, inlined locally.
`tests/test_live_namespace.py` runs the14 Qwen prototype tests from outside the
checkout in an isolated Python process that rejects every external `livemodule`
import. Set `BETTERSCALE_TEST_PACKAGE_ROOT` to an extracted/installed wheel root
to run that same boundary test against the distribution artifact.

The Qwen prototype imports the packaged model roots directly. No LiveInference checkout,
editable install, package discovery or extra PYTHONPATH is required.

## First selected architecture backend

`arch/ascend/graph.py` separately transfers the small ACLGraphBackend from the
same donor revision (`arch/ascend/runtime/aclgraph.py`). It specializes the common
capture lifecycle using torch-npu graph/stream primitives, with no native binary
intake and no import-time device initialization. Select it explicitly; no legacy
root export or runtime auto-detection is added. Its CPU protocol tests and the
Qwen GDN two-bank NPU probe qualify this narrow path, not full serving.

## Owned model roots

`llm/qwen35/root.py` owns QwenStateRoot and `llm/qwen35/state.py` its
model-specific declarations. The two-bank probe root and its lazy numerical
kernel are colocated; no implementation is imported from the prototype directory.
See [the model entry](llm/qwen35/README.md) for explicit construction and the
not-yet-qualified serving boundary. This is new BetterScale composition, not a
copy of the complete LiveInference serving root. The numerical kernel alone
requires the pinned vLLM Triton environment when executed.
