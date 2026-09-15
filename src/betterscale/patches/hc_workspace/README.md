# HC-pre: stop reserving a fixed 208 MiB workspace

This **TP-only** patch ships the host tiling library qualified on Ascend 910B2,
CANN 9.0.1 and our pinned vLLM-Ascend build. It replaces HC-pre's fixed minimum
with its computed requirement, retaining the existing 16 MiB runtime provision.
It does not change tensor schemas, tiling choices, device kernels or arithmetic.
DP continues to use the original vendor; no DP benefit is claimed by this release.

## Where it takes effect

`Worker.__init__` checks donor/config pins, then installs this patch **before**
`NPUWorker.__init__` registers Ascend operators. Installation copies the complete
native vendor into a process-private temporary directory and replaces only its
two host-tiler library files. API libraries, metadata and device kernels remain
native. Symlinks are dereferenced, so replacements cannot modify the donor.
The directory stays alive through all graph replay and is cleaned at process exit.
Multiprocessing exit cleanup is registered explicitly (ordinary Python atexit
is not sufficient for native worker processes); a forced SIGKILL can still leave
temporary files, as with other process-owned temporary storage.
It uses roughly 56 MiB of host filesystem space per TP worker, not NPU memory.

Native `enable_custom_op()` bootstraps `_CUSTOM_OP_BASE_DIR`; the patch redirects
that base and replaces the original vendor entry in `ASCEND_CUSTOM_OPP_PATH`.
There is exactly one selected `custom_transformer` tree. Users do not set another
OPP/LD_LIBRARY path or run a separate launcher. No installed donor file changes.
Unexpected binaries, late registration and nonqualified CANN fail closed.

## Evidence and limits

Same-hw3 TP8 dummy FULL, K5, buckets 24/4128, automatic KV with the same 1 GiB
safety: together with backing-level KV clear, mean per-rank KV budget increased
14.946 to 15.127 GiB (~186 MiB, 1.21%); READY reserved fell 360 MiB/rank. Exact
HC-pre outputs and changed-input FULL replay passed. These are capacity results,
not new real-weight quality or throughput results. Residual aliasing is **not**
included. Full evidence: `prototypes/peak-memory/hw3-fixes-results.json` in Git.

## Native artifact and rebuild

PyPI publishes an sdist containing the qualified `libcust_opmaster_rt2.0.so`.
`pip install` builds only the small Python wrapper wheel locally; it does **not**
compile CANN operators. The resulting wheel is Linux/aarch64, not `py3-none-any`.
PyPI does not accept that non-manylinux wheel tag, so the wheel is not uploaded;
we do not claim a CANN-dependent library is a portable manylinux binary.
Source control keeps its identity in `native.json` rather than build binaries.
A source checkout needs the qualified binary before packaging: either extract
it from the corresponding official PyPI sdist or
rebuild with `build_native.py --upstream <pinned-checkout> --cann <CANN-9.0.1>`.
The script uses `git archive` of the exact commit, applies `workspace.patch`,
and builds the native 34-operator A2 host library. It does not rebuild or replace
API/device binaries. A rebuilt artifact needs qualification and its own identity;
compiler-dependent binary differences must not bypass the package guard.

The underlying HC-pre/native sources retain Huawei's copyrights and CANN Open
Software License Agreement 1.0/2.0 where marked, **solely for Ascend processors**;
other upstream components retain Apache-2.0. See the distribution's license files
and `NATIVE_NOTICES.txt`. This is not a relicensing of the native library.
