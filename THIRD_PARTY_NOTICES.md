# Third-party notices

The pinned vLLM and vLLM-Ascend submodules retain their original copyright and
license files. This package adapts private execution paths from those pinned
Apache-2.0 projects; it does not claim those adaptations are an upstream API.

- `src/strengthen_dsv4/patches/qli_cpu/` adapts QLI metadata handling from
  vLLM-Ascend `attention/context_parallel/dsa_cp.py`.
- `src/strengthen_dsv4/patches/ordered_replay/` includes the attributed native
  `compilation/acl_graph.py` call body with the marked synchronization change.
- `src/strengthen_dsv4/patches/async_decode/_producer.py` adapts the native
  DSV4 input-preparation arithmetic from `worker/model_runner_v1.py`, retaining
  native speculative-decode kernels and metadata types. That source credits
  Huawei Technologies Co., Ltd. (2025) and the vLLM team (2025).
- `src/strengthen_dsv4/patches/target_full/_dp.py` adapts the native DSA metadata
  representation while continuing to invoke its original builder.

A copy of the license is in `licenses/Apache-2.0.txt`. This distribution does
not redistribute model weights, datasets, CANN, torch-npu or donor binaries.
