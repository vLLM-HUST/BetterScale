# Third-party notices

The pinned vLLM and vLLM-Ascend submodules retain their original copyright and
license files. This package adapts private execution paths from those pinned
Apache-2.0 projects; it does not claim those adaptations are an upstream API.

- `src/betterscale/patches/qli_cpu/` adapts QLI metadata handling from
  vLLM-Ascend `attention/context_parallel/dsa_cp.py`.
- `src/betterscale/patches/ordered_replay/` includes the attributed native
  `compilation/acl_graph.py` call body with the marked synchronization change.
- `src/betterscale/patches/async_decode/_producer.py` adapts the native
  DSV4 input-preparation arithmetic from `worker/model_runner_v1.py`, retaining
  native speculative-decode kernels and metadata types. That source credits
  Huawei Technologies Co., Ltd. (2025) and the vLLM team (2025).
- `src/betterscale/patches/target_full/_dp.py` adapts the native DSA metadata
  representation while continuing to invoke its original builder.

A copy of the license is in `licenses/Apache-2.0.txt`. This distribution does
not redistribute model weights, datasets, the CANN runtime, torch-npu or the
complete donor binaries.

The HC-pre patch distributes a modified native host-tiling library built from
vLLM-Ascend 9bf964cb. Its CANN-licensed components (Huawei Technologies Co., Ltd.,
2023–2026) remain under CANN Open Software License Agreement 1.0/2.0 where marked, solely for
Ascend processors; see `licenses/CANN-1.0.txt`, `licenses/CANN-2.0.txt` and the patch’s `NATIVE_NOTICES.txt`.
The sole source change removes HC-pre’s fixed workspace lower bound. Other
upstream components retain Apache-2.0. License text source:
https://gitee.com/ascend/shmem/raw/master/LICENSE (retrieved 2026-09-15).

CANN1.0 license text source: https://gitee.com/ascend/cann-ops/raw/master/LICENSE
(retrieved 2026-09-15).

The opt-in Qwen K-V recurrent kernel in
`src/betterscale/patches/qwen_gdn/decode_kv.py` adapts pinned vLLM's FLA kernel
(Apache-2.0), retaining its attribution to Songlin Yang and Yu Zhang and the
original flash-linear-attention MIT notice. The separately built AscendC H/O
library retains its original per-file notices under
`prototypes/qwen38-serving/ascendc_gdn`; it is not a system CANN replacement.
