# hw0 dual-source qualification

Active objective (2026-09-17): download Eco-Tech W8A8 directly on hw0 and run two
independent TP2 attention groups sharing E4. Do not transfer model weights from
hw2. The remote `/model` mount is read-only; this task uses the writable private
`/workspace/betterscale-hw0` directory instead. Initial free space was1.3TiB.

- `models/`: `download_model.py` downloads pinned Eco-Tech snapshot and only the
  two original Qwen shards needed for exact PLE integer repair. Neither is copied
  from hw2. Original partial snapshot is explicitly not a full original model.
- `runs/download/download.log`: full SDK log (progress output is noisy; read a
  bounded tail or receipt files). Initial bounded download PID32973, six-hour
  timeout; do not trust a reused PID without checking command/start identity.
- `models/*.receipt.json`: download completion and header/index checks.
- `repo/`: owned source snapshot plus private native overlay and matched server
  binary closure. These are code/build artifacts, not weights.
- `env/`: Python3.12 system-site-packages venv. Host torch2.10.0+cpu,
  torch_npu2.10.0.post2, CANN9.0.1 match the original gate. Local dependencies
  are installed into this venv, not upgraded globally.
- `environment.sh`: explicit model, PLE reference, Python and helper paths.
- `helpers/`: existing admission parser/supervisor/idle predicates, copied
  unchanged. Respect foreign tasks; no device reset or foreign process removal.

After environment and download validation:

```bash
source /workspace/betterscale-hw0/environment.sh
cd /workspace/betterscale-hw0/repo
bash prototypes/attention-client/qwen38/run_model.sh \
  0,1,2,3,4,5,6,7 --sources 2 --decode-graph
```

The launcher performs per-device admission and may wait up to6hours for a changed
window. On foreign occupancy it reclaims only this run. Do not retry model loads
without new availability evidence. Keep any retry bounded and interpret the
interrupted capsule as rejected, not a performance result.

First verify both sources' outputs, same-State eager/graph errors, owner call
counts and clean exits. Then use `analyze_concurrency.py <capsule>/roles` to
measure full-run paired waves and validate sampled same-layer pairs. This short
fixture is not a serving throughput or large-prefill qualification. Preserve
capsule identity and environment revisions when returning the results to hw2.

## Environment qualification and current waiter

Host-local private venv now pins transformers5.14.1 and numpy2.2.6 to the
single-source machine. Installed donor0.23 package requirements emit conflicts,
but importing this owned AttentionRoot loads **no external vllm/vllm_ascend
modules**; no installed donor was upgraded. This does not claim donor0.23 is
compatible with those private venv overrides.

`hw0-int8-gmm-result.json` passes eight dummy FULL graph cases using the actual
INT8 GEMM closure on admitted physical card2: both matrix geometries, dynamic
counts, empty groups and inactive-tail preservation. All integer outputs exact;
card released. This validates the new host's basic binary/math path, not full
model service.

`run-after-download.sh` (remote root directory) is the bounded continuation,
initial PID35910. It waits for both downloader receipts, executes
`validate_ple.py`, then enters eight-card admission and runs the two-source FULL
gate. Successful completion writes `runs/concurrency-result.json`;
`runs/campaign.log` retains startup/error state. On download failure it refuses
to launch, and on occupancy collision the existing supervisor reclaims our run.
Do not let a cancelled goal leave this waiter or the downloader alive.

At07:45 UTC the direct download held about22GiB locally and was still running;
no full-model dual-source qualification had occurred. Recheck receipt/process
identity rather than inferring completion from that size.


## Completion

Direct quantized snapshot completed11:29 UTC; the partial original snapshot
and exact PLE repair then passed. The scheduled first dual-source gate and the
subsequent warm dual-source/single-source controls completed normally; receipts
and interpretation are in README and `hw0-concurrency-result.json`. Downloader
32973/32974 and campaign35910 have exited. No continuing NPU job or waiter is
owned by this campaign. Preserve the downloaded snapshots for subsequent work.


The follow-up localized the wave52 pause to generation2 Python GC and qualified
same-GC-condition single/dual controls. Current authoritative performance receipt
is `hw0-gc-controlled-comparison.json`; prior raw timings remain preserved.
No production GC default was changed. Final comparison capsules are120738Z and
121027Z; both exited0, with all role outputs equal and all graph shadows0.
