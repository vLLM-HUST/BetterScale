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
