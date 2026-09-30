#!/bin/bash
set -euo pipefail
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh >/dev/null 2>&1; set -u
R=/root/my-ascend-workspace/runs/a2e2-revival/20260930-stage11-profile-recovery
PY=/workspace/my-ascend-workspace/runs/rp-legacy/20260903T155041Z-layout/rp-upstream-0.25.1/.venv/bin/python
exec "$PY" /root/my-ascend-workspace/runs/expert-event-graphs/20260924/admit-no-hoard.py --lease selected --devices 0,1,2,3 --observe-seconds 30 --wait-seconds 300 --run-seconds 1800 --output "$R/a4" -- "$PY" "$R/capsule/run_a2e2_profile.py" --run-dir "$R/p4"
