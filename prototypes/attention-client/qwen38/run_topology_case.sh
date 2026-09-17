#!/bin/bash
# hw0 equal-card campaign. Individual cases own admission; never kill outsiders.
set -euo pipefail
source /workspace/betterscale-hw0/environment.sh
cd /workspace/betterscale-hw0/repo
layout=$1
mode=$2
budget=$3
out=$4
[[ ! -e $out ]]
mkdir -p "$out"
extra=()
case "$layout" in
  tp2-e4) tp=2; sources=2; build=qwen38-server-rows1024-20260917; overlay=qwen38-native-runtime-20260917/overlay ;;
  tp2-ep8) tp=2; sources=4; build=qwen38-server-rows1024-20260917; overlay=qwen38-colocated-overlay-20260917; extra+=(--colocated) ;;
  tp1-e4) tp=1; sources=4; build=qwen38-e4-sources4-rows1024-20260917; overlay=qwen38-tp1-runtime-20260917/overlay ;;
  tp1-e3) tp=1; sources=5; build=qwen38-e3-sources5-rows1024-20260917; overlay=qwen38-tp1-runtime-20260917/overlay ;;
  tp1-ep8) tp=1; sources=8; build=qwen38-server-rows1024-20260917; overlay=qwen38-tp1-colocated-overlay-20260917 ; extra+=(--colocated) ;;
  *) echo "unknown layout" >&2; exit 2 ;;
esac
case "$mode" in
  capacity) extra+=(--capacity-probe) ;;
  trace) extra+=(--trace-plan /workspace/betterscale-hw0/runs/swe-traces-20260917/qwen38-swe40.json --trace-count 40 --trace-turns 2) ;;
  *) echo "unknown mode" >&2; exit 2 ;;
esac
export QWEN38_BUILD=$PWD/runs/$build
export QWEN38_OVERLAY=$PWD/runs/$overlay
export QWEN38_WAIT_SECONDS=1800
exec 9>/root/tp8.lock
flock -w 1800 9
# Keep the exact 40-session allocation and output budgets across all layouts.
# B40 is divisible by2,4,5,8; no replicated sessions or inactive padded sources.
set +e
bash "$PWD/prototypes/attention-client/qwen38/run_model.sh" 0,1,2,3,4,5,6,7 \
  --tp-size "$tp" --sources "$sources" --batch-size "$((40 / sources))" \
  --state-gib "$budget" --mtp-tokens 1 --decode-graph --trace-max-context 262144 \
  "${extra[@]}" > "$out/launch.log" 2>&1
status=$?
set -e
cp /tmp/betterscale-qwen38-model-capsule "$out/capsule"
printf '%s\n' "$status" > "$out/exit"
exit "$status"
