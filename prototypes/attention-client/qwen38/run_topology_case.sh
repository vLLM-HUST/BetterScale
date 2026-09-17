#!/bin/bash
# hw0 equal-card campaign. Individual cases own admission; never kill outsiders.
set -euo pipefail
source /workspace/betterscale-hw0/environment.sh
cd /workspace/betterscale-hw0/repo
layout=$1
mode=$2
budget=$3
out=$4
qsa=${5:-bounded128}
tokens=${6:-1024}
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
case "$qsa" in
  bounded128)
    if [[ $layout == *-ep8 ]]; then overlay=qwen38-bounded-qsa-colocated-20260917d;
    else overlay=qwen38-bounded-qsa-runtime-20260917d/overlay; fi ;;
  original) ;;
  *) echo "unknown QSA implementation" >&2; exit 2 ;;
esac
"$QWEN38_PYTHON" - "$out/parameters.json" "$layout" "$mode" "$budget" "$qsa" "$overlay" "$build" "$tokens" <<'PYMETA'
import json,sys
path,layout,mode,budget,qsa,overlay,build,tokens=sys.argv[1:]
with open(path,"w") as f:json.dump(dict(layout=layout,mode=mode,state_gib=float(budget),qsa=qsa,overlay=overlay,build=build,total_sessions=40,trace_turns=2,mtp_tokens=1,token_capacity=int(tokens)),f,indent=2)
PYMETA
export QWEN38_BUILD=$PWD/runs/$build
export QWEN38_OVERLAY=$PWD/runs/$overlay
export QWEN38_WAIT_SECONDS=1800
exec 9>/root/tp8.lock
if ! flock -w 3600 9; then echo "lease wait expired" > "$out/admission-failure"; exit 73; fi
# Keep the exact 40-session allocation and output budgets across all layouts.
# B40 is divisible by2,4,5,8; no replicated sessions or inactive padded sources.
set +e
bash "$PWD/prototypes/attention-client/qwen38/run_model.sh" 0,1,2,3,4,5,6,7 \
  --tp-size "$tp" --sources "$sources" --batch-size "$((40 / sources))" \
  --state-gib "$budget" --token-capacity "$tokens" --mtp-tokens 1 --decode-graph --trace-max-context 262144 \
  "${extra[@]}" > "$out/launch.log" 2>&1
status=$?
set -e
cp /tmp/betterscale-qwen38-model-capsule "$out/capsule"
printf '%s\n' "$status" > "$out/exit"
exit "$status"
