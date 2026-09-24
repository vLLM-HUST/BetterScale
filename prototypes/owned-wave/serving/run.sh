#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
runtime=/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env
: "${PROBE_CAPSULE:?fresh absolute capsule required}"
export OWNED_STATIC_FIA=${OWNED_STATIC_FIA:-1}
export OWNED_HOST_FIA=${OWNED_HOST_FIA:-1}
export BENCH_ARM=${BENCH_ARM:-owned}
case "$OWNED_STATIC_FIA" in 0|1) ;; *) echo "OWNED_STATIC_FIA must be 0 or 1" >&2; exit 2 ;; esac
case "$OWNED_HOST_FIA" in 0|1) ;; *) echo "OWNED_HOST_FIA must be 0 or 1" >&2; exit 2 ;; esac
export SWE_TRACE=${SWE_TRACE:-$repo/runs/owned-wave/swe-trace-v2/trace.json}
export HCCL_DETERMINISTIC=strict PYTHONHASHSEED=0
mkdir -p "$(dirname "$PROBE_CAPSULE")"
mkdir "$PROBE_CAPSULE" "$PROBE_CAPSULE/source" "$PROBE_CAPSULE/engine"
cp "$repo/prototypes/owned-wave/"{live_root,boundary}.py "$repo/prototypes/joint-wave/storage.py" "$PROBE_CAPSULE/source/"
cp "$repo/prototypes/owned-wave/serving/"*.py "$PROBE_CAPSULE/source/"
cp "$repo/prototypes/host-admission/admit_subset.py" "$PROBE_CAPSULE/source/admit.py"
cp -a /root/my-ascend-workspace/LiveInference/src/livemodule "$PROBE_CAPSULE/source/livemodule"
git -C /root/my-ascend-workspace/LiveInference rev-parse HEAD > "$PROBE_CAPSULE/liveinference-commit.txt"
git -C "$repo" rev-parse HEAD > "$PROBE_CAPSULE/base-commit.txt"
cp "$SWE_TRACE" "$PROBE_CAPSULE/trace.json"
export SWE_TRACE="$PROBE_CAPSULE/trace.json"
for file in worker/model_runner_v1.py attention/attention_v1.py attention/utils.py ascend_forward_context.py; do
  cmp "$repo/upstream/vllm-ascend/vllm_ascend/$file" "$runtime/lib/python3.12/site-packages/vllm_ascend/$file"
done
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
export ASCEND_RT_VISIBLE_DEVICES=${PROBE_DEVICES:-0,1} OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256
export VLLM_HOST_IP=127.0.0.1 MASTER_ADDR=127.0.0.1 MASTER_PORT=32041
export HCCL_NPU_SOCKET_PORT_RANGE=29500-29563 VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export PYTHONPATH="$PROBE_CAPSULE/source:$repo/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export ASCEND_CUSTOM_OPP_PATH="$runtime/lib/python3.12/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
if [[ ${OWNED_STATIC_FIA:-0} == 1 ]]; then
  cp "$repo/prototypes/owned-wave/fia-plan/"{static_plan,host_metadata}.cpp "$PROBE_CAPSULE/source/"
  plan_source=static_plan.cpp
  plan_libs=()
  if [[ ${OWNED_HOST_FIA:-0} == 1 ]]; then
    plan_source=host_metadata.cpp
    plan_libs=(-L/usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64 -lopapi)
  fi
  c++ -shared -fPIC -O2 -std=c++17 -I/usr/local/Ascend/cann-9.0.1/aarch64-linux/include \
    "$PROBE_CAPSULE/source/$plan_source" "${plan_libs[@]}" -ldl -o "$PROBE_CAPSULE/source/static_plan.so"
  export FIA_PLAN_LIBRARY="$PROBE_CAPSULE/source/static_plan.so"
  export LD_PRELOAD="$FIA_PLAN_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SERVING_OUTPUT="$PROBE_CAPSULE/engine"
exec "$runtime/bin/python" "$PROBE_CAPSULE/source/admit.py" --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --wait-seconds 600 --output "$PROBE_CAPSULE/run" -- \
  "$runtime/bin/python" "$PROBE_CAPSULE/source/serving_driver.py"
