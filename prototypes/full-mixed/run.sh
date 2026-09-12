#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime=${PROBE_RUNTIME:-/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env}
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
export ASCEND_RT_VISIBLE_DEVICES=${PROBE_DEVICES:-0} OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_HOST_IP=127.0.0.1 MASTER_ADDR=127.0.0.1 MASTER_PORT=30621
export HCCL_NPU_SOCKET_PORT_RANGE=27856-27919 VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTHONPATH="$root/prototypes/full-mixed${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export ASCEND_CUSTOM_OPP_PATH="$runtime/lib/python3.12/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
: "${PROBE_CAPSULE:?set a fresh capsule path}"
exec "$runtime/bin/python" "$root/prototypes/full-mixed/launch.py" --devices "$ASCEND_RT_VISIBLE_DEVICES" --output "$PROBE_CAPSULE" -- "$runtime/bin/python" "$root/prototypes/full-mixed/probe.py" --output "$PROBE_CAPSULE/engine" "$@"
