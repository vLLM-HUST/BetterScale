#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
runtime=/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env
: "${PROBE_CAPSULE:?choose a fresh capsule}"
devices=${PROBE_DEVICES:-0,1}
export QWEN_TP=${QWEN_TP:-2} QWEN_DP=${QWEN_DP:-1}
export HCCL_DETERMINISTIC=strict
mkdir -p "$(dirname "$PROBE_CAPSULE")"
mkdir "$PROBE_CAPSULE"
mkdir "$PROBE_CAPSULE/source"
cp "$repo/prototypes/joint-wave/"{storage,transport}.py "$PROBE_CAPSULE/source/"
cp "$repo/prototypes/joint-wave/qwen/"*.py "$PROBE_CAPSULE/source/"
cp "$repo/prototypes/attention-client/device-service/admit_subset.py" "$PROBE_CAPSULE/source/admit.py"
git -C "$repo" rev-parse HEAD > "$PROBE_CAPSULE/base-commit.txt"
for file in worker/model_runner_v1.py attention/attention_v1.py attention/utils.py ascend_forward_context.py; do
  cmp "$repo/upstream/vllm-ascend/vllm_ascend/$file" "$runtime/lib/python3.12/site-packages/vllm_ascend/$file"
  printf 'MATCH %s\n' "$file" >> "$PROBE_CAPSULE/native-source-comparison.txt"
done
cmp "$repo/upstream/vllm/vllm/model_executor/models/qwen3_moe.py" "$runtime/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_moe.py"
printf 'MATCH vllm/model_executor/models/qwen3_moe.py\n' >> "$PROBE_CAPSULE/native-source-comparison.txt"
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
export ASCEND_RT_VISIBLE_DEVICES="$devices" OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256
export VLLM_HOST_IP=127.0.0.1 MASTER_ADDR=127.0.0.1 MASTER_PORT=31841
export HCCL_NPU_SOCKET_PORT_RANGE=29200-29263 VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export PYTHONPATH="$PROBE_CAPSULE/source:$repo/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export ASCEND_CUSTOM_OPP_PATH="$runtime/lib/python3.12/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export QWEN_JOINT_OUTPUT="$PROBE_CAPSULE/engine"
mkdir "$QWEN_JOINT_OUTPUT"
exec "$runtime/bin/python" "$PROBE_CAPSULE/source/admit.py" --devices "$devices" \
  --wait-seconds 600 --output "$PROBE_CAPSULE/run" -- \
  "$runtime/bin/python" "$PROBE_CAPSULE/source/driver.py"
