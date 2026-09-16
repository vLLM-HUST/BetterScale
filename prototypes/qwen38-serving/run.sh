#!/usr/bin/env bash
set -euo pipefail
: "${CAPSULE:?fresh absolute capsule}" "${ARM:?sync/async/mtp1/mtp2/mtp3}"
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$source_dir" rev-parse --show-toplevel)
runtime=/workspace/my-ascend-workspace/runs/rp-legacy/20260903T155041Z-layout/rp-upstream-0.25.1/.venv
mkdir -p "$(dirname "$CAPSULE")"
mkdir "$CAPSULE" "$CAPSULE/source"
cp "$source_dir/"*.py "$CAPSULE/source/"
cp /root/my-ascend-workspace/runs/qwen38-27b-tp2-baseline/20260916-donor0251-v5/prompt.json "$CAPSULE/prompt.json"
git -C "$repo" rev-parse HEAD > "$CAPSULE/source-commit.txt"
cp /models/vllm-ascend-models/Qwen3.8-27B/config.json "$CAPSULE/model-config.json"
unset PYTHONPATH LD_PRELOAD TRACELOOM_CONTEXT_DIR
set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; set -u
export ASCEND_RT_VISIBLE_DEVICES=${PROBE_DEVICES:-0,1} OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
export PYTHONPATH="$CAPSULE/source${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export SERVING_PROFILE="$CAPSULE/profiles"
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256
export VLLM_HOST_IP=127.0.0.1 MASTER_ADDR=127.0.0.1 MASTER_PORT=32182
export HCCL_NPU_SOCKET_PORT_RANGE=29664-29727
unset ASCEND_CUSTOM_OPP_PATH
profile=()
if [[ ${PROFILE:-0} == 1 ]]; then profile=(--profile); fi
exec "$runtime/bin/python" /workspace/strengthen-dsv4/prototypes/attention-client/device-service/admit_subset.py \
 --devices "$ASCEND_RT_VISIBLE_DEVICES" --wait-seconds 1800 --output "$CAPSULE/admission" -- \
 "$runtime/bin/python" "$CAPSULE/source/service_probe.py" --capsule "$CAPSULE" --arm "$ARM" "${profile[@]}"
