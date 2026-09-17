#!/usr/bin/env bash
# hw3: a frozen capsule with source/, helpers/ and trace.json must exist first.
set -euo pipefail
: "${CAPSULE:?Set an absolute frozen capsule path on hw3}"
base=/workspace/my-ascend-workspace/runs/qwen27-partition-serving
python=/workspace/my-ascend-workspace/runs/liveinfer-online/20260907-donor-dspark-runtime/env/bin/python
unset PYTHONPATH LD_PRELOAD TRACELOOM_CONTEXT_DIR ASCEND_CUSTOM_OPP_PATH
set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; set -u
export PYTHONPATH="$CAPSULE/source:$base/runtime-source${PYTHONPATH:+:$PYTHONPATH}"
export PROBE_HELPERS="$CAPSULE/helpers" QWEN_MODEL_PATH="$base/model"
export OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256
export VLLM_HOST_IP=127.0.0.1 MASTER_ADDR=127.0.0.1
export BETTERSCALE_GDN_LIBRARY="$base/ascendc-gdn-build6/build/lib/libbs_gdn.so"
export SWE_DEVICE_PAIRS="${SWE_DEVICE_PAIRS:-0,1;6,7}"
exec "$python" "$CAPSULE/helpers/admit_subset.py" --devices "${SWE_DEVICE_PAIRS//;/,}" \
  --wait-seconds 1800 --runtime-seconds 3600 --output "$CAPSULE/admission" -- \
  "$python" "$CAPSULE/source/swe_compare.py"
