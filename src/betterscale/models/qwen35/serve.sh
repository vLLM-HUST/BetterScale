#!/usr/bin/env bash
# Serving entry only: selected-device lease/admission belongs to the caller.
set -euo pipefail
: "${BETTERSCALE_QWEN35_RUNTIME:?Prepare an isolated pinned donor first}"
: "${BETTERSCALE_FIA_LIBRARY:?Missing packaged FIA library}"
: "${PYTHON:?Use python -m betterscale serve-qwen}"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PYTHONPATH="$BETTERSCALE_QWEN35_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="$BETTERSCALE_FIA_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
export TASK_QUEUE_ENABLE=0 HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
export MTP_TOKENS=2 MTP_GDN_LAYOUT_FUSION=1
export BETTERSCALE_MTP_GREEDY=1 BETTERSCALE_GDN_SMALL_COPIES=1 MTP_PROFILE=0
exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$@"
