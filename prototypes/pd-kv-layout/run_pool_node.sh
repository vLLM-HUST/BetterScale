#!/usr/bin/env bash
# Experimental P4 or native D4 node on the restored pinned task runtime.
# Caller supplies explicit --kind, --bind, --peer, --port and a new --output.
set -eo pipefail
root=/workspace/betterscale-pd-runtime
package="$root/candidate-package-6-no-draft/betterscale"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh --cxx_abi=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$source_dir:${PYTHONPATH:-}"
export BETTERSCALE_GDN_LIBRARY="$package/patches/qwen_gdn/libbs_gdn.so"
export BETTERSCALE_GDN_HOST_LIBRARY="$package/patches/qwen_gdn/libbs_gdn_host.so"
export BETTERSCALE_FIA_LIBRARY="$package/patches/qwen_fia/libbs_fia.so"
export LD_PRELOAD="$BETTERSCALE_FIA_LIBRARY"
export VLLM_CACHE_ROOT="$root/model-cache" TASK_QUEUE_ENABLE=0 HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS=4
export VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
export MTP_TOKENS=2 MTP_GDN_LAYOUT_FUSION=1 BETTERSCALE_MTP_GREEDY=1 BETTERSCALE_GDN_SMALL_COPIES=1 MTP_PROFILE=0
export BETTERSCALE_MODEL_PATH="${BETTERSCALE_MODEL_PATH:-/workspace/models/Qwen3.5-35B-A3B}"
unset HCCL_DETERMINISTIC
exec "$root/venv/bin/python" "$source_dir/${BETTERSCALE_PD_NODE_ENTRY:-naive_pool_node.py}" "$@"
