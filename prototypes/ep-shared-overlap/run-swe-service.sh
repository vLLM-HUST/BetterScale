#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh --cxx_abi=1
variant=${1:?e16 or e36}; shift
case "$variant" in e16|e36) ;; *) exit 2;; esac
root=/workspace/overlap-swe
package=$root/$variant/src/betterscale
export PYTHONPATH="$root:$root/$variant/src:$root/owned-runtime:/workspace/ep-overlap-runtime/pinned:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=0 HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS=4
export VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
export MTP_TOKENS=2 MTP_GDN_LAYOUT_FUSION=1 BETTERSCALE_MTP_GREEDY=1 BETTERSCALE_GDN_SMALL_COPIES=1 MTP_PROFILE=0
export VLLM_CACHE_ROOT="$root/cache-$variant"
export BETTERSCALE_GDN_LIBRARY="$package/patches/qwen_gdn/libbs_gdn.so"
export BETTERSCALE_GDN_HOST_LIBRARY="$package/patches/qwen_gdn/libbs_gdn_host.so"
export BETTERSCALE_FIA_LIBRARY="$package/patches/qwen_fia/libbs_fia.so"
export BETTERSCALE_CP_LIBRARY="$package/patches/qwen_fia/context_parallel/libbs_fia_cp.so"
[ "$(npu-smi info | grep -c 'No running processes found')" -eq 2 ]
export LD_PRELOAD="$BETTERSCALE_FIA_LIBRARY"
exec "$root/venv/bin/python" "$root/swe_service.py" "$@"
