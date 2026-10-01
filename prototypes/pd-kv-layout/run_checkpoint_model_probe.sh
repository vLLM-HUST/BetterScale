#!/usr/bin/env bash
# Explicit task-local profile: real target requests never invoke draft forward.
set -eo pipefail
root=/workspace/betterscale-pd-runtime
package="$root/candidate-package-6-no-draft/betterscale"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh --cxx_abi=1
export ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_IF_BASE_PORT=29635
export PYTHONPATH="$root/candidate-package-6-no-draft:$root/owned-runtime:/workspace/BetterScale/prototypes/pd-kv-layout:${PYTHONPATH:-}"
export BETTERSCALE_GDN_LIBRARY="$package/patches/qwen_gdn/libbs_gdn.so"
export BETTERSCALE_GDN_HOST_LIBRARY="$package/patches/qwen_gdn/libbs_gdn_host.so"
export BETTERSCALE_FIA_LIBRARY="$package/patches/qwen_fia/libbs_fia.so"
export LD_PRELOAD="$BETTERSCALE_FIA_LIBRARY"
export VLLM_CACHE_ROOT="$root/model-cache" TASK_QUEUE_ENABLE=0 HCCL_OP_EXPANSION_MODE=AIV OMP_NUM_THREADS=4
export VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
export MTP_TOKENS=2 MTP_GDN_LAYOUT_FUSION=1 BETTERSCALE_MTP_GREEDY=1 BETTERSCALE_GDN_SMALL_COPIES=1 MTP_PROFILE=0
exec timeout --signal=TERM --kill-after=30 900 "$root/venv/bin/python" /workspace/BetterScale/prototypes/pd-kv-layout/checkpoint_model_probe.py --output "$1"
