#!/usr/bin/env bash
# Deployment entry, not an admission tool. Acquire the selected NPU leases first.
set -euo pipefail
: "${QWEN_MODEL_PATH:?Set the qualified Qwen27 checkpoint path}"
: "${BETTERSCALE_GDN_HOST_LIBRARY:?Set the qualified graph-pool host adapter path}"
: "${BETTERSCALE_GDN_LIBRARY:?Set the qualified owned-init K-V library path}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the admitted TP2 device pair}"
: "${VLLM_CACHE_ROOT:?Set a dedicated text-only compiler cache directory}"
: "${BETTERSCALE_FIA_LIBRARY:?Set the qualified wave FIA planner path}"
export LD_PRELOAD="$BETTERSCALE_FIA_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
export TASK_QUEUE_ENABLE=0
# Select device-side collectives before workers initialize HCCL or capture graphs.
export HCCL_OP_EXPANSION_MODE=AIV
exec "${PYTHON:-python}" -m vllm.entrypoints.cli.main serve "$QWEN_MODEL_PATH" \
  --host 127.0.0.1 --port "${SERVING_PORT:-32181}" --served-model-name qwen27 \
  --tensor-parallel-size 2 --distributed-executor-backend mp \
  --worker-cls betterscale.qwen_worker.MixedWorker --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 6442450944 --seed 17 \
  --enable-prefix-caching --mamba-cache-mode align --async-scheduling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --additional-config '{"enable_cpu_binding":false}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512,1024,1536,2048],"max_cudagraph_capture_size":2048}'
