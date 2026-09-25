#!/usr/bin/env bash
set -euo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TASK_QUEUE_ENABLE=0 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$BETTERSCALE_LIVE_TP" \
  --master_addr=127.0.0.1 --master_port="$BETTERSCALE_LIVE_DISTRIBUTED_PORT" \
  -m betterscale.live.llm.qwen35 "$@"
