#!/usr/bin/env bash
# Task-local launcher: only the user-assigned two-card hw112 container.
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh --cxx_abi=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS=ascend
export PYTHONPATH=/workspace/BetterScale-overlap/src:/workspace/ep-overlap-runtime/pinned:${PYTHONPATH:-}
arm=${1:?serial, donor or early}
name=${2:?unique output basename}
case "$arm" in serial|donor|early) ;; *) exit 2;; esac
case "$name" in *[!a-zA-Z0-9_-]*|'') exit 2;; esac
shift 2
output=/workspace/ep-overlap-runtime/$name
[ ! -e "$output" ] && [ ! -e "$output.log" ]
[ "$(npu-smi info | grep -c 'No running processes found')" -eq 2 ]
timeout 240s /workspace/ep-overlap-runtime/venv/bin/python -m torch.distributed.run \
  --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 27712 \
  /workspace/BetterScale-overlap/prototypes/ep-shared-overlap/probe.py \
  --arm "$arm" --output "$output" "$@" > "$output.log" 2>&1
cat "$output"/rank*.json
