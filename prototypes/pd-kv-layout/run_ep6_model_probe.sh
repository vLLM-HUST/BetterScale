#!/usr/bin/env bash
# hw180 task runtime, explicit six-card scope; caller owns hardware admission.
set -eo pipefail
root=/workspace/betterscale-pd-runtime
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh --cxx_abi=1
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7 TASK_QUEUE_ENABLE=0 OMP_NUM_THREADS=2
export PYTHONPATH="$root/candidate-ep6-native:$root/ep6-native-runtime:/workspace/BetterScale/prototypes/pd-kv-layout:${PYTHONPATH:-}"
exec timeout --signal=TERM --kill-after=30 660 "$root/venv/bin/python" /workspace/BetterScale/prototypes/pd-kv-layout/ep6_model_probe.py --output "$1" "${@:2}"
