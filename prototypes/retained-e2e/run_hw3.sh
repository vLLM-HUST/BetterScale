#!/bin/bash
set -euo pipefail
root=/workspace/my-ascend-workspace/runs/tp-continuation-20260914
runtime=/workspace/my-ascend-workspace/runs/liveinfer-online/20260907-donor-dspark-runtime/env
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_HOST_IP=127.0.0.1 MASTER_ADDR=127.0.0.1 MASTER_PORT=30621
export HCCL_NPU_SOCKET_PORT_RANGE=27856-27919
export PYTHONPATH="$root/retained-vs-native-v1${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export ASCEND_CUSTOM_OPP_PATH="$runtime/lib/python3.12/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PROBE_HELPERS="$root/helpers"
unset EARLY_BUDGET_OUTPUT EARLY_BUDGET_ORACLE EARLY_BUDGET_PROFILE || true
out="$root/$1"; topology=$2; policy=$3
[[ $topology == dp || $topology == tp ]]; [[ $policy == native || $policy == retained ]]
worker=vllm_ascend.worker.worker.NPUWorker
mode=FULL_DECODE_ONLY
if [[ $topology == dp ]]; then
 tp=1; dp=8; seats=16; local_seats=2; budget=1026; maxlen=16384; kv=8589934592; dsa=false
 captures='[6,12]'; maxcapture=12; export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
 if [[ $policy == retained ]]; then captures='[6,12,132,264,516,1026]'; maxcapture=1026; fi
else
 tp=8; dp=1; seats=4; local_seats=4; budget=4128; maxlen=15104; kv=12884901888; dsa=true
 captures='[24]'; maxcapture=24; export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
 worker=native_worker.NativeTPWorker
 if [[ $policy == retained ]]; then captures='[24,4128]'; maxcapture=4128; fi
fi
if [[ $policy == retained ]]; then worker=betterscale.worker.Worker; mode=FULL; fi
exec "$runtime/bin/python" "$root/retained-vs-native-v1/launch.py" --devices "$ASCEND_RT_VISIBLE_DEVICES" --output "$out" -- \
 "$runtime/bin/python" "$root/retained-vs-native-v1/service_bench.py" --output "$out/engine" --seats "$seats" --repeats 3 --quality-requests "$root/quality-inputs.json" -- \
 "$runtime/bin/vllm" serve /data/shared/models/DeepSeek-V4-Flash-0731-w8a8 --host 127.0.0.1 --port 30880 --served-model-name dsv4 \
 --tensor-parallel-size "$tp" --data-parallel-size "$dp" --data-parallel-size-local "$dp" --enable-expert-parallel --quantization ascend --dtype bfloat16 \
 --distributed-executor-backend mp --async-scheduling --worker-cls "$worker" \
 --max-model-len "$maxlen" --max-num-seqs "$local_seats" --max-num-batched-tokens "$budget" --kv-cache-memory-bytes "$kv" --no-enable-prefix-caching --seed 123 --block-size 128 \
 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
 --compilation-config "{\"cudagraph_mode\":\"$mode\",\"cudagraph_capture_sizes\":$captures,\"max_cudagraph_capture_size\":$maxcapture}" \
 --additional-config "{\"ascend_compilation_config\":{\"enable_npugraph_ex\":true,\"enable_static_kernel\":false},\"enable_cpu_binding\":false,\"enable_dsa_cp\":$dsa,\"multistream_overlap_shared_expert\":true}"
