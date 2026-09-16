#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.1}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/runs/attention-actual-gmm-build"}
HOST_ARCH=${HOST_ARCH:-$(uname -m)}
CATLASS_INCLUDE=${CATLASS_INCLUDE:-"$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/common/catlass/include"}

case "$HOST_ARCH" in
    aarch64) ;;
    *) echo "unsupported host architecture for this A2 recipe: $HOST_ARCH" >&2; exit 2 ;;
esac

CCEC="$CANN_ROOT/tools/ccec_compiler/bin/ccec"
LD_LLD="$CANN_ROOT/tools/ccec_compiler/bin/ld.lld"
SOURCE=${SOURCE:-"$ROOT/prototypes/attention-client/device-service/actual_gmm.cpp"}
OBJECT_NAME=${OBJECT_NAME:-actual_gmm}
TMP_OBJECT="$OUTPUT_DIR/${OBJECT_NAME}_tmp.o"
OBJECT="$OUTPUT_DIR/${OBJECT_NAME}.o"

test -x "$CCEC"
test -x "$LD_LLD"
mkdir -p "$OUTPUT_DIR"

"$CCEC" -c -x cce -O2 "$SOURCE" -o "$TMP_OBJECT" \
    -DCATLASS_ARCH=2201 \
    -DACTUAL_GMM_DFC=${ACTUAL_GMM_DFC:-0} \
    -I"$ROOT/upstream/vllm-ascend/csrc/mc2/dispatch_ffn_combine_bf16/op_kernel" \
    -DACTUAL_GMM_L0_K=${ACTUAL_GMM_L0_K:-64} \
    -DACTUAL_GMM_TILE_N=${ACTUAL_GMM_TILE_N:-256} \
    -DACTUAL_GMM_TILE_K=${ACTUAL_GMM_TILE_K:-256} \
    -DACTUAL_GMM_TILE_M=${ACTUAL_GMM_TILE_M:-128} \
    -DASCNEDC_DUMP \
    --cce-aicore-arch=dav-c220-cube \
    --cce-aicore-input-parameter-size=28000 \
    --cce-aicore-only \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -I"$CANN_ROOT/include/ascendc/basic_api" \
    -I"$CANN_ROOT/include/ascendc/basic_api/interface" \
    -I"$CANN_ROOT/include/ascendc/include/adv_api" \
    -I"$CANN_ROOT/include/ascendc/highlevel_api" \
    -I"$CANN_ROOT/aarch64-linux/ascendc/include/basic_api/impl" \
    -I"$CANN_ROOT/aarch64-linux/asc" \
    -I"$CANN_ROOT/aarch64-linux/asc/impl/basic_api" \
    -I"$CANN_ROOT/aarch64-linux/asc/impl/simt_api" \
    -I"$CANN_ROOT/aarch64-linux/asc/impl/micro_api" \
    -I"$CANN_ROOT/aarch64-linux/include/basic_api" \
    -I"$CANN_ROOT/aarch64-linux/include/simt_api" \
    -I"$CANN_ROOT/aarch64-linux/include/micro_api" \
    -I"$CANN_ROOT/include/hccl" \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm --cce-aicore-jump-expand=true \
    -I"$CATLASS_INCLUDE" \
    -std=c++17 -fstack-protector-all

test -s "$TMP_OBJECT"
"$LD_LLD" -m aicorelinux -Ttext=0 "$TMP_OBJECT" -static -o "$OBJECT"
test -s "$OBJECT"
file "$OBJECT"
LAUNCH_SOURCE=${LAUNCH_SOURCE:-"$ROOT/prototypes/attention-client/device-service/launch.cpp"}
g++ -shared -fPIC -O2 -std=c++17 "$LAUNCH_SOURCE" \
 -I"$CANN_ROOT/include" -L"$CANN_ROOT/lib64" -lascendcl \
 -Wl,-rpath,"$CANN_ROOT/lib64" -o "$OUTPUT_DIR/launch.so"
