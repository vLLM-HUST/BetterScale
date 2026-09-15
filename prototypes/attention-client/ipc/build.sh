#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.1}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/runs/attention-client-ipc-build"}
HOST_ARCH=${HOST_ARCH:-$(uname -m)}

case "$HOST_ARCH" in
    aarch64) ;;
    *) echo "unsupported host architecture for this A2 recipe: $HOST_ARCH" >&2; exit 2 ;;
esac

CCEC="$CANN_ROOT/tools/ccec_compiler/bin/ccec"
LD_LLD="$CANN_ROOT/tools/ccec_compiler/bin/ld.lld"
SOURCE="$ROOT/prototypes/attention-client/ipc/kernel.cpp"
TMP_OBJECT="$OUTPUT_DIR/pull_expert_client_tmp.o"
OBJECT="$OUTPUT_DIR/pull_expert_client.o"

test -x "$CCEC"
test -x "$LD_LLD"
mkdir -p "$OUTPUT_DIR"

"$CCEC" -c -x cce -O2 "$SOURCE" -o "$TMP_OBJECT" \
    -DASCNEDC_DUMP \
    --cce-aicore-arch=dav-c220-vec \
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
    -std=c++17 -fstack-protector-all

test -s "$TMP_OBJECT"
"$LD_LLD" -m aicorelinux -Ttext=0 "$TMP_OBJECT" -static -o "$OBJECT"
test -s "$OBJECT"
file "$OBJECT"
g++ -shared -fPIC -O2 -std=c++17 "$ROOT/prototypes/attention-client/ipc/launch.cpp" \
 -I"$CANN_ROOT/include" -L"$CANN_ROOT/lib64" -lascendcl \
 -Wl,-rpath,"$CANN_ROOT/lib64" -o "$OUTPUT_DIR/launch.so"
