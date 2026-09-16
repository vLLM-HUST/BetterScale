#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export OUTPUT_DIR=${OUTPUT_DIR:-"$repo/runs/qwen-next-build"}
src="$repo/prototypes/attention-client/device-service"
mkdir -p "$OUTPUT_DIR/source"
cp "$src/"{persistent_vector.cpp,persistent_cube.cpp,persistent_protocol.hpp,streaming_gmm.hpp,actual_gmm.cpp,launch.cpp} "$OUTPUT_DIR/source/"
cp "$repo/prototypes/attention-client/qwen-next/client_kernel.cpp" "$OUTPUT_DIR/source/"
# Frozen translation units enable only shape/topology adaptation, not new GEMM math.
for file in persistent_vector.cpp persistent_cube.cpp; do
  sed -i '1i#define QWEN_NEXT 1' "$OUTPUT_DIR/source/$file"
done
export LAUNCH_SOURCE="$OUTPUT_DIR/source/launch.cpp"
SOURCE="$OUTPUT_DIR/source/persistent_vector.cpp" OBJECT_NAME=persistent_vector bash "$src/build.sh"
SOURCE="$OUTPUT_DIR/source/client_kernel.cpp" OBJECT_NAME=queue_service bash "$src/build.sh"
SOURCE="$OUTPUT_DIR/source/persistent_cube.cpp" OBJECT_NAME=persistent_cube bash "$src/build_actual_gmm.sh"

printf '%s\n' '{"server_config_words":27,"client_config_words":15,"hidden":2048,"inner":512,"topk":10,"owners":4}' > "$OUTPUT_DIR/abi.json"
