#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export OUTPUT_DIR=${OUTPUT_DIR:-"$repo/runs/attention-persistent-build"}
src="$repo/prototypes/attention-client/device-service"
mkdir -p "$OUTPUT_DIR/source"
cp "$src/"{persistent_vector.cpp,persistent_cube.cpp,persistent_protocol.hpp,streaming_gmm.hpp,actual_gmm.cpp,launch.cpp} "$OUTPUT_DIR/source/"
export LAUNCH_SOURCE="$OUTPUT_DIR/source/launch.cpp"
SOURCE="$OUTPUT_DIR/source/persistent_vector.cpp" OBJECT_NAME=persistent_vector bash "$src/build.sh"
SOURCE="$OUTPUT_DIR/source/persistent_cube.cpp" OBJECT_NAME=persistent_cube bash "$src/build_actual_gmm.sh"
printf 'CATLASS_INCLUDE=%s\nACTUAL_GMM_DFC=%s\nTILE_M=%s\nTILE_N=%s\nTILE_K=%s\nL0_K=%s\n' \
  "${CATLASS_INCLUDE:-CANN9.0.1-installed-default}" "${ACTUAL_GMM_DFC:-0}" \
  "${ACTUAL_GMM_TILE_M:-128}" "${ACTUAL_GMM_TILE_N:-256}" \
  "${ACTUAL_GMM_TILE_K:-256}" "${ACTUAL_GMM_L0_K:-64}" > "$OUTPUT_DIR/build-options.txt"
(cd "$OUTPUT_DIR" && sha256sum persistent_cube.o persistent_vector.o launch.so) > "$OUTPUT_DIR/objects.sha256"
