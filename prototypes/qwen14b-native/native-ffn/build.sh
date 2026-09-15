#!/usr/bin/env bash
set -euo pipefail
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${BUILD_DIR:?provide a task-local, non-hidden build directory}"
cann=/usr/local/Ascend/cann-9.0.1
mkdir -p "$BUILD_DIR"
set +u; source "$cann/set_env.sh"; set -u
"$cann/bin/bisheng" -std=c++17 -O2 --asc-aicore-lang --npu-arch=dav-2201 \
  -fPIC -shared -DNATIVE_CM="${NATIVE_CM:-128}" -DNATIVE_CN="${NATIVE_CN:-256}" -DNATIVE_CK="${NATIVE_CK:-128}" -DNATIVE_PAIR="${NATIVE_PAIR:-0}" -DNATIVE_V3="${NATIVE_V3:-0}" -DNATIVE_SWIZZLE="${NATIVE_SWIZZLE:-0}" -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-ignored-attributes \
  -I"$cann/include" -I"$cann/aarch64-linux/ascendc/include/highlevel_api" \
  -I"$cann/aarch64-linux/ascendc/include/basic_api" \
  -I"$cann/aarch64-linux/asc" \
  "$src/kernel.cpp" -L"$cann/lib64" -lascendcl -lascendc_runtime -lruntime \
  -Wl,-rpath,"$cann/lib64" -o "$BUILD_DIR/libqwen_native_ffn.so"
sha256sum "$BUILD_DIR/libqwen_native_ffn.so" > "$BUILD_DIR/binary.sha256"
