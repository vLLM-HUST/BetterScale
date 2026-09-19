#!/usr/bin/env bash
set -euo pipefail
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${BUILD_DIR:?provide an explicit task-local build directory}"
cann=/usr/local/Ascend/cann-9.0.1
native="$cann/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3"
mkdir -p "$BUILD_DIR"
set +u; source "$cann/set_env.sh"; set -u
"$cann/bin/bisheng" -std=c++17 -O2 --asc-aicore-lang --npu-arch=dav-2201 \
 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-ignored-attributes \
 -I"$native" -I"$cann/include" -I"$cann/aarch64-linux/ascendc/include/highlevel_api" \
 -I"$cann/aarch64-linux/ascendc/include/basic_api" -I"$cann/aarch64-linux/asc" \
 "$src/panel_kernel.cpp" -L"$cann/lib64" -lascendcl -lascendc_runtime -lruntime \
 -Wl,-rpath,"$cann/lib64" -o "$BUILD_DIR/libbs_gate_up_panel.so"
