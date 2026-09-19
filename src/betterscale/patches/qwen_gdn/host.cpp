// SPDX-License-Identifier: Apache-2.0
// Framework-owned temporaries for the qualified raw AscendC H/O kernels.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <c10/core/DeviceGuard.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>

namespace {
using HLaunch = uint32_t (*)(uint32_t, void*, void*, void*, void*, void*, void*,
                            void*, void*, void*, void*, void*, void*, void*);
using OLaunch = uint32_t (*)(uint32_t, void*, void*, void*, void*, void*, void*,
                            void*, void*, void*, void*, void*);
void* library = nullptr;
HLaunch launch_h = nullptr;
OLaunch launch_o = nullptr;
std::string library_path;

void initialize(const std::string& path) {
  // Called once by the validated worker before any capture/model execution.
  TORCH_CHECK(!library || library_path == path, "Cannot change GDN library in a live process");
  if (library) return;
  library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(library, "Cannot load GDN library: ", dlerror());
  launch_h = reinterpret_cast<HLaunch>(dlsym(library, "aclrtlaunch_bs_gdn_h"));
  launch_o = reinterpret_cast<OLaunch>(dlsym(library, "aclrtlaunch_bs_gdn_o"));
  TORCH_CHECK(launch_h && launch_o, "GDN raw launch symbols missing");
  library_path = path;
}

int64_t workspace_size(int64_t cores, int64_t requests) {
  TORCH_CHECK(cores == 24 && requests > 0 && requests <= 9,
              "Outside qualified GDN workspace contract");
  auto align = [](int64_t n) { return (n + 511) / 512 * 512; };
  // Same reserved system prefix and user offsets as the immutable H/O tiling.
  const int64_t system = 16 * 1024 * 1024;
  const int64_t h = system + 2 * align(cores * 64 * 128 * 4 * 2)
      + align(cores * 128 * 128 * 4 * 2) + 2 * align((requests + 1) * 8);
  const int64_t o = system + 2 * align(cores * 64 * 128 * 4 * 2)
      + 2 * align(cores * 64 * 64 * 4 * 2) + align(64 * 64);
  return std::max(h, o);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> pool_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& w,
    const at::Tensor& u, const at::Tensor& g, at::Tensor bank,
    const at::Tensor& cu, const at::Tensor& state,
    const at::Tensor& indices, const at::Tensor& th, const at::Tensor& to,
    int64_t cores) {
  TORCH_CHECK(launch_h && launch_o, "Initialize qualified GDN library before capture");
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1,
              "GDN requires an NPU tensor");
  c10::DeviceGuard guard(q.device());
  const auto tokens = q.size(2);
  const auto chunks = indices.size(0);
  const auto requests = cu.numel() - 1;
  TORCH_CHECK(tokens > 0 && tokens <= 2048 && requests == 9,
              "Outside owned mixed GDN capacity");
  // Framework allocator allocations inside capture belong to its graph pool.
  // They are invocation-local: no bank/capacity/group retains a scratch arena.
  auto h = at::empty({1, 24, chunks, 128, 128}, q.options());
  auto v = at::empty({1, 24, tokens, 128}, q.options());
  auto output = at::empty_like(v);
  auto workspace = at::empty({workspace_size(cores, requests)}, q.options().dtype(at::kByte));
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto rc = launch_h(cores, stream, k.data_ptr(), w.data_ptr(), u.data_ptr(),
      g.data_ptr(), bank.data_ptr(), cu.data_ptr(), state.data_ptr(), h.data_ptr(),
      v.data_ptr(), bank.data_ptr(), workspace.data_ptr(), th.data_ptr());
  TORCH_CHECK(rc == 0, "GDN H launch failed: ", rc);
  rc = launch_o(cores, stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), h.data_ptr(),
      g.data_ptr(), cu.data_ptr(), indices.data_ptr(), output.data_ptr(),
      workspace.data_ptr(), to.data_ptr());
  TORCH_CHECK(rc == 0, "GDN O launch failed: ", rc);
  // H/V are exposed solely for the independent initial-H diagnostic oracle.
  // Production caller drops them immediately; output lives through its consumer.
  return {output, h, v};
}
}  // namespace

TORCH_LIBRARY(betterscale_gdn, m) {
  m.def("initialize(str path) -> ()", initialize);
  m.def("workspace_size(int cores, int requests) -> int", workspace_size);
  m.def("pool_forward(Tensor q, Tensor k, Tensor w, Tensor u, Tensor g, Tensor(a!) bank, Tensor cu, Tensor state, Tensor indices, Tensor th, Tensor to, int cores) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(betterscale_gdn, PrivateUse1, m) {
  m.impl("pool_forward", pool_forward);
}
