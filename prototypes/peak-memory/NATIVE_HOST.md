# HC-pre workspace-only native build

This is a native operator patch, not a Python-wheel feature. Keep the shared
donor installation untouched. Tested source: Ascend9bf964cb, CANN9.0.1,
Ascend910B2, Python3.12, native API/kernel package already used by the donor.
`hc_pre_workspace.patch` changes only the workspace lower bound; attrs, tensor
schemas, tiling serialization, tiling choices and kernel code are unchanged.

## Small coherent rebuild

1. Apply the patch in a private copy of the pinned `csrc` tree. The raw task's
   source and dependency products are in
   `runs/peak-memory-fixes-20260915/source/csrc`.
2. Take **the34 A2 operators** from the `ascend910b` branch of the pinned
   `csrc/build_aclnn.sh` (`CUSTOM_OPS_ARRAY`), including host-only metadata ops.
   Do not use `ALL`: the source tree also contains an alternative
   `lightning_indexer_vllm`; linking both definitions of `LIInfoParser` fails.
   The failed ALL build was retained, not retried unchanged.
3. Configure a fresh host build directory and build `cust_opmaster -j4`:

   ```text
   -G Ninja
   -DBUILD_OPEN_PROJECT=ON
   -DASCEND_OP_NAME=<semicolon-separated native A2 operator set>
   -DCANN_3RD_LIB_PATH=<this source tree's third_party>
   -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.1
   -DCHECK_COMPATIBLE=true
   -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=/usr/bin/ccache
   -DASCEND_COMPUTE_UNIT=ascend910b
   -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON
   -DENABLE_OPS_KERNEL=OFF -DENABLE_BUILD_PKG=OFF
   ```

   Some common stubs still write into `csrc/build` even with a different `-B`;
   serialize builds within one source tree and keep tested OPP copies detached.
   This avoids rebuilding unmodified API, protobuf and device binaries. The
   selected-op experiment independently rebuilt HcPre's two kernel objects;
   both were byte-identical to the original donor's objects. No changed ABI is
   being paired with old generated artifacts.
4. Copy the **complete original runtime vendor** into a new private package
   root. Dereference/check symlinks before replacing files; never write through
   a link into the original runtime. Replace only these two regular files with
   the newly built complete-A2 `libcust_opmaster_rt2.0.so`:

   ```text
   op_impl/ai_core/tbe/op_tiling/liboptiling.so
   op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
   ```

   Direct comparison of the hw3 envelope checked696 files: exactly these two
   differed. All API libraries, schemas, wrappers, configs and device binaries
   remained byte-identical. `native-closure.json` records the two library
   identities. This is one complete vendor root, NOT two partial vendors
   prepended under the same name.
5. Select the private root **before** custom-op registration. The fixture sets
   `vllm_ascend.utils._CUSTOM_OP_BASE_DIR` to its private package parent so native
   bootstrap cannot prepend the original vendor again. This fixture hook is
   not installed in the public Worker. Preserve CANN's built-in Python paths.
6. Run `hc_probe.py --rows 516 --reference <original-leaf-outputs> --vendor
   <complete-private-vendor> --output <fresh-output>` within normal admission.
   It must match original output bits, pass changed-input FULL replay and show
   the reduced workspace. Then use the same private vendor for the TP8 dummy
   model gate. No model weights are copied or loaded from checkpoint.

The hw3 run closure is
`/workspace/my-ascend-workspace/runs/peak-memory-hw3-20260915/`;
local receipts are under BetterScale's `runs/peak-memory-hw3-20260915/`.
The native package remains a task artifact. Shipping it needs a corresponding
native build/distribution; publishing an unchanged pure-Python wheel would not
install this fix.
