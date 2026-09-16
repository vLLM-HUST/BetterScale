#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
extern "C" int load_server(const char *path, const char *symbol, void **binary,
                           void **function) {
  int rc = aclrtBinaryLoadFromFile(path, nullptr, binary);
  if (rc)
    return rc;
  return aclrtBinaryGetFunction(*binary, symbol, function);
}
extern "C" int launch_blocks(void *fn, void *stream, void *config, void *audit,
                             void *trace, uint32_t blocks) {
  aclrtLaunchKernelAttr attrs[3]{};
  attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attrs[0].value.schemMode = 1;
  attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
  attrs[1].value.timeoutUs.timeoutLow = 10000000;
  attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  attrs[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
  aclrtLaunchKernelCfg cfg{};
  cfg.numAttrs = 3;
  cfg.attrs = attrs;
  struct {
    void *c;
    void *a;
    void *t;
  } args{config, audit, trace};
  return aclrtLaunchKernelWithHostArgs(fn, blocks, stream, &cfg, &args,
                                       sizeof(args), nullptr, 0);
}
extern "C" int unload_server(void *binary) { return aclrtBinaryUnLoad(binary); }

extern "C" int launch_server(void *fn, void *stream, void *config, void *a,
                             void *b) {
  return launch_blocks(fn, stream, config, a, b, 1);
}
