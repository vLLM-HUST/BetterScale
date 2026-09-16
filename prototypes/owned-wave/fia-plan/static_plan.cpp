// Bounded CANN 9.0.1/910B native FIA bootstrap. No attention arithmetic.
// Preload only in the owned candidate process; never install into CANN.
#include <acl/acl.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <memory>
#include <vector>
#include <string>

struct Plan {
  aclrtFuncHandle fn;
  uint32_t blocks;
  bool fd = false;
  std::vector<unsigned char> args;
  std::vector<aclrtPlaceHolderInfo> placeholders;
  std::vector<aclrtLaunchKernelAttr> attrs;
  uint32_t key_desc = 0, value_desc = 0, tiling = 0;
};
static std::atomic<uint64_t> selected{0};
static std::unique_ptr<Plan> pending;
static thread_local bool metadata_only = false;
static std::vector<std::unique_ptr<Plan>> plans;
using Launch = decltype(&aclrtLaunchKernelWithHostArgs);
static Launch original() {
  static auto fn = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "aclrtLaunchKernelWithHostArgs"));
  return fn;
}
static uint64_t word(const Plan &p, size_t off) {
  uint64_t result; memcpy(&result, p.args.data()+off, 8); return result;
}
static void put(Plan &p, size_t off, uint64_t value) {
  memcpy(p.args.data()+off, &value, 8);
}
extern "C" int plan_begin(uint64_t query) {
  if (!query || selected.load() || pending) return -1;
  selected.store(query); return 0;
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle fn, uint32_t blocks,
 aclrtStream stream, aclrtLaunchKernelCfg *cfg, void *args, size_t size,
 aclrtPlaceHolderInfo *ph, size_t nph) {
  bool intercepted=false;
  uint64_t q=0;
  if (size >= 16) memcpy(&q, static_cast<char*>(args)+8, 8);
  if (selected.load() && q==selected.load() && size>=296) {
    intercepted=true;
    auto p=std::make_unique<Plan>(); p->fn=fn; p->blocks=blocks;
    p->args.assign(static_cast<unsigned char*>(args),static_cast<unsigned char*>(args)+size);
    p->placeholders.assign(ph,ph+nph);
    if (cfg) p->attrs.assign(cfg->attrs,cfg->attrs+cfg->numAttrs);
    char name[512]{};
    auto getname=reinterpret_cast<decltype(&aclrtGetFunctionName)>(dlsym(RTLD_NEXT,"aclrtGetFunctionName"));
    int rc=getname(fn,sizeof(name),name);
    if (!metadata_only) fprintf(stderr,"FIA_PLAN name=%s rc=%d blocks=%u placeholders=%zu\n",name,rc,blocks,nph);
    if (!metadata_only) for (auto h:p->placeholders) fprintf(stderr,"FIA_BIND addr=%u data=%u\n",h.addrOffset,h.dataOffset);
    const char *prefix="FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da_";
    std::string variant = rc == 0 ? name : "";
    p->fd = variant == std::string(prefix)+"5100000000010200203";
    if (p->fd || variant == std::string(prefix)+"5000000000010200203") pending=std::move(p);
    selected.store(0);
  }
  if (metadata_only) return intercepted && pending ? ACL_SUCCESS : ACL_ERROR_INVALID_PARAM;
  return original()(fn,blocks,stream,cfg,args,size,ph,nph);
}
// Call only after synchronizing bootstrap's stream. Device memory is never borrowed.
extern "C" int plan_finish() {
  selected.store(0);
  if (!pending) return -1;
  // Failure consumes the candidate rather than poisoning future bootstraps.
  auto candidate = std::move(pending);
  auto &p = *candidate;
  unsigned seen = 0;
  for (auto h:p.placeholders) {
    unsigned bit = h.addrOffset==16 ? 1 : h.addrOffset==24 ? 2 :
        h.addrOffset==48 ? 4 : h.addrOffset==56 ? 8 : h.addrOffset==280 ? 16 : 0;
    if (!bit || (seen & bit) || h.dataOffset < 296 ||
        h.dataOffset % 8 || size_t(h.dataOffset)+8 > p.args.size()) return -2;
    seen |= bit;
    if (h.addrOffset==16) candidate->key_desc=h.dataOffset;
    else if (h.addrOffset==24) candidate->value_desc=h.dataOffset;
    else if (h.addrOffset==280) candidate->tiling=h.dataOffset;
    else if (h.addrOffset!=48 && h.addrOffset!=56) return -2;
  }
  if (!candidate->key_desc || !candidate->value_desc || !candidate->tiling || candidate->placeholders.size()!=5) return -3;
  if (candidate->key_desc+48>candidate->args.size() || candidate->value_desc+48>candidate->args.size() || candidate->tiling+96>candidate->args.size()) return -4;
  if (word(*candidate,candidate->key_desc)!=40 || word(*candidate,candidate->value_desc)!=40) return -5;
  if (seen != 31 || candidate->tiling != 296 ||
      word(p,p.key_desc+8) != 0x100000003ULL ||
      word(p,p.value_desc+8) != 0x100000003ULL) return -6;
  uint64_t sum = 0;
  for (size_t off : {56,64,72,80}) {
    auto n = word(p,p.tiling+off);
    if (n > (112ULL<<20)) return -7;
    sum += n;
  }
  if (p.fd) {
    if (p.tiling+168 > p.args.size()) return -7;
    auto lse=word(p,p.tiling+152), out=word(p,p.tiling+160);
    if(lse>(112ULL<<20) || out>(112ULL<<20)) return -7;
    sum += lse + out;
  }
  if (!sum || sum > (112ULL<<20) || sum != word(p,p.tiling+88)) return -7;
  std::vector<aclrtPlaceHolderInfo> kept;
  for (auto h:candidate->placeholders) if (h.addrOffset!=48 && h.addrOffset!=56) kept.push_back(h);
  candidate->placeholders=std::move(kept);
  plans.push_back(std::move(candidate)); return int(plans.size()-1);
}
extern "C" uint64_t plan_workspace(int id) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return 0;
  // Installed FAInfer plan: workSpaceSize at byte88; plus 910B system reserve.
  return word(*plans[id],plans[id]->tiling+88)+(16ULL<<20);
}
// Bind once before capture. K/V list descriptors remain immutable inline data.
extern "C" int plan_bind(int id, const uint64_t *ptrs) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return -1;
  auto &p=*plans[id];
  const size_t offsets[]={8,40,48,56,120,256,272};
  for(size_t i=0;i<7;i++) put(p,offsets[i],ptrs[i]);
  put(p,p.key_desc+40,ptrs[7]); put(p,p.value_desc+40,ptrs[8]);
  return 0;
}
extern "C" int plan_launch(int id, void *stream) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return -1;
  auto &p=*plans[id]; aclrtLaunchKernelCfg cfg{p.attrs.data(),p.attrs.size()};
  return original()(p.fn,p.blocks,stream,&cfg,p.args.data(),p.args.size(),p.placeholders.data(),p.placeholders.size());
}
extern "C" int plan_clone(int id) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return -1;
  plans.push_back(std::make_unique<Plan>(*plans[id]));
  return int(plans.size()-1);
}

// Caller must destroy graphs/retire all invocations first. IDs are never recycled.
extern "C" int plan_release(int id) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return -1;
  plans[id].reset(); return 0;
}

// Exact installed tiling payload through the next inline descriptor. Export
// baseline's plan unchanged; callers must retain native block count and variant.
extern "C" int plan_metadata(int id, void *dst, size_t capacity) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return -1;
  auto &p=*plans[id];
  if(p.key_desc<=p.tiling || p.key_desc-p.tiling>capacity) return -2;
  size_t bytes=p.key_desc-p.tiling;
  memcpy(dst,p.args.data()+p.tiling,bytes);
  return int(bytes);
}
extern "C" int plan_blocks(int id) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return -1;
  return plans[id]->blocks;
}
extern "C" int plan_is_fd(int id) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return -1;
  return plans[id]->fd;
}
extern "C" int plan_bind_metadata(int id, uint64_t address) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id] || !address) return -1;
  auto &p=*plans[id];
  std::vector<aclrtPlaceHolderInfo> kept;
  for(auto h:p.placeholders) if(h.addrOffset!=280) kept.push_back(h);
  p.placeholders=std::move(kept); put(p,280,address); return 0;
}

// Keep native split boundaries/reduction exactly; add only empty main-loop
// entries for extra launch blocks. Installed FAInfer coreInfo starts at200,
// with26-element arrays. CombineScale strides by actual launch block count.
// Scratch base is provisioned for all24 physical Cube cores by native tiling.
extern "C" int plan_pad_blocks(int id) {
  if(id<0 || size_t(id)>=plans.size() || !plans[id]) return -1;
  auto &p=*plans[id];
  if(!p.fd) return p.blocks==24 ? 0 : -2;
  auto u32=[&](size_t off) { uint32_t x; memcpy(&x,p.args.data()+p.tiling+off,4);return x; };
  if(p.key_desc-p.tiling!=2528 || p.blocks<1 || p.blocks>24 ||
     u32(172)!=p.blocks || u32(168)>26 || u32(32)==0) return -3;
  if(word(p,p.tiling+56)!=18874368 || word(p,p.tiling+64)!=9437184 ||
     word(p,p.tiling+72)!=18874368 || word(p,p.tiling+80)!=18874368) return -5;
  for(uint32_t core=0;core<p.blocks;++core) {
    if(u32(200+4*core)>u32(616+4*core) || u32(616+4*core)>=u32(32)) return -4;
  }
  for(uint32_t core=p.blocks;core<24;++core) {
    uint32_t start=1,end=0;
    memcpy(p.args.data()+p.tiling+200+4*core,&start,4);
    memcpy(p.args.data()+p.tiling+616+4*core,&end,4);
  }
  p.blocks=24; return 0;
}
