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

struct Plan {
  aclrtFuncHandle fn;
  uint32_t blocks;
  std::vector<unsigned char> args;
  std::vector<aclrtPlaceHolderInfo> placeholders;
  std::vector<aclrtLaunchKernelAttr> attrs;
  uint32_t key_desc = 0, value_desc = 0, tiling = 0;
};
static std::atomic<uint64_t> selected{0};
static std::unique_ptr<Plan> pending;
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
  uint64_t q=0;
  if (size >= 16) memcpy(&q, static_cast<char*>(args)+8, 8);
  if (selected.load() && q==selected.load() && size>=296) {
    auto p=std::make_unique<Plan>(); p->fn=fn; p->blocks=blocks;
    p->args.assign(static_cast<unsigned char*>(args),static_cast<unsigned char*>(args)+size);
    p->placeholders.assign(ph,ph+nph);
    if (cfg) p->attrs.assign(cfg->attrs,cfg->attrs+cfg->numAttrs);
    char name[512]{};
    auto getname=reinterpret_cast<decltype(&aclrtGetFunctionName)>(dlsym(RTLD_NEXT,"aclrtGetFunctionName"));
    int rc=getname(fn,sizeof(name),name);
    fprintf(stderr,"FIA_PLAN name=%s rc=%d blocks=%u placeholders=%zu\n",name,rc,blocks,nph);
    for (auto h:p->placeholders) fprintf(stderr,"FIA_BIND addr=%u data=%u\n",h.addrOffset,h.dataOffset);
    // Exact qualified binary variant only; reject generic/FD dispatch.
    if (rc==0 && strcmp(name,"FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da_5000000000010200203")==0) pending=std::move(p);
    selected.store(0);
  }
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
