#include <acl/acl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
static uintptr_t selected = 0;
extern "C" void select_query(uint64_t q) { selected = q; }
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle fn, uint32_t blocks,
 aclrtStream stream, aclrtLaunchKernelCfg *cfg, void *args, size_t size,
 aclrtPlaceHolderInfo *ph, size_t nph) {
 using F = decltype(&aclrtLaunchKernelWithHostArgs);
 static auto original = reinterpret_cast<F>(dlsym(RTLD_NEXT, "aclrtLaunchKernelWithHostArgs"));
 if (selected) {
   fprintf(stderr, "LAUNCH bytes=%zu blocks=%u first:", size, blocks);
   for (size_t i=0; i+8<=size && i<256; i+=8) {
     uint64_t word; memcpy(&word, static_cast<char*>(args)+i, 8);
     fprintf(stderr," %zu:%llx",i,(unsigned long long)word);
   }
   fprintf(stderr,"\n");
 }
 bool found = false;
 for (size_t i=0; selected && i+8<=size; i+=8)
   if (memcmp(static_cast<char*>(args)+i, &selected, 8)==0) found=true;
 if (found) {
   auto f = fopen("fia-launch.bin", "wb");
   if (f) { fwrite(args, 1, size, f); fclose(f); }
   char name[512]{};
   auto getname=reinterpret_cast<decltype(&aclrtGetFunctionName)>(dlsym(RTLD_NEXT,"aclrtGetFunctionName"));
   int rc=getname(fn,sizeof(name),name);
   f=fopen("fia-launch.meta", "w");
   if(f) {
     fprintf(f,"name %s\nname_rc %d\nblocks %u\nbytes %zu\n",name,rc,blocks,size);
     for(size_t i=0;i<nph;i++) fprintf(f,"placeholder %u %u\n",ph[i].addrOffset,ph[i].dataOffset);
     fclose(f);
   }
   fprintf(stderr,"FIA_CAPTURE blocks=%u bytes=%zu placeholders=%zu attrs=%zu\n",blocks,size,nph,cfg ? size_t(cfg->numAttrs):0);
   selected = 0;
 }
 return original(fn,blocks,stream,cfg,args,size,ph,nph);
}
