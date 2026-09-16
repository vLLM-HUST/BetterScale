// Use the installed ACLNN host planner, not a reimplementation of FD heuristics.
// The selected numerical launch is intercepted and suppressed. No model/runner
// call, Torch task-queue drain or device synchronization is needed per wave.
#include "static_plan.cpp"
#include <aclnnop/aclnn_fused_infer_attention_score.h>

static aclTensor *tensor(void *ptr, std::vector<int64_t> shape, aclDataType type) {
  std::vector<int64_t> strides(shape.size(),1);
  for (int i=int(shape.size())-2;i>=0;--i) strides[i]=strides[i+1]*shape[i+1];
  return aclCreateTensor(shape.data(),shape.size(),type,strides.data(),0,ACL_FORMAT_ND,
                         shape.data(),shape.size(),ptr);
}
// pointers: query, key, value, mask, table, output, workspace; all persistent.
extern "C" int plan_native_queries(const uint64_t *ptrs, int rows, int width,
 int heads, int kvheads, int pages, int columns, double scale,
 const int64_t *lengths, const int64_t *query_offsets, void *stream) {
  if (metadata_only || selected.load() || pending) return -20;
  if (rows < 1 || width < 1 || !query_offsets || !lengths) return -22;
  int64_t previous=0;
  for(int i=0;i<rows;++i) {
    if(query_offsets[i]<=previous || query_offsets[i]>(int64_t)rows*width ||
       lengths[i]<query_offsets[i]-previous) return -22;
    previous=query_offsets[i];
  }
  // The allocation keeps bucket capacity; the planner sees a contiguous live
  // prefix view. TND requires descriptor T == final actual query offset.
  auto tokens=query_offsets[rows-1];
  auto q=tensor((void*)ptrs[0],{tokens,heads,128},ACL_BF16);
  auto k=tensor((void*)ptrs[1],{pages,128,kvheads*128},ACL_BF16);
  auto v=tensor((void*)ptrs[2],{pages,128,kvheads*128},ACL_BF16);
  auto mask=tensor((void*)ptrs[3],{2048,2048},ACL_BOOL);
  auto table=tensor((void*)ptrs[4],{rows,columns},ACL_INT32);
  auto out=tensor((void*)ptrs[5],{tokens,heads,128},ACL_BF16);
  auto lse=tensor(nullptr,{0},ACL_FLOAT);
  auto kl=aclCreateTensorList(&k,1), vl=aclCreateTensorList(&v,1);
  auto qlen=aclCreateIntArray(query_offsets,rows);
  auto kvlen=aclCreateIntArray(lengths,rows);
  uint64_t workspace=0; aclOpExecutor *executor=nullptr;
  char layout[]="TND";
  int rc=aclnnFusedInferAttentionScoreGetWorkspaceSize(q,kl,vl,nullptr,mask,qlen,kvlen,
      nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,table,nullptr,nullptr,
      heads,scale,2147483647,0,layout,kvheads,3,0,128,0,false,out,lse,&workspace,&executor);
  if(!rc && workspace <= (128ULL<<20)) {
    selected.store(ptrs[0]); metadata_only=true;
    rc=aclnnFusedInferAttentionScore((void*)ptrs[6],workspace,executor,stream);
    metadata_only=false; selected.store(0);
  } else if(!rc) rc=-21;
  aclDestroyTensor(q); aclDestroyTensor(mask); aclDestroyTensor(table);
  aclDestroyTensor(out); aclDestroyTensor(lse);
  aclDestroyTensorList(kl); aclDestroyTensorList(vl);
  aclDestroyIntArray(qlen); aclDestroyIntArray(kvlen);
  if(rc) { pending.reset(); return rc>0 ? -rc : rc; }
  return plan_finish();
}
