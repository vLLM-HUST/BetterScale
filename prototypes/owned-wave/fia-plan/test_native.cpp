// CPU-only ABI rejection/lifecycle tests. No runtime or device execution.
#include "static_plan.cpp"
#include <cassert>

static void fixture() {
  pending=std::make_unique<Plan>();
  pending->args.resize(3008);
  pending->placeholders={{280,296},{16,2824},{24,2872},{48,2920},{56,2952}};
  for (size_t off : {2824,2872}) {
    put(*pending,off,40); put(*pending,off+8,0x100000003ULL);
  }
  for (size_t off : {56,64,72,80}) put(*pending,296+off,1024);
  put(*pending,296+88,4096);
}
int main() {
  fixture(); int id=plan_finish(); assert(id>=0);
  assert(plans[id]->placeholders.size()==3);
  assert(plan_workspace(id)==(16ULL<<20)+4096);
  int copy=plan_clone(id); assert(copy>=0);
  uint64_t ptrs[]={1,2,3,4,5,6,7,8,9};
  assert(plan_bind(copy,ptrs)==0);
  assert(word(*plans[copy],48)==3 && word(*plans[copy],56)==4);
  assert(word(*plans[copy],2824+40)==8);
  assert(word(*plans[id],48)==0); // clone never mutates the template
  assert(plan_release(copy)==0 && plan_release(copy)==-1);
  assert(plan_launch(copy,nullptr)==-1 && plan_clone(copy)==-1);
  assert(plan_release(id)==0);
  fixture(); pending->placeholders[4]=pending->placeholders[3];
  assert(plan_finish()==-2 && !pending);
  fixture(); pending->placeholders[4].dataOffset=3008;
  assert(plan_finish()==-2 && !pending);
  fixture(); put(*pending,2824+8,0);
  assert(plan_finish()==-6 && !pending);
  fixture(); put(*pending,296+88,4097);
  assert(plan_finish()==-7 && !pending);
  fixture(); id=plan_finish(); assert(id>=0); assert(plan_release(id)==0);
  puts("native FIA CPU admission/lifecycle PASS");
}
