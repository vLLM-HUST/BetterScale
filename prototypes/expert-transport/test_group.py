"""Execute actual candidate Group against unchanged Group on the CPU.

Checks complete map/count/boundary equivalence and UB bounds, not DMA visibility.
"""
import pathlib
import subprocess
import tempfile
import unittest
from build_group import patch

NATIVE=pathlib.Path(__file__).resolve().parents[2]/'src/betterscale/patches/expert_service/native'

class Group(unittest.TestCase):
    def test_full_group_equivalence(self):
        original=(NATIVE/'persistent_vector.cpp').read_text();candidate=patch(original)
        def body(s,name):
            start=s.index('__aicore__ inline void Group(')
            return s[start:s.index('__aicore__ inline void Command(',start)].replace('void Group(',f'void {name}(',1)
        with tempfile.TemporaryDirectory() as tmp:
            for local in [256,128,64]:
                cpp=pathlib.Path(tmp)/'group.cpp';binary=cpp.with_suffix('')
                cpp.write_text(HARNESS.replace('LOCAL_VALUE',str(local)).replace('ORIGINAL_BODY',body(original,'Original')).replace('CANDIDATE_BODY',body(candidate,'Candidate')))
                result=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                subprocess.run([str(binary)],check=True,timeout=30)

HARNESS=r'''
#include <vector>
#include <array>
#include <cassert>
#include <cstdint>
#include <algorithm>
#define __aicore__
#define __gm__
constexpr int SOURCES=7,TOPK=8,TOKENS=4096,MAP=TOKENS*TOPK+8;
constexpr int LOCAL_EXPERTS=LOCAL_VALUE,GROUPS=LOCAL_EXPERTS;
constexpr bool SINGLE_LAYER=true;
constexpr int EVENT_ID0=0;
enum class HardEvent {V_S};
template<HardEvent> void SetFlag(int){}
template<HardEvent> void WaitFlag(int){}
int ScalarMin(int a,int b){return std::min(a,b);}
struct Words {
 std::vector<int> a=std::vector<int>(196608/4);
 int GetValue(int i) const {assert(i>=0 && i<int(a.size()));return a[i];}
 void SetValue(int i,int v){assert(i>=0 && i<int(a.size()));a[i]=v;}
};
void Duplicate(Words &w,int32_t v,int n){for(int i=0;i<n;++i)w.SetValue(i,v);}
struct Transfer {Words words;
 void Read(int *p,int n,int offset=0){assert(n%8==0);for(int i=0;i<n;++i)words.SetValue(offset+i,p[i]);}
 void Write(int *p,int n){assert(n%8==0);for(int i=0;i<n;++i)p[i]=words.GetValue(i);}
};
struct Slot {int gen[SOURCES]={},rows[SOURCES]={},layer[SOURCES]={};
 int *ids[SOURCES];int live=0,boundary=0;};
ORIGINAL_BODY
CANDIDATE_BODY
struct Fixture {
 Slot s;Transfer io;std::array<int64_t,16> ptr={};
 std::vector<int> maps=std::vector<int>(SOURCES*MAP,-77),counts=std::vector<int>(GROUPS*2,-77);
 std::vector<int> seg0=std::vector<int>(GROUPS*2+512,-77),seg1=seg0;
 std::array<std::vector<int>,SOURCES> ids;
 Fixture(int rows,int sources,int pattern){
  ptr[5]=(int64_t)maps.data();ptr[6]=(int64_t)counts.data();ptr[9]=(int64_t)seg0.data();ptr[10]=(int64_t)seg1.data();
  for(int c=0;c<SOURCES;++c){
   int n=c<sources?std::max(1,rows-c):0;s.gen[c]=n?c+1:0;s.rows[c]=n;s.layer[c]=40;
   ids[c].resize(TOKENS*TOPK);
   for(int i=0;i<n*TOPK;++i)ids[c][i]=pattern==0?(i*71+c*17)%256:pattern==1?255:((i+c)%8);
   s.ids[c]=ids[c].data();
  }
 }
};
int main(){
 for(int rows:{1,3,96,127,1024,4096})for(int sources:{1,2,7})for(int pattern:{0,1,2})
 for(int owner=0;owner<256/LOCAL_EXPERTS;++owner)for(int mode:{0,1,2})for(int tail:{0,3}){
  Fixture a(rows,sources,pattern),b(rows,sources,pattern);
  Original(a.io,a.ptr.data(),a.s,owner,mode,tail);Candidate(b.io,b.ptr.data(),b.s,owner,mode,tail);
  assert(a.maps==b.maps && a.counts==b.counts && a.seg0==b.seg0 && a.seg1==b.seg1);
  assert(a.s.live==b.s.live && a.s.boundary==b.s.boundary);
 }
}
'''
if __name__=='__main__':unittest.main()
