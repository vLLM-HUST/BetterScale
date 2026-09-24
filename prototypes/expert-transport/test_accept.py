"""CPU execution of the emitted ID-validation loops; no NPU import."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from build_accept import patch
from betterscale.patches.expert_service import build as builder


def loop(text):
    start=text.index('    for (int i = 0; i < n * TOPK; ++i) {')
    end=text.index('    claimed[c] = gen;',start)
    return text[start:end]


class Admission(unittest.TestCase):
    def test_validation_preserved_and_only_unused_planned_cache_is_skipped(self):
        source=(Path(builder.__file__).parent/'native/persistent_vector.cpp').read_text()
        original=loop(source);candidate=loop(patch(source,True))
        code=r"""
#include <cassert>
#include <climits>
#include <vector>
constexpr int TOPK=8,EXPERTS=256;
namespace ExpertRoutePlan { constexpr int MIN_ROWS=1024; }
struct Words { std::vector<int> values; int GetValue(int i) { return values.at(i); } };
struct IO { Words words; };
struct Slot { std::vector<int> ids[1]={std::vector<int>(32768,-77)}; };
"""
        for name,body in [('original',original),('candidate',candidate)]:
            code+=f'int {name}(IO &io, Slot &s,int n) {{ int c=0;\n{body}\nreturn 0;}}\n'
        code+=r"""
int main() {
 for(int n:{1,3,96,1023,1024,4096}) {
  IO io;io.words.values.resize(n*TOPK);
  for(int i=0;i<n*TOPK;++i)io.words.values[i]=(i*17)%256;
  Slot a,b;
  assert(original(io,a,n)==0 && candidate(io,b,n)==0);
  for(int i=0;i<n*TOPK;++i)
    assert(b.ids[0][i]==(n<1024?a.ids[0][i]:-77));
  for(int pos:{0,n*TOPK/2,n*TOPK-1})
    for(int bad:{-1,256,INT_MIN,INT_MAX}) {
      int before=io.words.values[pos];io.words.values[pos]=bad;
      assert(original(io,a,n)==-1 && candidate(io,b,n)==-1);
      io.words.values[pos]=before;
    }
 }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'test.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2',str(root/'test.cpp'),'-o',str(root/'test')],check=True)
            subprocess.run([str(root/'test')],check=True)


if __name__=='__main__':unittest.main()
