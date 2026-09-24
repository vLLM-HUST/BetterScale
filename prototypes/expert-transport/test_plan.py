"""Wire placement and actual native-plan Group against the scalar reference."""
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import build_plan
from test_group import HARNESS,NATIVE

class Plan(unittest.TestCase):
    def test_emitted_wire_and_exact_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);baseline=root/'baseline';baseline.mkdir()
            for name in ['persistent_cube.o']:(baseline/name).write_bytes(b'CPU fixture')
            output=root/'emitted'
            with patch('sys.argv',['build_plan',str(output),'--baseline',str(baseline),'--min-plan-rows','1']),patch('build_plan.subprocess.run'):
                build_plan.main()
            abi=json.loads((output/'abi.json').read_text())
            self.assertEqual(abi['client_route_plan'],'native-v2-cap1')
            self.assertEqual(abi['sources_per_wave'],1)
            def body(source,name):
                start=source.index('__aicore__ inline void GroupLegacy(') if name=='Candidate' and '__aicore__ inline void GroupLegacy(' in source else source.index('__aicore__ inline void Group(')
                return source[start:source.index('__aicore__ inline void Command(',start)].replace('void Group(',f'void {name}(',1)
            candidate=body((output/'source/persistent_vector.cpp').read_text(),'Candidate')
            candidate=candidate.replace(str(build_plan.BASE),'0') # compact CPU-only wire allocation
            original=body((NATIVE/'persistent_vector.cpp').read_text(),'Original')
            harness=HARNESS.replace('LOCAL_VALUE','256').replace('ORIGINAL_BODY',original).replace('CANDIDATE_BODY',candidate)
            harness=harness.replace('constexpr bool SINGLE_LAYER=true;', '''constexpr bool SINGLE_LAYER=true;
constexpr int SOURCES_PER_WAVE=1,ROUTES=TOKENS*TOPK,STOP=0,LINE=16;
int SumSources(int *x){int sum=0;for(int c=0;c<SOURCES;++c)sum+=x[c];return sum;}
std::array<std::vector<int>,SOURCES> plans;
int64_t SourcePointer(int64_t*,int c){return (int64_t)plans[c].data();}
void Store(int *p,int v){*p=v;}
''')
            harness=harness.replace('for(int sources:{1,2,7})','for(int sources:{0,1})')
            old='Original(a.io,a.ptr.data(),a.s,owner,mode,tail);Candidate(b.io,b.ptr.data(),b.s,owner,mode,tail);'
            new='''for(int c=0;c<SOURCES;++c){
   plans[c].assign(ROUTES+512,0);int count[256]={},cursor[256]={};
   for(int i=0;i<b.s.rows[c]*TOPK;++i)++count[b.ids[c][i]];
   int sum=0;for(int g=0;g<256;++g){cursor[g]=sum;sum+=count[g];plans[c][ROUTES+g*2]=count[g];}
   for(int i=0;i<b.s.rows[c]*TOPK;++i)plans[c][i]=cursor[b.ids[c][i]]++;
  }
  int status[16]={};int64_t cfg[1]={(int64_t)status};
  Original(a.io,a.ptr.data(),a.s,owner,mode,tail);Candidate(b.io,b.ptr.data(),b.s,owner,mode,tail,cfg);
  assert(status[0]==0);'''
            self.assertIn(old,harness);harness=harness.replace(old,new)
            # Cursor is retained by the common Group body but unused in this candidate.
            cpp=root/'plan.cpp';cpp.write_text(harness);binary=root/'plan'
            result=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra',str(cpp),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            subprocess.run([str(binary)],check=True,timeout=30)

if __name__=='__main__':unittest.main()
