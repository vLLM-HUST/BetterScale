"""Host execution of actual device admission; no simulated GEMM claims."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace as S
from betterscale.patches.expert_service.build import emit
from betterscale.patches.expert_service.config import ServiceConfig

NATIVE = Path(__file__).resolve().parents[1]/'src/betterscale/patches/expert_service/native'

class Batching(unittest.TestCase):
    def test_build_binds_cap_for_both_placements_and_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            for placement in ('layer','expert'):
                for owners in (2,4):
                    for cap in (1,2,7):
                        out=emit(Path(tmp)/f'{placement}-{owners}-{cap}',draft_layers=1,
                                 placement=placement,owners=owners,sources_per_wave=cap)
                        abi=json.loads((out/'abi.json').read_text())
                        self.assertEqual(abi['sources_per_wave'],cap)
                        self.assertFalse(abi['early_return'])
                        self.assertIn(f'SOURCES_PER_WAVE = {cap};',(out/'source/persistent_protocol.hpp').read_text())
                        for n in ('launch.so','persistent_vector.o','persistent_cube.o','queue_service.o'):
                            (out/n).write_bytes(b'CPU fixture')
                        config=ServiceConfig('/control',str(out),owners,1,0,1,placement=placement)
                        config.check_build()
                        for invalid in (0,8,True,'7',None):
                            abi['sources_per_wave']=invalid
                            (out/'abi.json').write_text(json.dumps(abi))
                            with self.assertRaisesRegex(ValueError,'sources_per_wave'):config.check_build()
            for invalid in (0,8,True,1.5,'7'):
                with self.assertRaises(ValueError):emit(Path(tmp)/'bad',sources_per_wave=invalid)
                self.assertFalse((Path(tmp)/'bad').exists())

    def test_receipt_reports_executed_return_mode_and_batch_cap(self):
        path=NATIVE.parent/'persistent_engine.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
        finish=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='finish')
        scope={}
        exec(compile(ast.Module(body=[finish],type_ignores=[]),str(path),'exec'),scope)
        control=[[0]*16 for _ in range(128)]
        control[0][0]=control[43][0]=1
        def tensor(value):return S(cpu=lambda:S(tolist=lambda:value))
        for cap in (1,7):
            engine=S(streams=[],control=tensor(control),events=tensor([]),trace=tensor([]),
                     kernels=S(abi={'sources_per_wave':cap}),fine_pack=False,combined_return=False,
                     config=[S(item=lambda:0) for _ in range(29)])
            result=scope['finish'](engine)
            self.assertFalse(result['early_return'])
            self.assertEqual(result['sources_per_wave'],cap)

    @unittest.skipUnless(shutil.which('g++'),'host compiler required')
    def test_actual_accept_ready_coalescing_and_lifetime(self):
        source=(NATIVE/'persistent_vector.cpp').read_text()
        begin=source.index('__aicore__ inline int Accept(')
        accept=source[begin:source.index('__aicore__ inline void Group(',begin)]
        with tempfile.TemporaryDirectory() as tmp:
            for cap in (1,2,7):
                cpp=Path(tmp)/f'cap{cap}.cpp';binary=cpp.with_suffix('')
                cpp.write_text(HARNESS.replace('CAP_VALUE',str(cap)).replace('ACCEPT_BODY',accept))
                compiled=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                                         '-I',str(NATIVE),str(cpp),'-o',str(binary)],capture_output=True,text=True)
                self.assertEqual(compiled.returncode,0,compiled.stderr)
                subprocess.run([str(binary)],check=True,timeout=10)

HARNESS=r'''
#include <cassert>
#include <cstdint>
#include <algorithm>
#define __aicore__
#define __gm__
constexpr int SOURCES=7,TOPK=8,EXPERTS=256,LAYERS=41,TOKENS=4096;
constexpr int SOURCES_PER_WAVE=CAP_VALUE;
constexpr bool SINGLE_LAYER=true;
#include "priority_policy.hpp"
using namespace Persistent;
struct Words {int *p=nullptr;int GetValue(int i) const{return p[i];}};
struct Transfer {Words words;int Flag(int *p){return *p;}void Read(int *p,int){words.p=p;}};
int64_t SourcePointer(int64_t *cfg,int c){return ((int64_t*)cfg[27])[c];}
bool SameLayer(int *gen,int *layers,int layer){
 for(int c=0;c<SOURCES;++c)if(gen[c] && layers[c]!=layer)return false;
 return true;
}
struct Slot {int gen[SOURCES]={},rows[SOURCES]={},layer[SOURCES]={};
 int *ids[SOURCES];int priority=0,serviceRank=0,ticket=0;};
ACCEPT_BODY
struct Fixture {
 int frames[SOURCES][80]={},ids[SOURCES][8]={};
 int claimed[SOURCES]={},finished[SOURCES]={},closed[SOURCES]={};
 int64_t cfg[29]={},pointers[SOURCES]={},weights[82]={};
 Transfer io;Slot s;PriorityPolicy policy;
 Fixture(){
  std::fill(weights,weights+82,1);
  cfg[24]=1;cfg[25]=(int64_t)weights;cfg[26]=41;cfg[27]=(int64_t)pointers;
  for(int c=0;c<SOURCES;++c){pointers[c]=(int64_t)frames[c];s.ids[c]=ids[c];}
 }
 void ready(int c,int layer=40,int priority=0,int generation=1){
  frames[c][0]=generation;frames[c][8]=generation;frames[c][9]=layer;
  frames[c][10]=1;frames[c][11]=priority;
  for(int k=0;k<TOPK;++k)frames[c][64+k]=k+c;
 }
 int accept(){return Accept(io,cfg,s,claimed,finished,closed,policy);}
};
int main(){
 {Fixture f;assert(f.accept()==0);}
 {Fixture f;for(int c=0;c<SOURCES;++c)f.ready(c);
  assert(f.accept()==(1<<SOURCES_PER_WAVE)-1);assert(f.accept()==0);
  for(int c=0;c<SOURCES_PER_WAVE;++c){
   assert(f.claimed[c]==1 && f.s.layer[c]==40 && f.s.rows[c]==1);
   for(int k=0;k<TOPK;++k)assert(f.ids[c][k]==k+c);
  }
 }
 {Fixture f;f.ready(0);assert(f.accept()==1);f.ready(1);f.ready(2,39);f.ready(3,40,1);
  assert(f.accept()==(SOURCES_PER_WAVE>1?2:0));assert(!f.claimed[2] && !f.claimed[3]);}
 {Fixture f;f.ready(0,40,0,2);f.ready(1);assert(f.accept()==2);assert(!f.claimed[0]);}
 {Fixture f;f.ready(0);f.claimed[0]=1;f.ready(1);assert(f.accept()==2);}
 {Fixture f;f.frames[0][0]=-1;assert(f.accept()==0);assert(f.closed[0]==1);}
 {Fixture f;f.frames[0][0]=-2;assert(f.accept()==-1);}
 {Fixture f;f.ready(0);f.frames[0][64]=-1;assert(f.accept()==-1);}
 {Fixture f;f.ready(0);f.weights[80]=0;assert(f.accept()==-1);}
 {Fixture f;f.ready(0,41);assert(f.accept()==-1);}
}
'''
