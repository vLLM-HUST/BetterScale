"""Replace expensive per-route readiness with a conservative source-prefix gate.

Each mover prepares per-tile max input-token dependencies, then publishes only
chunk completion. Coordinator vector-reduces16 dependency catalogs and advances
a source-prefix watermark. AIC waits on one tile threshold/one progress line.
All publications follow payload DMA completion; no changed math or early DONE.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from betterscale.patches.expert_service import build as builder

MOVER=r'''
    if(kind==FETCH && n>=ExpertRoutePlan::MIN_ROWS) {
      constexpr int MAX_TILES=1024,DEP_OFFSET=32*LINE;
      int map[(TOKENS+VW-1)/VW*TOPK],expertIds[(TOKENS+VW-1)/VW*TOPK],used=0;
      io.Read((__gm__ int32_t*)ptr[5]+c*MAP+8,n*TOPK);
      for(int t=worker*8;t<n;t+=VW*8)
        for(int j=0;j<ScalarMin(8,n-t);++j)
          for(int k=0;k<TOPK;++k)map[used++]=io.words.GetValue((t+j)*TOPK+k);
      used=0;
      io.Read((__gm__ int32_t*)ptr[15]+c*ROUTES,n*TOPK);
      for(int t=worker*8;t<n;t+=VW*8)
        for(int j=0;j<ScalarMin(8,n-t);++j)
          for(int k=0;k<TOPK;++k)expertIds[used++]=io.words.GetValue((t+j)*TOPK+k);
      int starts[EXPERTS],tileBase[EXPERTS],tileCount=0,lastRow=0;
      io.Read((__gm__ int32_t*)ptr[6],EXPERTS*2);
      for(int e=0;e<EXPERTS;++e) {
        int end=io.words.GetValue(e*2);
        starts[e]=lastRow;tileBase[e]=tileCount;
        tileCount+=(end-lastRow+63)/64;lastRow=end;
      }
      if(tileCount>MAX_TILES){Store((__gm__ int32_t*)cfg[0]+STOP*LINE,-71);return;}
      int dependencies[MAX_TILES]={};used=0;
      for(int t=worker*8;t<n;t+=VW*8)
        for(int j=0;j<ScalarMin(8,n-t);++j)
          for(int k=0;k<TOPK;++k,++used) {
            int e=expertIds[used],tile=tileBase[e]+(map[used]-starts[e])/64;
            dependencies[tile]=ScalarMax(dependencies[tile],t+j+1);
          }
      auto ready=(__gm__ int32_t*)cfg[20]+slot*CAPACITY*LINE;
      int generation=Load((__gm__ int32_t*)cfg[0]+(PACK_EPOCH+slot)*LINE);
      auto fp=io.buf.Get<float>();
      for(int i=0;i<MAX_TILES;++i)fp.SetValue(i,float(dependencies[i]));
      io.Write(ready+DEP_OFFSET+worker*MAX_TILES,MAX_TILES);
      ready[worker*LINE+1]=0;Store(ready+worker*LINE,generation);
      V2LiteMovePipeline copy;copy.Init(io.words);
      int offset=0,sequence=0;
      for(int t=worker*8;t<n;t+=VW*8) {
        int tokens=ScalarMin(8,n-t);
        copy.RemoteFanout((__gm__ int32_t*)sources[c]+V2LITE_PAYLOAD_WORDS+t*HIDDEN/2,
                          (__gm__ int32_t*)ptr[1],map+offset,tokens);
        if(sequence) {
          WaitFlag<HardEvent::MTE3_S>(((sequence-1)&1)?EVENT_ID2:EVENT_ID1);
          ready[worker*LINE+1]=sequence;Refresh(ready+worker*LINE);
        }
        offset+=tokens*TOPK;++sequence;
      }
      if(sequence)WaitFlag<HardEvent::MTE3_S>(((sequence-1)&1)?EVENT_ID2:EVENT_ID1);
      // Empty/short last mover slices no longer constrain the final prefix.
      ready[worker*LINE+1]=(n+VW*8-1)/(VW*8);Refresh(ready+worker*LINE);
      copy.Finish();
'''
COORD=r'''
    // Dependency catalogs are immutable after each mover's epoch publication.
    for(int i=0;i<2;++i)if(s[i].stage!=EMPTY && SumSources(s[i].rows)>=ExpertRoutePlan::MIN_ROWS) {
      auto ready=(__gm__ int32_t*)cfg[20]+i*CAPACITY*LINE;
      int epoch=Load(ctrl+(PACK_EPOCH+i)*LINE),n=SumSources(s[i].rows);
      if(!s[i].inputPlanReady && Joined(ready,0,VW,epoch)) {
        constexpr int MAX_TILES=1024,DEP_OFFSET=32*LINE;
        io.Read(ready+DEP_OFFSET,VW*MAX_TILES);
        auto fp=io.buf.Get<float>();
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for(int w=1;w<VW;++w) {
          Max(fp,fp,fp[w*MAX_TILES],MAX_TILES);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        io.Write(ready+DEP_OFFSET+VW*MAX_TILES,MAX_TILES);
        Store(ready+16*LINE,0);s[i].inputPrefix=0;s[i].inputPlanReady=1;
      }
      if(s[i].inputPlanReady && s[i].inputPrefix<n) {
        int rounds=(n+VW*8-1)/(VW*8);
        for(int w=0;w<VW;++w) {
          Refresh(ready+w*LINE);rounds=ScalarMin(rounds,ready[w*LINE+1]);
        }
        int prefix=ScalarMin(n,rounds*VW*8);
        if(prefix>s[i].inputPrefix) {Store(ready+16*LINE,prefix);s[i].inputPrefix=prefix;}
      }
    }
'''
TILE=r'''
        if(pack && pack->ready) {
          constexpr int MAX_TILES=1024,DEP_OFFSET=32*Persistent::LINE;
          auto need=(__gm__ float*)(pack->ready+DEP_OFFSET+Persistent::VW*MAX_TILES);
          Persistent::Refresh((__gm__ int32_t*)(need+inputTileBase+coord.m()));
          int threshold=int(need[inputTileBase+coord.m()]);
          int64_t polls=0;
          while(Persistent::Load(pack->ready+16*Persistent::LINE)<threshold) {
            if(Persistent::Load(pack->stop) || ++polls>=pack->pollLimit) {
              Persistent::Store(pack->stop,-15);return;
            }
          }
        }
'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--baseline',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();shutil.copytree(a.baseline,root,ignore=shutil.ignore_patterns('runtime-wrapper-original'))
    native=root/'source';src=native/'persistent_vector.cpp';s=src.read_text()
    old='      if (!PlannedRouteIdsValid(io.words, io.buf.Get<float>(), n * TOPK))\n        return -1;'
    assert s.count(old)==1;s=s.replace(old,old+'\n      io.Write(s.ids[c],n*TOPK); // DMA metadata cache for tile dependencies, not scalar stores.')
    old='  __gm__ int32_t *ids[SOURCES]; int live = 0, boundary = 0;'
    assert s.count(old)==1;s=s.replace(old,old+'\n  int inputPlanReady=0,inputPrefix=0;')
    old='            s[i].stage = PULL;';assert s.count(old)==1
    s=s.replace(old,old+'\n            s[i].inputPlanReady=0;s[i].inputPrefix=0;')
    old='    if (cs < 0) {';assert s.count(old)==1;s=s.replace(old,COORD+old)
    old='slot.stage == PULL && SumSources(slot.rows)>=ExpertRoutePlan::MIN_ROWS'
    assert s.count(old)==1;s=s.replace(old,'slot.stage == PULL && slot.inputPlanReady && SumSources(slot.rows)>=ExpertRoutePlan::MIN_ROWS')
    # READY_UP after complete FETCH also waits for catalog reduction.
    old='          if (slot.stage != READY_UP && slot.stage != UP &&'
    assert s.count(old)==1;s=s.replace(old,'''          if(SumSources(slot.rows)>=ExpertRoutePlan::MIN_ROWS && !slot.inputPlanReady)continue;
'''+old)
    src.write_text(s)
    move=native/'bf16_move.hpp';s=move.read_text();start=s.index('    if(kind==FETCH && n>=ExpertRoutePlan::MIN_ROWS) {');end=s.index('    } else if(kind==FETCH) {',start)
    move.write_text(s[:start]+MOVER+s[end:])
    stream=native/'streaming_gmm.hpp';s=stream.read_text()
    s=s.replace('  uint32_t row = 0, nextCore = 0;','  uint32_t row = 0, nextCore = 0, inputTileBase=0;')
    start=s.index('        if(pack && pack->ready) {');end=s.index('        tile(x[',start)
    s=s[:start]+TILE+s[end:]
    old='    row = end;';assert s.count(old)==1
    s=s.replace(old,'    inputTileBase+=(rows+ACTUAL_GMM_TILE_M-1)/ACTUAL_GMM_TILE_M;\n'+old)
    stream.write_text(s)
    package=Path(builder.__file__).parent
    for unit,script in [('persistent_vector','build-vector.sh'),('persistent_cube','build-cube.sh')]:
        env=dict(os.environ,OUTPUT_DIR=str(root),SOURCE=str(native/f'{unit}.cpp'),OBJECT_NAME=unit,
                 LAUNCH_SOURCE=str(native/'launch.cpp'),ACTUAL_GMM_TILE_M='64')
        with (root/f'{unit}.compact-build.log').open('w') as log:
            subprocess.run(['bash',str(package/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/'experiment.json').write_text(json.dumps(dict(mode='cap1/layer/native1024/compact-source-prefix',tile_m=64,
        flags='per-mover chunk progress; vector-max immutable tile dependencies',baseline=str(a.baseline),runtime=str(root/'runtime')),indent=2))

if __name__=='__main__':main()
