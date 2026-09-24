"""Bounded cap1 input-readiness experiment atop direct peer chunk fanout.

Uses existing row-ready storage with publication after MTE3 completion. This is
intentionally a cost discriminator: per-row scalar flags may erase overlap.
Runtime is a private copied MOD source capsule, never an ambient modification.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from betterscale.patches.expert_service import build as builder

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--tile-m',type=int,choices=(64,128),default=64)
    a=p.parse_args();root=a.output.resolve()
    subprocess.run([sys.executable,str(Path(__file__).with_name('build_fused_pack.py')),
                    str(root),'--baseline',str(a.baseline)],check=True)
    native=root/'source';src=native/'persistent_vector.cpp';s=src.read_text()
    old='!(cfg[20] && slot.stage == PACK)'
    assert s.count(old)==1
    s=s.replace(old,'!(cfg[20] && slot.stage == PULL && SumSources(slot.rows)>=ExpertRoutePlan::MIN_ROWS)')
    # Existing small-frame REPACK cannot publish per-row flags.
    old='Store(ctrl + (PACK_EPOCH + vs) * LINE, vgen + 1);'
    assert s.count(old)==1;s=s.replace(old,'Store(ctrl + (PACK_EPOCH + vs) * LINE, 0);')
    old='              Command(ctrl, VCMD, ++vgen, FETCH, i, mask);'
    assert s.count(old)==1
    s=s.replace(old,'''              Store(ctrl+(PACK_EPOCH+i)*LINE,
                    SumSources(s[i].rows)>=ExpertRoutePlan::MIN_ROWS ? vgen+1 : 0);
'''+old)
    src.write_text(s)
    move=native/'bf16_move.hpp';s=move.read_text()
    # RemoteFanout only: create a scalar completion event per ping-pong slot.
    start=s.index('  __aicore__ inline void RemoteFanout(');end=s.index('  __aicore__ inline void Finish()',start)
    body=s[start:end].replace('    SetFlag<HardEvent::MTE3_MTE2>(event);++sequence;',
                             '    SetFlag<HardEvent::MTE3_S>(event);\n    SetFlag<HardEvent::MTE3_MTE2>(event);++sequence;')
    s=s[:start]+body+s[end:]
    start=s.index('    if(kind==FETCH && n>=ExpertRoutePlan::MIN_ROWS) {');end=s.index('    } else if(kind==FETCH) {',start)
    s=s[:start]+r'''    if(kind==FETCH && n>=ExpertRoutePlan::MIN_ROWS) {
      // Interleave8-token chunks across movers so early source prefixes arrive
      // early, rather than every mover completing a contiguous source slice.
      int map[(TOKENS+VW-1)/VW*TOPK],used=0;
      io.Read((__gm__ int32_t*)ptr[5]+c*MAP+8,n*TOPK);
      for(int t=worker*8;t<n;t+=VW*8)
        for(int j=0;j<ScalarMin(8,n-t);++j)
          for(int k=0;k<TOPK;++k)map[used++]=io.words.GetValue((t+j)*TOPK+k);
      auto ready=(__gm__ int32_t*)cfg[20]+slot*CAPACITY*LINE;
      int generation=Load((__gm__ int32_t*)cfg[0]+(PACK_EPOCH+slot)*LINE);
      V2LiteMovePipeline copy;copy.Init(io.words);
      int offset=0,previousOffset=0,previousCount=0,sequence=0;
      for(int t=worker*8;t<n;t+=VW*8) {
        int tokens=ScalarMin(8,n-t);
        copy.RemoteFanout((__gm__ int32_t*)sources[c]+V2LITE_PAYLOAD_WORDS+t*HIDDEN/2,
                          (__gm__ int32_t*)ptr[1],map+offset,tokens);
        // Publish the previous chunk while this chunk's DMA is in flight.
        if(sequence) {
          WaitFlag<HardEvent::MTE3_S>(((sequence-1)&1)?EVENT_ID2:EVENT_ID1);
          for(int k=0;k<previousCount;++k)
            Store(ready+map[previousOffset+k]*LINE,generation);
        }
        previousOffset=offset;previousCount=tokens*TOPK;offset+=previousCount;++sequence;
      }
      if(sequence) {
        WaitFlag<HardEvent::MTE3_S>(((sequence-1)&1)?EVENT_ID2:EVENT_ID1);
        for(int k=0;k<previousCount;++k)Store(ready+map[previousOffset+k]*LINE,generation);
      }
      copy.Finish();
''' +s[end:]
    move.write_text(s)
    cube=native/'persistent_cube.cpp';s=cube.read_text()
    old='      PackGate pack{cfg[20] ?'
    assert s.count(old)==1
    s=s.replace(old,'''      bool inputReadyGate=cfg[20] && Load(ctrl+(PACK_EPOCH+slot)*LINE)>0;
      PackGate pack{inputReadyGate ?''')
    cube.write_text(s)
    stream=native/'streaming_gmm.hpp';s=stream.read_text()
    start=s.index('      if (firstTile < tiles && pack && pack->ready) {')
    end=s.index('      for (uint32_t t = firstTile;',start)
    s=s[:start]+s[end:]
    old='        uint64_t col = coord.n() * ACTUAL_GMM_TILE_N;'
    assert s.count(old)==1
    s=s.replace(old,old+r'''
        if(pack && pack->ready) {
          int64_t polls=0;
          uint32_t tileEnd=row+m+ACTUAL_GMM_TILE_M;
          if(tileEnd>end)tileEnd=end;
          for(uint32_t r=row+m;r<tileEnd;++r)
            while(Persistent::Load(pack->ready+r*Persistent::LINE)!=pack->generation) {
              if(Persistent::Load(pack->stop) || ++polls>=pack->pollLimit) {
                Persistent::Store(pack->stop,-15);return;
              }
            }
        }
''')
    stream.write_text(s)
    package=Path(builder.__file__).parent
    for unit,script in [('persistent_vector','build-vector.sh'),('persistent_cube','build-cube.sh')]:
        env=dict(os.environ,OUTPUT_DIR=str(root),SOURCE=str(native/f'{unit}.cpp'),OBJECT_NAME=unit,
                 LAUNCH_SOURCE=str(native/'launch.cpp'),ACTUAL_GMM_TILE_M=str(a.tile_m))
        with (root/f'{unit}.overlap-build.log').open('w') as log:
            subprocess.run(['bash',str(package/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    abi=json.loads((root/'abi.json').read_text());abi['experimental_input_overlap']=True
    (root/'abi.json').write_text(json.dumps(abi,indent=2)+'\n')
    runtime=root/'runtime';shutil.copytree(package.parents[1],runtime/'betterscale',ignore=shutil.ignore_patterns('__pycache__'))
    engine=runtime/'betterscale/patches/expert_service/persistent_engine.py';s=engine.read_text()
    old="self.fine_pack=os.environ.get('BETTERSCALE_EXPERT_FINE_PACK','1')=='1'"
    assert s.count(old)==1;s=s.replace(old,"self.fine_pack=a.get('experimental_input_overlap',False) or os.environ.get('BETTERSCALE_EXPERT_FINE_PACK','1')=='1'")
    old="assert not a.get('pipelined_copy') or not self.fine_pack"
    assert s.count(old)==1;s=s.replace(old,"assert a.get('experimental_input_overlap',False) or not a.get('pipelined_copy') or not self.fine_pack")
    engine.write_text(s)
    (root/'experiment.json').write_text(json.dumps(dict(mode='cap1/layer/native1024/input-tile-readiness',
        tile_m=a.tile_m,flags='row64B/generation/MTE3-before-ready',runtime=str(runtime),baseline=str(a.baseline)),indent=2))

if __name__=='__main__':main()
