"""Cap1-only direct remote chunk fanout discriminator, not GEMM overlap yet.

Freeze the single admitted source, build its maps before FETCH, then transfer
8-token chunks once from peer memory and fan out locally from UB. No expanded
peer traffic, extra READY flags, early result or changed reduction order.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from betterscale.patches.expert_service import build as builder

METHOD=r'''
  __aicore__ inline void RemoteFanout(__gm__ int32_t *src,
                                     __gm__ int32_t *dst,int *map,int tokens) {
    auto event=(sequence&1)?EVENT_ID2:EVENT_ID1;
    auto local=scratch[(sequence&1)*8192];
    GlobalTensor<int32_t> in,out;in.SetGlobalBuffer(src);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    DataCopy(local,in,tokens*HIDDEN/2);
    SetFlag<HardEvent::MTE2_MTE3>(event);WaitFlag<HardEvent::MTE2_MTE3>(event);
    for(int t=0;t<tokens;++t)for(int k=0;k<TOPK;++k) {
      out.SetGlobalBuffer(dst+map[t*TOPK+k]*HIDDEN/2);
      DataCopy(out,local[t*HIDDEN/2],HIDDEN/2);
    }
    SetFlag<HardEvent::MTE3_MTE2>(event);++sequence;
  }
'''
FETCH=r'''
    if(kind==FETCH && n>=ExpertRoutePlan::MIN_ROWS) {
      int per=(n+VW-1)/VW,first=worker*per,last=ScalarMin(n,first+per);
      if(first>=last)continue;
      int map[(TOKENS+VW-1)/VW*TOPK];
      io.Read((__gm__ int32_t*)ptr[5]+c*MAP+8+first*TOPK,(last-first)*TOPK);
      for(int i=0;i<(last-first)*TOPK;++i)map[i]=io.words.GetValue(i);
      V2LiteMovePipeline copy;copy.Init(io.words);
      for(int t=first;t<last;t+=8)
        copy.RemoteFanout((__gm__ int32_t*)sources[c]+V2LITE_PAYLOAD_WORDS+t*HIDDEN/2,
                          (__gm__ int32_t*)ptr[1],map+(t-first)*TOPK,
                          ScalarMin(8,last-t));
      copy.Finish();
    } else if(kind==FETCH) {
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();root=builder.emit(a.output,draft_layers=1,route_plan_min_rows=1024)
    native=root/'source';src=native/'persistent_vector.cpp';s=src.read_text()
    old='            Descriptor(io, slots + i * 16, s[i]);'
    assert s.count(old)==1
    s=s.replace(old,'''            if (SumSources(s[i].rows)>=ExpertRoutePlan::MIN_ROWS)
              Group(io, slots+i*16, s[i], cfg[7], cfg[14], cfg[15], cfg);
            else
              Descriptor(io, slots + i * 16, s[i]);''')
    begin=s.index('      } else if (slot.stage == PULL) {')
    end=s.index('      } else {\n        if (vkind == REPACK)',begin)
    body=s[begin:end]
    prefix='      } else if (slot.stage == PULL) {'
    body=body.replace(prefix,prefix+'''
        if (SumSources(slot.rows)>=ExpertRoutePlan::MIN_ROWS) {
          slot.stage=READY_UP;
          vs=-1;
        } else {''',1)+'        }\n'
    s=s[:begin]+body+s[end:]
    src.write_text(s)
    move=native/'bf16_move.hpp';s=move.read_text()
    assert s.count('  __aicore__ inline void Finish() {')==1
    s=s.replace('  __aicore__ inline void Finish() {',METHOD+'  __aicore__ inline void Finish() {')
    assert s.count('    if(kind==FETCH) {')==1
    s=s.replace('    if(kind==FETCH) {',FETCH)
    move.write_text(s)
    for name in ('persistent_cube.o','queue_service.o'):shutil.copy2(a.baseline/name,root/name)
    env=dict(os.environ,OUTPUT_DIR=str(root),SOURCE=str(src),OBJECT_NAME='persistent_vector',
             LAUNCH_SOURCE=str(native/'launch.cpp'))
    with (root/'persistent_vector.build.log').open('w') as log:
        subprocess.run(['bash',str(Path(builder.__file__).parent/'build-vector.sh')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/'experiment.json').write_text(json.dumps(dict(
        mode='cap1/layer/native1024/direct-peer-fanout8',overlap=False,
        requires=dict(fine_pack=False,move_quantum=0,urgent=False),baseline=str(a.baseline)),indent=2))

if __name__=='__main__':main()
