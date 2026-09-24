"""Experimental cap1/layer wire extension using native client routing metadata.

The plan sits in existing aligned IPC padding beyond maximum hidden payload.
No default MOD source or ABI is changed. Multi-source batching is NOT supported.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from betterscale.patches.expert_service.build import emit

BASE=50176+4096*2048//2

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--min-plan-rows',type=int,default=1024);a=p.parse_args()
    assert 1<=a.min_plan_rows<=4096
    root=emit(a.output,draft_layers=1);native=root/'source'
    abi=json.loads((root/'abi.json').read_text());abi['client_route_plan']='native-v2-cap1';abi['client_route_plan_min_rows']=a.min_plan_rows
    source_bytes=((abi['payload_words']*4+abi['rows']*2048*2+(2<<20)-1)//(2<<20))*(2<<20)
    assert (BASE+4096*8+512)*4 <= source_bytes
    (root/'abi.json').write_text(json.dumps(abi,indent=2)+'\n')
    vector=native/'persistent_vector.cpp';s=vector.read_text()
    # Admission still independently validates all route IDs. Leave that cost in
    # the comparison rather than silently trusting a new wire producer.
    s=s.replace('void Read(__gm__ int32_t *p, int n) {','void Read(__gm__ int32_t *p, int n, int offset = 0) {',1)
    s=s.replace('DataCopy(words, g, n);','DataCopy(words[offset], g, n);',1)
    begin=s.index('__aicore__ inline void Group(');end=s.index('__aicore__ inline void Command(',begin)
    group=s[begin:end]
    old='''  for (int c = 0; c < SOURCES; ++c)
    if (s.gen[c])
      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          ++count[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                  s.ids[c][i] % LOCAL_EXPERTS];'''
    new=f'''  static_assert(SOURCES_PER_WAVE == 1 && LOCAL_EXPERTS == 256 && SINGLE_LAYER,
                "native plan prototype is cap1 whole-layer only");
  for (int c = 0; c < SOURCES; ++c) if (s.gen[c]) {{
    io.Read((__gm__ int32_t*)SourcePointer(cfg,c) + {BASE} + ROUTES, GROUPS*2);
    int total=0;
    for(int g=0;g<GROUPS;++g) {{
      count[g]=io.words.GetValue(g*2);total+=count[g];
      if(count[g]<0 || io.words.GetValue(g*2+1)!=0) {{
        Store((__gm__ int32_t*)cfg[0]+STOP*LINE,-61);return;
      }}
    }}
    if(total!=s.rows[c]*TOPK) {{Store((__gm__ int32_t*)cfg[0]+STOP*LINE,-62);return;}}
  }}'''
    assert group.count(old)==1;group=group.replace(old, f'  if(SumSources(s.rows)>={a.min_plan_rows}) {{\n'+new+'\n  } else {\n'+old+'\n  }')
    old='''    if (s.gen[c])
      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          io.words.SetValue(
              8 + i, cursor[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                            s.ids[c][i] % LOCAL_EXPERTS]++);'''
    new=f'''    if (s.gen[c]) io.Read((__gm__ int32_t*)SourcePointer(cfg,c)+{BASE},
                             (s.rows[c]*TOPK+7)/8*8,8);'''
    assert group.count(old)==1;group=group.replace(old, f'    if(s.rows[c]>={a.min_plan_rows}) {{\n'+new+'\n    } else {\n'+old+'\n    }')
    group=group.replace('int owner, int mode, int tailExperts)', 'int owner, int mode, int tailExperts, __gm__ int64_t *cfg)')
    s=s[:begin]+group+s[end:]
    s=s.replace('Group(io, ptr, slot, cfg[7], cfg[14], cfg[15]);','Group(io, ptr, slot, cfg[7], cfg[14], cfg[15], cfg);')
    vector.write_text(s)
    client=native/'client_kernel.cpp';s=client.read_text();needle='  if (quantized) {\n    // Eight compact'
    extra=f'''  // Native source-to-destination route map and int64 expert counts.
  if(n>={a.min_plan_rows}) {{
  for(int off=block*TILE;off<n*TOPK;off+=blocks*TILE) {{
    int count=n*TOPK-off<TILE?n*TOPK-off:TILE;
    io.Read((__gm__ int32_t*)cfg[16]+off,count);
    io.Write(src+{BASE}+off,count);
  }}
  if(block==0) {{io.Read((__gm__ int32_t*)cfg[17],512);io.Write(src+{BASE}+ROUTES,512);}}
  }}
'''
    assert s.count(needle)==1;s=s.replace(needle,extra+needle);client.write_text(s)
    shutil.copy2(a.baseline/'persistent_cube.o',root/'persistent_cube.o')
    package=Path(__file__).resolve().parents[2]/'src/betterscale/patches/expert_service'
    for unit,name in [('persistent_vector.cpp','persistent_vector'),('client_kernel.cpp','queue_service')]:
        env=dict(os.environ,OUTPUT_DIR=str(root),SOURCE=str(native/unit),OBJECT_NAME=name,LAUNCH_SOURCE=str(native/'launch.cpp'))
        with (root/f'{name}.build.log').open('w') as log:
            subprocess.run(['bash',str(package/'build-vector.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
