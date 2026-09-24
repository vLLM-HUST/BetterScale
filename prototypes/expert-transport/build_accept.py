"""Isolate planned-frame admission's redundant scalar-GM route cache.

Both diagnostic variants record coordinator Accept/Group envelopes (engine3,
kind1/2). Those clocks include scalar validation and source selection, not just
wire transfer. The candidate keeps independent ID range validation; it omits
only the ID cache unused by cap1 native Group. No default MOD source change.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from betterscale.patches.expert_service import build as builder


def patch(text, skip, vector=False):
    old="""      s.ids[c][i] = io.words.GetValue(i);
      if (s.ids[c][i] < 0 || s.ids[c][i] >= EXPERTS)
        return -1;"""
    assert text.count(old)==1
    if skip:
        text=text.replace(old,"""      int id = io.words.GetValue(i);
      if (id < 0 || id >= EXPERTS)
        return -1;
      // Planned cap1 Group consumes native counts/maps, not this GM cache.
      if (!(ExpertRoutePlan::MIN_ROWS && n >= ExpertRoutePlan::MIN_ROWS))
        s.ids[c][i] = id;""")
    if vector:
        assert not skip
        begin=text.index('    for (int i = 0; i < n * TOPK; ++i) {')
        end=text.index('    claimed[c] = gen;',begin)
        original=text[begin:end]
        text=text[:begin]+"""    if (ExpertRoutePlan::MIN_ROWS && n >= ExpertRoutePlan::MIN_ROWS) {
      if (!PlannedRouteIdsValid(io.words, io.buf.Get<float>(), n * TOPK))
        return -1;
    } else {
"""+original+'    }\n'+text[end:]
        text='#include "route_validate.hpp"\n'+text
    old='          int mask = Accept(io, cfg, s[i], claimed, finished, closed, policy);'
    assert text.count(old)==1
    text=text.replace(old,'''          uint64_t acceptBegin = GetSystemCycle();
          int mask = Accept(io, cfg, s[i], claimed, finished, closed, policy);
          if (mask > 0)
            Record(cfg, eventCount, 3, 1, i, acceptBegin, SumSources(s[i].rows)*TOPK);''')
    old='Group(io, ptr, slot, cfg[7], cfg[14], cfg[15], cfg);'
    assert text.count(old)==2
    text=text.replace(old,'''uint64_t groupBegin = GetSystemCycle();
              Group(io, ptr, slot, cfg[7], cfg[14], cfg[15], cfg);
              Record(cfg, eventCount, 3, 2, vs, groupBegin, slot.live);''')
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--skip-planned-ids',action='store_true')
    p.add_argument('--vector-validate',action='store_true')
    p.add_argument('--noinline-validator',action='store_true')
    a=p.parse_args()
    root=builder.emit(a.output,draft_layers=1,route_plan_min_rows=1024)
    native=root/'source';src=native/'persistent_vector.cpp'
    src.write_text(patch(src.read_text(),a.skip_planned_ids,a.vector_validate))
    if a.vector_validate:
        header=Path(__file__).with_name('route_validate.hpp').read_text()
        if a.noinline_validator:
            header=header.replace('__aicore__ inline bool PlannedRouteIdsValid(',
                                  '__aicore__ __attribute__((noinline)) bool PlannedRouteIdsValid(')
        (native/'route_validate.hpp').write_text(header)
    for name in ('persistent_cube.o','queue_service.o'):
        shutil.copy2(a.baseline/name,root/name)
    package=Path(builder.__file__).parent
    env=dict(os.environ,OUTPUT_DIR=str(root),SOURCE=str(src),OBJECT_NAME='persistent_vector',
             LAUNCH_SOURCE=str(native/'launch.cpp'))
    with (root/'persistent_vector.build.log').open('w') as log:
        subprocess.run(['bash',str(package/'build-vector.sh')],env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/'experiment.json').write_text(json.dumps(dict(skip_planned_ids=a.skip_planned_ids,
        noinline_validator=a.noinline_validator,vector_validate=a.vector_validate,coordinator_observer=True,mode='cap1/layer/native1024',baseline=str(a.baseline)),indent=2))


if __name__=='__main__': main()
