"""Emit a reversible routing-metadata DMA candidate, without changing MOD defaults."""
import argparse
from pathlib import Path
import shutil
import subprocess
import os
from betterscale.patches.expert_service.build import emit


def patch(source):
    before='void Read(__gm__ int32_t *p, int n) {'
    assert source.count(before)==1
    source=source.replace(before,'void Read(__gm__ int32_t *p, int n, int offset = 0) {',1)
    source=source.replace('DataCopy(words, g, n);','DataCopy(words[offset], g, n);',1)
    # ids used to be scalar-GM stores consumed by the same coordinator's scalar
    # reads. DMA consumers require a completed DMA producer, not that cache state.
    old = """      s.ids[c][i] = io.words.GetValue(i);
      if (s.ids[c][i] < 0 || s.ids[c][i] >= EXPERTS)
        return -1;
    }
    claimed[c] = gen;"""
    new = """      int id = io.words.GetValue(i);
      if (id < 0 || id >= EXPERTS)
        return -1;
    }
    io.Write(s.ids[c], (n * TOPK + 7) / 8 * 8);
    claimed[c] = gen;"""
    assert source.count(old)==1;source=source.replace(old,new)
    begin=source.index('__aicore__ inline void Group(');end=source.index('__aicore__ inline void Command(',begin)
    group=source[begin:end]
    old='''    if (s.gen[c])
      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          ++count[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                  s.ids[c][i] % LOCAL_EXPERTS];'''
    new='''    if (s.gen[c]) {
      io.Read(s.ids[c], (s.rows[c] * TOPK + 7) / 8 * 8);
      for (int i = 0; i < s.rows[c] * TOPK; ++i) {
        int id = io.words.GetValue(i);
        if (id / LOCAL_EXPERTS == owner)
          ++count[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) + id % LOCAL_EXPERTS];
      }
    }'''
    assert group.count(old)==1;group=group.replace(old,new)
    old='''    if (s.gen[c])
      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          io.words.SetValue(
              8 + i, cursor[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                            s.ids[c][i] % LOCAL_EXPERTS]++);'''
    new='''    if (s.gen[c]) {
      // Output map occupies [0,MAP). Input tile has disjoint UB ownership.
      constexpr int INPUT = (MAP + 7) / 8 * 8, TILE = 8192;
      static_assert((INPUT + TILE) * 4 <= 196608, "routing UB overlap");
      for (int first = 0; first < s.rows[c] * TOPK; first += TILE) {
        int n = ScalarMin(TILE, s.rows[c] * TOPK - first);
        io.Read(s.ids[c] + first, (n + 7) / 8 * 8, INPUT);
        for (int i = 0; i < n; ++i) {
          int id = io.words.GetValue(INPUT + i);
          if (id / LOCAL_EXPERTS == owner)
            io.words.SetValue(8 + first + i,
              cursor[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) + id % LOCAL_EXPERTS]++);
        }
      }
    }'''
    assert group.count(old)==1;group=group.replace(old,new)
    return source[:begin]+group+source[end:]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();root=emit(a.output,draft_layers=1)
    src=root/'source/persistent_vector.cpp';src.write_text(patch(src.read_text()))
    for name in ['persistent_cube.o','queue_service.o']:
        shutil.copy2(a.baseline/name,root/name)
    package=Path(__file__).resolve().parents[2]/'src/betterscale/patches/expert_service'
    env=dict(os.environ,OUTPUT_DIR=str(root),SOURCE=str(src),OBJECT_NAME='persistent_vector',LAUNCH_SOURCE=str(root/'source/launch.cpp'))
    with (root/'persistent_vector.build.log').open('w') as log:
        subprocess.run(['bash',str(package/'build-vector.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
