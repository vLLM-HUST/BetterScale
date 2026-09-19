"""Change only actual native1536 L2 panel metadata; frozen ND weights and Cube tile."""
import ctypes
import json
import os
from pathlib import Path
import statistics
import struct
import torch
import torch_npu

root = Path(os.environ['CAPSULE'])
torch.npu.set_device(0)
lib = ctypes.CDLL(str(root/'libbs_gate_up_panel.so'))
launch = lib.launch_panel
launch.argtypes = [ctypes.c_void_p]*5
launch.restype = ctypes.c_int
raw = (root/'native1536.bin').read_bytes()
assert len(raw) == 288
words = list(struct.unpack('<72I', raw))
assert words[:5] == [24, 1536, 17408, 5120, 5120]
assert words[8:11] == [128,256,64] and words[50:55] == [3,2,4,34,0]
variants = {'fork_native': words}
for name, panels in [('panel_1x2',[1,2,12,34,0]), ('panel_3x5',[3,5,4,14,0]), ('panel_1x5',[1,5,12,14,0])]:
    changed=words.copy();changed[50:55]=panels
    assert all(a==b for i,(a,b) in enumerate(zip(words,changed)) if not 50<=i<55)
    variants[name]=changed
receipt = dict(status='RUNNING', scope='ND BF16. Stock F.linear vs recompiled native base with same tiling vs same binary with only L2 panel fields changed. Not model speedup.', checks=[],timings=[])
def save(): (root/'receipt.json').write_text(json.dumps(receipt,indent=2))
try:
 with torch.inference_mode():
    torch.manual_seed(1909)
    w = (torch.randn((17408,5120),device='npu')/5120**.5).to(torch.bfloat16)
    original_w = w.clone()
    tilings = {name:torch.tensor(list(struct.pack('<72I',*data)),dtype=torch.uint8,device='npu')
               for name,data in variants.items()}
    banks = {}
    for arm in ('native', *variants):
        banks[arm] = []
        for bank in (0,1):
            x = torch.randn((1536,5120),device='npu',dtype=torch.bfloat16)
            guard = torch.full((1536*17408+256,),123.,device='npu',dtype=torch.bfloat16)
            y = guard[128:-128].view(1536,17408)
            def body():
                if arm == 'native': return torch.nn.functional.linear(x,w)
                code = launch(x.data_ptr(),w.data_ptr(),y.data_ptr(),tilings[arm].data_ptr(),torch.npu.current_stream().npu_stream)
                assert code == 0,code
                return y
            for _ in range(3): body()
            torch.npu.synchronize();g=torch.npu.NPUGraph()
            with torch.npu.graph(g): out=body()
            banks[arm].append(dict(g=g,x=x,y=out,guard=guard))
    for generation in range(4):
        value = torch.randn_like(x)
        ref = value.float() @ w.float().T
        for arm,bankset in banks.items():
            item=bankset[generation%2];item['x'].copy_(value);item['g'].replay();torch.npu.synchronize()
            error=item['y'].float()-ref
            passed=bool(torch.allclose(item['y'].float(),ref,atol=.03,rtol=.03))
            intact=bool((item['guard'][:128]==123).all() and (item['guard'][-128:]==123).all()) and torch.equal(item['x'],value) and torch.equal(w,original_w)
            row=dict(arm=arm,generation=generation,passed=passed,guards_inputs_intact=bool(intact),max_abs=float(error.abs().max()),rmse=float(error.square().mean().sqrt()))
            receipt['checks'].append(row);save();assert passed and intact,row
    for arm in (*banks, *reversed(banks)):
        for i in range(5):banks[arm][i%2]['g'].replay()
        torch.npu.synchronize();samples=[]
        for i in range(30):
            a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            a.record();banks[arm][i%2]['g'].replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000)
        receipt['timings'].append(dict(arm=arm,median_us=statistics.median(samples),samples_us=samples));save()
    receipt['status']='PASS'
except BaseException as exc:
 receipt.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
finally:save()
print([(r['arm'],r['median_us']) for r in receipt['timings']],flush=True)
