"""EP partition, build identity and device-client submission order contracts."""
import ast,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as S
from betterscale.patches.expert_service.build import emit
from betterscale.patches.expert_service.config import ServiceConfig

class ExpertPartition(unittest.TestCase):
    def test_all_layers_each_have_exact_expert_union_and_draft(self):
        code='''
from betterscale.patches.expert_service.config import ServiceConfig
ServiceConfig('/control','/build',2,1,0,1,placement='expert').bind('/model',96)
from betterscale.patches.expert_service.placement import Placement
for n in (2,4):
 p=Placement('expert',n)
 for layer in range(41):
  assert p.targets(layer)==tuple(range(n))
  assert all(layer in p.layers(o) for o in range(n))
  assert [e for o in range(n) for e in range(*p.experts(o))]==list(range(256))
'''
        subprocess.run([sys.executable,'-c',code],check=True,env={k:v for k,v in os.environ.items() if not k.startswith('BETTERSCALE_EXPERT_')})

    def test_build_rejects_cross_placement_and_owner_count(self):
        with tempfile.TemporaryDirectory() as d:
            for n in (2,4):
                out=emit(Path(d)/str(n),draft_layers=1,placement='expert',owners=n)
                abi=json.loads((out/'abi.json').read_text())
                self.assertFalse(abi['combined_return']);self.assertEqual(abi['local_experts'],256//n)
                self.assertIn(f'LOCAL_EXPERTS = {256//n}',(out/'source/persistent_protocol.hpp').read_text())
                self.assertIn(f'COLLECT_OWNERS={n},COLLECT_EXPERTS={256//n}',(out/'source/client_kernel.cpp').read_text())
                for f in ('launch.so','persistent_vector.o','persistent_cube.o','queue_service.o'):(out/f).write_bytes(b'fixture')
                ServiceConfig('/control',str(out),n,1,0,1,placement='expert').check_build()
                with self.assertRaises(ValueError):ServiceConfig('/control',str(out),n,1,0,1).check_build()
                with self.assertRaises(ValueError):ServiceConfig('/control',str(out),6-n,1,0,1,placement='expert').check_build()

    def test_four_owner_collect_pointers_do_not_overlap_layer_or_counter(self):
        import torch
        path=Path(__file__).resolve().parents[1]/'src/betterscale/patches/expert_service/persistent_remote.py'
        tree=ast.parse(path.read_text());bank=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Bank')
        def cpu(fn):
            def call(*args,**kwargs):kwargs['device']='cpu';return fn(*args,**kwargs)
            return call
        fake=S(empty_like=torch.empty_like,empty=cpu(torch.empty),zeros=cpu(torch.zeros),tensor=cpu(torch.tensor),bfloat16=torch.bfloat16,int32=torch.int32,int64=torch.int64)
        scope=dict(torch=fake,H=4,K=2);exec(compile(ast.Module(body=[bank],type_ignores=[]),str(path),'exec'),scope)
        peers={i:dict(local=100+i,output=200+i,counter=torch.zeros(8,dtype=torch.int32)) for i in range(4)}
        b=scope['Bank'](S(peers=peers,placement=S(mode='expert',owners=4),route_plan_min_rows=0),0,3)
        self.assertEqual(b.config[1:7].tolist(),[200,201,202,203,0,3])
        self.assertEqual(b.config[7].item(),peers[0]['counter'].data_ptr())

    def test_publish_all_before_one_collect_and_retire_all(self):
        # Execute the real Python call body on CPU tensors and a launch recorder.
        import torch
        path=Path(__file__).resolve().parents[1]/'src/betterscale/patches/expert_service/persistent_remote.py'
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PersistentRemote')
        call=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
        scope=dict(torch=torch,H=4,K=2);exec(compile(ast.Module(body=[call],type_ignores=[]),str(path),'exec'),scope)
        for count in (1,2,4):
            trace=[];banks={}
            for owner in range(count):banks[owner]=S(config=torch.zeros(16,dtype=torch.int64),x=torch.empty(3,4),id_storage=torch.empty(6,dtype=torch.int64),probs=torch.empty(6),output=torch.zeros(3,4))
            def launch(fn,config,*args):trace.append((fn,next(o for o,b in banks.items() if b.config is config)))
            obj=S(placement=S(targets=lambda _:tuple(range(count))),rows=3,python_submissions=0,route_plan_min_rows=0,
                  bank=lambda owner,n:banks[owner],kernels=S(call=launch),pack='pack',publish='publish',collect='collect',retire='retire',promote='promote',shared_callback=lambda *args:trace.append(('shared',None)))
            scope['__call__'](obj,40,torch.ones(5,4),torch.zeros(5,2,dtype=torch.int64),torch.ones(5,2))
            expected=[]
            for frame in range(2):
                expected += [(fn,o) for o in range(count) for fn in ('pack','publish')]
                if frame==0:expected += [('shared',None)]
                expected += [('collect',0)]+[('retire',o) for o in range(count)]
            self.assertEqual(trace,expected)

if __name__=='__main__':unittest.main()
