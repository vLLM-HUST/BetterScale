"""Partial-card leaf launchers must reject unsafe placement before any NPU import."""
from pathlib import Path
import subprocess,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parents[1]/'prototypes/expert-service-qualification'
class LeafPlacement(unittest.TestCase):
    def test_invalid_device_sets_never_create_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'must-not-launch'
            for script,extra in [('ep_leaf.py',['--owners','2']),('batch_leaf.py',[])]:
                for devices in ('4,4,6','4,5','4,5,8'):
                    result=subprocess.run([sys.executable,str(ROOT/script),'/unused/model',
                        '--build','/unused/build','--output',str(out),'--devices',devices,*extra],
                        capture_output=True,text=True,timeout=15)
                    self.assertEqual(result.returncode,2,result.stderr)
                    self.assertFalse(out.exists())
                    self.assertNotIn('torch_npu',result.stderr)

    def test_ep_supervisor_starts_all_owners_before_shared_card_clients(self):
        import importlib.util,json
        from unittest.mock import patch
        from types import SimpleNamespace as S
        spec=importlib.util.spec_from_file_location('ep_leaf_fixture',ROOT/'ep_leaf.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';calls=[]
            def launch(cmd,**kwargs):
                device=kwargs['env']['ASCEND_RT_VISIBLE_DEVICES']
                if '--owner' in cmd:
                    owner=int(cmd[cmd.index('--owner')+1]);calls.append(('owner',owner,device))
                    (out/f'control/e{owner}.sock').touch()
                    (out/f'owner{owner}.json').write_text(json.dumps(dict(completed={'0':2,'1':2},completed_counts=[2,2,0,0,0,0,0],waves=3)))
                else:
                    source=int(cmd[cmd.index('--source')+1]);calls.append(('client',source,device))
                    (out/f'client{source}.json').write_text(json.dumps(dict(receipt=dict(peer_generations={'0':2,'1':2}))))
                return S(poll=lambda:0,wait=lambda **kw:0,returncode=0)
            argv=[str(ROOT/'ep_leaf.py'),'/unused/model','--build','/unused/build',
                  '--output',str(out),'--owners','2','--sources','2','--shared-client-device','--devices','4,5,6']
            with patch.object(sys,'argv',argv),patch.object(module.subprocess,'Popen',launch):module.main()
            self.assertEqual(calls,[('owner',0,'5'),('owner',1,'6'),('client',0,'4'),('client',1,'4')])
            self.assertTrue((out/'PASS').exists())
