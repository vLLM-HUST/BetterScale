"""CPU-only group lifecycle tests; no native imports or child accelerator work."""
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import unittest
from types import SimpleNamespace as S
from unittest.mock import patch
from betterscale.patches.expert_service.config import ServiceConfig
from betterscale.patches.expert_service.deployment import Deployment, capture_sizes


class Child:
    def __init__(self):self.returncode=None;self.signals=[]
    def poll(self):return self.returncode
    def send_signal(self,sig):self.signals.append(sig)
    def wait(self,timeout):self.returncode=0;return 0


class Group(unittest.TestCase):
    def make(self,root,**overrides):
        args=S(output=Path(root)/'run',model='/model',build='/build',devices='0,1,2,3',
            sources=2,owners=2,max_seqs=16,max_model_len=262144,mtp_tokens=2,port_base=32510,kv_gib=32,
            lifetime=2400,qualification=None)
        vars(args).update(overrides)
        with patch.dict(os.environ,{'BETTERSCALE_EXPERT_EXTERNAL_WATCHDOG':'1'}),patch.object(ServiceConfig,'check_build'):
            return Deployment(args)

    def test_verification_capture_sizes_are_request_bounded(self):
        self.assertEqual(capture_sizes(16,2),[3,6,12,24,48])
        self.assertEqual(capture_sizes(32,0),[1,2,4,8,16,32])
        self.assertEqual(capture_sizes(5,2),[3,6,12,15])

    def test_synthetic_is_explicit_benchmark_only_and_requires_mtp2(self):
        with tempfile.TemporaryDirectory() as root:
            real=self.make(root)
            self.assertEqual(real.receipt['sampling_policy'],'real')
            benchmark=self.make(root,synthetic_acceptance_length=2.63)
            self.assertEqual(benchmark.receipt['sampling_policy'],'benchmark-only synthetic')
            self.assertEqual(benchmark.receipt['synthetic_acceptance_length'],2.63)
            for changes in (dict(mtp_tokens=0,synthetic_acceptance_length=2.63),
                            dict(synthetic_acceptance_length=float('nan')),
                            dict(synthetic_acceptance_length=3.1)):
                with self.assertRaises(ValueError):self.make(root,**changes)

    def test_drain_is_concurrent_then_clients_stop_and_owner_receipts_match(self):
        with tempfile.TemporaryDirectory() as root:
            group=self.make(root);group.output.mkdir();group.ready=True
            group.receipt['status']='READY';group.children=[Child() for _ in range(4)]
            barrier=threading.Barrier(2,timeout=2)
            def rpc(port,method):
                source=port-32510
                if method=='close_expert_service':barrier.wait()
                return {'peer_generations':{'0':11+source,'1':21+source}}
            group.rpc=rpc
            for owner in (0,1):
                (group.output/f'expert{owner}.json').write_text(json.dumps(dict(status='PASS',
                    completed={str(source):11+10*owner+source for source in (0,1)},
                    server_launch_mode='direct',persistent_kernel_launches=2,persistent_graph_launches=0)))
            group.close();group.close()
            self.assertEqual(group.receipt['status'],'PASS')
            self.assertEqual([c.signals for c in group.children],[[],[],[signal.SIGINT],[signal.SIGINT]])
            self.assertEqual(json.loads((group.output/'deployment.json').read_text())['exit_codes'],[0]*4)

    def test_partial_startup_cancel_stops_all_owned_children(self):
        with tempfile.TemporaryDirectory() as root:
            group=self.make(root);group.output.mkdir();group.children=[Child(),Child()]
            group.stopping=True
            with self.assertRaises(InterruptedError):group.check()
            group.receipt['status']='FAIL';group.close()
            self.assertEqual(group.receipt['status'],'FAIL')
            self.assertEqual([c.signals for c in group.children],[[signal.SIGINT],[signal.SIGINT]])

    def test_push_option_reaches_both_role_commands(self):
        with tempfile.TemporaryDirectory() as root:
            group=self.make(root,return_mode='push')
            commands=[]
            group.launch=lambda role,command,device:commands.append((role,command))
            group.wait=lambda predicate:None
            group.rpc=lambda port,method:{}
            group.start()
            self.assertEqual(group.receipt['return_mode'],'push')
            for role,command in commands:
                if role.startswith('expert'):
                    self.assertEqual(command[command.index('--return-mode')+1],'push')
                else:
                    config=json.loads(command[command.index('--additional-config')+1])
                    self.assertEqual(config['betterscale_experts']['return_mode'],'push')

    def test_short_smoke_model_length_is_explicitly_forwarded(self):
        with tempfile.TemporaryDirectory() as root:
            group=self.make(root,max_model_len=32768)
            commands=[]
            group.launch=lambda role,command,device:commands.append((role,command))
            group.wait=lambda predicate:None
            group.rpc=lambda port,method:{}
            group.start()
            attention=[command for role,command in commands if role.startswith('attention')]
            self.assertEqual(len(attention),2)
            for command in attention:
                self.assertEqual(command[command.index('--max-model-len')+1],'32768')
            self.assertEqual(group.receipt['max_model_len'],32768)

    def test_historical_model_length_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self.make(root).args.max_model_len,262144)
            with self.assertRaises(ValueError):self.make(root,max_model_len=262145)

    def test_one_owner_full_model_is_rejected_before_launch(self):
        with self.assertRaises(ValueError):ServiceConfig('/control','/build',1,7,0,1).validate()
