"""Captured draft handles and workspace must not leak across graph banks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

path=Path(__file__).resolve().parents[1]/'prototypes/qwen38-serving/mtp/draft_banks.py'
spec=importlib.util.spec_from_file_location('mtp_draft_banks',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def params():
    return NS(events={3:[],24:[]},workspaces={3:None,24:None},
              handles={3:[],24:[]},attn_params={3:[],24:[]})


class DraftBankTest(unittest.TestCase):
    def test_banks_are_disjoint_and_persist_across_calls(self):
        acl=NS(GraphParams=NS,_draft_graph_params=params(),_draft_graph_prefill_params=params())
        original=(acl._draft_graph_params,acl._draft_graph_prefill_params)
        owner=NS()
        for bank in (0,1):
            with module.graph_resources(acl,owner,bank):
                for p in (acl._draft_graph_params,acl._draft_graph_prefill_params):
                    self.assertEqual(p.handles[24],[])
                    p.handles[24].extend([(bank,0),(bank,1)])
                    p.workspaces[24]=object()
        for bank in (0,1,0):
            with module.graph_resources(acl,owner,bank):
                self.assertEqual(acl._draft_graph_params.handles[24],[(bank,0),(bank,1)])
        self.assertIs(acl._draft_graph_params,original[0])
        self.assertIs(acl._draft_graph_prefill_params,original[1])
        self.assertEqual(original[0].handles[24],[])
        self.assertIsNot(owner._mtp_graph_banks[('_draft_graph_params',0)].workspaces[24],
                         owner._mtp_graph_banks[('_draft_graph_params',1)].workspaces[24])

    def test_exception_and_nested_call_restore_ownership(self):
        acl=NS(GraphParams=NS,_draft_graph_params=params(),_draft_graph_prefill_params=None)
        original=acl._draft_graph_params;owner=NS()
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with module.graph_resources(acl,owner,0):
                outer=acl._draft_graph_params
                with module.graph_resources(acl,owner,0):
                    self.assertIs(acl._draft_graph_params,outer)
                raise RuntimeError('probe')
        self.assertIs(acl._draft_graph_params,original)
        self.assertIsNone(acl._draft_graph_prefill_params)
