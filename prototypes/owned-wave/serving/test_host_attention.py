"""CPU contracts for native variant selection, bounded admission and retirement."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
from collections import Counter
from host_attention import HostAttention


class HostAttentionTests(TestCase):
    def owner(self):
        a = object.__new__(HostAttention)
        a.root = SimpleNamespace(max_length=32768, frames={"d4b1": {}, "d4b1fd": {}})
        a.impls = [SimpleNamespace(num_heads=16, num_kv_heads=2)]
        a.lib = SimpleNamespace(
            plan_is_fd=Mock(return_value=1), plan_release=Mock(return_value=0)
        )
        a.native = Mock(return_value=(17, 23))
        a.export = Mock(return_value=object())
        a.wave_plans = 0
        a.dispatches = Counter()
        return a

    def test_native_variant_selects_same_bank_and_releases(self):
        a = self.owner()
        key, payload = a.prepare("d4b1", [4096] * 4)
        self.assertEqual(key, "d4b1fd")
        self.assertIs(payload, a.export.return_value)
        a.lib.plan_release.assert_called_once_with(17)
        self.assertEqual(a.wave_plans, 1)
        self.assertEqual(a.dispatches, {"1:23": 1})

    def test_actual_query_length_reaches_native_planner(self):
        a = self.owner()
        a.root.frames = {"p16b1": {}, "p16b1fd": {}}
        a.prepare("p16b1", [4096], 13)
        a.native.assert_called_once_with("p16b1", [4096], 13)

    def test_missing_variant_fails_closed_and_releases(self):
        a = self.owner()
        del a.root.frames["d4b1fd"]
        with self.assertRaisesRegex(RuntimeError, "uncaptured variant"):
            a.prepare("d4b1", [4096] * 4)
        a.lib.plan_release.assert_called_once_with(17)
        self.assertEqual(a.wave_plans, 0)

    def test_catalog_geometry_boundaries(self):
        a = self.owner()
        self.assertTrue(a.fd_eligible("d", 4))
        self.assertTrue(a.fd_eligible("p", 16))
        self.assertFalse(a.fd_eligible("p", 32))
        a.impls[0].num_kv_heads = 8
        self.assertFalse(a.fd_eligible("d", 4))
        a.root.max_length = 1024
        self.assertFalse(a.fd_eligible("p", 1))
