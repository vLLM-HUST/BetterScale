"""Scope and error tests: no NPU allocation or model construction."""

import unittest
from static_attention import StaticAttention


class Impl:
    def forward_impl(self, value):
        return ("original", value)


class StaticAttentionTests(unittest.TestCase):
    def owner(self):
        owner = object.__new__(StaticAttention)
        owner.impls = [Impl(), Impl()]
        owner.in_scope = False
        owner.forward = lambda key, impl, value: (key, value)
        return owner

    def test_instance_only_and_exception_restoration(self):
        owner = self.owner()
        unrelated = Impl()
        original = Impl.forward_impl
        with self.assertRaisesRegex(ValueError, "fixture"):
            with owner.scope("bank0"):
                self.assertEqual(owner.impls[0].forward_impl(3), ("bank0", 3))
                self.assertEqual(unrelated.forward_impl(3), ("original", 3))
                self.assertIs(Impl.forward_impl, original)
                raise ValueError("fixture")
        self.assertFalse(owner.in_scope)
        self.assertNotIn("forward_impl", owner.impls[0].__dict__)
        self.assertEqual(owner.impls[0].forward_impl(3), ("original", 3))

    def test_nested_scope_rejected_and_prior_override_restored(self):
        owner = self.owner()
        previous = lambda value: value
        owner.impls[0].forward_impl = previous
        with owner.scope("bank0"):
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                with owner.scope("bank1"):
                    pass
        self.assertIs(owner.impls[0].forward_impl, previous)

    def test_native_error_is_not_assert_side_effect(self):
        with self.assertRaisesRegex(RuntimeError, "bind failed: -6"):
            StaticAttention.check(-6, "bind")
