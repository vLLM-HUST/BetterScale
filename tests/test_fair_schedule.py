import unittest
from unittest.mock import patch
from betterscale.models.qwen35.fair_schedule import bind


class Compatibility(unittest.TestCase):
    def test_unknown_native_source_fails_closed(self):
        def native(self):
            return None

        with self.assertRaisesRegex(ValueError, "Unqualified native"):
            bind(native)

    def test_failure_does_not_patch_native_method(self):
        def native(self):
            return None

        original = native.__code__
        with patch(
            "betterscale.models.qwen35.fair_schedule.inspect.getsource",
            return_value='def schedule(self):\n    return "unknown"\n',
        ):
            with self.assertRaises(ValueError):
                bind(native)
        self.assertIs(native.__code__, original)
        self.assertIsNone(native(None))

    def test_known_balance_wrapper_needs_explicit_disabled_state(self):
        def native(self):
            return None

        from betterscale.models.qwen35.fair_schedule import BALANCE_SCHEDULE_SHA256

        with patch("betterscale.models.qwen35.fair_schedule.hashlib.sha256") as digest:
            digest.return_value.hexdigest.return_value = BALANCE_SCHEDULE_SHA256
            for enabled in (None, True):
                with self.assertRaisesRegex(ValueError, "balance scheduling disabled"):
                    bind(native, balance_enabled=enabled)


if __name__ == "__main__":
    unittest.main()
