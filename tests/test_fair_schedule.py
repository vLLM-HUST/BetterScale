import unittest
import ast
import inspect
import textwrap
from unittest.mock import patch
from betterscale.models.qwen35.fair_schedule import UNIFIED_SCHEDULE_SHA256, bind


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

    def test_unified_schedule_has_both_exact_grant_seams(self):
        def schedule(self):
            while self.running:
                request = self.running[0]
                num_new_tokens = 1
                if self.need_mamba_block_aligned_split:
                    pass
            if not preempted_reqs:
                while waiting:
                    request = request_queue.peek_request()
                    request_id = request.request_id
                    if self._is_blocked_waiting_status(request.status):
                        pass
                    num_new_tokens = 1
                    if self.need_mamba_block_aligned_split:
                        pass

        source = textwrap.dedent(inspect.getsource(schedule)).strip()
        with patch(
            "betterscale.models.qwen35.fair_schedule.inspect.getsource",
            return_value=source,
        ), patch(
            "betterscale.models.qwen35.fair_schedule.hashlib.sha256"
        ) as digest:
            digest.return_value.hexdigest.return_value = UNIFIED_SCHEDULE_SHA256
            transformed = bind(schedule)

        transformed_source = ast.unparse(ast.parse(source))
        self.assertNotIn("_prefill_round_robin.limit", transformed_source)
        instructions = list(__import__("dis").get_instructions(transformed))
        self.assertEqual(
            sum(i.argval == "_prefill_round_robin" for i in instructions), 3
        )


if __name__ == "__main__":
    unittest.main()
