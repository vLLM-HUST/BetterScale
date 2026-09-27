import unittest
from types import SimpleNamespace as NS
from betterscale.models.qwen35.prefill_round_robin import PrefillRoundRobin, prepare


class RoundRobin(unittest.TestCase):
    def test_rotate_start_not_last_served_and_fill_tail(self):
        ring = PrefillRoundRobin()
        rows = [("a", 20, True, False), ("b", 20, True, False), ("c", 20, True, False)]
        self.assertEqual(ring.plan(30, [], rows, admission_slots=0), {"a": 20, "b": 10})
        self.assertEqual(ring.plan(30, [], rows, admission_slots=0), {"b": 20, "c": 10})
        self.assertEqual(ring.plan(30, [], rows, admission_slots=0), {"c": 20, "a": 10})

    def test_waiting_competes_with_running_and_decode_is_reserved(self):
        ring = PrefillRoundRobin()
        rows = [("long", 100000, True, False), ("short", 100, True, True)]
        self.assertEqual(
            ring.plan(4096, [("decode", 3)], rows, admission_slots=1),
            {"decode": 3, "long": 4093},
        )
        self.assertEqual(
            ring.plan(4096, [("decode", 3)], rows, admission_slots=1),
            {"decode": 3, "short": 100, "long": 3993},
        )

    def test_blocked_head_does_not_block_circle_or_lose_membership(self):
        ring = PrefillRoundRobin()
        rows = [("blocked", 100, False, True), ("long", 10000, True, False)]
        self.assertEqual(ring.plan(100, [], rows, admission_slots=1), {"long": 100})
        self.assertIn("blocked", ring.ring)
        rows[0] = ("blocked", 100, True, True)
        ring.plan(100, [], rows, admission_slots=1)
        self.assertEqual(ring.plan(100, [], rows, admission_slots=1), {"blocked": 100})

    def test_turnover_and_full_execution_slots(self):
        ring = PrefillRoundRobin()
        rows = [("a", 100, True, False), ("b", 100, True, True)]
        ring.plan(100, [], rows, admission_slots=0)
        self.assertEqual(ring.plan(100, [], rows, admission_slots=0), {"a": 100})
        rows = [("b", 100, True, True), ("c", 100, True, True)]
        self.assertEqual(ring.plan(100, [], rows, admission_slots=1), {"b": 100})
        self.assertEqual(list(ring.ring), ["c", "b"])

    def test_zero_budget_and_failed_attempt_still_rotate(self):
        ring = PrefillRoundRobin()
        rows = [("a", 100, True, False), ("b", 100, True, False)]
        self.assertEqual(ring.plan(0, [], rows, admission_slots=0), {})
        self.assertEqual(ring.plan(10, [], rows, admission_slots=0), {"b": 10})
        self.assertEqual(ring.limit("a", 10), 0)
        self.assertEqual(ring.limit("b", 100), 10)

    def test_waiting_spec_padding_is_indivisible(self):
        ring = PrefillRoundRobin()
        rows = [("long", 4094, True, False), ("hot", 3, True, True)]
        self.assertEqual(
            ring.plan(4096, [], rows, admission_slots=1, minimum_grants={"hot": 3}),
            {"long": 4094},
        )
        self.assertEqual(
            ring.plan(4096, [], rows, admission_slots=1, minimum_grants={"hot": 3}),
            {"hot": 3, "long": 4093},
        )

    def test_preview_has_no_allocation_and_respects_fence(self):
        def req(key, prompt, computed, status="RUNNING"):
            return NS(
                request_id=key,
                num_computed_tokens=computed,
                num_prompt_tokens=prompt,
                num_tokens_with_spec=prompt,
                num_output_placeholders=0,
                max_tokens=20,
                next_decode_eligible_step=0,
                status=NS(name=status),
                all_token_ids=[],
                cache_salt=None,
                skip_reading_prefix_cache=False,
                num_tokens=prompt,
            )

        s = NS(
            running=[req("long", 10000, 0)],
            waiting=[req("hot", 105, 0, "WAITING")],
            skipped_waiting=[],
            current_step=0,
            max_model_len=262144,
            num_sampled_tokens_per_step=1,
            processed_step_seq=3,
            num_spec_tokens=2,
            max_num_running_reqs=16,
            num_waiting_for_streaming_input=0,
            max_num_scheduled_tokens=4096,
            _prefill_round_robin=PrefillRoundRobin(),
            _waiting_for_resident=lambda r: False,
            residents=NS(
                offer=lambda *a, **k: NS(seat=0, warm=True), seats=[NS(cursor=104)]
            ),
        )
        prepare(s)
        self.assertEqual(prepare(s), {"hot": 3, "long": 4093})
        s._waiting_for_resident = lambda r: True
        self.assertEqual(prepare(s), {"long": 4096})


if __name__ == "__main__":
    unittest.main()
