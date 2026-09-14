import dataclasses
import unittest
from types import SimpleNamespace as NS
from early_protocol import Proposal, Budget, Consumer, agree, propose


def schedule(ids=("a", "b")):
    return NS(
        num_scheduled_tokens={i: 6 for i in ids},
        total_num_scheduled_tokens=6 * len(ids),
        scheduled_new_reqs=[],
        finished_req_ids=set(),
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={i: [0] * 5 for i in ids},
        scheduled_cached_reqs=NS(
            req_ids=list(ids),
            resumed_req_ids=set(),
            num_computed_tokens=[128] * len(ids),
        ),
    )


class ProtocolTests(unittest.TestCase):
    caps = {6: (6, 2), 12: (12, 2)}

    def test_stable_budget_uses_schedule_not_acceptance(self):
        s = schedule()
        self.assertEqual(propose(7, s, ("b", "a"), self.caps), Proposal(7, 12, 2, True))

    def test_changes_decline(self):
        for field, value in [
            ("finished_req_ids", {"c"}),
            ("scheduled_new_reqs", [object()]),
            ("scheduled_encoder_inputs", {"a": [0]}),
            ("has_structured_output_requests", True),
            ("preempted_req_ids", {"a"}),
        ]:
            with self.subTest(field=field):
                s = schedule()
                setattr(s, field, value)
                self.assertFalse(propose(0, s, ("a", "b"), self.caps).eligible)
        s = schedule()
        s.scheduled_cached_reqs.resumed_req_ids = {"a"}
        self.assertFalse(propose(0, s, ("a", "b"), self.caps).eligible)

    def test_first_or_changed_set_is_native(self):
        self.assertFalse(propose(0, schedule(), (), self.caps).eligible)
        self.assertFalse(propose(0, schedule(), ("a", "c"), self.caps).eligible)

    def test_partial_draft_and_prefill_are_native(self):
        s = schedule()
        s.scheduled_spec_decode_tokens["a"] = [0] * 4
        self.assertFalse(propose(0, s, ("a", "b"), self.caps).eligible)
        s = schedule()
        s.scheduled_cached_reqs.num_computed_tokens[0] = 0
        self.assertFalse(propose(0, s, ("a", "b"), self.caps).eligible)
        s = schedule()
        s.num_scheduled_tokens["a"] = 132
        s.total_num_scheduled_tokens = 138
        self.assertFalse(propose(0, s, ("a", "b"), self.caps).eligible)

    def test_missing_bucket_does_not_capture(self):
        self.assertFalse(propose(0, schedule(), ("a", "b"), {6: (6, 2)}).eligible)

    def test_dummy_forces_group_fallback(self):
        b = agree([Proposal(0, 12, 2, True), propose(0, None, (), self.caps)])
        self.assertFalse(b.admitted)
        for rank in (0, 1):
            c = Consumer(rank)
            c.begin(b)
            self.assertIsNone(c.resolve(12, 2, False, True))
            c.end()

    def test_heterogeneous_seats_pad_to_max(self):
        b = agree([Proposal(0, 6, 2, True), Proposal(0, 12, 2, True)])
        for rank, n in enumerate((6, 12)):
            c = Consumer(rank)
            c.begin(b)
            self.assertEqual(c.resolve(n, 2, False, True), (12, (12, 12), 2))
            c.end()

    def test_wave_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            agree([Proposal(0, 6, 2, True), Proposal(1, 6, 2, True)])
        c = Consumer(0)
        with self.assertRaises(ValueError):
            c.begin(Budget(1, (6,), 2, True))

    def test_consumer_cannot_silently_fallback(self):
        for args in [(12, 2, False, True), (6, 0, False, True), (6, 2, False, False)]:
            c = Consumer(0)
            c.begin(Budget(0, (6,), 2, True))
            with self.assertRaises(ValueError):
                c.resolve(*args)

    def test_draft_does_not_consume_target_plan(self):
        c = Consumer(0)
        c.begin(Budget(0, (6,), 2, True))
        self.assertIsNone(c.resolve(6, 0, True, True))
        self.assertEqual(c.resolve(6, 2, False, True), (6, (6,), 2))
        c.end()

    def test_duplicate_or_missing_consumption_fails(self):
        c = Consumer(0)
        c.begin(Budget(0, (6,), 2, True))
        with self.assertRaises(ValueError):
            c.end()
        c.resolve(6, 2, False, True)
        with self.assertRaises(ValueError):
            c.resolve(6, 2, False, True)
        c.end()
        with self.assertRaises(ValueError):
            c.begin(Budget(0, (6,), 2, True))

    def test_finished_stable_resume_sequence(self):
        consumer = Consumer(0)
        for seq, admitted in enumerate([True, False, False, True]):
            b = Budget(seq, (6,), 2 if admitted else 0, admitted)
            consumer.begin(b)
            out = consumer.resolve(6, 2, False, True)
            self.assertEqual(out is not None, admitted)
            consumer.end()
        self.assertEqual(consumer.sequence, 4)

    def test_immutable_decision(self):
        b = Budget(0, (6,), 2, True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            b.sequence = 1


class PlainDecodeTests(unittest.TestCase):
    def plain(self, outputs=(5, 5)):
        s = schedule()
        s.num_scheduled_tokens = {"a": 1, "b": 1}
        s.total_num_scheduled_tokens = 2
        s.scheduled_spec_decode_tokens = {}
        s.scheduled_cached_reqs.num_output_tokens = list(outputs)
        return s

    def test_plain_decode_uses_its_own_captured_buckets(self):
        p = propose(
            0, self.plain(), ("a", "b"), {2: (2, 2)}, query_tokens=1, max_requests=4
        )
        self.assertTrue(p.eligible)
        b = agree([p, Proposal(0, 4, 2, True)], allowed_tokens=(1, 2, 4))
        c = Consumer(0)
        c.begin(b)
        self.assertEqual(c.resolve(2, 2, False, True), (4, (4, 4), 2))
        c.end()

    def test_one_token_prefill_tail_stays_native(self):
        for outputs in ((0, 5), ()):
            p = propose(
                0,
                self.plain(outputs),
                ("a", "b"),
                {2: (2, 2)},
                query_tokens=1,
                max_requests=4,
            )
            self.assertFalse(p.eligible)

    def test_plain_decode_rejects_speculative_tokens(self):
        s = self.plain()
        s.scheduled_spec_decode_tokens = {"a": [17]}
        self.assertFalse(
            propose(0, s, ("a", "b"), {2: (2, 2)}, query_tokens=1).eligible
        )

    def test_unknown_group_bucket_still_rejected(self):
        with self.assertRaises(ValueError):
            agree(
                [Proposal(0, 2, 2, True), Proposal(0, 5, 2, True)],
                allowed_tokens=(1, 2, 4),
            )


if __name__ == "__main__":
    unittest.main()
