import unittest
from schedule import NPlusTwo


def rows(wave, *, done=False, count=1, ranks=4):
    return [[wave.sequence, wave.generation, wave.length, 42 if count else -1,
             count, int(done), 0] for _ in range(ranks)]


class ScheduleTests(unittest.TestCase):
    def make(self, **kwargs):
        return NPlusTwo(**dict(width=32, max_tokens=7, grant_end=128, requests=2, ranks=4, **kwargs))

    def test_nplus2_and_old_generation_drain(self):
        s = self.make()
        first, second = s.next_wave(), s.next_wave()
        self.assertIsNone(s.next_wave())
        self.assertEqual((first.kind, second.kind), ('prefill', 'decode'))
        s.receive(rows(first))
        self.assertEqual(s.next_wave().sequence, 2)
        for seq in range(1, 7):
            wave = s.pending[0]
            self.assertEqual(wave.sequence, seq)
            s.receive(rows(wave, done=seq == 6))
            s.next_wave()
        self.assertEqual([(x.sequence, x.generation) for x in s.pending], [(7, 1), (8, 2)])
        self.assertEqual(s.pending[1].kind, 'prefill')
        s.receive(rows(s.pending[0], done=True, count=0))
        self.assertEqual(s.next_wave().generation, 2)

    def test_partial_quorum_stale_receipt_and_failed_rank_do_not_retire(self):
        for case in ('missing', 'generation', 'sequence', 'error', 'asymmetric'):
            s = self.make()
            wave = s.next_wave()
            batch = rows(wave)
            if case == 'missing':
                batch.pop()
            elif case == 'generation':
                batch[2][1] += 1
            elif case == 'sequence':
                batch[2][0] += 1
            elif case == 'error':
                batch[2][6] = 1
            else:
                batch[2][5] = 1
            with self.assertRaises(ValueError):
                s.receive(batch)
            self.assertEqual(len(s.pending), 1)
            self.assertEqual(s.terminal, set())

    def test_terminal_prefill_still_drains_already_issued_decode(self):
        s = NPlusTwo(width=32, max_tokens=1, grant_end=32, requests=1, ranks=4)
        a, b = s.next_wave(), s.next_wave()
        s.receive(rows(a, done=True))
        self.assertIsNone(s.next_wave())
        s.receive(rows(b, done=True, count=0))
        self.assertFalse(s.pending)
        self.assertIsNone(s.next_wave())

    def test_odd_turnover_exercises_both_prefill_banks(self):
        s = NPlusTwo(width=32, max_tokens=6, grant_end=128, requests=2, ranks=4)
        prefills = []
        while True:
            while (wave := s.next_wave()) is not None:
                if wave.kind == 'prefill':
                    prefills.append(wave.bank)
            if not s.pending:
                break
            wave = s.pending[0]
            ordinal = wave.sequence % 7
            s.receive(rows(wave, done=ordinal >= 5, count=int(ordinal <= 5)))
        self.assertEqual(prefills, [0, 1])

    def test_grant_is_admission_not_optimistic_length_prediction(self):
        with self.assertRaises(ValueError):
            NPlusTwo(width=32, max_tokens=7, grant_end=37, requests=2, ranks=4)


if __name__ == '__main__':
    unittest.main()
