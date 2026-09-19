import unittest
from window import WaveWindow


class WindowContracts(unittest.TestCase):
    def test_two_authorizations_do_not_require_acceptance(self):
        ledger = WaveWindow(position=100, granted_end=117)
        self.assertEqual(ledger.authorize().sequence, 0)
        self.assertEqual(ledger.authorize().sequence, 1)
        with self.assertRaisesRegex(RuntimeError, 'full'):
            ledger.authorize()

    def test_insufficient_grant_does_not_publish_a_wave(self):
        ledger = WaveWindow(position=100, granted_end=110)
        with self.assertRaisesRegex(RuntimeError, 'horizon'):
            ledger.authorize()
        self.assertEqual(ledger.next_sequence, 0)
        self.assertFalse(ledger.pending)

    def test_retirement_can_relieve_grant_backpressure(self):
        ledger = WaveWindow(position=128, granted_end=145)
        ledger.authorize()
        ledger.authorize()
        ledger.retire(sequence=0, generation=1, position=130, count=2, done=False)
        self.assertFalse(ledger.can_authorize())
        ledger.retire(sequence=1, generation=1, position=132, count=2, done=False)
        self.assertTrue(ledger.can_authorize())
        self.assertEqual(ledger.authorize().sequence, 2)

    def test_terminal_old_wave_drains_without_new_authorization(self):
        ledger = WaveWindow(position=100, granted_end=200)
        ledger.authorize()
        ledger.authorize()
        ledger.retire(sequence=0, generation=1, position=101, count=1, done=True)
        with self.assertRaisesRegex(RuntimeError, 'terminal'):
            ledger.authorize()
        ledger.retire(sequence=1, generation=1, position=101, count=0, done=True)
        self.assertFalse(ledger.pending)

    def test_wrong_generation_is_fail_atomic_and_poisoned(self):
        ledger = WaveWindow(position=100, granted_end=200)
        ledger.authorize()
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            ledger.retire(sequence=0, generation=2, position=101, count=1, done=False)
        self.assertEqual(ledger.position, 100)
        self.assertEqual(len(ledger.pending), 1)
        with self.assertRaises(RuntimeError):
            ledger.authorize()


if __name__ == '__main__':
    unittest.main()
