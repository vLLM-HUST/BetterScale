"""CPU-only tests for the capability revocation used by the hardware oracle."""
import unittest
from boundary import EXECUTION_METHODS, forbid_runner_execution


class BoundaryTests(unittest.TestCase):
    def test_every_existing_entry_is_revoked_and_restored(self):
        class Runner:
            pass
        runner = Runner()
        original = {}
        for name in EXECUTION_METHODS:
            original[name] = lambda: 'native'
            setattr(runner, name, original[name])
        with forbid_runner_execution(runner) as calls:
            for name in EXECUTION_METHODS:
                with self.assertRaisesRegex(RuntimeError, name):
                    getattr(runner, name)()
            self.assertEqual(calls, list(EXECUTION_METHODS))
        for name in EXECUTION_METHODS:
            self.assertIs(getattr(runner, name), original[name])

    def test_sparse_runner_and_exception_restore(self):
        class Runner:
            def execute_model(self):
                return 4
        runner = Runner()
        with self.assertRaises(ValueError):
            with forbid_runner_execution(runner) as calls:
                self.assertFalse(hasattr(runner, '_sample'))
                self.assertEqual(calls, [])
                raise ValueError('abort')
        self.assertEqual(runner.execute_model(), 4)


if __name__ == '__main__':
    unittest.main()
