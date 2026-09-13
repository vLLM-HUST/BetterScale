import unittest
import torch
from decode_metadata import DeviceOnly


class TransferBoundary(unittest.TestCase):
    def test_host_and_device_local_operations_are_allowed(self):
        with DeviceOnly('meta'):
            self.assertEqual((torch.ones(2) + 1).tolist(), [2, 2])
            self.assertEqual((torch.ones(2, device='meta') + 1).shape, (2,))

    def test_ingress_readback_and_scalar_extraction_are_rejected(self):
        cpu = torch.ones(1)
        device = torch.ones(1, device='meta')
        with DeviceOnly('meta'):
            with self.assertRaisesRegex(AssertionError, 'ingress'):
                cpu.to('meta')
            with self.assertRaisesRegex(AssertionError, 'readback'):
                device.to('cpu')
            with self.assertRaisesRegex(AssertionError, 'scalar readback'):
                device.item()
            with self.assertRaisesRegex(AssertionError, 'copy'):
                device.copy_(cpu)
