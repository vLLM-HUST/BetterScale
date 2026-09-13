import unittest
import torch
from torch._subclasses.fake_tensor import is_fake
from producer_shadow import HostProjection


class ProjectionTests(unittest.TestCase):
    def test_host_work_real_device_work_virtual_and_ingress_recorded(self):
        source=torch.arange(4,dtype=torch.float32)
        projection=HostProjection('meta')
        with projection:
            host=source+2
            output=host.to('meta')*3
        torch.testing.assert_close(host,source+2)
        self.assertTrue(is_fake(output))
        self.assertEqual(len(projection.ingress),1)
        self.assertEqual(projection.ingress[0]['bytes'],16)
        self.assertEqual(projection.readbacks,[])
