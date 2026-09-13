import unittest
from dataclasses import dataclass
import torch
from pingpong_buffers import CallPacket


@dataclass
class Metadata:
    rows: torch.Tensor
    bound: int


class PacketTests(unittest.TestCase):
    def test_independent_banks_preserve_views_and_repeated_objects(self):
        base=torch.arange(24,dtype=torch.int32).view(4,6)
        view=base[1:,::2]
        tree={'input':view,'metadata':Metadata(view,24),'base':base}
        a,b=CallPacket(tree),CallPacket(tree)
        self.assertIs(a.tree['input'],a.tree['metadata'].rows)
        self.assertEqual(a.tree['input'].untyped_storage().data_ptr(),a.tree['base'].untyped_storage().data_ptr())
        self.assertNotEqual(a.tree['base'].data_ptr(),b.tree['base'].data_ptr())
        self.assertEqual(a.bytes,base.numel()*base.element_size())
        base.add_(100); b.refresh(tree)
        torch.testing.assert_close(b.tree['input'],view)
        torch.testing.assert_close(a.tree['input'],view-100)

    def test_reject_changed_shape_before_partial_refresh(self):
        src=[torch.ones(3),torch.ones(4)];p=CallPacket(src)
        with self.assertRaises(AssertionError):p.refresh([torch.zeros(3),torch.zeros(5)])
        torch.testing.assert_close(p.tree[0],torch.ones(3))

    def test_reject_alias_split(self):
        t=torch.ones(3);p=CallPacket([t,t])
        with self.assertRaisesRegex(AssertionError,'alias'):p.refresh([t,t.clone()])

    def test_heterogeneous_alias_bytes(self):
        raw=torch.arange(32,dtype=torch.uint8)
        p=CallPacket([raw,raw.view(torch.int32)])
        raw.add_(1);p.refresh([raw,raw.view(torch.int32)])
        self.assertEqual(p.bytes,32)
        torch.testing.assert_close(p.tree[1],raw.view(torch.int32))

from types import SimpleNamespace as NS
from pingpong_sources import HostField, HostSourceSlots


class Event:
    def __init__(self): self.waits=0;self.records=0
    def synchronize(self): self.waits+=1
    def record(self): self.records+=1


class HostSourceTests(unittest.TestCase):
    def test_wait_only_reused_slot_and_preserve_logical_cpu_state(self):
        owner=NS(cpu=torch.zeros(3))
        field=HostField(owner,'cpu','np')
        slots=HostSourceSlots(NS(),[field],Event)
        with slots.scope(): owner.cpu.fill_(7)
        first=owner.cpu
        with slots.scope():
            self.assertIsNot(owner.cpu,first)
            self.assertEqual(owner.np.tolist(),[7,7,7])
            owner.cpu.fill_(9)
        self.assertEqual([e.waits for e in slots.events],[0,0])
        with slots.scope():
            self.assertIs(owner.cpu,first)
            self.assertEqual(owner.np.tolist(),[9,9,9])
        self.assertEqual([e.waits for e in slots.events],[1,0])

    def test_exception_poisoned_generation(self):
        slots=HostSourceSlots(NS(),[],Event)
        with self.assertRaises(ValueError):
            with slots.scope():raise ValueError('failed step')
        with self.assertRaises(AssertionError):
            with slots.scope():pass
        self.assertEqual(slots.events[0].records,1)

@dataclass
class WithConstant:
    rows: torch.Tensor
    full_compress_cos: torch.Tensor


class ConstantTests(unittest.TestCase):
    def test_explicit_immutable_table_shared_and_address_guarded(self):
        t=WithConstant(torch.ones(4),torch.ones(1000))
        p=CallPacket(t,shared_fields=('full_compress_cos',))
        self.assertIs(p.tree.full_compress_cos,t.full_compress_cos)
        self.assertEqual(p.bytes,16)
        p.refresh(t)
        with self.assertRaisesRegex(AssertionError,'Immutable'):
            p.refresh(WithConstant(torch.ones(4),t.full_compress_cos.clone()))

class DummyScopeTests(unittest.TestCase):
    def test_dummy_does_not_inherit_previous_exact_bound_oracle(self):
        bounds=NS(admitted=True,skipped=True)
        state=HostSourceSlots(NS(_cross_step_bounds=bounds),[],Event)
        with state.scope():
            self.assertFalse(bounds.admitted)
            self.assertFalse(bounds.skipped)

class NativeViewTests(unittest.TestCase):
    def test_metadata_leading_view_uses_full_backing_not_partial_copy(self):
        t=torch.arange(8,dtype=torch.int32)
        p=CallPacket(((),{},Metadata(t[:4],4)),native_views=True)
        t.add_(10);p.refresh(((),{},Metadata(t[:3],3)))
        torch.testing.assert_close(p.tree[2].rows,t[:4])
        with self.assertRaisesRegex(AssertionError,'backing'):
            p.refresh(((),{},Metadata(t[:3].clone(),3)))

class SharedDAGTests(unittest.TestCase):
    def test_shared_layer_metadata_visited_once_without_hiding_alias_split(self):
        from unittest.mock import patch
        tensor=torch.arange(12)
        meta=Metadata(tensor,12)
        source={str(i):meta for i in range(60)}
        packet=CallPacket(source)
        with patch.object(packet,'raw',wraps=packet.raw) as raw:
            packet.refresh(source)
            self.assertEqual(raw.call_count,2)
        changed=dict(source);changed['59']=Metadata(tensor.clone(),12)
        with self.assertRaisesRegex(AssertionError,'alias'):
            packet.refresh(changed)
