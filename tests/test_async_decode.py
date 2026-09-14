import unittest
from dataclasses import dataclass
import torch
from betterscale.patches.async_decode._packet import CallPacket


def refresh(packet, source):
    copies = {}
    packet._bind(packet.tree, source, copies, seen=set())
    for destination, origin in copies.values():
        destination.copy_(origin)


@dataclass
class Metadata:
    rows: torch.Tensor
    bound: int


class PacketTests(unittest.TestCase):
    def test_independent_banks_preserve_views_and_repeated_objects(self):
        base = torch.arange(24, dtype=torch.int32).view(4, 6)
        view = base[1:, ::2]
        tree = {"input": view, "metadata": Metadata(view, 24), "base": base}
        a, b = CallPacket(tree), CallPacket(tree)
        self.assertIs(a.tree["input"], a.tree["metadata"].rows)
        self.assertEqual(
            a.tree["input"].untyped_storage().data_ptr(),
            a.tree["base"].untyped_storage().data_ptr(),
        )
        self.assertNotEqual(a.tree["base"].data_ptr(), b.tree["base"].data_ptr())
        self.assertEqual(a.bytes, base.numel() * base.element_size())
        base.add_(100)
        refresh(b, tree)
        torch.testing.assert_close(b.tree["input"], view)
        torch.testing.assert_close(a.tree["input"], view - 100)

    def test_reject_changed_shape_before_partial_refresh(self):
        src = [torch.ones(3), torch.ones(4)]
        p = CallPacket(src)
        with self.assertRaises(AssertionError):
            refresh(p, [torch.zeros(3), torch.zeros(5)])
        torch.testing.assert_close(p.tree[0], torch.ones(3))

    def test_reject_alias_split(self):
        t = torch.ones(3)
        p = CallPacket([t, t])
        with self.assertRaisesRegex(AssertionError, "alias"):
            refresh(p, [t, t.clone()])

    def test_heterogeneous_alias_bytes(self):
        raw = torch.arange(32, dtype=torch.uint8)
        p = CallPacket([raw, raw.view(torch.int32)])
        raw.add_(1)
        refresh(p, [raw, raw.view(torch.int32)])
        self.assertEqual(p.bytes, 32)
        torch.testing.assert_close(p.tree[1], raw.view(torch.int32))


from types import SimpleNamespace as NS
from betterscale.patches.async_decode._host import HostField, HostSourceSlots


class Event:
    def __init__(self):
        self.waits = 0
        self.records = 0

    def synchronize(self):
        self.waits += 1

    def record(self):
        self.records += 1


class HostSourceTests(unittest.TestCase):
    def test_wait_only_reused_slot_and_preserve_logical_cpu_state(self):
        owner = NS(cpu=torch.zeros(3))
        field = HostField(owner, "cpu", "np")
        slots = HostSourceSlots(NS(), [field], Event)
        with slots.scope():
            owner.cpu.fill_(7)
        first = owner.cpu
        with slots.scope():
            self.assertIsNot(owner.cpu, first)
            self.assertEqual(owner.np.tolist(), [7, 7, 7])
            owner.cpu.fill_(9)
        self.assertEqual([e.waits for e in slots.events], [0, 0])
        with slots.scope():
            self.assertIs(owner.cpu, first)
            self.assertEqual(owner.np.tolist(), [9, 9, 9])
        self.assertEqual([e.waits for e in slots.events], [1, 0])

    def test_exception_poisoned_generation(self):
        slots = HostSourceSlots(NS(), [], Event)
        with self.assertRaises(ValueError):
            with slots.scope():
                raise ValueError("failed step")
        with self.assertRaises(AssertionError):
            with slots.scope():
                pass
        self.assertEqual(slots.events[0].records, 1)


@dataclass
class WithConstant:
    rows: torch.Tensor
    full_compress_cos: torch.Tensor


class ConstantTests(unittest.TestCase):
    def test_explicit_immutable_table_shared_and_address_guarded(self):
        t = WithConstant(torch.ones(4), torch.ones(1000))
        p = CallPacket(t, shared_fields=("full_compress_cos",))
        self.assertIs(p.tree.full_compress_cos, t.full_compress_cos)
        self.assertEqual(p.bytes, 16)
        refresh(p, t)
        with self.assertRaisesRegex(AssertionError, "Immutable"):
            refresh(p, WithConstant(torch.ones(4), t.full_compress_cos.clone()))


class DummyScopeTests(unittest.TestCase):
    def test_dummy_does_not_inherit_previous_exact_bound_oracle(self):
        bounds = NS(admitted=True, skipped=True)
        state = HostSourceSlots(NS(_cross_step_bounds=bounds), [], Event)
        with state.scope():
            self.assertFalse(bounds.admitted)
            self.assertFalse(bounds.skipped)


class NativeViewTests(unittest.TestCase):
    def test_metadata_leading_view_uses_full_backing_not_partial_copy(self):
        t = torch.arange(8, dtype=torch.int32)
        p = CallPacket(((), {}, Metadata(t[:4], 4)), native_views=True)
        t.add_(10)
        refresh(p, ((), {}, Metadata(t[:3], 3)))
        torch.testing.assert_close(p.tree[2].rows, t[:4])
        with self.assertRaisesRegex(AssertionError, "backing"):
            refresh(p, ((), {}, Metadata(t[:3].clone(), 3)))


class SharedDAGTests(unittest.TestCase):
    def test_shared_layer_metadata_visited_once_without_hiding_alias_split(self):
        from unittest.mock import patch

        tensor = torch.arange(12)
        meta = Metadata(tensor, 12)
        source = {str(i): meta for i in range(60)}
        packet = CallPacket(source)
        with patch.object(packet, "raw", wraps=packet.raw) as raw:
            refresh(packet, source)
            self.assertEqual(raw.call_count, 2)
        changed = dict(source)
        changed["59"] = Metadata(tensor.clone(), 12)
        with self.assertRaisesRegex(AssertionError, "alias"):
            refresh(packet, changed)


import unittest
import torch
from betterscale.patches.async_decode._metadata import DeviceOnly


class TransferBoundary(unittest.TestCase):
    def test_startup_restores_input_banks_even_on_capture_failure(self):
        from unittest.mock import patch
        from betterscale.patches.async_decode._warmup import preserve_inputs

        carrier = NS(cpu=torch.arange(4), gpu=torch.arange(4) + 10)
        field = HostField(carrier, "cpu", "np")
        original_cpu = carrier.cpu
        runner = NS(
            _host_source_slots=NS(fields=[field]),
            _cross_step_bounds=NS(build_args="original"),
            num_computed_tokens=torch.tensor([7]),
            positions=torch.tensor([8]),
            seq_lens=torch.tensor([9]),
            attn_state="original",
        )
        with patch.object(torch, "npu", NS(synchronize=lambda: None), create=True):
            with self.assertRaisesRegex(ValueError, "capture failed"):
                with preserve_inputs(runner):
                    field.select(1)
                    carrier.cpu.zero_()
                    carrier.gpu.zero_()
                    runner.positions.zero_()
                    runner.attn_state = "dummy"
                    runner._cross_step_bounds.build_args = "dummy"
                    raise ValueError("capture failed")
        self.assertIs(carrier.cpu, original_cpu)
        self.assertEqual(carrier.np.tolist(), [0, 1, 2, 3])
        self.assertEqual(carrier.gpu.tolist(), [10, 11, 12, 13])
        self.assertEqual(field.slots[1].tolist(), [0, 1, 2, 3])
        self.assertEqual(runner.positions.tolist(), [8])
        self.assertEqual(runner.attn_state, "original")
        self.assertEqual(runner._cross_step_bounds.build_args, "original")

    def test_missing_metadata_shape_never_captures_online(self):
        from unittest.mock import Mock
        from betterscale.patches.async_decode._metadata import DecodeMetadata

        metadata = DecodeMetadata.__new__(DecodeMetadata)
        metadata.r = NS(
            max_num_reqs=2,
            _cross_step_bounds=NS(admitted=True),
            cache_config=NS(kv_sharing_fast_prefill=False),
            model_config=NS(enable_return_routed_experts=False),
            optimistic_seq_lens_cpu=torch.zeros(2),
        )
        metadata.producer = NS(active=object())
        metadata.entries = {}
        metadata.capture = Mock(side_effect=AssertionError("online capture"))
        metadata.native = Mock(return_value="native")
        request = dict(
            num_reqs=1,
            num_reqs_padded=2,
            num_tokens=6,
            num_tokens_padded=12,
            max_query_len=6,
            use_spec_decode=True,
        )
        self.assertEqual(metadata.build(**request), "native")
        metadata.capture.assert_not_called()
        self.assertFalse(metadata.entries)

    def test_startup_shapes_follow_captured_descriptors(self):
        from betterscale.patches.async_decode._warmup import shapes

        @dataclass(frozen=True)
        class Descriptor:
            num_reqs: int
            num_tokens: int

        runner = NS(
            max_num_reqs=2,
            _cross_step_bounds=NS(max_requests=2),
            model=NS(
                _decode_pair=NS(
                    packets=[
                        {
                            Descriptor(2, 6): object(),
                            Descriptor(2, 12): object(),
                            Descriptor(2, 516): object(),
                        }
                    ]
                )
            ),
        )
        self.assertEqual(shapes(runner), [(2, 2, 12), (1, 2, 12), (1, 2, 6)])

    def test_local_decode_with_global_prefill_padding_stays_native(self):
        from betterscale.patches.async_decode._metadata import DecodeMetadata

        metadata = DecodeMetadata.__new__(DecodeMetadata)
        metadata.r = NS(max_num_reqs=2, _cross_step_bounds=NS(admitted=True))
        metadata.producer = NS(active=object())
        calls = []
        metadata.native = lambda *a, **kw: calls.append((a, kw)) or "native"
        request = dict(
            num_reqs=2, num_reqs_padded=2, num_tokens=12, num_tokens_padded=516
        )
        self.assertEqual(metadata.build(**request), "native")
        self.assertEqual(calls, [((), request)])

    def test_host_and_device_local_operations_are_allowed(self):
        with DeviceOnly("meta"):
            self.assertEqual((torch.ones(2) + 1).tolist(), [2, 2])
            self.assertEqual((torch.ones(2, device="meta") + 1).shape, (2,))

    def test_ingress_readback_and_scalar_extraction_are_rejected(self):
        cpu = torch.ones(1)
        device = torch.ones(1, device="meta")
        with DeviceOnly("meta"):
            with self.assertRaisesRegex(AssertionError, "ingress"):
                cpu.to("meta")
            with self.assertRaisesRegex(AssertionError, "readback"):
                device.to("cpu")
            with self.assertRaisesRegex(AssertionError, "scalar readback"):
                device.item()
            with self.assertRaisesRegex(AssertionError, "copy"):
                device.copy_(cpu)
