"""Two pinned host source slots, separate from single-copy device feedback.

Keep native H2D/device preparation ordered on its existing stream. Bank reuse
waits for that bank's last input-preparation event, not the adjacent wave's.
This does NOT make arbitrary receipt consumers or cross-stream copies safe.
"""
from contextlib import contextmanager
import torch


class HostField:
    def __init__(self, owner, name, numpy_name=None):
        self.owner, self.name, self.numpy_name = owner, name, numpy_name
        initial = getattr(owner, name)
        assert isinstance(initial, torch.Tensor) and initial.device.type == 'cpu'
        assert initial.is_contiguous() and initial.storage_offset() == 0
        other = torch.empty_like(initial, pin_memory=initial.is_pinned())
        other.copy_(initial)
        self.slots = [initial, other]

    def select(self, index):
        current = getattr(self.owner, self.name)
        target = self.slots[index]
        if current is not target:
            target.copy_(current)
        setattr(self.owner, self.name, target)
        if self.numpy_name is not None:
            setattr(self.owner, self.numpy_name, target.numpy())


class HostSourceSlots:
    def __init__(self, runner, fields, event_factory):
        self.runner, self.fields = runner, fields
        self.events = [event_factory(), event_factory()]
        self.used = [False, False]
        self.sequence = 0
        self.failed = False
        self.busy = False

    @contextmanager
    def scope(self):
        assert not self.failed and not self.busy, 'Invalid host source-slot lifecycle'
        bounds = getattr(self.runner, '_cross_step_bounds', None)
        if bounds is not None:
            # Dummy waves bypass _prepare_inputs. Never inherit a real wave
            # admission or rebuild their deliberately invalidated slot map.
            bounds.admitted = bounds.skipped = False
        bank = self.sequence % 2
        if self.used[bank]:
            self.events[bank].synchronize()  # actual DMA source reuse, not a device-only wait
        for field in self.fields:
            field.select(bank)
        self.runner.prepare_inputs_event = self.events[bank]
        self.busy = True
        try:
            yield
        except BaseException:
            self.failed = True
            raise
        finally:
            self.events[bank].record()
            self.used[bank] = True
            self.sequence += 1
            self.busy = False

    def receipt(self):
        return dict(stage='two-host-source-slots-native-device-stream', waves=self.sequence,
                    failed=self.failed, fields=[f.name for f in self.fields],
                    extra_pinned_bytes=sum(f.slots[1].numel()*f.slots[1].element_size() for f in self.fields),
                    independent_h2d_stream=False)


def install(worker):
    r = worker.model_runner
    assert r.use_compress and r.use_async_spec_decode and not r.use_dcp
    assert not r.supports_mm_inputs and not r.enable_prompt_embeds and not r._has_gdn
    assert not r.uses_mrope and r.uses_xdrope_dim == 0
    assert r.vllm_config.parallel_config.pipeline_parallel_size == 1
    # The native accepted-count event/CPU destination are feedback, not banked.
    # Native preparation continues to wait on that event when required.
    assert '_host_source_slots' not in r.__dict__
    torch.npu.synchronize()  # explicit installation transition only
    fields=[]
    # Every mutable CpuGpuBuffer read by the admitted text DSV4 preparation.
    # GPU destinations are intentionally unchanged in this stage; num_computed_tokens,
    # sampled counts/IDs, draft tokens, weights and KV remain persistent State.
    for name in ('input_ids','query_start_loc','prev_num_draft_tokens','prev_positions',
                 'req_indices','query_pos','num_scheduled_tokens','discard_request_indices',
                 'discard_request_mask','num_decode_draft_tokens','num_accepted_tokens',
                 'num_draft_tokens','group_len','group_key_idx','group_key_cache_idx'):
        carrier=getattr(r,name)
        fields.append(HostField(carrier,'cpu','np' if hasattr(carrier,'np') else None))
    for table in r.input_batch.block_table.block_tables:
        for name in ('block_table','slot_mapping'):
            carrier=getattr(table,name)
            fields.append(HostField(carrier,'cpu','np' if hasattr(carrier,'np') else None))
    for name, numpy_name in (('_positions_cpu_buf','_positions_np_buf'),
                              ('_dsa_positions_cpu_buf','_dsa_positions_np_buf'),
                              ('optimistic_seq_lens_cpu',None)):
        fields.append(HostField(r,name,numpy_name))
    for name, numpy_name in (('num_computed_tokens_cpu_tensor','num_computed_tokens_cpu'),
                              ('num_prompt_tokens_cpu_tensor','num_prompt_tokens')):
        fields.append(HostField(r.input_batch,name,numpy_name))
    state=r._host_source_slots=HostSourceSlots(r,fields,lambda:torch.npu.Event())
    state.original_scope=r.synchronize_input_prep
    r.synchronize_input_prep=state.scope
    return state.receipt()
