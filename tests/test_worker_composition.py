"""One real entry with isolated native doubles; no accelerator initialization."""

import subprocess
import sys
import unittest


BOOT = """
import sys, types
from types import SimpleNamespace as S
from unittest.mock import patch
calls=[]
native=types.ModuleType('vllm_ascend.worker.worker')
class Native:
    def __init__(self, config, *a, **kw):
        self.vllm_config=config
        self.model_runner=S(model='model')
        calls.append('native-init')
    def _init_device(self): calls.append('native-device'); return 'device'
    def determine_available_memory(self): calls.append('native-memory'); return 123
    def load_model(self): calls.append('native-load'); return 'loaded'
    def compile_or_warm_up_model(self): calls.append('native-warm'); return 'warm'
native.NPUWorker=Native
sys.modules[native.__name__]=native
import betterscale.worker as entry
from betterscale.models import select, qwen
assert 'betterscale.models.dsv4' not in sys.modules
assert 'betterscale.patches.auto_kv' not in sys.modules
hf=S(model_type='qwen3_5_text',num_hidden_layers=64,hidden_size=5120,
     head_dim=256,num_attention_heads=24,num_key_value_heads=4,
     linear_num_key_heads=16,linear_num_value_heads=48,
     linear_key_head_dim=128,linear_value_head_dim=128,linear_conv_kernel_dim=4)
c=S(model_config=S(hf_config=S(text_config=hf),hf_text_config=hf,
      dtype='torch.bfloat16',quantization=None,max_model_len=8192,multimodal_config=None),
    parallel_config=S(tensor_parallel_size=2,data_parallel_size=1,pipeline_parallel_size=1,
      enable_expert_parallel=False,decode_context_parallel_size=1,prefill_context_parallel_size=1),
    scheduler_config=S(max_num_seqs=8,max_num_batched_tokens=2048,scheduler_cls=None,async_scheduling=True),
    speculative_config=None,lora_config=None,kv_transfer_config=None,load_config=S(load_format='auto'),
    cache_config=S(enable_prefix_caching=True,mamba_cache_mode='align'),
    compilation_config=S(cudagraph_mode='FULL',cudagraph_capture_sizes=[1,2,4,8,*qwen.PREFILLS]))
"""


class Composition(unittest.TestCase):
    def run_case(self, code):
        subprocess.run([sys.executable, "-c", BOOT + code], check=True)

    def test_owned_selection_and_leaf_lifecycle(self):
        self.run_case(
            """
from betterscale.patches import qwen_gdn, qwen_fia, qwen_layout
from betterscale.patches.qwen_gdn.execution import forward_core
with patch.object(qwen,'check_runtime',side_effect=lambda *a: calls.append('pins')), \
     patch.object(qwen_gdn,'check_library',side_effect=lambda: calls.append('gdn-check')), \
     patch.object(qwen_fia,'check_library',side_effect=lambda: calls.append('fia-check')), \
     patch.object(qwen_gdn,'install',side_effect=lambda: calls.append('gdn-install')), \
     patch.object(qwen_fia,'install',side_effect=lambda: calls.append('fia-install')), \
     patch.object(qwen_layout,'pack_conv_weights',side_effect=lambda model,consumer: calls.append(('pack',model,consumer))):
    w=entry.Worker(c)
    assert w._init_device()=='device'
    assert w.determine_available_memory()==123
    assert w.load_model()=='loaded'
    assert w.compile_or_warm_up_model()=='warm'
assert calls==['pins','pins','gdn-check','fia-check','gdn-install','native-init','fia-install',
 'native-device','native-memory','native-load',('pack','model',forward_core),'native-warm'], calls
assert 'betterscale.models.dsv4' not in sys.modules
assert 'betterscale.patches.auto_kv' not in sys.modules
c.cache_config.enable_prefix_caching=False; c.cache_config.mamba_cache_mode='none'
assert select(c)[1]=='qwen-owned' # not the retired single-prefill path
"""
        )

    def test_native_mtp_has_no_owned_dependencies(self):
        self.run_case(
            """
from betterscale.patches import qwen_layout
c.speculative_config=S(method='mtp',num_speculative_tokens=2,enforce_eager=False)
c.cache_config.enable_prefix_caching=False
c.compilation_config.cudagraph_mode='FULL_AND_PIECEWISE'
c.compilation_config.cudagraph_capture_sizes=[1,2,4,8,16,24]
assert select(c)[1]=='qwen-mtp'
with patch.object(qwen,'check_runtime',side_effect=lambda: calls.append('pins')), \
     patch.object(qwen_layout,'pack_conv_weights',side_effect=lambda model,consumer: calls.append(('pack',consumer))):
    w=entry.Worker(c); w.load_model()
assert calls==['pins','native-init','native-load',('pack',None)], calls
assert 'betterscale.patches.qwen_fia' not in sys.modules
"""
        )

    def test_invalid_config_and_missing_library_do_not_install(self):
        self.run_case(
            """
c.parallel_config.tensor_parallel_size=4
with patch.object(qwen,'check',side_effect=AssertionError('check too early')):
    try: entry.Worker(c)
    except ValueError: pass
    else: raise AssertionError('invalid config accepted')
assert entry._route is None and not calls
c.parallel_config.tensor_parallel_size=2
with patch.object(qwen,'check',side_effect=RuntimeError('missing library')):
    try: entry.Worker(c)
    except RuntimeError: pass
    else: raise AssertionError('missing library accepted')
assert entry._route is None and not calls
"""
        )

    def test_conflicts_and_partial_install_poison(self):
        self.run_case(
            """
with patch.object(qwen,'check'), patch.object(qwen,'before_init',side_effect=RuntimeError('partial install')):
    try: entry.Worker(c)
    except RuntimeError as e: assert str(e)=='partial install'
    else: raise AssertionError('partial install accepted')
assert entry._failed
with patch.object(qwen,'check',side_effect=AssertionError('must not retry')):
    try: entry.Worker(c)
    except RuntimeError as e: assert 'fresh process' in str(e)
    else: raise AssertionError('poison ignored')
entry._failed=False; entry._route='dsv4-tp'
with patch.object(qwen,'check',side_effect=AssertionError('must not mix')):
    try: entry.Worker(c)
    except RuntimeError as e: assert 'conflict' in str(e)
    else: raise AssertionError('mixed process accepted')
"""
        )

    def test_legacy_names_are_aliases_not_more_workers(self):
        self.run_case(
            """
from betterscale.qwen_worker import Worker, MixedWorker
assert Worker is MixedWorker is entry.Worker
"""
        )

    def test_repeat_leaf_install_does_not_rewrap(self):
        self.run_case(
            """
from betterscale.patches import qwen_gdn
Builder=type('Builder',(),{})
sys.modules['vllm_ascend.ops.gdn_attn_builder']=S(AscendGDNAttentionMetadataBuilder=Builder)
from betterscale.patches.qwen_gdn import metadata, execution, publication
with patch.object(qwen_gdn,'check_library',side_effect=lambda: calls.append('library')), \
     patch.object(metadata,'install',side_effect=lambda: calls.append('metadata')), \
     patch.object(execution,'install',side_effect=lambda: calls.append('execution')), \
     patch.object(publication,'install',side_effect=lambda: calls.append('publication')):
    qwen_gdn.install(); qwen_gdn.install()
assert calls==['library','metadata','execution','publication'], calls
"""
        )
