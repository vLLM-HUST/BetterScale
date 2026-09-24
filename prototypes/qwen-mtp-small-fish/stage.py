"""Stage explicit minimal donor changes; never edit a seed or installed runtime."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f'Expected {count} exact source occurrences: {before!r}')
    return text.replace(before, after)


def change_method(text, cls, method, transform):
    tree = ast.parse(text)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method)
    lines = text.splitlines(keepends=True)
    before = ''.join(lines[fn.lineno-1:fn.end_lineno])
    after = transform(before)
    return ''.join(lines[:fn.lineno-1]) + after + ''.join(lines[fn.end_lineno:])


def proposer_source(text):
    return change_method(text, 'AscendSpecDecodeBaseProposer', '_run_merged_draft',
        lambda s: replace_once(s,
            'if get_ascend_config().enable_reduce_sample:',
            'if get_ascend_config().enable_reduce_sample or getattr(self, "_betterscale_draft_greedy", False):', 2))


def logits_source(text):
    return change_method(text, 'AscendLogitsProcessor', '_get_logits_normal',
        lambda s: replace_once(s,
            'if not get_ascend_config().enable_reduce_sample:',
            'if not (get_ascend_config().enable_reduce_sample or getattr(self, "_betterscale_draft_greedy", False)):', 2))


def gdn_source(text):
    # Preserve native behavior unless the owned-consumer lifecycle sets its flag.
    # Deepest first: inserted nested lines must not match the next pattern.
    for indent in ('                ', '            '):
        before = f'\n{indent}b = b.contiguous()\n{indent}a = a.contiguous()'
        after = f'\n{indent}if not getattr(self, "_betterscale_strided_gates", False):\n{indent}    b = b.contiguous()\n{indent}    a = a.contiguous()'
        text = replace_once(text, before, after)
    return text


def mixed_source(text):
    text = replace_once(text, '        self.capacity = capacity',
        '        self.capacity = capacity\n'
        '        self.shared_qkv_pack = os.environ.get("BETTERSCALE_GDN_SMALL_COPIES", "0") == "1"')
    return replace_once(text, '        transformed = torch.empty_like(x)',
        '        # Both convolution roles consume the same immutable packed input.\n'
        '        if self.shared_qkv_pack:\n'
        '            x = x.contiguous()\n'
        '        transformed = torch.empty_like(x)')


def stage(seed, runtime, output):
    if output.exists():
        raise FileExistsError(output)
    # Verify the baseline source identity before allowing any new pin values.
    pins_path = seed/'package/betterscale/qwen_pins.json'
    pins = json.loads(pins_path.read_text())
    for item in pins['source_files']:
        if item['path'].startswith('vllm_ascend/'):
            path = runtime/item['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Seed/runtime pin mismatch: {path}')
    if not (seed/'draft_sampling.py').is_file():
        raise ValueError('Control must already include request-bounded draft sampling')
    shutil.copytree(seed, output, ignore=shutil.ignore_patterns('__pycache__', '*.log', 'runtime-source'))
    shutil.copytree(runtime/'vllm_ascend', output/'runtime-source/vllm_ascend',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    transforms = {
        'runtime-source/vllm_ascend/spec_decode/llm_base_proposer.py': proposer_source,
        'runtime-source/vllm_ascend/ops/vocab_parallel_embedding.py': logits_source,
        'runtime-source/vllm_ascend/ops/gdn.py': gdn_source,
        'mixed_core.py': mixed_source,
        'service_adapter.py': lambda s: replace_once(s,
            '        _GDN_PATCH_TARGET._forward_core = forward_core',
            '        _GDN_PATCH_TARGET._forward_core = forward_core\n'
            '        from small_fish_runtime import install as install_small_fish\n'
            '        install_small_fish()'),
    }
    diffs, changes = [], []
    for relative, transform in transforms.items():
        path = output/relative
        old = path.read_text()
        new = transform(old)
        compile(new, str(path), 'exec')
        path.write_text(new)
        diffs.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                     fromfile='seed/'+relative, tofile='candidate/'+relative))
        changes.append({'path':relative,'before':hashlib.sha256(old.encode()).hexdigest(),
                        'after':hashlib.sha256(new.encode()).hexdigest()})
    shutil.copyfile(Path(__file__).with_name('runtime.py'), output/'small_fish_runtime.py')
    # Add explicit identities for both newly touched donor files, including those
    # not covered by the old package manifest. Do not weaken any existing pin.
    manifest = output/'package/betterscale/qwen_pins.json'
    by_path = {r['path']:r for r in pins['source_files']}
    for relative in transforms:
        if not relative.startswith('runtime-source/'):
            continue
        name = relative.removeprefix('runtime-source/')
        by_path.setdefault(name, {'path':name,'distribution':'vllm-ascend'})['sha256'] = hashlib.sha256((output/relative).read_bytes()).hexdigest()
    pins['source_files'] = list(by_path.values())
    manifest.write_text(json.dumps(pins,indent=2)+'\n')
    (output/'small-fish.diff').write_text(''.join(diffs))
    (output/'small-fish-source.json').write_text(json.dumps({
        'status':'UNQUALIFIED','seed':str(seed),'runtime':str(runtime),'changes':changes,
        'flags':['BETTERSCALE_MTP_GREEDY','BETTERSCALE_GDN_SMALL_COPIES'],
        'control':'Both flags0; same request-bounded sampling and runtime pins',
        'nonchanges':['global enable_reduce_sample','target logits/sampler','model/attention capacities',
                      'GDN state/conv ownership','output initialization','native binaries'],
    },indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('seed',type=Path)
    p.add_argument('runtime',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    stage(a.seed.resolve(),a.runtime.resolve(),a.output.resolve())
