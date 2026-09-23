"""Stage request-bounded sampling without editing a live/frozen source capsule."""
import argparse
from pathlib import Path
import shutil


def install_in_capsule(output):
    module = Path(__file__).with_name('draft_sampling.py')
    target = output / module.name
    assert not target.exists(), target
    path = output / 'draft_banks.py'
    source = path.read_text()
    hook = '    install_live_draft_rows()\n'
    assert source.count(hook) == 1, 'Expected the qualified live-row publication guard'
    source = source.replace(hook, hook +
        '    from draft_sampling import install as install_request_sampling\n'
        '    install_request_sampling()\n')
    compile(source, str(path), 'exec')
    shutil.copyfile(module, target)
    path.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    shutil.copytree(args.seed, args.output, ignore=shutil.ignore_patterns('__pycache__', '*.log'))
    install_in_capsule(args.output)


if __name__ == '__main__':
    main()
