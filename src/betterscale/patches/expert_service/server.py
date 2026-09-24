"""Expert-role process entry. Run only under an admitted bounded supervisor.

This module does not acquire another lease: deployment holds the shared lease
through all attention/expert processes, admission, watchdog and cleanup.
"""
import argparse
import json
import os
from pathlib import Path
from .config import ServiceConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--build', required=True)
    parser.add_argument('--owners', type=int, required=True)
    parser.add_argument('--sources', type=int, required=True)
    parser.add_argument('--owner', type=int, required=True)
    parser.add_argument('--draft-layers', type=int, choices=(0, 1), default=0)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    config = ServiceConfig(args.control, args.build, args.owners, args.sources, 0, args.draft_layers)
    config.validate()
    config.check_build()
    if not 0 <= args.owner < args.owners:
        parser.error('owner outside configured expert placement')
    if os.environ.get('BETTERSCALE_EXPERT_EXTERNAL_WATCHDOG') != '1':
        parser.error('requires an admitted bounded external watchdog')
    if not Path(args.control).is_dir():
        parser.error('supervisor must provide a private control directory')
    config.bind(args.model, 1)
    # Native imports deliberately follow CPU admission. Physical device scoping
    # is supplied by the supervisor; the role owns exactly one visible NPU.
    import torch
    import torch_npu  # noqa: F401
    if torch.npu.device_count() != 1:
        raise RuntimeError('Expert role requires exactly one visible device')
    torch.npu.set_device(0)
    from .persistent_server import serve
    from .placement import Placement
    receipt = serve(Path(args.control), Placement('layer', args.owners), args.owner, args.sources)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
