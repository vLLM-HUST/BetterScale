"""One candidate load: warmed1024/1536 timing and six matched diagnostic forwards."""
import json
import os
from pathlib import Path
import subprocess
import sys

from step_compare import reclaimed


def main():
    root = Path(os.environ['CAPSULE'])
    out = root / 'candidate'
    out.mkdir()
    receipt = dict(status='RUNNING', source_commit=os.environ['SWE_SOURCE_COMMIT'],
                   scope='Same qualified candidate; no native reload.1024 prompt vs1536 first chunk of2048. Six profile forwards in ABBA order, separate from unprofiled event timing.')
    try:
        reclaimed(root)
        env = os.environ.copy()
        env.update(ASCEND_RT_VISIBLE_DEVICES='6,7', COMPARE_NO_MTP='candidate', ELASTIC_CANDIDATE='1',
                   TASK_QUEUE_ENABLE='0', VLLM_CACHE_ROOT=str(root / 'cache-candidate'),
                   SERVING_PROFILE=str(out / 'profiles'), MASTER_PORT='32382',
                   HCCL_NPU_SOCKET_PORT_RANGE='30000-30063', VLLM_SERVER_DEV_MODE='1',
                   STEP_PREFILL_DELTA='1', LD_PRELOAD=env['BETTERSCALE_FIA_LIBRARY'])
        with (out / 'run.log').open('w') as log:
            subprocess.run([sys.executable, str(root / 'source/step_probe.py'), str(out)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        result = json.loads((out / 'receipt.json').read_text())
        assert result['status'] == 'PASS', result.get('error')
        reclaimed(root)
        receipt['status'] = 'PASS'
    except BaseException as exc:
        receipt.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (root / 'comparison.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
