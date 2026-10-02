"""Canonical live-prefix/zero-padding descriptor without importing NPU modules."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest

source=Path(__file__).resolve().parents[1]/'src/betterscale/models/qwen35/draft_fia.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compact_padding')
ns={'copy':copy, 'EXECUTION':16}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
compact=ns['compact_padding']


@pytest.mark.parametrize('live,capacity',[(1,16),(1,4096),(8,48),(16,4096),(3,3)])
def test_compacts_all_trailing_padding_without_changing_live_queries(live,capacity):
    lengths=[4000+i for i in range(live)]+[0]*(capacity-live)
    ends=list(range(1,capacity+1))
    m=SimpleNamespace(seq_lens_list=lengths,actual_seq_lengths_q=ends,
                      seq_lens=lengths.copy(),_mtp_device_seq_lens=lengths.copy())
    result=compact(m)
    assert result.actual_seq_lengths_q==ends[:live]+([capacity] if capacity>live else [])
    assert result.seq_lens_list==lengths[:live]+([0] if capacity>live else [])
    assert result._mtp_device_seq_lens[:live]==lengths[:live]
    assert m.seq_lens_list==lengths and m.actual_seq_lengths_q==ends


@pytest.mark.parametrize('lengths',[[4,0,5],[0,0],[4]*17])
def test_rejects_holes_empty_live_prefix_or_seventeenth_live_request(lengths):
    with pytest.raises(ValueError):
        compact(SimpleNamespace(seq_lens_list=lengths,actual_seq_lengths_q=list(range(1,len(lengths)+1))))
