import pytest
from decode_graph_policy import capacity,KEYS


def test_restored_tail_and_native_verify_share_small_graph_bank():
    assert capacity(1,1,[1],[99],[100])==3
    assert capacity(15,5,[3]*5,[100]*5,[100]*5)==24
    assert capacity(20,8,[1,1,3,3,3,3,3,3],[99,99,100,100,100,100,100,100],[100]*8)==24
    assert capacity(48,16,[3]*16,[100]*16,[100]*16)==48
    assert not set(KEYS)&{16,32,64,128,256,512,1024,1536,2048,4096}


@pytest.mark.parametrize("args",[
    (4,1,[4],[99],[100]), (1,1,[1],[98],[100]),
    (1,1,[1],[0],[100]), (49,17,[3]*17,[100]*17,[100]*17),
    (2,1,[1],[100],[100]), (0,0,[],[],[]),
])
def test_bulk_or_unrestored_state_never_falls_back_to_mixed(args):
    with pytest.raises(ValueError):capacity(*args)
