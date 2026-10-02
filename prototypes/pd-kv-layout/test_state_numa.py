import pytest
from state_numa import cpus_from_list, resolve_node

def test_visibility_remaps_to_physical_numa():
    mapping=[6,0,7,1,4,2,5,3]
    assert resolve_node(mapping,"4,5",0)==(4,4)
    assert resolve_node(mapping,"4,5",1)==(5,2)
    assert resolve_node(mapping,None,7)==(7,3)
    assert cpus_from_list("0-2,5,7-8")=={0,1,2,5,7,8}

@pytest.mark.parametrize("mapping,visible,index",[
    ([6,0],"0,0",0),([6,0],"2",0),([6,0],"0,1",2),
    ([True,0],None,0),([],None,0),([6,-1],None,0)])
def test_bad_topology_rejected(mapping,visible,index):
    with pytest.raises(ValueError):resolve_node(mapping,visible,index)
