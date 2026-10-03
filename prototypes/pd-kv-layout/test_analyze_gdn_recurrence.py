import math
import torch
from analyze_gdn_recurrence import recurrence


def test_two_token_scalar_recurrence_embedded_in_real_geometry():
    q=torch.zeros(2,8,128);q[:,:,0]=1
    v=torch.ones(2,16,128);v[1]*=2
    values=dict(q=q.reshape(2,-1),k=q.reshape(2,-1),v=v.reshape(2,-1),
                g=torch.full((2,16),math.log(.5)),beta=torch.full((2,16),.5))
    result=recurrence(values,torch.float64)
    # First state=.5. Second state=.5*.5+(2-.5*.5)*.5=1.125.
    expected=torch.tensor([.5,1.125],dtype=torch.float64)[:,None].expand(-1,2048)/math.sqrt(128)
    torch.testing.assert_close(result,expected,rtol=1e-7,atol=1e-9)


def test_zero_beta_preserves_zero_initial_state():
    values=dict(q=torch.ones(3,1024),k=torch.ones(3,1024),v=torch.ones(3,2048),
                g=torch.zeros(3,16),beta=torch.zeros(3,16))
    assert torch.count_nonzero(recurrence(values,torch.float64))==0

