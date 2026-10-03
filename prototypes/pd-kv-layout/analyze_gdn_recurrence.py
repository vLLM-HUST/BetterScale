"""Independent CPU gated-delta recurrence on captured first-layer activations.

Equation and grouped-query mapping follow the independent gdn_resume_probe
oracle. FP64 and FP32 are both evaluated; no candidate kernel is executed.
This isolates recurrence given preprocessed inputs, not convolution correctness
or a whole-model accuracy qualification.
"""
import argparse
import json
from pathlib import Path
import torch


def recurrence(values, dtype, initial=None, return_state=False):
    q=values['q'].reshape(-1,8,128).to(dtype).repeat_interleave(2,dim=1)*128**-.5
    k=values['k'].reshape(-1,8,128).to(dtype).repeat_interleave(2,dim=1)
    v=values['v'].reshape(-1,16,128).to(dtype)
    g=values['g'].to(dtype);beta=values['beta'].to(dtype)
    state=torch.zeros((16,128,128),dtype=dtype) if initial is None else initial.reshape(16,128,128).to(dtype).clone()
    outputs=[]
    for token in range(len(q)):
        state=state*g[token].exp()[:,None,None]
        delta=(v[token]-(state*k[token,:, :,None]).sum(1))*beta[token,:,None]
        state=state+k[token,:,:,None]*delta[:,None,:]
        outputs.append((state*q[token,:,:,None]).sum(1).reshape(-1))
    result=torch.stack(outputs)
    return (result,state) if return_state else result


def error(actual,reference):
    x=actual.double();y=reference.double();delta=x-y
    return dict(max_abs=float(delta.abs().max()),mean_abs=float(delta.abs().mean()),
                rmse=float(delta.square().mean().sqrt()),
                relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(y)),
                reference_max_abs=float(y.abs().max()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4);rows=[]
    for rank in (2,3):
        paths=sorted(args.root.glob(f'rank{rank}-prefill*.pt'),
                     key=lambda p:int(p.stem.split('prefill')[1]))
        states=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
        for index,prefix in enumerate(states):
            if prefix['actual_tokens']!=280:continue
            warm=states[index+1]
            cold=next(d for d in states[index+2:] if d['actual_tokens']==281)
            assert warm['actual_tokens']==1
            keys=('q','k','v','g','beta')
            combined={k:torch.cat([prefix['gdn'][k],warm['gdn'][k]]) for k in keys}
            full={k:cold['gdn'][k] for k in keys}
            warm_reference=recurrence(combined,torch.float64)
            cold_reference=recurrence(full,torch.float64)
            cold32=recurrence(full,torch.float32)
            row=dict(rank=rank,prefix_index=prefix['index'],
                     preprocessed_equal={k:torch.equal(combined[k],full[k]) for k in keys},
                     preprocessed_error={k:error(combined[k],full[k]) for k in keys},
                     reference_warm_cold=error(warm_reference[-1:],cold_reference[-1:]),
                     fp32_vs_fp64=error(cold32,cold_reference),
                     cold_all=error(cold['gdn']['gdn_core'],cold_reference),
                     cold_last=error(cold['gdn']['gdn_core'][-1:],cold_reference[-1:]),
                     warm_last=error(warm['gdn']['gdn_core'],warm_reference[-1:]),
                     warm_cold_observed=error(warm['gdn']['gdn_core'],cold['gdn']['gdn_core'][-1:]))
            row["cold_last_vs_rounded_reference"]=error(cold['gdn']['gdn_core'][-1:],cold_reference[-1:].bfloat16())
            row["warm_last_vs_rounded_reference"]=error(warm['gdn']['gdn_core'],warm_reference[-1:].bfloat16())
            if 'state_before' in warm['gdn']:
                assert prefix['gdn']['state_used'].item()==0
                assert warm['gdn']['state_used'].item()==1
                assert cold['gdn']['state_used'].item()==0
                _,prefix_state=recurrence(prefix['gdn'],torch.float64,return_state=True)
                resumed_output,resumed_state=recurrence(warm['gdn'],torch.float64,
                    initial=warm['gdn']['state_before'],return_state=True)
                row['local_state_carry_exact']=torch.equal(prefix['gdn']['state_after'],warm['gdn']['state_before'])
                row['prefix_state_vs_fp64']=error(prefix['gdn']['state_after'],prefix_state.reshape(1,-1))
                row['warm_output_given_actual_state']=error(warm['gdn']['gdn_core'],resumed_output)
                row['warm_state_given_actual_state']=error(warm['gdn']['state_after'],resumed_state.reshape(1,-1))
                row['warm_cold_final_state']=error(warm['gdn']['state_after'],cold['gdn']['state_after'])
            rows.append(row)
    assert rows
    (args.root/'recurrence-oracle.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

