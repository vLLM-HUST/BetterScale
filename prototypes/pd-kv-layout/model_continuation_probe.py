"""Bounded greedy model continuity gate: warm, cold and uninterrupted generation.

Run against an explicitly selected task-owned loopback service. Retain failures;
exact tokens are the contract, not just plausible output or HTTP success.
"""
import argparse,json,time,urllib.request,urllib.error
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--port',type=int,default=18180)
p.add_argument('--server-log',type=Path)
p.add_argument('--wait-seconds',type=int,default=900)
p.add_argument('--tokens',type=int,default=64)
a=p.parse_args();assert 0<a.port<65536 and 1<=a.tokens<=128 and 0<=a.wait_seconds<=900
a.output.mkdir(exist_ok=False,parents=True);url=f'http://127.0.0.1:{a.port}'
deadline=time.monotonic()+a.wait_seconds
while True:
 try:
  with urllib.request.urlopen(url+'/health',timeout=3) as r:
   if r.status==200:break
 except (urllib.error.URLError,TimeoutError):pass
 if a.server_log and 'Traceback' in a.server_log.read_text():raise RuntimeError('startup failure; inspect server log')
 if time.monotonic()>deadline:raise TimeoutError('service readiness deadline')
 time.sleep(10)
print('READY',flush=True)
def call(name,prompt,salt,n):
 data=dict(model='qwen35-moe',prompt=prompt,max_tokens=n,temperature=0,
           return_token_ids=True,ignore_eos=True,cache_salt=salt,logprobs=5)
 start=time.monotonic()
 try:
  with urllib.request.urlopen(urllib.request.Request(url+'/v1/completions',
      data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:res=json.load(r)
 except urllib.error.HTTPError as e:
  (a.output/(name+'-error.txt')).write_bytes(e.read());raise
 (a.output/(name+'.json')).write_text(json.dumps(res,indent=2))
 print(name,res['usage'],round(time.monotonic()-start,3),flush=True);return res
prompt=('Observation: the blue key opens the north gate; the red key opens the south gate.\n'*220)+'\nQuestion: Which key opens the north gate?\nAnswer:'
salt=a.output.name
first=call('prefix',prompt,salt,a.tokens)['choices'][0]
continued=first['prompt_token_ids']+first['token_ids']
w=call('warm',continued,salt,a.tokens);c=call('cold',continued,salt+'-cold',a.tokens)
u=call('uninterrupted',first['prompt_token_ids'],salt+'-uninterrupted',2*a.tokens)['choices'][0]
def difference(x,y):return next((i for i,(v,z) in enumerate(zip(x,y)) if v!=z),None) if len(x)==len(y) else 'length'
wi=w['choices'][0];ci=c['choices'][0];suffix=u['token_ids'][a.tokens:]
row=dict(prefix_equal=first['token_ids']==u['token_ids'][:a.tokens],warm_cold_first_diff=difference(wi['token_ids'],ci['token_ids']),
    warm_uninterrupted_first_diff=difference(wi['token_ids'],suffix),cold_uninterrupted_first_diff=difference(ci['token_ids'],suffix),
    warm_cached=w['usage']['prompt_tokens_details']['cached_tokens'],cold_cached=c['usage']['prompt_tokens_details']['cached_tokens'])
i=row['warm_cold_first_diff']
if isinstance(i,int):
 row['first_difference_top_logprobs']={n:r['logprobs']['top_logprobs'][i] for n,r in [('warm',wi),('cold',ci)]}
row['passed']=row['prefix_equal'] and all(row[k] is None for k in ('warm_cold_first_diff','warm_uninterrupted_first_diff','cold_uninterrupted_first_diff')) and row['warm_cached']>0 and row['cold_cached']==0
(a.output/'receipt.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
assert row['passed'],'continuity gate failed; preserve response/receipt evidence'
