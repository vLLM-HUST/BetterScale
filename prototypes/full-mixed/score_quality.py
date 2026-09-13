"""Use the pinned upstream OpenCompass score, not a new retrieval metric."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

p=argparse.ArgumentParser()
p.add_argument('capsule',type=Path)
p.add_argument('--evaluator-repo',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--tokenizer',type=Path,required=True)
a=p.parse_args()
commit='60a28a727d3b7807eb3554928f3530d04c948452'
relative='opencompass/datasets/longbench/evaluators.py'
source=subprocess.check_output(['git','-C',str(a.evaluator_repo),'show',f'{commit}:{relative}'],text=True)
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='LongBenchRetrievalEvaluator')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='score')
namespace=dict(re=re,List=list)
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),f'{commit}:{relative}','exec'),namespace)
evaluate=lambda pred,gold:namespace['score'](SimpleNamespace(language='en'),pred,gold)
assert (a.capsule/'exit.txt').read_text().strip()=='0'
result=json.loads((a.capsule/'engine/quality-result.json').read_text())
assert result['status']=='COMPLETED_UNSCORED'
inputs=json.loads((a.capsule/'engine/quality-inputs.json').read_text())
gold={str(r['request_id']):r['gold'] for r in json.loads(a.reference.read_text())['rows']}
assert len(result['requests'])==len(inputs)==32
assert {str(r['request_id']) for r in result['requests']}=={str(r['request_id']) for r in inputs}<=set(gold)
from tokenizers import Tokenizer
tokenizer=Tokenizer.from_file(str(a.tokenizer))
rows=[]
for r in result['requests']:
    text=tokenizer.decode(r['token_ids'],skip_special_tokens=True)
    ref=gold[str(r['request_id'])]
    rows.append(dict(**r,prediction=text,gold=ref,score=evaluate([text],[ref])['score']))
receipt=dict(scope=result['scope'],evaluator_commit=commit,method='Unmodified score AST from pinned Git object',
             count=len(rows),score=evaluate([r['prediction'] for r in rows],[r['gold'] for r in rows]),
             correct=sum(r['score']==100 for r in rows),all_correct=all(r['score']==100 for r in rows),rows=rows)
(a.capsule/'quality-score.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},indent=2))
