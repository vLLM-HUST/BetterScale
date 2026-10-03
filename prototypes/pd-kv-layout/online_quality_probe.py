"""Bounded real-chat retrieval and exact warm/cold continuation diagnostics."""
import argparse
import asyncio
import json
from pathlib import Path
import uuid
from online_coordinator import Coordinator
from naive_pd_coordinator import owner_for


async def run(args):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    args.output.mkdir(parents=True,exist_ok=False)
    names=[];counts=[0]*4
    label="quality-"+uuid.uuid4().hex[:10]
    for i in range(10000):
        name=f"{label}-{i}";owner=owner_for(name)
        if counts[owner]<2:names.append(name);counts[owner]+=1
        if len(names)==8:break
    cases=[]
    for i,name in enumerate(names):
        code=f"blue-gate-{73100+i}"
        text=("The north gate access code is "+code+".\n"+
              "The garden has trees. The south gate is closed.\n"*(100+13*i)+
              "What is the north gate access code? Reply with only the exact code.")
        tokens=tokenizer.apply_chat_template([dict(role="user",content=text)],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=tokenizer.encode(tokens,add_special_tokens=False)
        cases.append((name,code,tokens))
    trace=(args.output/"control-events.jsonl").open("w")
    def record(event):trace.write(json.dumps(event)+"\n");trace.flush()
    c=Coordinator(args.output/"directory.db",args.p,args.d,trace=record)
    await c.start()
    try:
        first=await asyncio.gather(*(c.submit(name,tokens,len(tokenizer.encode(code,add_special_tokens=False))) for name,code,tokens in cases))
        checks=[]
        for (name,code,tokens),value in zip(cases,first):
            text=tokenizer.decode(value["token_ids"])
            checks.append(dict(session=name,code=code,text=text,retrieved=code in text))
        (args.output/"first.json").write_text(json.dumps(first,indent=2))
        (args.output/"retrieval.json").write_text(json.dumps(checks,indent=2))
        assert all(v["retrieved"] for v in checks),"real-chat retrieval failed"
        # Exact terminal continuation: cold D oracle sees identical token IDs,
        # while P/D sees transferred State plus a new user turn.
        suffix=tokenizer.encode("<|im_end|>\n<|im_start|>user\nRepeat the exact north gate access code.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",add_special_tokens=False)
        second=await asyncio.gather(*(c.submit(v["session"],v["full_tokens"]+suffix,len(first[i]["token_ids"])) for i,v in enumerate(first)))
        (args.output/"second.json").write_text(json.dumps(second,indent=2))
        cold=[]
        for i,v in enumerate(first):
            cold.append(await c.d.rpc(0,"generate",owner=v["owner"],tokens=v["full_tokens"]+suffix,
                                      salt=f"{label}-cold-{i}",n=len(v["token_ids"])))
        comparisons=[]
        for i,(a,b) in enumerate(zip(second,cold)):
            comparisons.append(dict(session=a["session"],
                exact=a["token_ids"]==b["token_ids"],
                first_difference=next((j for j,(x,y) in enumerate(zip(a["token_ids"],b["token_ids"])) if x!=y),None),
                warm_text=tokenizer.decode(a["token_ids"]),cold_text=tokenizer.decode(b["token_ids"])))
        (args.output/"cold.json").write_text(json.dumps(cold,indent=2))
        (args.output/"comparison.json").write_text(json.dumps(comparisons,indent=2))
        assert all(cases[i][1] in v["warm_text"] and cases[i][1] in v["cold_text"] for i,v in enumerate(comparisons))
        assert all(v["exact"] for v in comparisons),"preserve mismatch; do not relabel numerical drift automatically"
        print(json.dumps(dict(ok=True,retrievals=24,exact_continuations=8)),flush=True)
    finally:
        await c.close()
        trace.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p",default="http://10.244.1.16:55581")
    p.add_argument("--d",default="http://10.244.2.32:55586")
    p.add_argument("--model",default="/workspace/models/Qwen3.5-35B-A3B")
    p.add_argument("--output",type=Path,required=True)
    asyncio.run(run(p.parse_args()))
