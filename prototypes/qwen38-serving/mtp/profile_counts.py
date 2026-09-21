"""Export short native profiles and summarize exact target/draft graph evidence."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess


def inspect(db):
    with sqlite3.connect(f'file:{db.resolve()}?mode=ro',uri=True) as c:
        assert c.execute('pragma quick_check').fetchone()[0]=='ok'
        graphs=[]
        for launch,start,duration,evidence in c.execute(
                'select launch_id,start_ns,dur_us,evidence_level from traceloom_graph_launch order by start_ns').fetchall():
            rows=c.execute('''select e.op_type,count(*),sum(m.dur_us)
                from traceloom_graph_body_member m join traceloom_event e using(event_id)
                where m.launch_id=? group by e.op_type order by sum(m.dur_us) desc''',(launch,)).fetchall()
            fia=sum(n for op,n,d in rows if op and op.startswith('FusedInferAttentionScore'))
            kind='target' if fia==16 else ('draft' if 1<=fia<=4 else 'unclassified')
            graphs.append(dict(launch_id=launch,start_ns=start,duration_us=duration,
                evidence=evidence,kind=kind,fia_count=fia,
                operators=[dict(type=op,count=n,sum_us=d) for op,n,d in rows]))
        targets=[g for g in graphs if g['kind']=='target' and g['evidence']=='exact_direct']
        for current,following in zip(targets,targets[1:]):
            current['target_period_us']=(following['start_ns']-current['start_ns'])/1000
            current['after_target_us']=current['target_period_us']-current['duration_us']
        api=c.execute('''select s.value,count(*),sum(a.endNs-a.startNs)/1000.0
            from CANN_API a join STRING_IDS s on s.id=a.name
            where s.value like '%FusedInferAttention%' or s.value like '%Update%'
               or s.value='aclmdlRIExecuteAsync'
            group by s.value''').fetchall()
    return dict(graphs=graphs,apis=api,scope='Rank-local profiled graph envelopes. after_target includes draft, sampling and metadata, NOT pure idle. Operator sums may overlap.')


def main():
    p=argparse.ArgumentParser();p.add_argument('capsule',type=Path)
    p.add_argument('--traceloom',type=Path,required=True)
    p.add_argument('--rules-config',type=Path);a=p.parse_args()
    out=a.capsule/'traceloom';out.mkdir(exist_ok=True)
    manifest=[]
    for phase in ('decode','mixed'):
        for rank in (0,1):
            sources=list((a.capsule/'profiles'/phase/'profile').glob(f'rank{rank}_*/PROF_*'))
            assert len(sources)==1,sources
            dest=a.capsule/'native-graph'/phase/f'rank{rank}'/sources[0].name
            if not dest.exists():shutil.copytree(sources[0],dest)
            name=f'{a.capsule.name}-{phase}-rank{rank}'
            if not list(dest.glob('msprof_*.db')):
                with (out/f'{name}-msprof.log').open('w') as log:
                    subprocess.run(['msprof','--export=on','--type=db',f'--output={dest}'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
            db=out/f'{name}.db';timeline=out/f'{name}.perfetto.json.gz'
            if not db.exists():
                with (out/f'{name}-analyze.log').open('w') as log:
                    subprocess.run([str(a.traceloom),str(dest),'--threads','4','--output',str(db)] + (['--rules-config',str(a.rules_config)] if a.rules_config else []),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            if not timeline.exists():
                with (out/f'{name}-export.log').open('w') as log:
                    subprocess.run([str(a.traceloom),'export-perfetto',str(db),'--output',str(timeline)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            result=inspect(db)
            schedule=json.loads((a.capsule/'profiles'/phase/f'schedule-rank{rank}.json').read_text())
            result['dispatch']=[r for r in schedule['events'] if r['event']=='dispatch']
            result.update(phase=phase,rank=rank,timeline=str(timeline.resolve()))
            (out/f'{name}-summary.json').write_text(json.dumps(result,indent=2))
            manifest.append(dict(name=name,timeline=str(timeline),exact_graphs=len(result['graphs'])))
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print(name,'exported',flush=True)

if __name__=='__main__':main()
