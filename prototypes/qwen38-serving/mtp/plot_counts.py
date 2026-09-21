"""Dependency-free SVG of measured count points and observed three-cohort range."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('summary',type=Path);a=p.parse_args()
rows=json.loads(a.summary.read_text())['throughput']
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="410" viewBox="0 0 1120 410">',
     '<rect width="1120" height="410" fill="white"/>',
     '<style>text{font-family:Arial,sans-serif;fill:#172033;font-size:12px}.title{font-size:21px;font-weight:bold}.grid{stroke:#e2e8f0;stroke-width:1}</style>',
     '<text x="34" y="32" class="title">Qwen27 TP2: MTP count versus end-to-end output throughput</text>',
     '<text x="34" y="55">APC enabled · 6 GiB KV budget · 3073–3129 input / 64 output tokens · median and observed range (3 cohorts)</text>']
for panel,c in enumerate((1,4,8)):
    left=60+panel*367; top=105; width=285; height=220
    selected=[r for r in rows if r['concurrency']==c]
    maximum=max(r['max'] for r in selected)*1.12
    def point(k,y):return left+k*width/4,top+height-y/maximum*height
    svg.append(f'<text x="{left}" y="87" font-weight="bold">Concurrency {c} · output tokens/s</text>')
    for i in range(5):
        value=maximum*i/4; y=point(0,value)[1]
        svg += [f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" class="grid"/>',f'<text x="{left-9}" y="{y+4}" text-anchor="end">{value:.0f}</text>']
        x=point(i,0)[0];svg.append(f'<text x="{x}" y="{top+height+21}" text-anchor="middle">{i}</text>')
    for arm,color in (('native','#64748b'),('candidate','#007f73')):
        points=[]
        for r in sorted((r for r in selected if r['arm']==arm),key=lambda r:r['mtp_tokens']):
            x,y=point(r['mtp_tokens'],r['output_tokens_per_second']);points.append(f'{x},{y}')
            lo=point(r['mtp_tokens'],r['min'])[1];hi=point(r['mtp_tokens'],r['max'])[1]
            svg += [f'<line x1="{x}" y1="{lo}" x2="{x}" y2="{hi}" stroke="{color}" stroke-width="2" opacity=".5"/>',f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>']
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
    svg.append(f'<text x="{left+width/2}" y="{top+height+40}" text-anchor="middle">MTP draft-token count (0 = disabled)</text>')
svg += ['<text x="34" y="389">Native: grey   Candidate: teal. K0 reuses more prefix tokens than K≥1; curves compare serving configurations, not isolated MTP kernel speed.</text>','</svg>']
out=a.summary.with_name('mtp-counts.svg');out.write_text('\n'.join(svg));print(out)
