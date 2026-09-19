"""Decode only verified TCube/L2 prefixes; V2's different ABI remains opaque."""
import csv
import json
from pathlib import Path
import statistics
import struct
import sys

root = Path(sys.argv[1])
fields = '''usedCoreNum M N Ka Kb singleCoreM singleCoreN singleCoreK baseM baseN baseK depthA1 depthB1 stepM stepN isBias transLength iterateOrder shareMode shareL1Size shareL0CSize shareUbSize batchM batchN singleBatchM singleBatchN stepKa stepKb depthAL1CacheUB depthBL1CacheUB dbL0A dbL0B dbL0C ALayoutInfoB ALayoutInfoS ALayoutInfoN ALayoutInfoG ALayoutInfoD BLayoutInfoB BLayoutInfoS BLayoutInfoN BLayoutInfoG BLayoutInfoD CLayoutInfoB CLayoutInfoS1 CLayoutInfoN CLayoutInfoG CLayoutInfoS2 BatchNum mxTypePara'''.split()
assert len(fields) == 50
result = dict(scope='hw3 device5; selected eager ND BF16 kernel, profiler replay. PMU counters overlap; do not sum into latency. V2 ABI not decoded.', cases=[])
for n in (1024,1408,1536,1792):
    folders = list((root/'measurements'/str(n)).glob('OPPROF_*'))
    assert len(folders) == 1
    folder = folders[0]
    def rows(name):
        with (folder/name).open() as file: return list(csv.DictReader(file))
    basic = rows('OpBasicInfo.csv'); assert len(basic) == 1
    data = (folder/'dump/kernel_data/input_tiling.bin').read_bytes()
    case = dict(tokens=n, kernel=basic[0], tiling_bytes=len(data), counters={})
    if n != 1024:
        assert len(data) == 288 and 'MatMulV3_' in basic[0]['Op Name']
        values = struct.unpack('<72I',data)
        cube = dict(zip(fields,values[:50]))
        assert [cube[k] for k in ('usedCoreNum','M','N','Ka','Kb')] == [24,n,17408,5120,5120]
        l2 = dict(zip(('mTileCntL2','nTileCntL2','mTileBlock','nTileBlock','calOrder'), values[50:55]))
        assert l2['mTileCntL2']*l2['mTileBlock']*cube['baseM'] >= n
        assert l2['nTileCntL2']*l2['nTileBlock']*cube['baseN'] >= 17408
        case.update(cube=cube,l2=l2,panel_rows=l2['mTileBlock']*cube['baseM'],
                    panel_columns=l2['nTileBlock']*cube['baseN'],
                    panel_weight_mib=l2['nTileBlock']*cube['baseN']*5120*2/1024**2)
    for filename, names in [('L2Cache.csv',['aic_read_hit_rate(%)']),
                            ('PipeUtilization.csv',['aic_cube_time(us)','aic_mte2_time(us)','aic_cube_ratio','aic_mte2_ratio'])]:
        records=rows(filename);assert len(records)==24
        for name in names:
            v=[float(r[name]) for r in records]
            case['counters'][name]=dict(min=min(v),median=statistics.median(v),max=max(v))
    result['cases'].append(case)
(root/'tiling-summary.json').write_text(json.dumps(result,indent=2))
for c in result['cases']:
    print(c['tokens'],c.get('l2'),{k:round(v['median'],3) for k,v in c['counters'].items()})
