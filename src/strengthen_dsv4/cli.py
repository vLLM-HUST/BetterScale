"""One explicit serving entry; no package installation or donor source edits."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import sys

from .compat import check_runtime
from .config import PROFILES, engine_options


def command(options, host, port, served_model_name):
    args=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',options['model'],
          '--host',host,'--port',str(port),'--served-model-name',served_model_name]
    for name,value in options.items():
        if name=='model': continue
        flag='--'+name.replace('_','-')
        if isinstance(value,bool):
            args.append(flag if value else '--no-'+name.replace('_','-'))
        else:
            args.extend((flag,json.dumps(value,separators=(',',':')) if isinstance(value,dict) else str(value)))
    return args


def main(argv=None):
    parser=argparse.ArgumentParser(description='Qualified TP8/DSACP/K5/four-seat DSV4 serving')
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('check',help='Check pinned runtime versions and source without loading a model')
    for action in ('plan','serve'):
        p=sub.add_parser(action)
        p.add_argument('--model',required=True)
        p.add_argument('--profile',choices=PROFILES,default='optimized')
        p.add_argument('--kv-gib',type=float,default=12)
        p.add_argument('--artifacts',type=Path)
        p.add_argument('--host',default='127.0.0.1')
        p.add_argument('--port',type=int,default=8000)
        p.add_argument('--served-model-name',default='dsv4')
    args=parser.parse_args(argv)
    if args.action=='check':
        print(json.dumps(check_runtime(),indent=2));return
    if not 1<=args.port<=65535:
        parser.error('port must be between1 and65535')
    artifacts=args.artifacts or Path('runs')/('serve-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    options=engine_options(args.model,artifacts,args.profile,args.kv_gib)
    launch=command(options,args.host,args.port,args.served_model_name)
    environment=dict(OMP_NUM_THREADS='4',TASK_QUEUE_ENABLE='1',
        HCCL_CONNECT_TIMEOUT='120',HCCL_EXEC_TIMEOUT='120',HCCL_BUFFSIZE='256',
        HCCL_OP_EXPANSION_MODE='AIV',VLLM_ASCEND_ENABLE_FLASHCOMM1='1',
        PYTORCH_NPU_ALLOC_CONF='expandable_segments:True')
    if args.action=='plan':
        print(json.dumps(dict(argv=launch,environment=environment,engine_options=options),indent=2));return
    checked=check_runtime()
    vendor=metadata.distribution('vllm-ascend').locate_file('vllm_ascend/_cann_ops_custom/vendors/custom_transformer')
    if not vendor.is_dir():
        raise RuntimeError('Pinned Ascend custom-op vendor directory is missing')
    environment['ASCEND_CUSTOM_OPP_PATH']=str(vendor)
    os.environ['LD_LIBRARY_PATH']=str(vendor/'op_api/lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    if artifacts.exists() and any(artifacts.iterdir()):
        parser.error('artifact directory is not empty; use a fresh directory')
    artifacts.mkdir(parents=True,exist_ok=True)
    (artifacts/'launch.json').write_text(json.dumps(dict(
        profile=args.profile,compatibility=checked,engine_options=options,
        host=args.host,port=args.port,served_model_name=args.served_model_name,
        environment=environment),indent=2))
    os.environ.update(environment)
    print(f'strengthen-dsv4: {args.profile}; receipts: {artifacts.resolve()}',flush=True)
    os.execv(sys.executable,launch)


if __name__=='__main__':
    main()
