"""One-time task-local native overlay; refuse already-staged version pins."""
import json,shutil,hashlib
from pathlib import Path
r=Path("/workspace/overlap-swe")
for variant in ("e16","e36"):
 p=r/variant/"src/betterscale"
 pins=p/"qwen35_pins.json"; data=json.loads(pins.read_text()); assert data["versions"]["torch-npu"]=="2.10.0.post2";data["versions"]["torch-npu"]="2.10.0.post4";pins.write_text(json.dumps(data,indent=2)+"\n")
 for src in (r/"native/patches").rglob("*"):
  if src.is_file() and src.name!="libbs_gdn_host37.so":
   dest=p/src.relative_to(r/"native");dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 shutil.copy2(r/"native/patches/qwen_gdn/libbs_gdn_host37.so",p/"patches/qwen_gdn/libbs_gdn_host.so")
 native=p/"patches/qwen_gdn/native.json";meta=json.loads(native.read_text());meta["host_sha256"]=hashlib.sha256((p/"patches/qwen_gdn/libbs_gdn_host.so").read_bytes()).hexdigest();meta["qualification"]="CANN9.1/post4 experimental SWE qualification; host from original E36 source";native.write_text(json.dumps(meta,indent=2)+"\n")
 shutil.copy2(r/"moe_overlap.py",p/"models/qwen35/moe_overlap.py")
 f=p/"models/qwen35/integration.py";s=f.read_text();assert s.count("def before_init(worker, config):")==1;s=s.replace("def before_init(worker, config):","def before_init(worker, config):\n    from .moe_overlap import configure\n    configure(config)");f.write_text(s)
 print("staged",variant,meta["host_sha256"])
