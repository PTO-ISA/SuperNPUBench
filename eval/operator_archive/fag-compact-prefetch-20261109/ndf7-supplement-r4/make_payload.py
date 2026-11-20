#!/usr/bin/env python3
"""Assemble operator-r3 from the immutable downloaded operator-r2 and this overlay."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--previous-operator-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():p.error('choose new output directory')
source=Path(__file__).resolve().parent/'payload_source';expected=json.loads((source/'files.json').read_text())['files']
a.out.mkdir(parents=True)
for x in expected:
 rel=Path(x['path']);src=source/rel
 if not src.exists():src=a.previous_operator_root/rel
 if src.is_symlink() or not src.is_file() or src.stat().st_size!=x['bytes'] or hashlib.sha256(src.read_bytes()).hexdigest()!=x['sha256']:raise SystemExit('Missing/mismatched authoritative input: '+str(rel))
 dst=a.out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
shutil.copyfile(source/'files.json',a.out/'files.json')
subprocess.run([sys.executable,str(a.out/'verify.py')],check=True)
