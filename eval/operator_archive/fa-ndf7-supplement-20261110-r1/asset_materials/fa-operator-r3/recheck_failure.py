#!/usr/bin/env python3
"""Recheck saved nonzero outputs; successful audit still reports numerical FAIL."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
ROOT=Path(__file__).resolve().parent/'numerical_failure'
def values(blob):return [struct.unpack('<f',struct.pack('<I',x<<16))[0] for x, in struct.iter_unpack('<H',blob)]
def bad(a,b):return sum(not math.isfinite(x) or abs(x-y)>1e-3+2e-2*abs(y) for x,y in zip(a,b))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():p.error('choose new output record')
    m=json.loads((ROOT/'identity.json').read_text())
    for name,digest in m['files'].items():
        f=ROOT/name
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise SystemExit('FAIL: numerical evidence identity '+name)
    names=['srcq','srck','srcv','srcq_scale','srck_scale','srcv_scale']
    for name in names:
        if (ROOT/(name+'.bin')).read_bytes()!=(ROOT/(name+'_readback.bin')).read_bytes():raise SystemExit('FAIL: historical input readback '+name)
    actual,ref=values((ROOT/'res.bin').read_bytes()),values((ROOT/'golden.bin').read_bytes())
    if len(actual)!=32768 or len(ref)!=32768 or not all(math.isfinite(x) for x in ref) or not any(ref):raise SystemExit('FAIL: vacuous or malformed reference')
    observed=bad(actual,ref);zero=bad([0.0]*len(ref),ref);nan=bad([float('nan')]*len(ref),ref)
    if observed!=m['expected_bad'] or not zero or nan!=len(ref):raise SystemExit('FAIL: failure record or negative control differs')
    r={'audit_status':'pass','numerical_status':'fail','bad':observed,'total':len(ref),'negative_zero_bad':zero,'negative_nan_bad':nan,'historical_input_readbacks_match':True,'scope':m['scope']}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
