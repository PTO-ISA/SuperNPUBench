#!/usr/bin/env python3
"""Run one frozen-config gfsim performance run and emit the NDF trace set.

  <prefix>_swim.log            raw gfsim stdout
  <prefix>_swim.json           full SwimLane JSON (counters included)
  <prefix>_swim_nocounters.json  same trace with ph:"C" counters removed
  <prefix>_swim.json.gz / <prefix>_swim_nocounters.json.gz
  <prefix>_meta.json           bytes/sha256/cycles/kernel span

Config: config/random300.json (soc_random, 300-cycle latency, 128 B/cycle,
256 KiB ShareTReg); optional RWDB/TAG_WAY overrides.
"""
import argparse, gzip, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def strip_counters(src, dst):
    data=json.load(src.open())
    ev=data.get("traceEvents", data)
    kept=[e for e in ev if e.get("ph")!="C"]
    if isinstance(data,dict): data["traceEvents"]=kept
    else: data=kept
    json.dump(data, dst.open('w'))
    return len(ev), len(kept)

ts_re=re.compile(r'"ts":\s*(\d+)')
dur_re=re.compile(r'"dur":\s*(\d+)')
name_re=re.compile(r'"name":\s*"(\d+) (TLOAD|TSTORE) PE\d+"')

def scan_span(path):
    first_load=None; last_store=0; inblk=False; buf=[]; pend=None
    with open(path) as f:
        for line in f:
            if not inblk:
                if line.rstrip('\n')=='  {': inblk=True; buf=[line]
                continue
            buf.append(line)
            if line.rstrip('\n') in ('  },','  }'):
                inblk=False; blk=''.join(buf); buf=[]
                m=name_re.search(blk)
                if not m: continue
                t=ts_re.search(blk); d=dur_re.search(blk)
                if not t: continue
                ts=int(t.group(1)); dur=int(d.group(1)) if d else 0
                if m.group(2)=='TLOAD':
                    if first_load is None or ts<first_load: first_load=ts
                else:
                    last_store=max(last_store, ts+dur)
    return first_load, last_store

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gfsim', required=True)
    ap.add_argument('--elf', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--rwdb', default=None)
    ap.add_argument('--way', default=None)
    ap.add_argument('--timeout', type=int, default=1800)
    a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    prefix=out/f"{a.name}_swim"
    common=['-s','softcore.multiThreadNum=4','-s','cube.enable_internal_acc=true',
            '-s','bctrl.vec_cell_sched_enable=true','-s','core.soc_random=true',
            '-s','core.mean_read_latency=300','-s','core.mean_write_latency=300',
            '-s','core.read_bandwidth_limit=128','-s','core.write_bandwidth_limit=128',
            '-s','core.sharetreg_size_bytes=262144','-s','core.sharetreg_ssb_count=64']
    if a.rwdb: common+=['-s',f'sl2.RWDB_ENTRY_COUNT={a.rwdb}']
    if a.way: common+=['-s',f'sl2.TAG_WAY_COUNT={a.way}']
    cmd=[a.gfsim,'--test-finisher','1','-f',a.elf,'--swimlane','1',
         '--swimfile',str(prefix)]+common
    t0=time.time()
    model_root=Path(a.gfsim).resolve().parents[1]
    proc=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, timeout=a.timeout, cwd=model_root)
    (out/f"{a.name}_swim.log").write_text(proc.stdout)
    if proc.returncode!=0:
        print(f"gfsim rc={proc.returncode}", file=sys.stderr); return proc.returncode
    cycles=None
    for line in proc.stdout.splitlines():
        if line.strip().startswith('Total Cycles'):
            cycles=int(re.search(r'(\d+)\s*$', line).group(1)); break
    full=Path(str(prefix)+'.json')
    fl,ls=scan_span(full)
    noc=Path(str(prefix)+'_nocounters.json')
    total,kept=strip_counters(full,noc)
    for p in (full,noc):
        with gzip.open(str(p)+'.gz','wb',compresslevel=6) as g, open(p,'rb') as f:
            for b in iter(lambda: f.read(1<<20), b''): g.write(b)
    meta={'elf':a.elf,'elf_sha256':sha256(a.elf),'cycles':cycles,
          'first_tload':fl,'last_tstore_end':ls,
          'kernel_span':(ls-fl) if (fl is not None and ls) else None,
          'events_total':total,'events_nocounters':kept,
          'files':{}}
    for p in (full,noc,Path(str(full)+'.gz'),Path(str(noc)+'.gz'),out/f"{a.name}_swim.log"):
        meta['files'][p.name]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    (out/f"{a.name}_meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print(f"{a.name}: cycles={cycles} span={meta['kernel_span']} ({fl}..{ls}) "
          f"events={total}->{kept} wall={time.time()-t0:.0f}s")
    return 0

if __name__=='__main__':
    sys.exit(main())
