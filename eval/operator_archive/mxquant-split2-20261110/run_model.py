#!/usr/bin/env python3
"""Hash-pinned timing runtime adapter, selected explicitly by replay.py."""
import hashlib, os, resource, sys
from pathlib import Path
model=Path(os.environ['MXQUANT_TIMING_MODEL_ROOT']).resolve()
exe=model/'bin/gfsim'
expected='bebcf1197d78cacdf65109ad1ad76d42604d7996b37c298d92063f82435b7224'
if hashlib.sha256(exe.read_bytes()).hexdigest()!=expected:
    raise SystemExit('Timing runtime hash mismatch; historical timing identity required')
for name in ('RMS_IDEAL_L2_READ_BANK', 'RMS_IDEAL_L2_WRITE_BANK'):
    os.environ.pop(name, None)
os.environ['RMS_TAIL_BYPASS']='1'
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
os.nice(15)
os.chdir(model)
os.execv(str(exe),[str(exe),*sys.argv[1:]])
