#!/usr/bin/env python3
"""Compare the fixed network's RES_CHECK files with its independent goldens."""
import argparse
import json
from pathlib import Path
import numpy as np

def load(path,count,dtype):
    dtype=np.dtype(dtype)
    if path.stat().st_size!=count*dtype.itemsize:
        raise ValueError(f'{path}: expected {count*dtype.itemsize} bytes')
    return np.fromfile(path,dtype=dtype)

def compare(directory):
    directory=Path(directory)
    tiling=load(directory/'tiling_info.bin',24,'<u4')
    pe=int(tiling[6])
    if pe not in (1,4): raise ValueError('invalid PE count')
    status=load(directory/'status.bin',pe*4,'<u4').reshape(pe,4)
    expected=np.array([[1,i+1,0,256//pe] for i in range(pe)],dtype='<u4')
    if not np.array_equal(status,expected):
        raise ValueError(f'incomplete/wrong PE ownership: {status.tolist()}')
    guards=load(directory/'output_guards.bin',16,'<f4')
    if not np.array_equal(guards.view('<u4'),np.full(16,-987.25,dtype='<f4').view('<u4')):
        raise ValueError('output guard bytes modified')
    metrics={}
    for actual_name,golden_name,count in [
        ('loss.bin','negative_log_likelihood.golden.bin',256),
        ('log_alpha.bin','log_alpha.golden.bin',256*195*149)]:
        actual=load(directory/actual_name,count,'<f4')
        golden=load(directory/golden_name,count,'<f4')
        close=np.isclose(actual,golden,rtol=1e-4,atol=2e-5,equal_nan=False)
        if not close.all():
            i=int(np.flatnonzero(~close)[0])
            raise ValueError(f'{actual_name}[{i}]: got {actual[i]}, expected {golden[i]}')
        finite=np.isfinite(golden)
        metrics[actual_name]=dict(elements=count,
            max_absolute_error=float(np.max(np.abs(actual[finite]-golden[finite]))))
    metrics['pe_status']=status.tolist()
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cmp-dir',required=True,type=Path)
    a=p.parse_args()
    result=compare(a.cmp_dir)
    (a.cmp_dir/'accuracy.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ACCURACY_PASS '+json.dumps(result))
