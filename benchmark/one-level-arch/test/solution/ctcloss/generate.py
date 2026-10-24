"""Exact user-supplied ops-nn Ctcloss_v2 shape, deterministic synthetic inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import math
GUARD = -987.25

def logadd(values):
    m = max(values)
    return m if m == -math.inf else m + math.log(sum(math.exp(x-m) for x in values))

def oracle(probs, labels, length, blank, time, states):
    extended = [blank]
    for label in labels:
        extended += [label, blank]
    alpha = [[-math.inf]*states for _ in range(time)]
    if length:
        alpha[0][0] = probs[0][blank]
        if labels:
            alpha[0][1] = probs[0][labels[0]]
        for t in range(1, length):
            for s, label in enumerate(extended):
                parents = [alpha[t-1][s]]
                if s:
                    parents.append(alpha[t-1][s-1])
                if s > 1 and label != extended[s-2]:
                    parents.append(alpha[t-1][s-2])
                alpha[t][s] = logadd(parents) + probs[t][label]
        end = [alpha[length-1][len(extended)-1]]
        if labels:
            end.append(alpha[length-1][len(extended)-2])
        loss = -logadd(end)
    else:
        loss = math.inf if labels else 0.0
    return alpha, loss



def generate_network(out, seed=20261024):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    T,N,C,U,S=195,256,41,74,149
    rng=np.random.default_rng(seed)
    logits=rng.uniform(-4,4,(T,N,C))
    maximum=logits.max(axis=2,keepdims=True)
    log_probs=(logits-maximum-np.log(np.exp(logits-maximum).sum(axis=2,keepdims=True))).astype('<f4')
    targets=rng.integers(1,C,(N,U),dtype=np.int32).astype('<i4')
    targets[0,40:]=0
    il=np.full(N,T,dtype='<i4')
    tl=np.full(N,U,dtype='<i4'); tl[0]=40
    assert log_probs.shape==(195,256,41) and targets.shape==(256,74)
    assert np.array_equal(il,np.array([195]*256,dtype=np.int32))
    assert np.array_equal(tl,np.array([40]+[74]*255,dtype=np.int32))
    p=log_probs.astype(np.float64)
    assert np.max(np.abs(np.exp(p).sum(axis=2)-1))<1e-6
    extended=np.zeros((N,S),dtype=np.int32)
    extended[:,1::2]=targets
    valid=np.arange(S)[None,:]<(2*tl[:,None]+1)
    skip=np.zeros((N,S),dtype=bool)
    skip[:,2:]=(extended[:,2:]!=0)&(extended[:,2:]!=extended[:,:-2])
    skip &= valid
    alpha=np.full((N,T,S),-np.inf,dtype=np.float64)
    alpha[:,0,0]=p[0,:,0]
    alpha[:,0,1]=p[0,np.arange(N),targets[:,0]]
    # Independent time-first FP64 DP, vectorized across samples and states.
    left=np.full((N,S),-np.inf,dtype=np.float64)
    left2=np.full_like(left,-np.inf)
    for t in range(1,T):
        previous=alpha[:,t-1,:]
        left[:,1:]=previous[:,:-1]
        left2[:,2:]=previous[:,:-2]
        parents=np.logaddexp(np.logaddexp(previous,left),np.where(skip,left2,-np.inf))
        alpha[:,t,:]=np.where(valid,parents+p[t,np.arange(N)[:,None],extended],-np.inf)
    b=np.arange(N)
    loss=-np.logaddexp(alpha[b,T-1,2*tl],alpha[b,T-1,2*tl-1])
    assert np.isfinite(loss).all()
    assert np.isneginf(alpha[0,:,81:]).all()
    # Cross-check three complete samples with the preexisting scalar FP64 oracle.
    for sample in (0,1,255):
        reference,reference_loss=oracle(log_probs[:,sample,:].tolist(),
            targets[sample,:tl[sample]].tolist(),T,0,T,S)
        reference=np.asarray(reference,dtype=np.float64)
        np.testing.assert_allclose(alpha[sample],reference,rtol=1e-12,atol=1e-10)
        assert np.isclose(loss[sample],reference_loss,rtol=1e-12,atol=1e-10)
    arrays={'log_probs.bin':log_probs,'targets.bin':targets,
            'input_lengths.bin':il,'target_lengths.bin':tl}
    for name,value in arrays.items(): value.tofile(out/name)
    alpha32=alpha.astype('<f4'); loss32=loss.astype('<f4')
    alpha32.tofile(out/'log_alpha.golden.bin')
    loss32.tofile(out/'negative_log_likelihood.golden.bin')
    for name,values in [('alpha',alpha32),('loss',loss32)]:
        flat=values.ravel()
        checked=np.full(flat.size+8,GUARD,dtype='<f4'); checked[4:-4]=flat
        checked.tofile(out/(name+'.golden'))
        np.full(flat.size+8,GUARD,dtype='<f4').tofile(out/(name+'.initial.bin'))
    kAlpha=N*T*S+8; kLoss=N+8
    header=f"""#pragma once
#include <cstdint>
constexpr unsigned kCases=1,kTime={T},kBatch={N},kClasses={C},kTargetWidth={U};
constexpr unsigned kInputElements={T*N*C},kTargetElements={N*U},kAlpha={kAlpha},kLoss={kLoss};
extern "C" {{
extern const float ctc_lp[kInputElements];
extern const int32_t ctc_targets[kTargetElements],ctc_lengths[kBatch],ctc_target_lengths[kBatch];
extern float ctc_alpha[kAlpha],ctc_loss[kLoss];
}}
"""
    (out/'data.hpp').write_text(header)
    assembly=[]
    for symbol,file,section in [('ctc_lp','log_probs.bin','.rodata'),
        ('ctc_targets','targets.bin','.rodata'),('ctc_lengths','input_lengths.bin','.rodata'),
        ('ctc_target_lengths','target_lengths.bin','.rodata'),
        ('ctc_alpha','alpha.initial.bin','.data'),('ctc_loss','loss.initial.bin','.data')]:
        alignment=4096 if section=='.data' else 32
        assembly += [f'.section {section}',f'.balign {alignment}',f'.global {symbol}',
            f'.type {symbol},@object',symbol+':',f'.incbin "{out/file}"',
            f'.size {symbol}, .-{symbol}']
    (out/'data.S').write_text('\n'.join(assembly)+'\n')
    case=dict(name='ops_nn_ctcloss_v2',lp=0,targets=0,lengths=0,offsets=0,
        alpha=4,loss=4,T=T,N=N,C=C,states=S,blank=0,
        lpts=N*C,lpbs=C,lpcs=1,ts=1,abs=T*S,ats=S,ass=1)
    (out/'cases.json').write_text(json.dumps([case],indent=2)+'\n')
    spec=dict(name='ops_nn_Ctcloss_v2',data_origin='deterministic synthetic, not captured network tensors',
        seed=seed,numpy_version=np.__version__,rng='default_rng/PCG64',attributes=dict(blank=0,reduction='none',zero_infinity=False),
        inputs=[dict(name=name,shape=list(value.shape),dtype=str(value.dtype),
            bytes=value.nbytes,file=name+'.bin') for name,value in [
            ('log_probs',log_probs),('targets',targets),('input_lengths',il),('target_lengths',tl)]],
        input_lengths=[195]*256,target_lengths=[40]+[74]*255,
        outputs=[dict(name='negative_log_likelihood',shape=[N],dtype='float32',
            file='negative_log_likelihood.golden.bin'),
            dict(name='log_alpha',shape=[N,T,S],dtype='float32',file='log_alpha.golden.bin')],
        first_sample_valid_states=81,allocated_states=S,
        unused_alpha_states='-Inf; sample 0 states 81..148',
        target_layout='contiguous padded [N,74]; first sample padding filled with zero',
        strides_in_elements=dict(log_probs=[N*C,C,1],targets=[U,1],log_alpha=[T*S,S,1]),
        oracle='time-first NumPy FP64 DP; full scalar FP64 cross-check of samples 0,1,255',
        host_checks='PASS; no simulator accuracy/performance claim',
        loss_min=float(loss.min()),loss_max=float(loss.max()),
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.bin')})
    (out/'network-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    print('[network-data][PASS] exact shapes/dtypes/lengths; FP64 oracle checked',flush=True)
    return [case]

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=20261024)
    a=p.parse_args(); generate_network(a.out,a.seed)
