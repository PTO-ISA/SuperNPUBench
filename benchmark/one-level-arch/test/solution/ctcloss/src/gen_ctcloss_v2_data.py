"""Exact user-supplied ops-nn Ctcloss_v2 shape, deterministic synthetic inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import math

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



def generate_network(out, seed=20261024, pe_num=4):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    if pe_num not in (1,4): raise ValueError("pe_num must be 1 or 4")
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
    # Unwritten logical outputs remain zero in BSS; retain full write coverage
    # without a guest-side fill of all 7.4M alpha elements before launch.
    assert not np.any(np.isclose(np.float32(0),alpha32,rtol=1e-4,atol=2e-5)), 'zero BSS could match an unwritten output'
    assert not np.any(np.isclose(np.float32(0),loss32,rtol=1e-4,atol=2e-5)), 'zero BSS could match an unwritten loss'
    alpha32.tofile(out/'log_alpha.golden.bin')
    loss32.tofile(out/'negative_log_likelihood.golden.bin')
    if pe_num==1:
        starts=[0,N,N,N]; ends=[N,N,N,N]
    else:
        groups=(N+1)//2; q,r=divmod(groups,4)
        first=[q*pe+min(pe,r) for pe in range(4)]
        last=[first[pe]+q+(pe<r) for pe in range(4)]
        starts=[min(x*2,N) for x in first]; ends=[min(x*2,N) for x in last]
    words=[T,N,C,U,S,0,pe_num,0,N*C,C,1,1,T*S,S,1,0,*starts,*ends]
    np.asarray(words,dtype='<u4').tofile(out/'tiling_info.bin')
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
        tiling_info=dict(file='tiling_info.bin',abi='24 little-endian uint32 words',pe_num=pe_num,
                         batch_begin=starts,batch_end=ends),
        loss_min=float(loss.min()),loss_max=float(loss.max()),
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/name for name in [*arrays,'log_alpha.golden.bin','negative_log_likelihood.golden.bin','tiling_info.bin']]})
    (out/'network-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    print('[network-data][PASS] exact shapes/dtypes/lengths; FP64 oracle checked',flush=True)
    return spec

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',required=True)
    p.add_argument('--pe-num',type=int,choices=[1,4],default=4)
    p.add_argument('--seed',type=int,default=20261024)
    a=p.parse_args(); generate_network(a.out_dir,a.seed,a.pe_num)
