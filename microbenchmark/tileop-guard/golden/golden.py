#!/usr/bin/env python3
"""Host-side independent golden for the tileop-guard res_check precision path.

Two modes:
  golden.py gen   <case> <chkdir>   # write CHK_DIR/in_*.bin (host owns inputs)
  golden.py check <case> <chkdir>   # read in_*.bin + out.bin, compare, exit 0/1

Independence principle: the reference is implemented here in numpy purely from
the *interface semantics* (SuperNPUBench docs/intrinsics + SuperScalarModel ISA
spec). It does NOT read the emulator's tile implementation, so it is a true
independent oracle. Inputs are generated here (not on device) because the tile
backend mangles device-side memory fills (see project memory / plan step 0).
"""
import sys, os
import numpy as np

# --------------------------------------------------------------------------
# input generators (deterministic, varied, mixed-sign, non-degenerate).
# div-family avoids zero divisors; shift-family clamps to [0,31].
# --------------------------------------------------------------------------
def seq_f32(n, base, step, seed=0):
    i = np.arange(n, dtype=np.float32)
    # mixed sign, non-monotone, nonzero
    v = base + i * step + np.float32(3.0) * np.sin((i + seed) * np.float32(0.7))
    v = v.astype(np.float32)
    v[v == 0] = np.float32(0.5)
    return v

def seq_i32(n, base, step, seed=0):
    i = np.arange(n, dtype=np.int64)
    v = (base + i * step + ((i + seed) * 131 % 257) - 128).astype(np.int32)
    return v

def shifts_i32(n):
    return (np.arange(n, dtype=np.int64) % 31).astype(np.int32)

def f32_to_f16bits(a):
    return a.astype(np.float16).view(np.uint16)

# --------------------------------------------------------------------------
# registry: case -> spec dict.
#   fam: binary|scalar|unary|reduce|expand|matmul|convert
#   op : numpy semantic key
#   M,N[,K], dt_in, dt_out, scalar (for *S), axis/bcast (reduce/expand), eps
# --------------------------------------------------------------------------
F32 = 'f32'; I32 = 'i32'; F16 = 'f16'

def R(**kw):
    return kw

REG = {}

# ---- VEC binary (same dtype in/out) ----
for name, op in [('tadd','add'),('tsub','sub'),('tmul','mul'),('tdiv','div'),
                 ('tmax','max'),('tmin','min')]:
    REG[name] = R(fam='binary', op=op, M=16, N=16, dt=F32, eps=1e-4)
for name, op in [('tand','and'),('tor','or'),('txor','xor'),('trem','rem'),
                 ('tshl','shl'),('tshr','shr')]:
    REG[name] = R(fam='binary', op=op, M=16, N=16, dt=I32, eps=0)

# ---- VEC unary ----
for name, op in [('tabs','abs'),('tneg','neg'),('trelu','relu')]:
    REG[name] = R(fam='unary', op=op, M=16, N=16, dt=F32, eps=1e-5)
REG['tnot'] = R(fam='unary', op='not', M=16, N=16, dt=I32, eps=0)
# tcvt (f32->i32) left run-only: rounding mode (trunc vs RNE) not pinned by docs.
REG['tfma'] = R(fam='ternary', op='fma', M=16, N=16, dt=F32, eps=1e-4)

# ---- VEC scalar (*S) ----
SCAL = np.float32(1.75)
for name, op in [('tadds','add'),('tsubs','sub'),('tmuls','mul'),('tdivs','div'),
                 ('tmaxs','max'),('tmins','min')]:
    REG[name] = R(fam='scalar', op=op, M=16, N=16, dt=F32, scalar=float(SCAL), eps=1e-4)
for name, op in [('tands','and'),('tors','or'),('txors','xor'),('trems','rem'),
                 ('tshls','shl'),('tshrs','shr')]:
    REG[name] = R(fam='scalar', op=op, M=16, N=16, dt=I32, scalar=5, eps=0)
REG['texpands'] = R(fam='fill', op='fill', M=16, N=16, dt=F32, scalar=float(SCAL), eps=0)

# ---- SFU reduce (row: reduce over cols -> per-row scalar at out[r*N+0];
#                  col: reduce over rows -> per-col scalar at out[0*N+c]=out[c]) ----
for name, op in [('trowsum','sum'),('trowmax','max'),('trowmin','min'),('trowprod','prod')]:
    REG[name] = R(fam='reduce', op=op, foot='row', M=16, N=16, dt=F32, eps=1e-3)
for name, op in [('tcolsum','sum'),('tcolmax','max'),('tcolmin','min'),('tcolprod','prod')]:
    REG[name] = R(fam='reduce', op=op, foot='col', M=16, N=16, dt=F32, eps=1e-3)

# ---- CUBE matmul family (f16 inputs, f32 accumulate; wide eps for f16 rounding) ----
REG['tmatmul']        = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F32, eps=2e-2)
REG['tmatmul_acc']    = R(fam='matmul', M=32, N=32, K=32, post='acc',  dt_out=F32, eps=2e-2)
REG['tmatmul_bias']   = R(fam='matmul', M=32, N=32, K=32, post='bias', dt_out=F32, eps=2e-2)
REG['tmatmul_relu']   = R(fam='matmul', M=32, N=32, K=32, post='relu', dt_out=F16, eps=3e-2)
REG['tmatmul_rowmax'] = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F32, eps=2e-2)
REG['tmatmul_f16']    = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F16, eps=3e-2)

# ---- FIXP convert (matmul then cast to fp16 via fixp::convert) ----
REG['convert'] = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F16, eps=3e-2)

# --------------------------------------------------------------------------
NP = {F32: np.float32, I32: np.int32, F16: np.float16}

def np_read(path, dt):
    return np.fromfile(path, dtype=NP[dt])

def gen(case, chkdir):
    s = REG[case]; fam = s['fam']; op = s.get('op', '')
    os.makedirs(chkdir, exist_ok=True)
    # --- matmul family (f16 A/B, optional f32 C / bias) ---
    if fam == 'matmul':
        M, N, K = s['M'], s['N'], s['K']
        ii = np.arange(M * K, dtype=np.float32)
        jj = np.arange(K * N, dtype=np.float32)
        A = (np.sin(ii * np.float32(0.13)) * np.float32(0.8)).astype(np.float16)
        B = (np.cos(jj * np.float32(0.21)) * np.float32(0.8)).astype(np.float16)
        A.tofile(os.path.join(chkdir, 'in_a.bin'))
        B.tofile(os.path.join(chkdir, 'in_b.bin'))
        if s['post'] == 'acc':
            kk = np.arange(M * N, dtype=np.float32)
            C = (np.sin(kk * np.float32(0.07)) * np.float32(0.5)).astype(np.float32)
            C.tofile(os.path.join(chkdir, 'in_c.bin'))
        if s['post'] == 'bias':
            bias = (np.arange(N, dtype=np.float32) * np.float32(0.01) + np.float32(0.5)).astype(np.float32)
            bias.tofile(os.path.join(chkdir, 'in_bias.bin'))
        return
    # --- elementwise / reduce families ---
    n = s['M'] * s['N']; dt = s['dt']
    # --- src0 (in_a) ---
    if fam == 'reduce':
        # bounded positive near 1 so prod stays finite / well-conditioned
        i = np.arange(n, dtype=np.float32)
        a = (np.float32(1.0) + np.float32(0.3) * np.sin(i * np.float32(0.6))).astype(np.float32)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    if dt == I32:
        if op in ('shl', 'shr'):
            # positive & bounded so logical==arithmetic shift and no overflow
            a = (np.abs(seq_i32(n, 1, 3, 0)) & 0x000FFFFF).astype(np.int32)
        elif op == 'rem':
            a = (np.abs(seq_i32(n, 5, 2, 0)) & 0x0000FFFF).astype(np.int32)
        else:
            a = seq_i32(n, 1, 2, 0)
    else:
        a = seq_f32(n, 1.0, 0.1, 0)
    a.tofile(os.path.join(chkdir, 'in_a.bin'))
    # --- src1 (in_b) for binary/ternary ---
    if fam in ('binary', 'ternary'):
        if dt == I32:
            if op in ('shl', 'shr'):
                b = shifts_i32(n)                       # 0..30
            elif op == 'rem':
                b = (np.abs(seq_i32(n, 0, 1, 5)) % 17 + 1).astype(np.int32)  # 1..17
            else:
                b = seq_i32(n, 3, 1, 9)
        else:
            b = seq_f32(n, 2.0, 0.07, 9)
            if op == 'div':
                b[np.abs(b) < 0.5] = np.float32(0.9)    # avoid /0
        b.tofile(os.path.join(chkdir, 'in_b.bin'))
    if fam == 'ternary':
        c = seq_f32(n, 0.5, 0.05, 21)
        c.tofile(os.path.join(chkdir, 'in_c.bin'))

def _elt(op, a, b=None, s=None):
    if op == 'add': return a + (b if b is not None else s)
    if op == 'sub': return a - (b if b is not None else s)
    if op == 'mul': return a * (b if b is not None else s)
    if op == 'div': return a / (b if b is not None else s)
    if op == 'max': return np.maximum(a, b if b is not None else s)
    if op == 'min': return np.minimum(a, b if b is not None else s)
    if op == 'and': return a & (b if b is not None else np.int32(s))
    if op == 'or':  return a | (b if b is not None else np.int32(s))
    if op == 'xor': return a ^ (b if b is not None else np.int32(s))
    if op == 'rem': return np.remainder(a, b if b is not None else np.int32(s))
    if op == 'shl': return (a.astype(np.int32) << (b if b is not None else np.int32(s))).astype(np.int32)
    if op == 'shr': return (a.astype(np.int32) >> (b if b is not None else np.int32(s))).astype(np.int32)
    raise KeyError(op)

def ref(case, chkdir):
    s = REG[case]; dt = s['dt']; fam = s['fam']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), dt)
    if fam == 'binary':
        b = np_read(os.path.join(chkdir, 'in_b.bin'), dt)
        return _elt(s['op'], a, b=b), dt
    if fam == 'scalar':
        return _elt(s['op'], a, s=s['scalar']), dt
    if fam == 'ternary':  # fma = a*b + c
        b = np_read(os.path.join(chkdir, 'in_b.bin'), dt)
        c = np_read(os.path.join(chkdir, 'in_c.bin'), dt)
        return a * b + c, dt
    if fam == 'fill':
        return np.full(a.shape, s['scalar'], dtype=np.float32), dt
    if fam == 'unary':
        op = s['op']
        if op == 'abs':  return np.abs(a), dt
        if op == 'neg':  return -a, dt
        if op == 'relu': return np.maximum(a, np.float32(0)), dt
        if op == 'not':  return ~a, dt
        if op == 'cvt_f32_i32': return a.astype(np.int32), s.get('dt_out', I32)
    raise KeyError(fam)

def check_reduce(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    fn = {'sum': np.sum, 'max': np.max, 'min': np.min, 'prod': np.prod}[s['op']]
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    o = np_read(out_path, F32)
    if s['foot'] == 'row':
        red = fn(a, axis=1)                       # length M
        got = o.reshape(M, N)[:, 0]               # out[r*N+0]
    else:
        red = fn(a, axis=0)                       # length N
        got = o.reshape(M, N)[0, :]               # out[0*N+c]
    eps = np.float32(s.get('eps', 1e-3))
    atol = eps + eps * np.abs(red.astype(np.float32))
    bad = np.flatnonzero(np.abs(got.astype(np.float32) - red.astype(np.float32)) > atol)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] REDUCE MISMATCH {bad.size}/{red.size} first@{i} got={got[i]} ref={red[i]}',
          file=sys.stderr)
    return 1

def check_matmul(case, chkdir):
    s = REG[case]; M, N, K = s['M'], s['N'], s['K']
    A = np.fromfile(os.path.join(chkdir, 'in_a.bin'), np.float16).reshape(M, K).astype(np.float32)
    B = np.fromfile(os.path.join(chkdir, 'in_b.bin'), np.float16).reshape(K, N).astype(np.float32)
    D = A @ B
    post = s['post']
    if post == 'acc':
        D = D + np.fromfile(os.path.join(chkdir, 'in_c.bin'), np.float32).reshape(M, N)
    elif post == 'bias':
        D = D + np.fromfile(os.path.join(chkdir, 'in_bias.bin'), np.float32).reshape(1, N)
    elif post == 'relu':
        D = np.maximum(D, np.float32(0))
    out_dt = s['dt_out']
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    ref_f = D.astype(NP[out_dt]).astype(np.float32).reshape(-1)
    got = np.fromfile(out_path, NP[out_dt]).astype(np.float32).reshape(-1)[:ref_f.size]
    eps = np.float32(s.get('eps', 2e-2))
    atol = eps + eps * np.abs(ref_f)
    bad = np.flatnonzero(np.abs(got - ref_f) > atol)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] MATMUL MISMATCH {bad.size}/{ref_f.size} first@{i} got={got[i]} ref={ref_f[i]}',
          file=sys.stderr)
    return 1

def check(case, chkdir):
    s = REG[case]
    if s['fam'] == 'matmul':
        return check_matmul(case, chkdir)
    if s['fam'] == 'reduce':
        return check_reduce(case, chkdir)
    r, out_dt = ref(case, chkdir)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, out_dt)
    r = r.astype(NP[out_dt]).reshape(-1)
    got = got.reshape(-1)[:r.size]
    if out_dt == I32:
        bad = np.flatnonzero(got != r)
    else:
        eps = np.float32(s.get('eps', 1e-4))
        atol = eps + eps * np.abs(r.astype(np.float32))
        bad = np.flatnonzero(np.abs(got.astype(np.float32) - r.astype(np.float32)) > atol)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] MISMATCH {bad.size}/{r.size}  first@{i} got={got[i]} ref={r[i]}',
          file=sys.stderr)
    return 1

def main():
    if len(sys.argv) < 4:
        print('usage: golden.py gen|check <case> <chkdir>', file=sys.stderr); return 2
    mode, case, chkdir = sys.argv[1], sys.argv[2], sys.argv[3]
    if case not in REG:
        # not registered => no golden for this case (run-only). Signal via 3.
        return 3
    if mode == 'gen':   gen(case, chkdir);   return 0
    if mode == 'check': return check(case, chkdir)
    print('unknown mode', mode, file=sys.stderr); return 2

if __name__ == '__main__':
    sys.exit(main())
