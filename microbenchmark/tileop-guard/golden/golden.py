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
I16 = 'i16'; U32 = 'u32'; U16 = 'u16'

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
# reinterpret_tile: bitcast fp32->int32, clear sign bit (TANDS 0x7fffffff), then a
# native fp32 op re-tags the backing; host reads back |x| as fp32. golden = abs.
REG['reinterpret_tile'] = R(fam='unary', op='abs', M=16, N=16, dt=F32, eps=0)
# reinterpret_tcmp: same bitcast but consumed by TCMP -> run-fail witness (emulator
# TCMP handler rejects reinterpret-view sources). NO golden (crashes before store);
# would be where(int_bits(a)>int_bits(b), tru, prior) once the model accepts views.
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

# ---- TCMP/TCMPS produce a packed predicate consumed by TSEL. End-to-end golden:
#      out = where(a <cmp> b|s, tru, prior). tsel chains TCMP<LT> -> TSEL. ----
REG['tsel']  = R(fam='cmpsel',   mode='lt', M=16, N=16, dt=I32, eps=0)
REG['tcmp']  = R(fam='cmpsel',   mode='gt', M=16, N=16, dt=I32, eps=0)
REG['tcmps'] = R(fam='cmpsel_s', mode='gt', M=16, N=16, dt=I32, scalar=0, eps=0)
# ---- TSELS: masked select between a tile source and a scalar; out = where(a>b, src, SVAL) ----
REG['tsels'] = R(fam='tsels', mode='gt', M=16, N=16, dt=I32, scalar=777, eps=0)

# ---- SFU transcendental (elementwise; positive bounded domain so log/sqrt/
#      recip/rsqrt stay in-domain and exp does not overflow). SFU is a hardware
#      approximation, so tolerance is relative (calibrated per op, see check). ----
REG['texp']   = R(fam='transcend', op='exp',   M=16, N=16, dt=F32, eps=1e-5)
REG['tlog']   = R(fam='transcend', op='log',   M=16, N=16, dt=F32, eps=1e-5)
REG['trecip'] = R(fam='transcend', op='recip', M=16, N=16, dt=F32, eps=1e-5)
REG['tsqrt']  = R(fam='transcend', op='sqrt',  M=16, N=16, dt=F32, eps=1e-5)
REG['trsqrt'] = R(fam='transcend', op='rsqrt', M=16, N=16, dt=F32, eps=1e-5)

# ---- TQUANT / TDEQUANT (explicit float multiplier + integer zeroPoint) ----
#   TQUANT:  q = clamp(round_RNE(src*mult) + zp, -128, 127)  (S8, saturate)
#   TDEQUANT: dst = (src - zp) * mult                         (fp32)
REG['tquant']   = R(fam='quant',   M=8, N=256, mult=1.0, zp=0, eps=0)  # emu ignores mult/zp
REG['tdequant'] = R(fam='dequant', M=8, N=256, mult=2.0, zp=0, eps=0)

# ---- TPART* : elementwise binary over the common valid region (full-valid here
#      => plain elementwise). "PART" = partial-valid-region, not segmented. ----
for name, op in [('tpartadd','add'),('tpartmul','mul'),('tpartmax','max'),('tpartmin','min')]:
    REG[name] = R(fam='binary', op=op, M=16, N=16, dt=F32, eps=1e-4)

# ---- SFU layout / sort (deterministic reorder; exact golden) ----
REG['ttrans']  = R(fam='transpose', M=16, N=16, dt=F32, eps=0)
REG['tconcat'] = R(fam='concat', M=16, N=8, dt=F32, eps=0)   # [MxN | MxN] -> Mx2N
REG['tsort']   = R(fam='sort', M=32, N=32, dt=F32, eps=1e-6, desc=0)

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

# ---- TCI (create index / "vci"): self-generated iota, NO input ----
#   ascending col k = start+k, descending = start-k; ValidRow=1 (TCI.md).
REG['tci']      = R(fam='iota', M=1, N=64, dt=I32, start=0,   desc=0, vc=64, eps=0)
REG['tci_desc'] = R(fam='iota', M=1, N=64, dt=I32, start=100, desc=1, vc=64, eps=0)
REG['tci_s16']  = R(fam='iota', M=1, N=64, dt=I16, start=0,   desc=0, vc=64, eps=0)
REG['tci_u32']  = R(fam='iota', M=1, N=64, dt=U32, start=0,   desc=0, vc=64, eps=0)
REG['tci_u16']  = R(fam='iota', M=1, N=64, dt=U16, start=0,   desc=0, vc=64, eps=0)

# ---- MGATHER / MSCATTER / MGATHER_MASK (GM byte-offset addressing) ----
REG['mgather']      = R(fam='gather',      BM=8, BN=1024, OM=8, ON=32, eps=0)
REG['mscatter']     = R(fam='scatter',     BM=8, BN=1024, OM=8, ON=32, eps=0)
REG['mgather_mask'] = R(fam='gather_mask', BM=8, BN=1024, OM=8, ON=32, eps=0)
REG['mscatter_mask'] = R(fam='scatter_mask', BM=8, BN=1024, OM=8, ON=32, eps=0)

# ---- expand-arith: fused broadcast + binary op. row broadcasts a per-row scalar
#      (src1[i,0], M x 1 source); col broadcasts a per-col scalar (src1[0,j],
#      1 x N source). EXPDIF = exp(src0-src1) (base-e, softmax). Semantics from
#      docs/intrinsics/t{row,col}expand{op}.md. Positive bounded inputs so div
#      has nonzero divisor and expdif's exp(a-b) stays in fp32 range. ----
for op in ['add', 'sub', 'mul', 'div', 'max', 'min', 'expdif']:
    REG['trowexpand' + op] = R(fam='expandarith', op=op, foot='row', M=16, N=16, eps=1e-5)
    REG['tcolexpand' + op] = R(fam='expandarith', op=op, foot='col', M=16, N=16, eps=1e-5)

# NOTE: TROWEXPAND/TCOLEXPAND (copy-expand) left run-only — the correct broadcast
# source now runs, but the model's fill width/height is pinned to the source
# valid dim (=1), a degenerate expand not worth asserting. See the demo comments.

# ---- TROW/TCOL ARGMAX/ARGMIN: index output forced to UINT32; reduce shape ----
#   src distinct per line (perm) so the arg index is unambiguous.
REG['trowargmax'] = R(fam='argreduce', op='max', foot='row', M=16, N=16, eps=0)
REG['trowargmin'] = R(fam='argreduce', op='min', foot='row', M=16, N=16, eps=0)
REG['tcolargmax'] = R(fam='argreduce', op='max', foot='col', M=16, N=16, eps=0)
REG['tcolargmin'] = R(fam='argreduce', op='min', foot='col', M=16, N=16, eps=0)

# ---- TEXTRACT / TINSERT (sub-tile copy at (indexRow,indexCol)) ----
# Signatures corrected from the header (TEXTRACT/TINSERT(dst,src,indexRow,indexCol)),
# but left run-only (NO golden REG): TEXTRACT is rejected by gfrun's descriptor
# contract (same TEPL family as TIMG2COL); TINSERT runs but the model writes a
# fresh tile with only a partial (4x8) window of the patch — semantics are not
# pinned by docs, so we do not assert a golden. check_extract/check_insert kept
# below for when the model/doc contract is settled.

# --------------------------------------------------------------------------
NP = {F32: np.float32, I32: np.int32, F16: np.float16,
      I16: np.int16, U32: np.uint32, U16: np.uint16}

def np_read(path, dt):
    return np.fromfile(path, dtype=NP[dt])

def gen(case, chkdir):
    s = REG[case]; fam = s['fam']; op = s.get('op', '')
    os.makedirs(chkdir, exist_ok=True)
    # --- TCI: self-generated iota, no input files ---
    if fam == 'iota':
        return
    # --- transcendental: positive bounded input in [0.5, 8.0] (log/sqrt/recip/
    #     rsqrt domain-safe; exp(8)=2981 stays in fp32 range). ---
    if fam == 'transcend':
        n = s['M'] * s['N']
        i = np.arange(n, dtype=np.float32)
        a = (np.float32(0.5) + np.float32(7.5) * (np.float32(0.5) *
             (np.float32(1.0) + np.sin(i * np.float32(0.37))))).astype(np.float32)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- MGATHER / MSCATTER / MGATHER_MASK: host owns base + byte offsets ---
    if fam in ('gather', 'scatter', 'gather_mask', 'scatter_mask'):
        BNE = s['BM'] * s['BN']; ONE = s['OM'] * s['ON']
        base = (np.arange(BNE, dtype=np.float32) * np.float32(0.5) + np.float32(1.0))
        idx = ((np.arange(ONE, dtype=np.int64) * 101 + 7) % BNE)   # injective (101 coprime 8192)
        off = (idx * 4).astype(np.uint32)                          # U32 byte displacement
        base.tofile(os.path.join(chkdir, 'in_base.bin'))
        off.tofile(os.path.join(chkdir, 'in_off.bin'))
        if fam in ('scatter', 'scatter_mask'):
            src = -(np.arange(ONE, dtype=np.float32) + np.float32(1.0))   # distinct from base
            src.tofile(os.path.join(chkdir, 'in_src.bin'))
        if fam in ('gather_mask', 'scatter_mask'):
            mask = ((np.arange(ONE, dtype=np.int64) % 3) != 0).astype(np.uint8)
            mask.tofile(os.path.join(chkdir, 'in_mask.bin'))
        return
    # --- TCMP/TCMPS end-to-end: compare operands + prior + true ---
    if fam in ('cmpsel', 'cmpsel_s'):
        n = s['M'] * s['N']
        a = seq_i32(n, 1, 2, 0)
        prior = seq_i32(n, -1, -1, 4)
        tru = seq_i32(n, 100, 3, 9)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        if fam == 'cmpsel':
            b = seq_i32(n, 3, 1, 9)                # tile-tile compare operand
            b.tofile(os.path.join(chkdir, 'in_b.bin'))
            prior.tofile(os.path.join(chkdir, 'in_c.bin'))
            tru.tofile(os.path.join(chkdir, 'in_d.bin'))
        else:                                       # tile-scalar
            prior.tofile(os.path.join(chkdir, 'in_b.bin'))
            tru.tofile(os.path.join(chkdir, 'in_c.bin'))
        return
    # --- quant: fp32 source spanning +-256 so round+zp saturates at S8 edges ---
    if fam == 'quant':
        M, N = s['M'], s['N']
        n = M * N
        i = np.arange(n, dtype=np.float32)
        a = (np.float32(512.0) * (i / np.float32(n)) - np.float32(256.0) +
             np.float32(0.5) * np.sin(i * np.float32(0.9))).astype(np.float32)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- dequant: int8 source spanning full [-128,127] ---
    if fam == 'dequant':
        M, N = s['M'], s['N']
        n = M * N
        a = (((np.arange(n, dtype=np.int64) * 63) % 256) - 128).astype(np.int8)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- transpose: single MxN source ---
    if fam == 'transpose':
        M, N = s['M'], s['N']
        a = seq_f32(M * N, 1.0, 0.1, 5)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- concat: two MxN sources (distinct ranges) ---
    if fam == 'concat':
        M, N = s['M'], s['N']
        a = seq_f32(M * N, 1.0, 0.1, 1)
        b = seq_f32(M * N, 100.0, 0.1, 2)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        b.tofile(os.path.join(chkdir, 'in_b.bin'))
        return
    # --- sort: per-row distinct values (a per-row permutation so ordering is
    #     unambiguous); host owns the MxN source. ---
    if fam == 'sort':
        M, N = s['M'], s['N']
        r = np.arange(M)[:, None]; c = np.arange(N)[None, :]
        # distinct within a row and shuffled: value = ((5*c+r) % N) + r*N, float.
        a = (((5 * c + r) % N) + r * N).astype(np.float32)
        a.reshape(-1).tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- expand-arith: src0 MxN + broadcast src1. row: src1 is M per-row scalars;
    #     col: src1 is a full MxN whose row 0 holds the N per-col scalars. Inputs
    #     positive-bounded [0.5,4.5] (div divisor nonzero; expdif exp(a-b) safe). ---
    if fam == 'expandarith':
        M, N = s['M'], s['N']
        ii = np.arange(M * N, dtype=np.float32)
        a = (np.float32(0.5) + np.float32(2.0) *
             (np.float32(1.0) + np.sin(ii * np.float32(0.29)))).astype(np.float32)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        if s['foot'] == 'row':
            jj = np.arange(M, dtype=np.float32)
            b = (np.float32(0.5) + np.float32(2.0) *
                 (np.float32(1.0) + np.cos(jj * np.float32(0.41)))).astype(np.float32)
        else:
            jj = np.arange(N, dtype=np.float32)
            row0 = (np.float32(0.5) + np.float32(2.0) *
                    (np.float32(1.0) + np.cos(jj * np.float32(0.41)))).astype(np.float32)
            b = np.tile(row0, (M, 1)).reshape(-1)          # full MxN, every row = row0
        b.tofile(os.path.join(chkdir, 'in_b.bin'))
        return
    # --- copy-expand: full MxN source; row reads col 0 (src[r,0]), col reads
    #     row 0 (src[0,c]). The valid extent (=source valid dims) drives fill. ---
    if fam == 'copyexpand':
        M, N = s['M'], s['N']
        full = seq_f32(M * N, 1.0, 0.1, 7).reshape(M, N)
        full.reshape(-1).tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- ARGMAX/ARGMIN: src distinct per row & per col (double perm) ---
    if fam == 'argreduce':
        M, N = s['M'], s['N']
        r = np.arange(M)[:, None]; c = np.arange(N)[None, :]
        # (7*c + 3*r) % N is a per-row permutation (7 coprime 16); add r to also
        # separate columns so per-col reductions have a unique arg too.
        a = ((7 * c + 3 * r) % N + r * N).astype(np.float32)
        a.reshape(-1).tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- TSELS: compare operands (a,b) + tile source ---
    if fam == 'tsels':
        n = s['M'] * s['N']
        a = seq_i32(n, 1, 2, 0)
        b = seq_i32(n, 3, 1, 9)
        src = seq_i32(n, 100, 3, 5)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        b.tofile(os.path.join(chkdir, 'in_b.bin'))
        src.tofile(os.path.join(chkdir, 'in_c.bin'))
        return
    # --- TEXTRACT: host owns the big src tile ---
    if fam == 'extract':
        SNE = s['SM'] * s['SN']
        src = seq_f32(SNE, 1.0, 0.1, 3)
        src.tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- TINSERT: host owns base + patch ---
    if fam == 'insert':
        DNE = s['DM'] * s['DN']; SNE = s['SM'] * s['SN']
        base = seq_f32(DNE, 1.0, 0.1, 3)
        patch = -(seq_f32(SNE, 2.0, 0.07, 9))
        base.tofile(os.path.join(chkdir, 'in_a.bin'))
        patch.tofile(os.path.join(chkdir, 'in_b.bin'))
        return
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
    if fam == 'identity':
        return a, dt
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

def check_iota(case, chkdir):
    s = REG[case]; dt = s['dt']; vc = s['vc']; start = s['start']; desc = s['desc']
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, dt)[:vc]
    k = np.arange(vc, dtype=np.int64)
    seq = (start - k) if desc else (start + k)      # TCI.md: asc start+k / desc start-k
    ref = seq.astype(NP[dt])                          # astype wraps modulo element width
    bad = np.flatnonzero(got != ref)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] TCI MISMATCH {bad.size}/{vc} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_quant(case, chkdir):
    s = REG[case]
    a = np.fromfile(os.path.join(chkdir, 'in_a.bin'), np.float32).astype(np.float64)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np.fromfile(out_path, np.int8).astype(np.int64)
    q = np.rint(a * s['mult']) + s['zp']           # np.rint = round-half-to-even = RNE
    ref = np.clip(q, -128, 127).astype(np.int64)
    n = min(got.size, ref.size)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] QUANT MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]} src={a[i]}',
          file=sys.stderr)
    return 1

def check_dequant(case, chkdir):
    s = REG[case]
    a = np.fromfile(os.path.join(chkdir, 'in_a.bin'), np.int8).astype(np.float64)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np.fromfile(out_path, np.float32).astype(np.float64)
    ref = (a - s['zp']) * s['mult']
    n = min(got.size, ref.size)
    eps = 1e-4 + 1e-4 * np.abs(ref[:n])
    bad = np.flatnonzero(np.abs(got[:n] - ref[:n]) > eps)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] DEQUANT MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_transpose(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(N, M)
    ref = a.T
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] TRANSPOSE MISMATCH {bad.size} first@{i}', file=sys.stderr)
    return 1

def check_concat(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    b = np_read(os.path.join(chkdir, 'in_b.bin'), F32).reshape(M, N)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(M, 2 * N)
    ref = np.concatenate([a, b], axis=1)
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] CONCAT MISMATCH {bad.size} first@{i} got={got.reshape(-1)[i]} ref={ref.reshape(-1)[i]}',
          file=sys.stderr)
    return 1

def check_sort(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(M, N)
    ref = np.sort(a, axis=1)
    if s.get('desc', 0):
        ref = ref[:, ::-1]
    eps = np.float32(s.get('eps', 1e-6))
    bad = np.flatnonzero(np.abs(got - ref).reshape(-1) > eps)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] SORT MISMATCH {bad.size} first@{i} got={got.reshape(-1)[i]} ref={ref.reshape(-1)[i]}',
          file=sys.stderr)
    return 1

def check_expandarith(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']; op = s['op']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N).astype(np.float64)
    braw = np_read(os.path.join(chkdir, 'in_b.bin'), F32).astype(np.float64)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(M, N).astype(np.float64)
    if s['foot'] == 'row':
        bc = braw[:M].reshape(M, 1)                 # per-row scalar, broadcast over cols
    else:
        bc = braw.reshape(M, N)[0, :].reshape(1, N)  # per-col scalar (row 0), over rows
    fn = {'add': lambda x, y: x + y, 'sub': lambda x, y: x - y,
          'mul': lambda x, y: x * y, 'div': lambda x, y: x / y,
          'max': np.maximum, 'min': np.minimum,
          'expdif': lambda x, y: np.exp(x - y)}[op]   # EXPDIF base-e (softmax)
    ref = fn(a, bc)
    eps = float(s.get('eps', 1e-4))
    atol = eps + eps * np.abs(ref)
    err = np.abs(got - ref)
    print(f'[{case}] expandarith max_abs={err.max():.3e} '
          f'max_rel={(err/(np.abs(ref)+1e-30)).max():.3e} eps={eps:.1e}', file=sys.stderr)
    bad = np.flatnonzero(err.reshape(-1) > atol.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] EXPANDARITH MISMATCH {bad.size}/{ref.size} first@{i} '
          f'got={got.reshape(-1)[i]} ref={ref.reshape(-1)[i]}', file=sys.stderr)
    return 1

def check_transcend(case, chkdir):
    s = REG[case]
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32).astype(np.float64)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).astype(np.float64)
    # TLOG is base-2 (verified: TLOG(4.25)=2.0875=log2(4.25), not ln); TEXP is
    # base-e (verified PASS against np.exp). Asymmetric bases confirmed empirically.
    fn = {'exp': np.exp, 'log': np.log2, 'recip': lambda x: 1.0 / x,
          'sqrt': np.sqrt, 'rsqrt': lambda x: 1.0 / np.sqrt(x)}[s['op']]
    ref = fn(a)
    n = min(got.size, ref.size)
    got, ref = got[:n], ref[:n]
    eps = float(s.get('eps', 3e-3))
    atol = eps + eps * np.abs(ref)                    # relative + small absolute
    err = np.abs(got - ref)
    bad = np.flatnonzero(err > atol)
    rel = err / (np.abs(ref) + 1e-30)
    print(f'[{case}] transcend max_abs={err.max():.3e} max_rel={rel.max():.3e} eps={eps:.1e}',
          file=sys.stderr)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] TRANSCEND MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_gather(case, chkdir):
    base = np.fromfile(os.path.join(chkdir, 'in_base.bin'), np.float32)
    off = np.fromfile(os.path.join(chkdir, 'in_off.bin'), np.uint32).astype(np.int64)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np.fromfile(out_path, np.float32)
    ref = base[off // 4]                              # addr = base + byte displacement
    n = min(got.size, ref.size)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] GATHER MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_scatter(case, chkdir):
    base = np.fromfile(os.path.join(chkdir, 'in_base.bin'), np.float32)
    src = np.fromfile(os.path.join(chkdir, 'in_src.bin'), np.float32)
    off = np.fromfile(os.path.join(chkdir, 'in_off.bin'), np.uint32).astype(np.int64)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np.fromfile(out_path, np.float32)
    ref = base.copy()
    ref[off // 4] = src                              # injective offsets -> deterministic
    n = min(got.size, ref.size)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] SCATTER MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_gather_mask(case, chkdir):
    base = np.fromfile(os.path.join(chkdir, 'in_base.bin'), np.float32)
    off = np.fromfile(os.path.join(chkdir, 'in_off.bin'), np.uint32).astype(np.int64)
    mask = np.fromfile(os.path.join(chkdir, 'in_mask.bin'), np.uint8)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np.fromfile(out_path, np.float32)
    ref = np.where(mask == 1, base[off // 4], np.float32(0.0)).astype(np.float32)
    n = min(got.size, ref.size)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] GATHER_MASK MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_scatter_mask(case, chkdir):
    base = np.fromfile(os.path.join(chkdir, 'in_base.bin'), np.float32)
    src = np.fromfile(os.path.join(chkdir, 'in_src.bin'), np.float32)
    off = np.fromfile(os.path.join(chkdir, 'in_off.bin'), np.uint32).astype(np.int64)
    mask = np.fromfile(os.path.join(chkdir, 'in_mask.bin'), np.uint8)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np.fromfile(out_path, np.float32)
    ref = base.copy()
    m = mask == 1
    ref[(off // 4)[m]] = src[m]                       # injective offsets -> deterministic
    n = min(got.size, ref.size)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] SCATTER_MASK MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_extract(case, chkdir):
    s = REG[case]
    src = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(s['SM'], s['SN'])
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(s['DM'], s['DN'])
    ref = src[s['OR']:s['OR'] + s['DM'], s['OC']:s['OC'] + s['DN']]
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] EXTRACT MISMATCH {bad.size} first@{i}', file=sys.stderr)
    return 1

def check_insert(case, chkdir):
    s = REG[case]
    base = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(s['DM'], s['DN'])
    patch = np_read(os.path.join(chkdir, 'in_b.bin'), F32).reshape(s['SM'], s['SN'])
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(s['DM'], s['DN'])
    ref = base.copy()
    ref[s['OR']:s['OR'] + s['SM'], s['OC']:s['OC'] + s['SN']] = patch
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] INSERT MISMATCH {bad.size} first@{i}', file=sys.stderr)
    return 1

def _cmp_mask(mode, a, rhs):
    if mode == 'gt': return a > rhs
    if mode == 'lt': return a < rhs
    if mode == 'ge': return a >= rhs
    if mode == 'le': return a <= rhs
    if mode == 'eq': return a == rhs
    if mode == 'ne': return a != rhs
    raise KeyError(mode)

def check_cmpsel(case, chkdir):
    s = REG[case]
    a = np_read(os.path.join(chkdir, 'in_a.bin'), I32)
    if s['fam'] == 'cmpsel':
        b = np_read(os.path.join(chkdir, 'in_b.bin'), I32)
        prior = np_read(os.path.join(chkdir, 'in_c.bin'), I32)
        tru = np_read(os.path.join(chkdir, 'in_d.bin'), I32)
        mask = _cmp_mask(s['mode'], a, b)
    else:
        prior = np_read(os.path.join(chkdir, 'in_b.bin'), I32)
        tru = np_read(os.path.join(chkdir, 'in_c.bin'), I32)
        mask = _cmp_mask(s['mode'], a, np.int32(s['scalar']))
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, I32)
    ref = np.where(mask, tru, prior).astype(np.int32)
    n = min(got.size, ref.size)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] CMPSEL MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_copyexpand(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(M, N)
    src = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    if s['foot'] == 'row':
        ref = np.repeat(src[:, 0:1], N, axis=1)                # dst[r,c] = src[r,0]
    else:
        ref = np.repeat(src[0:1, :], M, axis=0)                # dst[r,c] = src[0,c]
    eps = np.float32(s.get('eps', 1e-4))
    atol = eps + eps * np.abs(ref)
    bad = np.flatnonzero(np.abs(got - ref) > atol)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] COPYEXPAND MISMATCH {bad.size} first@{i}', file=sys.stderr)
    return 1

def check_argreduce(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    o = np_read(out_path, U32)
    fn = np.argmax if s['op'] == 'max' else np.argmin
    if s['foot'] == 'row':
        ref = fn(a, axis=1).astype(np.uint32)      # per-row -> length M
        got = o.reshape(M, N)[:, 0]                 # index at out[r*N+0]
    else:
        ref = fn(a, axis=0).astype(np.uint32)       # per-col -> length N
        got = o.reshape(M, N)[0, :]                 # index at out[0*N+c]
    bad = np.flatnonzero(got.astype(np.int64) != ref.astype(np.int64))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] ARGREDUCE MISMATCH {bad.size} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_tsels(case, chkdir):
    s = REG[case]
    a = np_read(os.path.join(chkdir, 'in_a.bin'), I32)
    b = np_read(os.path.join(chkdir, 'in_b.bin'), I32)
    src = np_read(os.path.join(chkdir, 'in_c.bin'), I32)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, I32)
    mask = _cmp_mask(s['mode'], a, b)
    ref = np.where(mask, src, np.int32(s['scalar'])).astype(np.int32)   # dst = mask ? src : SVAL
    n = min(got.size, ref.size)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] TSELS MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check(case, chkdir):
    s = REG[case]
    if s['fam'] == 'matmul':
        return check_matmul(case, chkdir)
    if s['fam'] == 'tsels':
        return check_tsels(case, chkdir)
    if s['fam'] == 'argreduce':
        return check_argreduce(case, chkdir)
    if s['fam'] == 'copyexpand':
        return check_copyexpand(case, chkdir)
    if s['fam'] in ('cmpsel', 'cmpsel_s'):
        return check_cmpsel(case, chkdir)
    if s['fam'] == 'extract':
        return check_extract(case, chkdir)
    if s['fam'] == 'insert':
        return check_insert(case, chkdir)
    if s['fam'] == 'reduce':
        return check_reduce(case, chkdir)
    if s['fam'] == 'iota':
        return check_iota(case, chkdir)
    if s['fam'] == 'transcend':
        return check_transcend(case, chkdir)
    if s['fam'] == 'expandarith':
        return check_expandarith(case, chkdir)
    if s['fam'] == 'quant':
        return check_quant(case, chkdir)
    if s['fam'] == 'dequant':
        return check_dequant(case, chkdir)
    if s['fam'] == 'transpose':
        return check_transpose(case, chkdir)
    if s['fam'] == 'concat':
        return check_concat(case, chkdir)
    if s['fam'] == 'sort':
        return check_sort(case, chkdir)
    if s['fam'] == 'gather':
        return check_gather(case, chkdir)
    if s['fam'] == 'scatter':
        return check_scatter(case, chkdir)
    if s['fam'] == 'gather_mask':
        return check_gather_mask(case, chkdir)
    if s['fam'] == 'scatter_mask':
        return check_scatter_mask(case, chkdir)
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
