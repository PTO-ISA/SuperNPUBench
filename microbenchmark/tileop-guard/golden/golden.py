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
I16 = 'i16'; U32 = 'u32'; U16 = 'u16'; S8 = 's8'

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
#   TQUANT (pto-spec normative, format-conversion/TQUANT.md): "compute x*multiplier
#   + zero_point, then apply the selected rounding mode"; Sat=1 clamps to S8/U8.
#   → q = clamp(round_RNE(src*mult) + zp, -128, 127). multiplier/zeroPoint are
#   NORMATIVELY MANDATORY (omitted B.IOR defaults 1.0/0 only). golden pins real
#   mult/zp; the emulator IGNORES them → this case is expected to land in
#   PRECISION-FAIL, witnessing model gap gfrun-5 (NOT a golden regression).
#   TDEQUANT: dst = (src - zp) * mult                         (fp32)
REG['tquant']   = R(fam='quant',   M=8, N=256, mult=0.5, zp=1, eps=0)  # real mult/zp per spec
REG['tdequant'] = R(fam='dequant', M=8, N=256, mult=2.0, zp=0, eps=0)

# ---- TCVT (numeric conversion fp32 -> s32). Rounding mode pinned empirically
#      below (round=...). Inputs span both signs with varied fractions so the
#      chosen mode is actually exercised at the .5 boundary. ----
REG['tcvt'] = R(fam='cvt', M=16, N=16, din=F32, dout=I32, round='rne', eps=0)

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
# TMATMUL_MX FP16 pair (no scales): identical math to a plain f16 matmul; the
# convenience overload just omits the scale operand. Golden = A@B, same family.
REG['tmatmul_mx']     = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F32, eps=2e-2)
# TGEMV: D(1,N) = Vec(1,K) @ Mtx(K,N). GEMV is a matmul with M=1 (in_a=vec,
# in_b=mtx). doc arg order TGEMV(Dst,Mtx,Vec) but math source A=Vec, B=Mtx.
REG['tgemv']          = R(fam='matmul', M=1, N=32, K=32, post='none', dt_out=F32, eps=2e-2)

# ---- FIXP convert (matmul then cast to fp16 via fixp::convert) ----
REG['convert'] = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F16, eps=3e-2)

# ---- FIXP B.FPATR matrix post-process (matmul + quant/activation/reduction) ----
# Golden pins the *spec* post-process contract (pto-spec arch/profile/
# matrix-postprocess.asl + matrix-quantization.asl). Shared FP19 carriers
# (must match the demo make_*_quant constants exactly):
#   FP19 16.0 = 0x20C00 ; 8.0 = 0x20800 ; 1.0 = 0x1FC00 ; 0.5 = 0x1F800.
# S8 quant (QF322S8Pre / VQF322S8Pre): offset width 9 (S9 intermediate). Scale
# folded into a single multiplier per spec MatrixSelectedMultiplier; positive ->
# scale, negative under LReLU/PReLU -> slope. Then round+sat S9, +offset, encode.
#
# Tier 1 -- keep_acc reductions: main D published is the *plain* fp32 matmul
# (RowMax/GroupMax/MaxAbs write only the *auxiliary* destinations, which these
# demos do not store). So golden = matmul, post=none. Guards that enabling the
# reduction path does not corrupt D.
REG['rowmax_acc'] = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F32, eps=2e-2)
REG['group_max']  = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F32, eps=2e-2)
REG['chain']      = R(fam='matmul', M=32, N=32, K=32, post='none', dt_out=F32, eps=2e-2)
# Tier 2 -- quantizing post-process: D itself is requantized.
#   scale=16.0 (0x20C00) keeps D*scale in +-40 -> rich, non-saturating S8 spread.
#   offset=5 exercises the S9 offset path. itol=1 absorbs f16-matmul boundary
#   rounding (host f32 accum vs cube accum) without masking a scale/offset bug.
REG['s8_scalar']      = R(fam='mquant', M=32, N=32, K=32, qmode='s8', fp19=0x20C00, off=5, dt_out=S8, itol=1)
REG['scalar_generic'] = R(fam='mquant', M=32, N=32, K=32, qmode='s8', fp19=0x20C00, off=5, dt_out=S8, itol=1)
REG['s8_vector']      = R(fam='mquant', M=32, N=32, K=32, qmode='s8', fp19=0x20C00, off=5, dt_out=S8, itol=1)
REG['vquant_f16']     = R(fam='mquant', M=32, N=32, K=32, qmode='f16', fp19=0x20C00, off=0, dt_out=F16, eps=3e-2)
# Tier 3 -- activation fused with the multiplier. lrelu: pos*scale, neg*slope,
# then S8 quant. prelu: fixp::f16() convert (scale=1.0), pos*1, neg*slope, fp16.
REG['lrelu'] = R(fam='mquant', M=32, N=32, K=32, qmode='s8',  fp19=0x20C00, off=5,
                 relu='lrelu', slope_fp19=0x20800, dt_out=S8, itol=1)
REG['prelu'] = R(fam='mquant', M=32, N=32, K=32, qmode='f16', fp19=0x1FC00, off=0,
                 relu='prelu', slope_fp19=0x1F800, dt_out=F16, eps=3e-2)
# cscale: TMATMUL_ACC with per-row U8 exponent on the initial accumulator C
# (pto-spec cube.asl MatrixInitialAccumulatorValue -> C/2^exp), then A@B on top.
# exp=1 (all rows) -> d = A@B + C/2.
REG['cscale'] = R(fam='cscale', M=32, N=32, K=32, cexp=1, dt_out=F32, eps=2e-2)

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
# MGATHER_CAS: per-element compare-and-swap on GM. observedOld[i]=*(base+off[i])
# (always the pre-swap value); if *(base+off[i])==expected[i] then it is set to
# replacement[i], else unchanged. Injective offsets; ~half lanes hit / half miss.
# Golden checks BOTH observedOld (out.bin) and the post-CAS backing (out_mem.bin).
REG['mgather_cas']  = R(fam='cas', BM=8, BN=1024, OM=8, ON=32, eps=0)

# ---- expand-arith: fused broadcast + binary op. row broadcasts a per-row scalar
#      (src1[i,0], M x 1 source); col broadcasts a per-col scalar (src1[0,j],
#      1 x N source). EXPDIF = exp(src0-src1) (base-e, softmax). Semantics from
#      docs/intrinsics/t{row,col}expand{op}.md. Positive bounded inputs so div
#      has nonzero divisor and expdif's exp(a-b) stays in fp32 range. ----
for op in ['add', 'sub', 'mul', 'div', 'max', 'min', 'expdif']:
    REG['trowexpand' + op] = R(fam='expandarith', op=op, foot='row', M=16, N=16, eps=1e-5)
    REG['tcolexpand' + op] = R(fam='expandarith', op=op, foot='col', M=16, N=16, eps=1e-5)

# TROWEXPAND/TCOLEXPAND (copy-expand). Authoritative semantics (pto-spec normative
# ASL): TROWEXPAND "broadcasts one one-column source bit-for-bit across every valid
# destination column" -> dst[r,c]=src[r,0]; TCOLEXPAND broadcasts a one-row source
# across every valid row -> dst[r,c]=src[0,c]. golden pins the full M x N broadcast.
# The emulator pins the fill extent to the source valid dim (=1), a degenerate
# expand, so these are expected to land in PRECISION-FAIL, witnessing that model
# contract gap (header/impl vs spec). foot=row: src is an M-vector column; foot=col:
# src is an M x N tile whose valid row 0 holds the N-vector.
REG['trowexpand'] = R(fam='copyexpand', foot='row', M=16, N=16, eps=0)
REG['tcolexpand'] = R(fam='copyexpand', foot='col', M=16, N=16, eps=0)

# ---- TROW/TCOL ARGMAX/ARGMIN: index output forced to UINT32; reduce shape ----
#   src distinct per line (perm) so the arg index is unambiguous.
REG['trowargmax'] = R(fam='argreduce', op='max', foot='row', M=16, N=16, eps=0)
REG['trowargmin'] = R(fam='argreduce', op='min', foot='row', M=16, N=16, eps=0)
REG['tcolargmax'] = R(fam='argreduce', op='max', foot='col', M=16, N=16, eps=0)
REG['tcolargmin'] = R(fam='argreduce', op='min', foot='col', M=16, N=16, eps=0)

# ---- TEXTRACT / TINSERT (sub-tile copy at (indexRow,indexCol)) ----
# Authoritative semantics (pto-spec normative ASL): TINSERT "inserts a source Tile
# into a snapshotted OLD destination at encoded row/column offsets" — source0 is the
# persistent old destination (base PRESERVED outside the window), source1 the patch.
# -> dst = base.copy(); dst[OR:OR+SM, OC:OC+SN] = patch. golden pins that. The
# emulator writes a fresh tile with only a partial window and does not preserve base,
# so TINSERT is expected to land in PRECISION-FAIL (model contract gap). TEXTRACT
# stays run-fail (gfrun descriptor contract rejects it); its golden (check_extract)
# is kept READY so it auto-validates once the model implements the extract path.
REG['tinsert'] = R(fam='insert', DM=16, DN=32, SM=8, SN=16, OR=4, OC=8, eps=0)
# TEXTRACT (pto-spec normative ASL): dst[j,i] = src[OR+j, OC+i] (copy the rectangle
# beginning at the encoded row/col offset). READY golden — the case currently
# run-fails (gfrun descriptor contract), so it stays run-fail until the model
# implements the extract path, then check_extract auto-validates.
REG['textract'] = R(fam='extract', SM=16, SN=32, DM=8, DN=16, OR=4, OC=8, eps=0)

# ---- TMRGSORT / TGATHER / TSCATTER (irregular-and-complex). READY goldens: all
#      three currently run-fail (model stubs/gaps); goldens encode the pto-spec
#      normative semantics and auto-validate once the model implements them. ----
#   TMRGSORT: stably merge two sorted single-row streams -> sorted(concat).
#   TGATHER : dst[r,c] = value[idx[r,c], c]  (row-index gather, per column).
#   TSCATTER: dst[idx[r,c], c] = src[r,c]     (row-index scatter; idx injective/col).
REG['tmrgsort'] = R(fam='mrgsort', H=128, W=256, eps=0)
REG['tgather']  = R(fam='tgather', M=16, N=16, eps=0)
REG['tscatter'] = R(fam='tscatter', M=16, N=16, eps=0)

# ---- TFILLPAD (layout/init): copy the VALID source rectangle into dst and fill
#   the physical padding with zero. Authoritative semantics: TFILLPAD.md ("复制
#   有效源区域, 并将绑定标量写入 padding") + the page's worked example (InputTile
#   valid 9x9 inside a 16x16 physical, OutputTile PadValue::Zero); cpu_sim
#   TFillPad.hpp static_assert PadVal==Zero confirms zero-pad only. src is a
#   VR x VC valid region in a physical M x N tile -> dst[i,j] = src[i,j] for
#   i<VR & j<VC, else 0.
REG['tfillpad'] = R(fam='fillpad', M=16, N=16, VR=9, VC=9, eps=0)

# ---- range::Assemble: a destination-side range carrier layered over TLOAD. The
#   carrier only retargets the load; the data is an identity copy of the source
#   GM tile -> out == in. golden = identity (transitive TLOAD coverage made
#   explicit with a numeric check).
REG['range_assemble'] = R(fam='identity', M=4, N=8, dt=F32, eps=0)

# ---- region TileArray + TASSEMBLY: NF row-major fragments (PM x FN) laid out
#   block-major in memory are column-concatenated into a PM x (NF*FN) parent
#   (TileArrayRegionAsm.cpp assembles 4 x 32x16 -> 32x64). golden = concat over
#   columns. The case currently run-fails (region producer path not yet
#   implemented in the model); the golden is READY and auto-validates once it is.
REG['region_tilearray'] = R(fam='regionasm', PM=32, PN=64, FN=16, NF=4, eps=0)

# ---- TTRI: self-generated triangular matrix (no input). pto-spec normative ASL
#   (TTRI.asl): the 1-arg C++ overload omits B.IOR -> diagonal 0 + LOWER
#   orientation; logical element [r,c] is typed 1.0 iff c <= r+diagonal (here
#   c <= r), else 0.0 (exact FP32 0/1 encodings). golden = tril(ones).
REG['ttri'] = R(fam='ttri', M=16, N=16, diag=0, upper=0, eps=0)

# --------------------------------------------------------------------------
NP = {F32: np.float32, I32: np.int32, F16: np.float16,
      I16: np.int16, U32: np.uint32, U16: np.uint16, S8: np.int8}

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
    # --- MGATHER_CAS: host owns backing + byte offsets + expected + replacement.
    #     expected is set so even lanes HIT (== current slot) and odd lanes MISS. ---
    if fam == 'cas':
        BNE = s['BM'] * s['BN']; ONE = s['OM'] * s['ON']
        base = (np.arange(BNE, dtype=np.float32) * np.float32(0.5) + np.float32(1.0))
        idx = ((np.arange(ONE, dtype=np.int64) * 101 + 7) % BNE)   # injective
        off = (idx * 4).astype(np.uint32)
        cur = base[idx]                                            # current slot values
        rep = -(np.arange(ONE, dtype=np.float32) + np.float32(1.0))  # distinct replacements
        exp = cur.copy()                                           # even lanes: exact match -> HIT
        miss = (np.arange(ONE) % 2) == 1
        exp[miss] = cur[miss] + np.float32(1000.0)                 # odd lanes: mismatch -> MISS
        base.tofile(os.path.join(chkdir, 'in_base.bin'))
        off.tofile(os.path.join(chkdir, 'in_off.bin'))
        exp.astype(np.float32).tofile(os.path.join(chkdir, 'in_exp.bin'))
        rep.astype(np.float32).tofile(os.path.join(chkdir, 'in_rep.bin'))
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
    # --- cvt: fp32 source with varied fractional parts, both signs; includes
    #     exact .5 midpoints so the rounding mode is exercised. ---
    if fam == 'cvt':
        M, N = s['M'], s['N']
        n = M * N
        i = np.arange(n, dtype=np.float32)
        # base ramp across [-N/2, +N/2) plus a fractional wobble hitting .5/.25/.75
        base = (i - np.float32(n // 2)).astype(np.float32)
        frac = (np.float32(0.5) * ((i.astype(np.int64) % 4).astype(np.float32) - 1.0))  # {-0.5,0,0.5,1.0}
        a = (base + frac).astype(np.float32)
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
    # --- copy-expand: match the demo's source read. row: demo loads an M x 1
    #     column (M floats) -> dst[r,c]=src[r]. col: demo loads an M x N tile whose
    #     valid row 0 is the N-vector -> dst[r,c]=src[c]. Nonzero values so the
    #     demo's `if(src[i]==0)` fallback is a no-op under RES_CHECK. ---
    if fam == 'copyexpand':
        M, N = s['M'], s['N']
        if s['foot'] == 'row':
            col = ((np.arange(M, dtype=np.float32) + np.float32(1.0)) * np.float32(0.5))
            col.tofile(os.path.join(chkdir, 'in_a.bin'))
        else:
            row0 = ((np.arange(N, dtype=np.float32) + np.float32(1.0)) * np.float32(0.25))
            full = np.tile(row0, (M, 1)).reshape(-1).astype(np.float32)   # every row = row0
            full.tofile(os.path.join(chkdir, 'in_a.bin'))
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
    # --- TMRGSORT: two ascending-sorted single-row streams (disjoint interleave) ---
    if fam == 'mrgsort':
        H = s['H']
        a = (2.0 * np.arange(H, dtype=np.float32))          # 0,2,4,... sorted
        b = (2.0 * np.arange(H, dtype=np.float32) + 1.0)    # 1,3,5,... sorted, disjoint
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        b.tofile(os.path.join(chkdir, 'in_b.bin'))
        return
    # --- TGATHER: value MxN + per-coordinate ROW index in [0,M) (nontrivial) ---
    if fam == 'tgather':
        M, N = s['M'], s['N']
        r = np.arange(M)[:, None]; c = np.arange(N)[None, :]
        val = ((np.arange(M * N, dtype=np.float32) + 1.0) * np.float32(0.1))
        idx = ((7 * r + 3 * c) % M).astype(np.int32)        # picks a row per (r,c)
        val.tofile(os.path.join(chkdir, 'in_a.bin'))
        idx.reshape(-1).tofile(os.path.join(chkdir, 'in_idx.bin'))
        return
    # --- TSCATTER: src MxN + per-column PERMUTATION row index (injective/col) ---
    if fam == 'tscatter':
        M, N = s['M'], s['N']
        r = np.arange(M)[:, None]; c = np.arange(N)[None, :]
        src = ((np.arange(M * N, dtype=np.float32) + 1.0) * np.float32(0.1))
        idx = ((r + c) % M).astype(np.int32)                # each column is a perm of [0,M)
        src.tofile(os.path.join(chkdir, 'in_a.bin'))
        idx.reshape(-1).tofile(os.path.join(chkdir, 'in_idx.bin'))
        return
    # --- TFILLPAD: full physical M x N nonzero source (golden reads only the
    #     VR x VC valid rectangle; the rest is what the op must zero out). ---
    if fam == 'fillpad':
        M, N = s['M'], s['N']
        a = seq_f32(M * N, 1.0, 0.13, 3)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- region TileArray: NF contiguous PM x FN blocks, each block distinct so
    #     a mis-ordered assembly is caught. ---
    if fam == 'regionasm':
        PM, PN = s['PM'], s['PN']
        a = seq_f32(PM * PN, 1.0, 0.05, 7)
        a.tofile(os.path.join(chkdir, 'in_a.bin'))
        return
    # --- TTRI: self-generated, no input files ---
    if fam == 'ttri':
        return
    # --- cscale: matmul + fp32 accumulator C (scaled per-row) ---
    if fam == 'cscale':
        M, N, K = s['M'], s['N'], s['K']
        ii = np.arange(M * K, dtype=np.float32)
        jj = np.arange(K * N, dtype=np.float32)
        A = (np.sin(ii * np.float32(0.13)) * np.float32(0.8)).astype(np.float16)
        B = (np.cos(jj * np.float32(0.21)) * np.float32(0.8)).astype(np.float16)
        kk = np.arange(M * N, dtype=np.float32)
        C = (np.sin(kk * np.float32(0.07)) * np.float32(0.5)).astype(np.float32)
        A.tofile(os.path.join(chkdir, 'in_a.bin'))
        B.tofile(os.path.join(chkdir, 'in_b.bin'))
        C.tofile(os.path.join(chkdir, 'in_c.bin'))
        return
    # --- matmul + fixp post-process families (f16 A/B, optional f32 C / bias) ---
    if fam in ('matmul', 'mquant'):
        M, N, K = s['M'], s['N'], s['K']
        ii = np.arange(M * K, dtype=np.float32)
        jj = np.arange(K * N, dtype=np.float32)
        A = (np.sin(ii * np.float32(0.13)) * np.float32(0.8)).astype(np.float16)
        B = (np.cos(jj * np.float32(0.21)) * np.float32(0.8)).astype(np.float16)
        A.tofile(os.path.join(chkdir, 'in_a.bin'))
        B.tofile(os.path.join(chkdir, 'in_b.bin'))
        if s.get('post') == 'acc':
            kk = np.arange(M * N, dtype=np.float32)
            C = (np.sin(kk * np.float32(0.07)) * np.float32(0.5)).astype(np.float32)
            C.tofile(os.path.join(chkdir, 'in_c.bin'))
        if s.get('post') == 'bias':
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

def fp19_to_f32(v):
    """Decode a 19-bit FP19 carrier (pto-spec arch/data-types/fp19.asl):
    1 sign [18], 8-bit bias-127 exponent [17:10], 10-bit fraction [9:0]."""
    v &= 0x7ffff
    sign = (v >> 18) & 1
    exp = (v >> 10) & 0xff
    frac = v & 0x3ff
    if exp == 0:
        mag = float(frac) * (2.0 ** -136)      # subnormal (unused for our normals)
    else:
        mag = (1.0 + frac / 1024.0) * (2.0 ** (exp - 127))
    return -mag if sign else mag

def check_mquant(case, chkdir):
    """B.FPATR matrix post-process golden (pto-spec matrix-postprocess.asl).
    D = A@B (fp32); a single multiplier folds scale+activation:
      positive -> scale ; negative under LReLU/PReLU -> slope (replaces scale).
    S8 path: round+sat S9 intermediate, +offset, encode S8 (RNE).
    F16 path: (D*mult) encoded to fp16 (RNE)."""
    s = REG[case]; M, N, K = s['M'], s['N'], s['K']
    A = np.fromfile(os.path.join(chkdir, 'in_a.bin'), np.float16).reshape(M, K).astype(np.float32)
    B = np.fromfile(os.path.join(chkdir, 'in_b.bin'), np.float16).reshape(K, N).astype(np.float32)
    D = (A @ B).astype(np.float64)
    scale = fp19_to_f32(s['fp19'])
    relu = s.get('relu', 'none')
    if relu in ('lrelu', 'prelu'):
        slope = fp19_to_f32(s['slope_fp19'])
        # spec source_negative: value<0 -> slope; value>=0 (incl +0) -> scale.
        mult = np.where(D < 0.0, slope, scale)
    else:
        mult = scale
    act = D * mult
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    if s['qmode'] == 's8':
        off = s.get('off', 0)
        inter = np.clip(np.rint(act), -256, 255)          # S9 round+sat (RNE)
        ref = np.clip(inter + off, -128, 127).astype(np.int64)
        got = np.fromfile(out_path, np.int8).astype(np.int64).reshape(-1)[:ref.size]
        itol = int(s.get('itol', 0))
        d = np.abs(got - ref.reshape(-1))
        bad = np.flatnonzero(d > itol)
        if bad.size == 0:
            return 0
        i = int(bad[0])
        print(f'[{case}] MQUANT-S8 MISMATCH {bad.size}/{ref.size} first@{i} '
              f'got={got[i]} ref={ref.reshape(-1)[i]} (scale={scale} off={off} tol={itol})',
              file=sys.stderr)
        return 1
    # f16 floating encode
    ref = act.astype(np.float16).astype(np.float32).reshape(-1)
    got = np.fromfile(out_path, np.float16).astype(np.float32).reshape(-1)[:ref.size]
    eps = np.float32(s.get('eps', 3e-2))
    atol = eps + eps * np.abs(ref)
    bad = np.flatnonzero(np.abs(got - ref) > atol)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] MQUANT-F16 MISMATCH {bad.size}/{ref.size} first@{i} '
          f'got={got[i]} ref={ref[i]} (scale={scale})', file=sys.stderr)
    return 1

def check_cscale(case, chkdir):
    """TMATMUL_ACC + cscale golden (pto-spec cube.asl / matrix-postprocess.asl):
    d = A@B + C / 2^exp (per-row exponent; here uniform exp)."""
    s = REG[case]; M, N, K = s['M'], s['N'], s['K']
    A = np.fromfile(os.path.join(chkdir, 'in_a.bin'), np.float16).reshape(M, K).astype(np.float32)
    B = np.fromfile(os.path.join(chkdir, 'in_b.bin'), np.float16).reshape(K, N).astype(np.float32)
    C = np.fromfile(os.path.join(chkdir, 'in_c.bin'), np.float32).reshape(M, N)
    D = (A @ B) + C / np.float32(2.0 ** s['cexp'])
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    ref = D.astype(np.float32).reshape(-1)
    got = np.fromfile(out_path, np.float32).reshape(-1)[:ref.size]
    eps = np.float32(s.get('eps', 2e-2))
    atol = eps + eps * np.abs(ref)
    bad = np.flatnonzero(np.abs(got - ref) > atol)
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] CSCALE MISMATCH {bad.size}/{ref.size} first@{i} got={got[i]} ref={ref[i]}',
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

def check_cvt(case, chkdir):
    s = REG[case]
    a = np.fromfile(os.path.join(chkdir, 'in_a.bin'), NP[s['din']]).astype(np.float64)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, s['dout']).astype(np.int64)
    mode = s['round']
    if mode == 'rne':
        ref = np.rint(a)                 # round-half-to-even
    elif mode == 'rtz':
        ref = np.trunc(a)                # toward zero
    elif mode == 'floor':
        ref = np.floor(a)
    elif mode == 'ceil':
        ref = np.ceil(a)
    else:
        print(f'[{case}] unknown round mode {mode}', file=sys.stderr); return 2
    ref = ref.astype(np.int64)
    n = min(got.size, ref.size)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] CVT MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]} src={a[i]} mode={mode}',
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
    # Authoritative semantics (pto-spec normative ASL contract): TLOG = same-type
    # NATURAL logarithm (ln) — docs/tile/.../transcendental/TLOG.md states "natural
    # logarithm" 3x + log(1)=+0/log(0)=-inf/log(neg)=NaN; SuperNPUBench tlog.md and
    # tileop-usage TLOG.md agree. golden pins the DESIGN intent (ln). The emulator
    # currently computes log2 (measured TLOG(4.25)=2.0875) → this case is expected to
    # land in PRECISION-FAIL, witnessing model gap gfrun-6 (NOT a golden regression).
    # TEXP is base-e (matches spec + emulator).
    fn = {'exp': np.exp, 'log': np.log, 'recip': lambda x: 1.0 / x,
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

def check_cas(case, chkdir):
    s = REG[case]; ONE = s['OM'] * s['ON']
    base0 = np.fromfile(os.path.join(chkdir, 'in_base.bin'), np.float32)
    off = np.fromfile(os.path.join(chkdir, 'in_off.bin'), np.uint32).astype(np.int64)
    exp = np.fromfile(os.path.join(chkdir, 'in_exp.bin'), np.float32)
    rep = np.fromfile(os.path.join(chkdir, 'in_rep.bin'), np.float32)
    old_path = os.path.join(chkdir, 'out.bin')
    mem_path = os.path.join(chkdir, 'out_mem.bin')
    if not os.path.exists(old_path) or not os.path.exists(mem_path):
        print(f'[{case}] MISSING out.bin/out_mem.bin', file=sys.stderr); return 1
    got_old = np.fromfile(old_path, np.float32)[:ONE]
    got_mem = np.fromfile(mem_path, np.float32)
    slot = off // 4
    cur = base0[slot]                                   # pre-swap values (injective offsets)
    hit = (cur == exp)                                  # CAS succeeds iff current == expected
    # 1) observedOld is always the pre-swap value
    ref_old = cur
    bad = np.flatnonzero(got_old[:ONE] != ref_old)
    if bad.size:
        i = int(bad[0])
        print(f'[{case}] CAS observedOld MISMATCH {bad.size}/{ONE} first@{i} '
              f'got={got_old[i]} ref={ref_old[i]}', file=sys.stderr); return 1
    # 2) backing after CAS: hit lanes -> replacement, miss lanes -> unchanged
    ref_mem = base0.copy()
    ref_mem[slot] = np.where(hit, rep, cur)
    bad = np.flatnonzero(got_mem != ref_mem)
    if bad.size:
        i = int(bad[0])
        print(f'[{case}] CAS backing MISMATCH {bad.size}/{ref_mem.size} first@slot{i} '
              f'got={got_mem[i]} ref={ref_mem[i]}', file=sys.stderr); return 1
    return 0

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

def check_mrgsort(case, chkdir):
    s = REG[case]; H, W = s['H'], s['W']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32)
    b = np_read(os.path.join(chkdir, 'in_b.bin'), F32)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32)
    ref = np.sort(np.concatenate([a[:H], b[:H]]))          # stable ascending merge
    n = min(got.size, ref.size, W)
    bad = np.flatnonzero(got[:n] != ref[:n])
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] MRGSORT MISMATCH {bad.size}/{n} first@{i} got={got[i]} ref={ref[i]}',
          file=sys.stderr)
    return 1

def check_tgather(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']
    val = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    idx = np_read(os.path.join(chkdir, 'in_idx.bin'), I32).reshape(M, N)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(M, N)
    c = np.arange(N)[None, :]
    ref = val[idx, c]                                      # dst[r,c] = val[idx[r,c], c]
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] TGATHER MISMATCH {bad.size} first@{i}', file=sys.stderr)
    return 1

def check_tscatter(case, chkdir):
    s = REG[case]; M, N = s['M'], s['N']
    src = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    idx = np_read(os.path.join(chkdir, 'in_idx.bin'), I32).reshape(M, N)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(M, N)
    ref = np.zeros((M, N), np.float32)
    c = np.broadcast_to(np.arange(N)[None, :], (M, N))
    ref[idx, c] = src                                     # dst[idx[r,c], c] = src[r,c]
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] TSCATTER MISMATCH {bad.size} first@{i}', file=sys.stderr)
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
    raw = np_read(os.path.join(chkdir, 'in_a.bin'), F32)
    if s['foot'] == 'row':
        col = raw[:M].reshape(M, 1)                             # M x 1 broadcast column
        ref = np.repeat(col, N, axis=1)                        # dst[r,c] = src[r]
    else:
        row0 = raw.reshape(M, N)[0:1, :]                        # 1 x N broadcast row (valid row 0)
        ref = np.repeat(row0, M, axis=0)                       # dst[r,c] = src[c]
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

def check_fillpad(case, chkdir):
    s = REG[case]; M, N, VR, VC = s['M'], s['N'], s['VR'], s['VC']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32).reshape(M, N)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(M, N)
    ref = np.zeros((M, N), np.float32)
    ref[:VR, :VC] = a[:VR, :VC]                       # copy valid rect, zero the pad
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] FILLPAD MISMATCH {bad.size}/{M*N} first@{i} '
          f'got={got.reshape(-1)[i]} ref={ref.reshape(-1)[i]}', file=sys.stderr)
    return 1

def check_regionasm(case, chkdir):
    s = REG[case]; PM, PN, FN, NF = s['PM'], s['PN'], s['FN'], s['NF']
    a = np_read(os.path.join(chkdir, 'in_a.bin'), F32)
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(PM, PN)
    frags = [a[k * PM * FN:(k + 1) * PM * FN].reshape(PM, FN) for k in range(NF)]
    ref = np.concatenate(frags, axis=1)               # PM x (NF*FN) column concat
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] REGIONASM MISMATCH {bad.size}/{PM*PN} first@{i} '
          f'got={got.reshape(-1)[i]} ref={ref.reshape(-1)[i]}', file=sys.stderr)
    return 1

def check_ttri(case, chkdir):
    s = REG[case]; M, N, diag = s['M'], s['N'], s['diag']
    out_path = os.path.join(chkdir, 'out.bin')
    if not os.path.exists(out_path):
        print(f'[{case}] MISSING out.bin', file=sys.stderr); return 1
    got = np_read(out_path, F32).reshape(M, N)
    r = np.arange(M)[:, None]; c = np.arange(N)[None, :]
    mask = (c >= r + diag) if s['upper'] else (c <= r + diag)   # ASL: lower iff c<=r+diag
    ref = mask.astype(np.float32)
    bad = np.flatnonzero(got.reshape(-1) != ref.reshape(-1))
    if bad.size == 0:
        return 0
    i = int(bad[0])
    print(f'[{case}] TTRI MISMATCH {bad.size}/{M*N} first@{i} '
          f'got={got.reshape(-1)[i]} ref={ref.reshape(-1)[i]}', file=sys.stderr)
    return 1

def check(case, chkdir):
    s = REG[case]
    if s['fam'] == 'matmul':
        return check_matmul(case, chkdir)
    if s['fam'] == 'mquant':
        return check_mquant(case, chkdir)
    if s['fam'] == 'cscale':
        return check_cscale(case, chkdir)
    if s['fam'] == 'fillpad':
        return check_fillpad(case, chkdir)
    if s['fam'] == 'regionasm':
        return check_regionasm(case, chkdir)
    if s['fam'] == 'ttri':
        return check_ttri(case, chkdir)
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
    if s['fam'] == 'mrgsort':
        return check_mrgsort(case, chkdir)
    if s['fam'] == 'tgather':
        return check_tgather(case, chkdir)
    if s['fam'] == 'tscatter':
        return check_tscatter(case, chkdir)
    if s['fam'] == 'reduce':
        return check_reduce(case, chkdir)
    if s['fam'] == 'iota':
        return check_iota(case, chkdir)
    if s['fam'] == 'transcend':
        return check_transcend(case, chkdir)
    if s['fam'] == 'expandarith':
        return check_expandarith(case, chkdir)
    if s['fam'] == 'cvt':
        return check_cvt(case, chkdir)
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
    if s['fam'] == 'cas':
        return check_cas(case, chkdir)
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
