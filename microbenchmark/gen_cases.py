#!/usr/bin/env python3
# Microbench case generator.
#
# Emits one .cpp per (opcode x dtype x tile-size) case plus a compile.all list
# for each family (cube / vector / memory). Intrinsic naming follows the
# PTO 0.57.1 reference; sources are structural and may not compile until
# pto_tileop.hpp aligns to these names.
#
# Usage: python3 gen_cases.py
from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.abspath(__file__))

DTYPE = {
    "fp16": "__half",
    "bf16": "__bf16",
    "fp32": "float",
    "i8":   "int8_t",
    "i16":  "int16_t",
    "i32":  "int32_t",
}

def load_fixp_modes() -> list[str]:
    text = open(os.path.join(ROOT, "fixp", "compile.all"), errors="replace").read()
    match = re.search(r"^MODES=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise RuntimeError("cannot find fixp MODES array")
    return re.sub(r"#.*", "", match.group(1)).split()


FIXP_MODES = load_fixp_modes()

# Formal corpus coverage is limited to modes with a deterministic independent
# golden profile. Benchmark-only modes may remain in fixp/compile.all without
# silently expanding the ASL execution contract.
ASL_FIXP_MODES = (
    "keep_acc keep_acc_relu f16 f16_relu bf16 bf16_relu "
    "s_reqs8 s_deqf16 s_shifts16 s_qf_s4 s_qf_s16 s_qf_s8 s_qf_hif8 s_qf_fp8 "
    "s_qf_f32 s_qf_f16 s_qf_bf16 s_qs_bf16 "
    "v_reqs8 v_deqf16 v_shifts16 v_qf_s4 v_qf_s16 v_qf_s8 v_qf_hif8 "
    "v_qf_f16 v_qf_bf16 v_qf_fp8 v_qf_f32 v_qs_bf16 "
    "s8_relu s8_lrelu v_s8_relu f16_prelu s8_prelu "
    "rowmax rowmax_init groupmax_8 groupmax_16 groupmax_128 rowgroup_maxabs "
    "f16_groupmax s8_rowmax bias acc mx mxbias mxacc gemv gemv_bias gemv_acc "
    "gemv_mx gemv_mx_bias gemv_mx_acc bias_s8 acc_s8 mx_s8 gemv_s8 gemv_mx_s8 "
    "shared s8_shared vqf_s8_prelu legacy3"
).split()

# ---- case spec ----
# kind selects the bench template + call lambda.
@dataclass
class Case:
    op: str
    kind: str
    dtypes: tuple[str, ...]
    size: tuple[int, int]          # (M, N); cube uses (M,N,K) via cube_kind
    cube: bool = False             # if True, size is (M,N,K)


# ============ vector (TEPL) cases ============
V = []  # noqa
M16 = (16, 16)

# mode 0 tile-tile binary arithmetic/bitwise
for op in ["TADD", "TSUB", "TMUL", "TDIV", "TREm".replace("REm", "REM"),
           "TAND", "TOR", "TXOR", "TSHL", "TSHR", "TMAX", "TMIN", "TCMP"]:
    if op in ("TAND", "TOR", "TXOR", "TSHL", "TSHR"):
        dt = ("i16", "i32")
    elif op == "TREM":
        dt = ("fp16", "fp32", "i32")  # re-test full dtype set
    elif op == "TCMP":
        dt = ("fp16", "fp32", "i32")
    else:
        dt = ("fp16", "fp32", "i16", "i32")
    V.append(Case(op, "binary", dt, M16))

# mode 0 unary
for op, dt in [
    ("TABS", ("fp16", "fp32", "bf16")),
    ("TNOT", ("i16", "i32")),
    ("TNEG", ("fp16", "fp32", "i16", "i32")),
    ("TEXP", ("fp16", "fp32")),
    ("TLOG", ("fp16", "fp32")),
    ("TRECIP", ("fp16", "fp32")),
    ("TSQRT", ("fp16", "fp32")),
    ("TRSQRT", ("fp16", "fp32")),
    ("TRELU", ("fp16", "fp32")),
    ("TCVT", ("fp16", "fp32")),
]:
    V.append(Case(op, "unary", dt, M16))

# mode 0 ternary
for op in ["TSEL"]:
    V.append(Case(op, "ternary", ("fp16", "fp32"), M16))

# mode 0 partial-valid
for op in ["TPARTADD", "TPARTMUL", "TPARTMAX", "TPARTMIN"]:
    V.append(Case(op, "binary", ("fp16", "fp32"), M16))

# mode 1 tile-scalar (1 tile + scalar)
# arithmetic scalar ops: float dtypes
for op in ["TADDS", "TSUBS", "TMULS", "TDIVS", "TREMS", "TMAXS", "TMINS", "TCMPS"]:
    V.append(Case(op, "scalar", ("fp16", "fp32"), M16))

# bitwise/shift scalar ops: integer dtypes (ISA v0.58 only allows integer)
for op in ["TANDS", "TORS", "TXORS", "TSHLS", "TSHRS"]:
    V.append(Case(op, "scalar", ("i16", "i32"), M16))

# mode 1 scalar broadcast
V.append(Case("TEXPANDS", "scalarbcast", ("fp16", "fp32"), M16))

# mode 3 contiguous integer sequence generation
# TCI supports signed/unsigned 16-bit and 32-bit integer tiles. The generator's
# current dtype table exposes the two signed variants.
V.append(Case("TCI", "sequence", ("i16", "i32"), (1, 64)))

# mode 2 reduce
for op in ["TROWSUM", "TROWMAX", "TROWMIN", "TROWPROD",
           "TCOLSUM", "TCOLMAX", "TCOLMIN", "TCOLPROD"]:
    V.append(Case(op, "reduce", ("fp16", "fp32", "i32"), M16))

# mode 2 argmax / argmin
for op in ["TROWARGMAX", "TROWARGMIN", "TCOLARGMAX", "TCOLARGMIN"]:
    V.append(Case(op, "reduce", ("fp16", "fp32"), M16))

# mode 2 expand (1 src broadcast)
for op in ["TROWEXPAND", "TCOLEXPAND"]:
    V.append(Case(op, "unary", ("fp16", "fp32"), M16))

# mode 2 expand (2 src: data M×N + row/col scalar vector). Per tileop-usage doc:
#   row expand arith: src1 = M×1 ; col expand arith: src1 = 1×N
for op in ["TROWEXPANDADD", "TROWEXPANDSUB", "TROWEXPANDMUL", "TROWEXPANDDIV",
           "TROWEXPANDMAX", "TROWEXPANDMIN", "TROWEXPANDEXPDIF"]:
    V.append(Case(op, "expand_row", ("fp16", "fp32"), M16))
for op in ["TCOLEXPANDADD", "TCOLEXPANDSUB", "TCOLEXPANDMUL", "TCOLEXPANDDIV",
           "TCOLEXPANDMAX", "TCOLEXPANDMIN", "TCOLEXPANDEXPDIF"]:
    V.append(Case(op, "expand_col", ("fp16", "fp32"), M16))

# mode 3 complex
# TCONCAT: dst = [src0 | src1] along cols (src0=M×N0, src1=M×N1, dst=M×(N0+N1))
V.append(Case("TCONCAT", "concat", ("fp16", "fp32"), M16))
V.append(Case("THISTOGRAM", "hist", ("i16", "i32"), M16))

# Opcodes the toolchain does not yet expose (or needs special layout like TCVT's
# NZ requirement). Kept here as a skip list; re-enable when pto_tileop.hpp aligns.
VECTOR_SKIP = {
    # These operations were removed from PTO ISA v0.58.5.
    "TPARTADD", "TPARTMUL", "TPARTMAX", "TPARTMIN",
    # need fractal/NZ layout (32-byte align) — plain RowMajor Mx1/Nx1 output fails:
    "TROWMAX", "TROWMIN", "TROWPROD", "TROWSUM", "TROWARGMAX", "TROWARGMIN",
    "TCOLSUM", "TCOLMAX", "TCOLMIN", "TCOLPROD", "TCOLARGMAX", "TCOLARGMIN",
    # The checked-out API source has TSELECT, but the installed main compiler
    # headers do not expose it yet. Keep it visible in coverage.json.
    "TSEL",
    # The installed assembler rejects the B.DATR encodings emitted by these
    # headers. Keep the operations visible as unsupported compiler coverage.
    "TCMP", "TCMPS", "THISTOGRAM",
}
# BF16 TABS reaches the frontend but crashes during instruction selection in
# the installed compiler. FP16/FP32 remain active.
for case in V:
    if case.op == "TABS":
        case.dtypes = tuple(dt for dt in case.dtypes if dt != "bf16")
V = [c for c in V if c.op not in VECTOR_SKIP]


# ============ memory (TLSU) cases ============
ME = []
for op, kind, dt, sz in [
    ("TLOAD", "load", ("fp16", "fp32", "i32"), M16),
    ("TLOAD", "load", ("fp16", "fp32"), (32, 32)),
    ("TSTORE", "store", ("fp16", "fp32", "i32"), M16),
    ("MGATHER", "gather", ("fp16", "fp32", "i32"), M16),
    ("MSCATTER", "scatter", ("fp16", "fp32", "i32"), M16),
    ("MGATHER_MASK", "gather_mask", ("fp16", "fp32"), M16),
    ("MSCATTER_MASK", "scatter_mask", ("fp16", "fp32"), M16),
]:
    ME.append(Case(op, kind, dt, sz))

MEMORY_SKIP = {"MGATHER_MASK", "MSCATTER_MASK"}
ME = [case for case in ME if case.op not in MEMORY_SKIP]


# ============ matrix (TMA/CUBE direct operations) cases ============
# These benchmark shapes intentionally keep each input at or below 8 KiB;
# this is a workload choice, not the architectural 256 KiB-per-PE capacity.
# TGEMV* are not yet exposed by the toolchain; only TMATMUL* land.
C = []

def csize(dt):
    if dt == "fp32":
        return (32, 32, 32)   # 32x32x4B = 4KB
    if dt == "fp16":
        return (64, 64, 64)   # 64x64x2B = 8KB
    if dt == "i8":
        return (64, 64, 64)   # 64x64x1B = 4KB
    return (64, 64, 64)

def cube(op, kind, dt, sz=None):
    if sz is None:
        sz = csize(dt)
    C.append(Case(op, kind, (dt,), sz, cube=True))

cube("TMATMUL", "matmul", "fp32", (32, 32, 32))
cube("TMATMUL", "matmul", "fp16", (16, 32, 32))
cube("TMATMUL", "matmul", "fp16", (32, 64, 64))
cube("TMATMUL", "matmul", "bf16", (32, 64, 64))
cube("TMATMUL", "matmul", "i8", (32, 64, 64))
cube("TMATMUL_ACC", "matmul_acc", "fp32", (32, 32, 32))
cube("TMATMUL_ACC", "matmul_acc", "fp16", (32, 64, 64))
cube("TMATMUL_ACC", "matmul_acc", "bf16", (32, 64, 64))
cube("TMATMUL_BIAS", "matmul_bias", "fp32", (32, 32, 32))
cube("TMATMUL_BIAS", "matmul_bias", "fp16", (32, 64, 64))
cube("TMATMUL_BIAS", "matmul_bias", "bf16", (32, 64, 64))

# TODO: re-enable when toolchain exposes TGEMV/TGEMV_ACC/TGEMV_BIAS/TGEMV_MX.
# cube("TGEMV", "gemv", "fp16")
# cube("TGEMV", "gemv", "fp32")
# cube("TGEMV_ACC", "gemv_acc", "fp16")
# cube("TGEMV_BIAS", "gemv_bias", "fp16")
# cube("TGEMV_MX", "gemv_mx", "fp16")


# ============ scalar (GPR ALU) cases ============
# Plain C + volatile; compiler emits scalar micro-ISA (misa_g/l/f). ~27 opcodes.
#   cat=bin : binary op, both thr and lat
#   cat=un  : unary op wrapped as op(x+y) to vary input, both thr and lat
#   cat=ld  : pure load (return y), thr only
#   cat=st  : pure store, thr only
#   cat=cv  : IN->OUT conversion, thr only
SDTYPE = {"i32": "int32_t", "i64": "int64_t", "fp32": "float", "f64": "double"}
UT = {"i32": "uint32_t", "i64": "uint64_t"}

# (op, cat, dtypes, lambda_tpl)  lambda_tpl uses {T} (return cast) and {ut} (unsigned cast)
SCALAR_OPS = [
    ("add", "bin", ("i32", "i64", "fp32", "f64"), "[](auto x,auto y){{return x+y;}}"),
    ("sub", "bin", ("i32", "i64", "fp32", "f64"), "[](auto x,auto y){{return x-y;}}"),
    ("mul", "bin", ("i32", "i64", "fp32", "f64"), "[](auto x,auto y){{return x*y;}}"),
    ("div", "bin", ("i32", "i64", "fp32", "f64"), "[](auto x,auto y){{return x/y;}}"),
    ("and", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x & ({ut})y);}}"),
    ("or",  "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x | ({ut})y);}}"),
    ("xor", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x ^ ({ut})y);}}"),
    ("sll", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x << (y & 31));}}"),
    ("srl", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x >> (y & 31));}}"),
    ("sra", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(x >> (y & 31));}}"),
    ("slt", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(x < y);}}"),
    ("max", "bin", ("i32", "i64", "fp32", "f64"), "[](auto x,auto y){{return x<y?y:x;}}"),
    ("min", "bin", ("i32", "i64", "fp32", "f64"), "[](auto x,auto y){{return x<y?x:y;}}"),
    ("mod", "bin", ("i32", "i64"), "[](auto x,auto y){{return x%y;}}"),
    ("abs", "un", ("i32", "fp32", "f64"), "[](auto x,auto y){{auto t=x+y; return t<0?-t:t;}}"),
    ("neg", "un", ("i32", "i64", "fp32", "f64"), "[](auto x,auto y){{return -(x+y);}}"),
    ("not", "un", ("i32", "i64"), "[](auto x,auto y){{return ({T})(~({ut})(x+y));}}"),
    ("popc", "un", ("i32", "i64"), "[](auto x,auto y){{return ({T})__builtin_popcountll((unsigned long long)(x+y));}}"),
    ("clz", "un", ("i32", "i64"), "[](auto x,auto y){{return ({T})__builtin_clzll((unsigned long long)(x+y));}}"),
    ("sqrt", "un", ("fp32", "f64"), "[](auto x,auto y){{return ({T})std::sqrt((double)(x+y));}}"),
    ("ld", "ld", ("i32", "i64", "fp32", "f64"), "[](auto x,auto y){{return y;}}"),
]
# (op, cat, [(in,out), ...])
SCALAR_CV = [
    ("i2f", "cv", [("i32", "fp32"), ("i32", "f64")]),
    ("f2i", "cv", [("fp32", "i32"), ("f64", "i32")]),
    ("f2f_widen", "cv", [("fp32", "f64")]),
    ("f2f_narrow", "cv", [("f64", "fp32")]),
]
SCALAR_ST = [("st", "st", ("i32", "i64", "fp32", "f64"))]


# ============ emission ============
def lower(op: str) -> str:
    return op.lower()

def case_name(c: Case, dt: str) -> str:
    base = lower(c.op)
    if c.cube:
        m, n, k = c.size
        return f"{base}_{dt}_{m}x{n}x{k}"
    m, n = c.size
    return f"{base}_{dt}_{m}x{n}"


def vector_reference(c: Case, dt: str) -> tuple[str, int]:
    """Return C++ scalar oracle and the number of valid destination elements."""
    op = c.op.removeprefix("TPART") if c.op.startswith("TPART") else c.op.removeprefix("T")
    m, n = c.size
    count = m * n
    cast = DTYPE[dt]
    is_float = dt.startswith("fp") or dt == "bf16"

    def binary(lhs: str, rhs: str, opname: str = op) -> str:
        table = {"ADD": f"{lhs}+{rhs}", "SUB": f"{lhs}-{rhs}",
                 "MUL": f"{lhs}*{rhs}", "DIV": f"{lhs}/{rhs}",
                 "AND": f"{lhs}&{rhs}", "OR": f"{lhs}|{rhs}",
                 "XOR": f"{lhs}^{rhs}", "SHL": f"{lhs}<<{rhs}",
                 "SHR": f"{lhs}>>{rhs}",
                 "MAX": f"({lhs}>{rhs}?{lhs}:{rhs})",
                 "MIN": f"({lhs}<{rhs}?{lhs}:{rhs})"}
        if opname in ("REM", "REMS"):
            return f"std::fmod((double){lhs},(double){rhs})" if is_float else f"{lhs}%{rhs}"
        return table[opname.removesuffix("S")]

    if c.kind == "binary":
        expr = binary("a[i]", "b[i]")
        code = f"for (int i=0;i<M*N;++i) ref[i]=({cast})({expr});"
    elif c.kind in ("expand_row", "expand_col"):
        rhs = "b[i/N]" if c.kind == "expand_row" else "b[i%N]"
        opname = op.replace("ROWEXPAND", "").replace("COLEXPAND", "")
        if opname == "EXPDIF":
            expr = f"std::exp((double)a[i]-(double){rhs})"
        else:
            expr = binary("a[i]", rhs, opname)
        code = f"for (int i=0;i<M*N;++i) ref[i]=({cast})({expr});"
    elif c.kind == "concat":
        k = 32 // {"fp16": 2, "fp32": 4}[dt]
        count = m * 2 * k
        code = (f"constexpr int K={k}; for(int r=0;r<M;++r) for(int j=0;j<K;++j) "
                "{ ref[r*2*K+j]=a[r*K+j]; ref[r*2*K+K+j]=b[r*K+j]; }")
    elif c.op == "TROWEXPAND":
        code = "for(int i=0;i<M*N;++i) ref[i]=a[i/N];"
    elif c.op == "TCOLEXPAND":
        code = "for(int i=0;i<M*N;++i) ref[i]=a[i%N];"
    elif c.kind == "unary":
        exprs = {"ABS": "std::fabs((double)a[i])", "NOT": "~a[i]", "NEG": "-a[i]",
                 "EXP": "std::exp((double)a[i])", "LOG": "std::log((double)a[i])",
                 "RECIP": "1.0/(double)a[i]", "SQRT": "std::sqrt((double)a[i])",
                 "RSQRT": "1.0/std::sqrt((double)a[i])",
                 "RELU": "a[i]>0?a[i]:0", "CVT": "a[i]"}
        code = f"for(int i=0;i<M*N;++i) ref[i]=({cast})({exprs[op]});"
    elif c.kind == "scalar":
        expr = binary("a[i]", "s", op)
        code = f"for(int i=0;i<M*N;++i) ref[i]=({cast})({expr});"
    elif c.kind == "scalarbcast":
        code = "for(int i=0;i<M*N;++i) ref[i]=s;"
    elif c.kind == "sequence":
        code = f"for(int i=0;i<M*N;++i) ref[i]=({cast})(s+i);"
    else:
        code = "for(int i=0;i<M*N;++i) ref[i]=c[i]; // no independent oracle"
    return code, count


def emit_vector(c: Case, dt: str) -> str:
    ct = DTYPE[dt]
    m, n = c.size
    op = c.op
    # concat dst can be up to M*2K (fp16 K=16 -> 32 cols); size arrays for the worst case
    arr = m * 32 if c.kind == "concat" else m * n
    head = f'''#include "vector_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated by gen_cases.py
// {op} ({c.kind}) {dt} {c.size[0]}x{c.size[1]}
int main() {{
    constexpr int M = {m}, N = {n};
    {ct} a[{arr}], b[{arr}], d[{arr}], c[{arr}];
    fill_const(a, {arr}, ({ct})2); fill_const(b, {arr}, ({ct})1);
    fill_const(d, {arr}, ({ct})3); zero(c, {arr});
    BENCHSTART;
'''
    ref_code, ref_count = vector_reference(c, dt)
    prep = f"    fill_const(b, {arr}, ({ct})3);\n" if c.kind == "concat" else ""
    head = head.replace("    BENCHSTART;\n", prep + "    BENCHSTART;\n")
    tail = ("    BENCHEND;\n"
            "#ifdef CROSS_MODEL_CORPUS\n"
            f"    publish_cross_model_result(c, {ref_count});\n"
            "#endif\n"
            "#ifdef RES_CHECK\n"
            f"    {ct} ref[{arr}]; zero(ref, {arr}); {ref_code}\n"
            f"    return verify(c,ref,{ref_count},({ct})verify_epsilon<{ct}>(),"
            f"({ct})verify_epsilon<{ct}>()) ? 0 : 1;\n"
            "#else\n    return 0;\n#endif\n}\n")
    if c.kind == "binary":
        body = f"    bench_binary<{ct},M,N>(c,a,b,[](auto& dst,auto& s0,auto& s1){{ {op}(dst,s0,s1); }});\n"
    elif c.kind == "expand_row":
        body = f"    bench_expand_row<{ct},M,N>(c,a,b,[](auto& dst,auto& s0,auto& s1){{ {op}(dst,s0,s1); }});\n"
    elif c.kind == "expand_col":
        body = f"    bench_expand_col<{ct},M,N>(c,a,b,[](auto& dst,auto& s0,auto& s1){{ {op}(dst,s0,s1); }});\n"
    elif c.kind == "concat":
        # src0=M×K, src1=M×K, dst=M×2K (K=32/sizeof(D)); arrays sized for max (fp16 K=16->dst 32 cols)
        body = f"    bench_concat<{ct},M>(c,a,b,[](auto& dst,auto& s0,auto& s1){{ {op}(dst,s0,s1); }});\n"
    elif c.op == "TROWEXPAND":
        body = f"    bench_expand_copy_row<{ct},M,N>(c,a,[](auto& dst,auto& s){{ {op}(dst,s); }});\n"
    elif c.op == "TCOLEXPAND":
        body = f"    bench_expand_copy_col<{ct},M,N>(c,a,[](auto& dst,auto& s){{ {op}(dst,s); }});\n"
    elif c.kind == "unary":
        body = f"    bench_unary<{ct},M,N>(c,a,[](auto& dst,auto& s){{ {op}(dst,s); }});\n"
    elif c.kind == "ternary":
        body = f"    bench_select<{ct},M,N>(c,a,b,[](auto& dst,auto& cond,auto& s0,auto& s1){{ TSELECT(dst,cond,s0,s1); }});\n"
    elif c.kind == "reduce":
        body = f"    bench_reduce<{ct},M,N>(c,a,[](auto& dst,auto& s){{ {op}(dst,s); }});\n"
    elif c.kind == "scalar":
        body = f"    {ct} s = ({ct})0.5;\n    bench_scalar<{ct},M,N>(c,a,s,[](auto& dst,auto& s0,auto& sc){{ {op}(dst,s0,sc); }});\n"
    elif c.kind == "scalar3":
        # TSELS signature is (dst, src0, scalar, src1).
        call = f"{op}(dst,s0,s1,sc)" if c.op == "TSELS" else f"{op}(dst,s0,s1,sc)"
        body = f"    {ct} s = ({ct})0.5;\n    bench_scalar3<{ct},M,N>(c,a,b,s,[](auto& dst,auto& s0,auto& s1,auto& sc){{ {call}; }});\n"
    elif c.kind == "scalarbcast":
        body = f"    {ct} s = ({ct})0.5;\n    bench_scalar_bcast<{ct},M,N>(c,s,[](auto& dst,auto& sc){{ {op}(dst,sc); }});\n"
    elif c.kind == "sequence":
        body = f"    {ct} s = ({ct})7;\n    bench_scalar_bcast<{ct},M,N>(c,s,[](auto& dst,auto& sc){{ {op}(dst,sc); }});\n"
    elif c.kind == "gather":
        body = f"    bench_gather<{ct},M,N>(c,a,b,[](auto& dst,auto& s,auto& idx){{ {op}(dst,s,idx); }});\n"
    elif c.kind == "hist":
        body = f"    bench_hist<{ct},M,N>(c,a,b,0,[](auto& dst,auto& s,auto& idx,auto b){{ {op}(dst,s,idx,b); }});\n"
    else:
        body = f"    // unhandled kind {c.kind}\n"
    return head + body + tail


def emit_memory(c: Case, dt: str) -> str:
    ct = DTYPE[dt]
    m, n = c.size
    op = c.op
    head = f'''#include "memory_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated by gen_cases.py
// {op} ({c.kind}) {dt} {c.size[0]}x{c.size[1]}
int main() {{
    constexpr int M = {m}, N = {n};
    {ct} a[M*N], c[M*N];
    int32_t idx[M*N]; uint16_t mask[M*N];
    fill_const(a, M*N, ({ct})2); fill_idx(idx, M*N); fill_const(mask, M*N, (uint16_t)1); zero(c, M*N);
    for (int i=0;i<M*N;++i) idx[i] *= sizeof({ct}); // gather/scatter offsets are bytes
    BENCHSTART;
'''
    if c.kind in ("load", "store", "mov", "gather", "gather_mask"):
        ref = f"for(int i=0;i<M*N;++i) ref[i]=a[idx[i]/sizeof({ct})];" if "gather" in c.kind else \
              "for(int i=0;i<M*N;++i) ref[i]=a[i];"
    else:
        ref = f"for(int i=0;i<M*N;++i) ref[idx[i]/sizeof({ct})]=a[i];"
    tail = ("    BENCHEND;\n"
            "#ifdef CROSS_MODEL_CORPUS\n"
            "    publish_cross_model_result(c, M*N);\n"
            "#endif\n"
            "#ifdef RES_CHECK\n"
            f"    {ct} ref[M*N]; zero(ref,M*N); {ref}\n"
            f"    return verify(c,ref,M*N,({ct})verify_epsilon<{ct}>(),"
            f"({ct})verify_epsilon<{ct}>()) ? 0 : 1;\n"
            "#else\n    return 0;\n#endif\n}\n")
    if c.kind == "load":
        body = f"    bench_load<{ct},M,N>(c,a);\n"
    elif c.kind == "store":
        body = f"    bench_load<{ct},M,N>(c,a);  // load then store\n"
    elif c.kind == "mov":
        body = f"    bench_mov<{ct},M,N>(c,a);\n"
    elif c.kind == "gather":
        body = f"    bench_gather<{ct},M,N>(c,a,idx);\n"
    elif c.kind == "scatter":
        body = f"    bench_scatter<{ct},M,N>(c,a,idx);\n"
    elif c.kind == "gather_mask":
        body = f"    bench_gather_mask<{ct},M,N>(c,a,idx,mask);\n"
    elif c.kind == "scatter_mask":
        body = f"    bench_scatter_mask<{ct},M,N>(c,a,idx,mask);\n"
    else:
        body = f"    // unhandled kind {c.kind}\n"
    return head + body + tail


def emit_cube(c: Case, dt: str) -> str:
    ct = DTYPE[dt]
    acc = "int32_t" if dt == "i8" else "float"
    m, n, k = c.size
    op = c.op
    declarations = (f"    static {ct} a[M*K], b[K*N];\n"
                    f"    static {acc} bias[N], initial[M*N], c[M*N], ref[M*N];")
    value = "1" if dt == "i8" else "0.25"
    init = ("    fill_const(c, M*N, (float)1); zero(ref, M*N);"
            if dt == "bf16" else
            f"    fill_const(a, M*K, ({ct}){value}); fill_const(b, K*N, ({ct}){value});\n"
            f"    fill_const(bias, N, ({acc})1); fill_const(initial, M*N, ({acc})1);\n"
            "    zero(c, M*N); zero(ref, M*N);")
    head = f'''#include "cube_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated by gen_cases.py
// {op} ({c.kind}) {dt} {m}x{n}x{k}
int main() {{
    constexpr int M = {m}, N = {n}, K = {k};
{declarations}
{init}
    BENCHSTART;
'''
    publish = "#ifdef CROSS_MODEL_CORPUS\n    publish_cross_model_result(c, M*N);\n#endif\n"
    tail = (f"    BENCHEND;\n{publish}    return verify(c,ref,M*N,cube_epsilon<{acc}>()) ? 0 : 1;\n}}\n"
            if dt == "bf16" else
            f"    BENCHEND;\n    reference_matmul<{ct},{acc},M,N,K>(ref,a,b);\n{publish}    return verify(c,ref,M*N,cube_epsilon<{acc}>()) ? 0 : 1;\n}}\n")
    if c.kind == "matmul":
        body = f"    bench_matmul<{ct},M,N,K>(c,a,b);\n"
    elif c.kind == "matmul_acc":
        body = f"    bench_matmul_acc<{ct},M,N,K>(c,initial,a,b);\n"
        tail = (f"    BENCHEND;\n{publish}    return verify(c,ref,M*N,cube_epsilon<{acc}>()) ? 0 : 1;\n}}\n"
                if dt == "bf16" else
                f"    BENCHEND;\n    reference_matmul<{ct},{acc},M,N,K>(ref,a,b,initial);\n{publish}    return verify(c,ref,M*N,cube_epsilon<{acc}>()) ? 0 : 1;\n}}\n")
    elif c.kind == "matmul_bias":
        body = f"    bench_matmul_bias<{ct},M,N,K>(c,a,b,bias);\n"
        tail = (f"    BENCHEND;\n{publish}    return verify(c,ref,M*N,cube_epsilon<{acc}>()) ? 0 : 1;\n}}\n"
                if dt == "bf16" else
                f"    BENCHEND;\n    reference_matmul<{ct},{acc},M,N,K>(ref,a,b,nullptr,bias);\n{publish}    return verify(c,ref,M*N,cube_epsilon<{acc}>()) ? 0 : 1;\n}}\n")
    elif c.kind == "gemv":
        body = f"    bench_gemv<{ct},M,N,K>(c,a,b);\n"
    elif c.kind == "gemv_acc":
        body = f"    bench_gemv_acc<{ct},M,N,K>(c,a,b);\n"
    elif c.kind == "gemv_bias":
        body = f"    bench_gemv_bias<{ct},M,N,K>(c,a,b,bias);\n"
    elif c.kind == "gemv_mx":
        body = f"    bench_gemv_mx<{ct},M,N,K>(c,a,as,b,bs);\n"
    else:
        body = f"    // unhandled kind {c.kind}\n"
    return head + body + tail


def gen_family(family: str, cases: list[Case], emitter):
    src_dir = os.path.join(ROOT, family, "src")
    os.makedirs(src_dir, exist_ok=True)
    # clean old generated
    for f in os.listdir(src_dir):
        if f.endswith(".cpp"):
            os.remove(os.path.join(src_dir, f))
    names = []
    for c in cases:
        for dt in c.dtypes:
            if dt not in DTYPE:
                continue  # skip placeholder dtypes (e.g. bf16)
            name = case_name(c, dt)
            names.append(name)
            with open(os.path.join(src_dir, name + ".cpp"), "w") as f:
                f.write(emitter(c, dt))
    if len(names) != len(set(names)):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise RuntimeError(f"duplicate {family} testcase names: {duplicates}")
    # compile.all: run every case, preserve all failures, and fail closed.
    lines = ["#!/bin/bash", "set -u", f'echo "=== {family} ==="',
             "failures=()", "run_case() {", "  local testcase=$1",
             "  echo \"  $testcase\"",
             "  if make TESTCASE=\"$testcase\" diss; then",
             "    echo \"PASS: $testcase\"", "  else",
             "    echo \"FAIL: $testcase\"", "    failures+=(\"$testcase\")",
             "  fi", "}"]
    for n in sorted(names):
        lines.append(f"run_case {n}")
    lines += ['if [ "${#failures[@]}" -ne 0 ]; then',
              f'  echo "=== {family} FAILED: ${{failures[*]}} ==="', "  exit 1", "fi",
              f'echo "=== {family} completed: all {len(names)} cases passed ==="']
    with open(os.path.join(ROOT, family, "compile.all"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(os.path.join(ROOT, family, "compile.all"), 0o755)
    return len(names)


def gen_scalar():
    src_dir = os.path.join(ROOT, "scalar", "src")
    os.makedirs(src_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f.endswith(".cpp"):
            os.remove(os.path.join(src_dir, f))
    names = []

    def write(name, body):
        names.append(name)
        with open(os.path.join(src_dir, name + ".cpp"), "w") as f:
            f.write(body)

    # bin / un / ld opcodes
    for op, cat, dtypes, lam_tpl in SCALAR_OPS:
        for dt in dtypes:
            ct = SDTYPE[dt]
            ut = UT.get(dt, "uint32_t")
            lam = lam_tpl.format(T=ct, ut=ut)
            metrics = ("thr", "lat") if cat in ("bin", "un") else ("thr",)
            for m in metrics:
                fn = "bench_throughput" if m == "thr" else "bench_latency"
                ref_fn = "reference_throughput" if m == "thr" else "reference_latency"
                write(f"{op}_{dt}_{m}", f'''#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: {op} ({cat}) {dt} {m}
int main() {{
    {ct} a[16], b[16];
    for (int i = 0; i < 16; ++i) {{ a[i] = ({ct})(i * 0.7 + 1); b[i] = ({ct})(i * 0.3 + 2); }}
    volatile {ct} sink = ({ct})0;
    auto scalar_op = {lam};
    BENCHSTART;
    {ct} r = {fn}<{ct}>(a, b, scalar_op);
    BENCHEND;
    sink = r;
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_scalar(r);
#endif
#ifdef RES_CHECK
    {ct} ref = {ref_fn}<{ct}>(a, b, scalar_op);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}}
''')

    # store opcodes
    for op, cat, dtypes in SCALAR_ST:
        for dt in dtypes:
            ct = SDTYPE[dt]
            write(f"{op}_{dt}_thr", f'''#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: {op} ({cat}) {dt} throughput
int main() {{
    {ct} out[16], val = ({ct})5;
    for (int i = 0; i < 16; ++i) out[i] = ({ct})0;
    BENCHSTART;
    bench_store<{ct}>(out, val);
    BENCHEND;
    volatile {ct} sink = out[0];
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_result(out, 16);
#endif
#ifdef RES_CHECK
    for (int i = 0; i < 16; ++i) if (out[i] != val) return 1;
#endif
    return 0;
}}
''')

    # conversion opcodes
    for op, cat, pairs in SCALAR_CV:
        for indt, outdt in pairs:
            ict = SDTYPE[indt]
            oct = SDTYPE[outdt]
            write(f"{op}_{indt}_to_{outdt}_thr", f'''#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: {op} ({cat}) {indt}->{outdt} throughput
int main() {{
    {ict} b[16];
    for (int i = 0; i < 16; ++i) b[i] = ({ict})(i * 0.7 + 1);
    volatile {oct} sink = ({oct})0;
    BENCHSTART;
    {oct} r = bench_cv<{ict}, {oct}>(b);
    BENCHEND;
    sink = r;
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_scalar(r);
#endif
#ifdef RES_CHECK
    {oct} ref = ({oct})0;
    for (int lane = 0; lane < 8; ++lane)
        ref = ({oct})(ref + ({oct})b[(1023 * 8 + lane) & 15]);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}}
''')

    if len(names) != len(set(names)):
        raise RuntimeError("duplicate scalar testcase names")
    lines = ["#!/bin/bash", "set -u", 'echo "=== scalar ==="',
             "failures=()", "run_case() {", "  local testcase=$1",
             "  echo \"  $testcase\"",
             "  if make TESTCASE=\"$testcase\" diss; then",
             "    echo \"PASS: $testcase\"", "  else",
             "    echo \"FAIL: $testcase\"", "    failures+=(\"$testcase\")",
             "  fi", "}"]
    for n in sorted(names):
        lines.append(f"run_case {n}")
    lines += ['if [ "${#failures[@]}" -ne 0 ]; then',
              '  echo "=== scalar FAILED: ${failures[*]} ==="', "  exit 1", "fi",
              f'echo "=== scalar completed: all {len(names)} cases passed ==="']
    with open(os.path.join(ROOT, "scalar", "compile.all"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(os.path.join(ROOT, "scalar", "compile.all"), 0o755)
    return len(names)


def main():
    nv = gen_family("vector", V, emit_vector)
    nm = gen_family("memory", ME, emit_memory)
    nc = gen_family("cube", C, emit_cube)
    ns = gen_scalar()
    active = []
    for family, cases in (("vector", V), ("memory", ME), ("cube", C)):
        for case in cases:
            for dt in case.dtypes:
                if dt in DTYPE:
                    active.append({"name": case_name(case, dt), "family": family,
                                   "operation": case.op, "dtype": dt,
                                   "shape": list(case.size), "status": "active"})
    active.extend({"name": name, "family": "scalar", "operation": name.split("_")[0],
                   "status": "active"}
                  for name in sorted(p[:-4] for p in os.listdir(os.path.join(ROOT, "scalar", "src"))
                                     if p.endswith(".cpp")))
    active.extend({"name": f"fixp_tmatmul_{mode}_M32_N32_K32_tM32_tN32_tK32",
                   "family": "fixp",
                   "operation": "TGEMV" if mode.startswith("gemv") else "TMATMUL",
                   "mode": mode, "shape": [32, 32, 32], "status": "active"}
                  for mode in ASL_FIXP_MODES)
    unsupported = [
        {"name": "fixp_tmatmul_lrelu_only", "family": "fixp",
         "operation": "TMATMUL", "status": "unsupported",
         "reason": "main compiler assembler rejects empty LRELU_ONLY B.IOR operand stream"},
        {"name": "tsel_fp16_16x16", "family": "vector", "operation": "TSELECT",
         "dtype": "fp16", "shape": [16, 16], "status": "unsupported",
         "reason": "installed compiler TileOP headers do not expose TSELECT"},
        {"name": "tmatmul_fp16_64x64x64", "family": "cube", "operation": "TMATMUL",
         "dtype": "fp16", "shape": [64, 64, 64], "status": "unsupported",
         "reason": "CUBE_M32 supports at most 32 logical rows; M=64 requires operator tiling"},
        {"name": "tsel_fp32_16x16", "family": "vector", "operation": "TSELECT",
         "dtype": "fp32", "shape": [16, 16], "status": "unsupported",
         "reason": "installed compiler TileOP headers do not expose TSELECT"},
        {"name": "tabs_bf16_16x16", "family": "vector", "operation": "TABS",
         "dtype": "bf16", "shape": [16, 16], "status": "unsupported",
         "reason": "main compiler crashes during BF16 TABS instruction selection"},
    ]
    for op, dtypes in (("TCMP", ("fp16", "fp32", "i32")),
                       ("TCMPS", ("fp16", "fp32")),
                       ("THISTOGRAM", ("i16", "i32"))):
        for dtype in dtypes:
            unsupported.append({"name": f"{op.lower()}_{dtype}_16x16",
                                "family": "vector", "operation": op,
                                "dtype": dtype, "shape": [16, 16],
                                "status": "unsupported",
                                "reason": "main compiler assembler rejects emitted B.DATR encoding"})
    for dt in ("fp16", "fp32", "i32"):
        unsupported.append({"name": f"tmov_{dt}_16x16", "family": "memory",
                            "operation": "TMOV", "dtype": dt, "shape": [16, 16],
                            "status": "unsupported",
                            "reason": "main compiler assembler rejects generic TMOV B.DATR"})
    for dt in ("fp16", "fp32"):
        unsupported.append({"name": f"tmov_{dt}_32x32", "family": "memory",
                            "operation": "TMOV", "dtype": dt, "shape": [32, 32],
                            "status": "unsupported",
                            "reason": "main compiler assembler rejects generic TMOV B.DATR"})
    for op in ("MGATHER_MASK", "MSCATTER_MASK"):
        for dt in ("fp16", "fp32"):
            unsupported.append({"name": f"{op.lower()}_{dt}_16x16",
                                "family": "memory", "operation": op,
                                "dtype": dt, "shape": [16, 16],
                                "status": "unsupported",
                                "reason": "main compiler assembler rejects emitted masked B.IOT encoding"})
    for op in ("TPARTADD", "TPARTMUL", "TPARTMAX", "TPARTMIN"):
        for dtype in ("fp16", "fp32"):
            unsupported.append({"name": f"{op.lower()}_{dtype}_16x16",
                                "family": "vector", "operation": op,
                                "dtype": dtype, "shape": [16, 16],
                                "status": "unsupported",
                                "reason": "operation removed from PTO ISA v0.58.5"})
    with open(os.path.join(ROOT, "coverage.json"), "w") as f:
        json.dump({"schema_version": 1, "active": active, "unsupported": unsupported,
                   "notes": ["TLOAD_ND2NZ is not a PTO operation and is intentionally absent",
                             "ASL FIXP coverage is the deterministic ASL_FIXP_MODES subset; benchmark-only modes remain in fixp/compile.all"]},
                  f, indent=2)
        f.write("\n")
    print(f"generated: vector={nv} memory={nm} cube={nc} scalar={ns} total={nv+nm+nc+ns}")


if __name__ == "__main__":
    main()
