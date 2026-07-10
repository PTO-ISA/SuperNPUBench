#!/usr/bin/env python3
# Microbench case generator.
#
# Emits one .cpp per (opcode x dtype x tile-size) case plus a compile.all list
# for each family (cube / vector / memory). Intrinsic naming follows the
# DavinciOO/PTO reference (OPERATOR_REFERENCE.md); sources are structural and
# may not compile until pto_tileop.hpp aligns to these names.
#
# Usage: python3 gen_cases.py
from __future__ import annotations

import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.abspath(__file__))

DTYPE = {
    "fp16": "__half",
    "fp32": "float",
    "i8":   "int8_t",
    "i16":  "int16_t",
    "i32":  "int32_t",
}

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
for op in ["TADD", "TSUB", "TMUL", "TDIV", "TREm".replace("REm", "REM"), "TFMOD",
           "TAND", "TOR", "TXOR", "TSHL", "TSHR", "TMAX", "TMIN", "TCMP", "TPRELU"]:
    if op in ("TAND", "TOR", "TXOR", "TSHL", "TSHR"):
        dt = ("i16", "i32")
    elif op in ("TREM", "TFMOD"):
        dt = ("fp16", "fp32", "i32")  # re-test full dtype set
    elif op == "TCMP":
        dt = ("fp16", "fp32", "i32")
    elif op == "TPRELU":
        dt = ("fp16", "fp32")
    else:
        dt = ("fp16", "fp32", "i16", "i32")
    V.append(Case(op, "binary", dt, M16))

# mode 0 unary
for op, dt in [
    ("TABS", ("fp16", "f32", "i16", "i32")),
    ("TNOT", ("i16", "i32")),
    ("TNEG", ("fp16", "fp32", "i16", "i32")),
    ("TEXP", ("fp16", "fp32")),
    ("TLOG", ("fp16", "fp32")),
    ("TRECIP", ("fp16", "fp32")),
    ("TSQRT", ("fp16", "fp32")),
    ("TRSQRT", ("fp16", "f32")),
    ("TRELU", ("fp16", "fp32")),
    ("TCVT", ("fp16", "fp32")),
]:
    V.append(Case(op, "unary", dt, M16))

# mode 0 ternary
for op in ["TADDC", "TSUBC", "TSEL"]:
    V.append(Case(op, "ternary", ("fp16", "fp32"), M16))

# mode 0 partial-valid
for op in ["TPARTADD", "TPARTMUL", "TPARTMAX", "TPARTMIN"]:
    V.append(Case(op, "binary", ("fp16", "fp32"), M16))

# mode 1 tile-scalar (1 tile + scalar)
# tile-scalar ops only support float dtypes on current toolchain
for op in ["TADDS", "TSUBS", "TMULS", "TDIVS", "TREMS", "TFMODS",
           "TANDS", "TORS", "TXORS", "TSHLS", "TSHRS",
           "TMAXS", "TMINS", "TCMPS", "TLRELU", "TAXPY"]:
    V.append(Case(op, "scalar", ("fp16", "f32"), M16))

# mode 1 tile-scalar fused (2 tile + scalar)
for op in ["TADDSC", "TSUBSC", "TSELS"]:
    V.append(Case(op, "scalar3", ("fp16", "fp32"), M16))

# mode 1 scalar broadcast
V.append(Case("TEXPANDS", "scalarbcast", ("fp16", "fp32"), M16))

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

# mode 2 expand (2 src: data + row/col vector)
for op in ["TROWEXPANDADD", "TROWEXPANDSUB", "TROWEXPANDMUL", "TROWEXPANDDIV",
           "TROWEXPANDMAX", "TROWEXPANDMIN", "TROWEXPANDEXPDIF",
           "TCOLEXPANDADD", "TCOLEXPANDSUB", "TCOLEXPANDMUL", "TCOLEXPANDDIV",
           "TCOLEXPANDMAX", "TCOLEXPANDMIN", "TCOLEXPANDEXPDIF"]:
    V.append(Case(op, "binary", ("fp16", "fp32"), M16))

# mode 3 complex
V.append(Case("TCONCAT", "binary", ("fp16", "fp32"), M16))
V.append(Case("TGATHERB", "gather", ("fp16", "fp32"), M16))
V.append(Case("THISTOGRAM", "hist", ("fp16", "fp32"), M16))

# Opcodes the toolchain does not yet expose (or needs special layout like TCVT's
# NZ requirement). Kept here as a skip list; re-enable when pto_tileop.hpp aligns.
VECTOR_SKIP = {
    # need fractal/NZ layout (32-byte align) — plain RowMajor Mx1/Nx1 output fails:
    "TROWMAX", "TROWMIN", "TROWPROD", "TROWSUM", "TROWARGMAX", "TROWARGMIN",
    "TCOLSUM", "TCOLMAX", "TCOLMIN", "TCOLPROD", "TCOLARGMAX", "TCOLARGMIN",
    # toolchain inline-asm syntax bug ("unknown token in expression"):
    "TADDC", "TSUBC",
    # signature mismatch (PTO arity != bench template); TODO align:
    "TSEL", "TGATHERB",
}
V = [c for c in V if c.op not in VECTOR_SKIP]


# ============ memory (TLSU) cases ============
ME = []
for op, kind, dt, sz in [
    ("TLOAD", "load", ("fp16", "fp32", "i32"), M16),
    ("TLOAD", "load", ("fp16", "fp32"), (32, 32)),
    ("TSTORE", "store", ("fp16", "fp32", "i32"), M16),
    ("TMOV", "mov", ("fp16", "fp32", "i32"), M16),
    ("TMOV", "mov", ("fp16", "fp32"), (32, 32)),
    ("MGATHER", "gather", ("fp16", "fp32", "i32"), M16),
    ("MSCATTER", "scatter", ("fp16", "fp32", "i32"), M16),
    ("MGATHER_MASK", "gather_mask", ("fp16", "fp32"), M16),
    ("MSCATTER_MASK", "scatter_mask", ("fp16", "fp32"), M16),
    ("TLOAD_ND2NZ", "load_layout", ("fp16", "fp32"), M16),
]:
    ME.append(Case(op, kind, dt, sz))


# ============ cube (CUBE) cases ============
# Tile allocation must be <= 8KB per active profile (imm4=3..9). Square tile
# constraint: S^2 * sizeof(dtype) <= 8192  =>  fp16/i8: S<=64..90, fp32: S<=45.
# TGEMV* are not yet exposed by the toolchain; only TMATMUL* + ACCCVT land.
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

for dt in ("fp16", "fp32", "i8"):
    cube("TMATMUL", "matmul", dt)
cube("TMATMUL_BIAS", "matmul_bias", "fp16")
cube("TMATMUL_BIAS", "matmul_bias", "fp32")
cube("TMATMUL_MX", "matmul_mx", "fp16")        # e4m3 placeholder -> fp16
# matmul.ac backend still crashes (llvm.linx.blk.matmul.ac SelectionDAG) on latest toolchain
# cube("TMATMUL_ACC", "matmul_acc", "fp16")
# cube("TMATMUL_ACC", "matmul_acc", "fp32")
cube("ACCCVT", "acccvt", "fp16")
cube("ACCCVT", "acccvt", "fp32")
cube("ACCCVT_NZ2ND", "acccvt", "fp16")

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
SDTYPE = {"i32": "int32_t", "i64": "int64_t", "f32": "float", "f64": "double"}
UT = {"i32": "uint32_t", "i64": "uint64_t"}

# (op, cat, dtypes, lambda_tpl)  lambda_tpl uses {T} (return cast) and {ut} (unsigned cast)
SCALAR_OPS = [
    ("add", "bin", ("i32", "i64", "f32", "f64"), "[](auto x,auto y){{return x+y;}}"),
    ("sub", "bin", ("i32", "i64", "f32", "f64"), "[](auto x,auto y){{return x-y;}}"),
    ("mul", "bin", ("i32", "i64", "f32", "f64"), "[](auto x,auto y){{return x*y;}}"),
    ("div", "bin", ("i32", "i64", "f32", "f64"), "[](auto x,auto y){{return x/y;}}"),
    ("and", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x & ({ut})y);}}"),
    ("or",  "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x | ({ut})y);}}"),
    ("xor", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x ^ ({ut})y);}}"),
    ("sll", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x << (y & 31));}}"),
    ("srl", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(({ut})x >> (y & 31));}}"),
    ("sra", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(x >> (y & 31));}}"),
    ("slt", "bin", ("i32", "i64"), "[](auto x,auto y){{return ({T})(x < y);}}"),
    ("max", "bin", ("i32", "i64", "f32", "f64"), "[](auto x,auto y){{return x<y?y:x;}}"),
    ("min", "bin", ("i32", "i64", "f32", "f64"), "[](auto x,auto y){{return x<y?x:y;}}"),
    ("mod", "bin", ("i32", "i64"), "[](auto x,auto y){{return x%y;}}"),
    ("abs", "un", ("i32", "f32", "f64"), "[](auto x,auto y){{auto t=x+y; return t<0?-t:t;}}"),
    ("neg", "un", ("i32", "i64", "f32", "f64"), "[](auto x,auto y){{return -(x+y);}}"),
    ("not", "un", ("i32", "i64"), "[](auto x,auto y){{return ({T})(~({ut})(x+y));}}"),
    ("popc", "un", ("i32", "i64"), "[](auto x,auto y){{return ({T})__builtin_popcountll((unsigned long long)(x+y));}}"),
    ("clz", "un", ("i32", "i64"), "[](auto x,auto y){{return ({T})__builtin_clzll((unsigned long long)(x+y));}}"),
    ("sqrt", "un", ("f32", "f64"), "[](auto x,auto y){{return ({T})std::sqrt((double)(x+y));}}"),
    ("ld", "ld", ("i32", "i64", "f32", "f64"), "[](auto x,auto y){{return y;}}"),
]
# (op, cat, [(in,out), ...])
SCALAR_CV = [
    ("i2f", "cv", [("i32", "f32"), ("i32", "f64")]),
    ("f2i", "cv", [("f32", "i32"), ("f64", "i32")]),
    ("f2f_widen", "cv", [("f32", "f64")]),
    ("f2f_narrow", "cv", [("f64", "f32")]),
]
SCALAR_ST = [("st", "st", ("i32", "i64", "f32", "f64"))]


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


def emit_vector(c: Case, dt: str) -> str:
    ct = DTYPE[dt]
    m, n = c.size
    n_elems = m * n
    name = case_name(c, dt)
    op = c.op
    head = f'''#include "vector_bench.hpp"
// auto-generated by gen_cases.py
// {op} ({c.kind}) {dt} {c.size[0]}x{c.size[1]}
int main() {{
    constexpr int M = {m}, N = {n};
    {ct} a[M*N], b[M*N], d[M*N], c[M*N];
    fill_seq(a, M*N); fill_seq(b, M*N); fill_seq(d, M*N); zero(c, M*N);
    BENCHSTART;
'''
    tail = "    BENCHEND;\n    return 0;\n}\n"
    if c.kind == "binary":
        body = f"    bench_binary<{ct},M,N>(c,a,b,[](auto& dst,auto& s0,auto& s1){{ {op}(dst,s0,s1); }});\n"
    elif c.kind == "unary":
        body = f"    bench_unary<{ct},M,N>(c,a,[](auto& dst,auto& s){{ {op}(dst,s); }});\n"
    elif c.kind == "ternary":
        body = f"    bench_ternary<{ct},M,N>(c,a,b,d,[](auto& dst,auto& s0,auto& s1,auto& s2){{ {op}(dst,s0,s1,s2); }});\n"
    elif c.kind == "reduce":
        body = f"    bench_reduce<{ct},M,N>(c,a,[](auto& dst,auto& s){{ {op}(dst,s); }});\n"
    elif c.kind == "scalar":
        body = f"    {ct} s = ({ct})0.5;\n    bench_scalar<{ct},M,N>(c,a,s,[](auto& dst,auto& s0,auto& sc){{ {op}(dst,s0,sc); }});\n"
    elif c.kind == "scalar3":
        # TADDSC/TSUBSC/TSELS signature is (dst, src0, scalar, src1)
        call = f"{op}(dst,s0,sc,s1)" if c.op in ("TADDSC", "TSUBSC", "TSELS") else f"{op}(dst,s0,s1,sc)"
        body = f"    {ct} s = ({ct})0.5;\n    bench_scalar3<{ct},M,N>(c,a,b,s,[](auto& dst,auto& s0,auto& s1,auto& sc){{ {call}; }});\n"
    elif c.kind == "scalarbcast":
        body = f"    {ct} s = ({ct})0.5;\n    bench_scalar_bcast<{ct},M,N>(c,s,[](auto& dst,auto& sc){{ {op}(dst,sc); }});\n"
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
    name = case_name(c, dt)
    op = c.op
    head = f'''#include "memory_bench.hpp"
// auto-generated by gen_cases.py
// {op} ({c.kind}) {dt} {c.size[0]}x{c.size[1]}
int main() {{
    constexpr int M = {m}, N = {n};
    {ct} a[M*N], c[M*N], idx[M*N], mask[M*N];
    fill_seq(a, M*N); fill_idx(idx, M*N); fill_const(mask, M*N, ({ct})1); zero(c, M*N);
    BENCHSTART;
'''
    tail = "    BENCHEND;\n    return 0;\n}\n"
    if c.kind == "load":
        body = f"    bench_load<{ct},M,N>(c,a);\n"
    elif c.kind == "load_layout":
        body = f"    bench_load_layout<{ct},M,N>(c,a);\n"
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
    ct = DTYPE.get(dt, "__half")  # low-precision placeholder falls back to __half
    m, n, k = c.size
    op = c.op
    head = f'''#include "cube_bench.hpp"
// auto-generated by gen_cases.py
// {op} ({c.kind}) {dt} {m}x{n}x{k}
int main() {{
    constexpr int M = {m}, N = {n}, K = {k};
    {ct} a[M*K], b[K*N], bias[1*N], as[M*K], bs[K*N], c[M*N];
    fill_seq(a, M*K); fill_seq(b, K*N); fill_seq(bias, N);
    fill_const(as, M*K, ({ct})1); fill_const(bs, K*N, ({ct})1); zero(c, M*N);
    BENCHSTART;
'''
    tail = "    BENCHEND;\n    return 0;\n}\n"
    if c.kind == "matmul":
        body = f"    bench_matmul<{ct},M,N,K>(c,a,b);\n"
    elif c.kind == "matmul_acc":
        body = f"    bench_matmul_acc<{ct},M,N,K>(c,a,b);\n"
    elif c.kind == "matmul_bias":
        body = f"    bench_matmul_bias<{ct},M,N,K>(c,a,b,bias);\n"
    elif c.kind == "matmul_mx":
        body = f"    bench_matmul_mx<{ct},M,N,K>(c,a,as,b,bs);\n"
    elif c.kind == "gemv":
        body = f"    bench_gemv<{ct},M,N,K>(c,a,b);\n"
    elif c.kind == "gemv_acc":
        body = f"    bench_gemv_acc<{ct},M,N,K>(c,a,b);\n"
    elif c.kind == "gemv_bias":
        body = f"    bench_gemv_bias<{ct},M,N,K>(c,a,b,bias);\n"
    elif c.kind == "gemv_mx":
        body = f"    bench_gemv_mx<{ct},M,N,K>(c,a,as,b,bs);\n"
    elif c.kind == "acccvt":
        body = f"    bench_acccvt<{ct},M,N,K>(c,a,b);\n"
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
    # compile.all
    lines = ["#!/bin/bash", f'echo "=== {family} ==="']
    for n in sorted(names):
        lines.append(f'echo "  {n}"; make TESTCASE={n}')
    lines.append('echo "=== Done ==="')
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
                write(f"{op}_{dt}_{m}", f'''#include "scalar_bench.hpp"
// auto-generated: {op} ({cat}) {dt} {m}
int main() {{
    {ct} a[16], b[16];
    for (int i = 0; i < 16; ++i) {{ a[i] = ({ct})(i * 0.7 + 1); b[i] = ({ct})(i * 0.3 + 2); }}
    volatile {ct} sink = ({ct})0;
    BENCHSTART;
    {ct} r = {fn}<{ct}>(a, b, {lam});
    BENCHEND;
    sink = r;
    return 0;
}}
''')

    # store opcodes
    for op, cat, dtypes in SCALAR_ST:
        for dt in dtypes:
            ct = SDTYPE[dt]
            write(f"{op}_{dt}_thr", f'''#include "scalar_bench.hpp"
// auto-generated: {op} ({cat}) {dt} throughput
int main() {{
    {ct} out[16], val = ({ct})5;
    for (int i = 0; i < 16; ++i) out[i] = ({ct})0;
    BENCHSTART;
    bench_store<{ct}>(out, val);
    BENCHEND;
    volatile {ct} sink = out[0];
    return 0;
}}
''')

    # conversion opcodes
    for op, cat, pairs in SCALAR_CV:
        for indt, outdt in pairs:
            ict = SDTYPE[indt]
            oct = SDTYPE[outdt]
            write(f"{op}_{indt}_to_{outdt}_thr", f'''#include "scalar_bench.hpp"
// auto-generated: {op} ({cat}) {indt}->{outdt} throughput
int main() {{
    {ict} b[16];
    for (int i = 0; i < 16; ++i) b[i] = ({ict})(i * 0.7 + 1);
    volatile {oct} sink = ({oct})0;
    BENCHSTART;
    {oct} r = bench_cv<{ict}, {oct}>(b);
    BENCHEND;
    sink = r;
    return 0;
}}
''')

    lines = ["#!/bin/bash", 'echo "=== scalar ==="']
    for n in sorted(names):
        lines.append(f'echo "  {n}"; make TESTCASE={n}')
    lines.append('echo "=== Done ==="')
    with open(os.path.join(ROOT, "scalar", "compile.all"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(os.path.join(ROOT, "scalar", "compile.all"), 0o755)
    return len(names)


def main():
    nv = gen_family("vector", V, emit_vector)
    nm = gen_family("memory", ME, emit_memory)
    nc = gen_family("cube", C, emit_cube)
    ns = gen_scalar()
    print(f"generated: vector={nv} memory={nm} cube={nc} scalar={ns} total={nv+nm+nc+ns}")


if __name__ == "__main__":
    main()
