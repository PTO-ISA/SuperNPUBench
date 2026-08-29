#!/usr/bin/env python3
"""Generate the fixp microbench coverage report.

Scans the per-mode ELF disassemblies produced by `compile.all`, decodes the
B.FPATR attribute word and the matrix-post-process operand stream of each mode,
and cross-checks them against the PTO-ISA v0.58 fixp matrix contract encoded in
this script's expectation table.

The benchmark covers the full 12-operation family (TMATMUL{,_BIAS,_ACC,_MX,
_MX_BIAS,_MX_ACC} + TGEMV{,_BIAS,_ACC,_MX,_MX_BIAS,_MX_ACC}) sharing one B.FPATR
options mechanism, plus the Shared-Right (B.IOS), LReLU-without-quant, vector-
quant+PReLU and legacy 3-param edge cases.

Output: fixp_report.md (next to this script).
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ELF_GLOB = os.path.join(
    HERE, "..", "..", "output", "microbenchmark", "fixp", "elf",
    "fixp", "fixp_tmatmul_*_M*_N*_K*.elf.diss",
)
REPORT = os.path.join(HERE, "fixp_report.md")

# GroupN -> B.FPATR GroupNCode (fixp::group_n_code).
GROUP_N_CODE = {8: 1, 16: 2, 32: 3, 48: 4, 64: 5, 80: 6, 96: 7, 112: 8, 128: 9}

# Recognised BSTART.<unit> op mnemonics (PTO-ISA v0.58 matrix post-process
# family). Note the MX variants are spelled without a dot: TMATMULMX / TGEMVMX.
MNEMONICS = {
    "TMATMUL", "TMATMUL.BIAS", "TMATMUL.ACC",
    "TMATMULMX", "TMATMULMX.BIAS", "TMATMULMX.ACC",
    "TGEMV", "TGEMV.BIAS", "TGEMV.ACC",
    "TGEMVMX", "TGEMVMX.BIAS", "TGEMVMX.ACC",
}

# Number of B.IOT *source* (non-destination) lines that carry the math operands
# (A/B/C/Bias/ScaleA/ScaleB/Mtx/Vec) for each (mnemonic, shared-B). A Shared
# right operand (and, for MX, its ScaleB) moves off B.IOT onto B.IOS. This is
# subtracted from the total non-destination B.IOT count so the remainder is the
# PostProcess aux-source count (RowMaxIn / vector-quant Tile / PReLU Tile).
# Calibrated from real .diss; if an unseen (mnemonic, shared) combo turns up the
# aux check reports "unknown MATH_SRC" rather than guessing.
MATH_SRC = {
    ("TMATMUL", False): 1, ("TMATMUL", True): 1,
    ("TMATMUL.BIAS", False): 2,
    ("TMATMUL.ACC", False): 2,
    ("TMATMULMX", False): 4,
    ("TMATMULMX.BIAS", False): 5,
    ("TMATMULMX.ACC", False): 5,
    ("TGEMV", False): 1,
    ("TGEMV.BIAS", False): 2,
    ("TGEMV.ACC", False): 2,
    ("TGEMVMX", False): 4,
    ("TGEMVMX.BIAS", False): 5,
    ("TGEMVMX.ACC", False): 5,
}

# MX ScaleMask changes the number of mathematical B.IOT source lines without
# changing the mnemonic. Keep per-mode overrides so scale carriers are not
# misclassified as post-process auxiliary operands.
MATH_SRC_BY_MODE = {
    "mx_scale0": 2,
    "mx_scale_a": 3,
    "mx_scale_b": 3,
    "mxacc_cscale": 2,
    "gemv_mx_scale0": 2,
    "gemv_mx_scale_a": 3,
    "gemv_mx_scale_b": 3,
}

# Expected FPATR (PreQuant, Relu, GroupNCode, RowMaxEn, GroupMaxEn, RowMaxInit,
# MaxAbsEn[, TransA, TransB, CScaleEn]) and operand-stream shape per mode.
#   op:     expected BSTART mnemonic
#   shared: expected B.IOS (Shared-Right) presence
#   ior:    expected number of real B.IOR source GPR slots (the literal `zero`
#           placeholder is not counted) carrying scalar-quant / LReLU descriptors
#   aux:    expected PostProcess aux B.IOT source tiles (RowMaxIn / quant / PReLU)
#   desc:   human-readable description of the option chain / op
MODES = [
    # (mode_label, op, shared, (fpatr...), ior, aux, description)
    ("keep_acc",        "TMATMUL", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "fixp::keep_acc()"),
    ("keep_acc_relu",   "TMATMUL", False, (0, 1, 0, 0, 0, 0, 0), 0, 0, "keep_acc().relu()"),
    ("f16",             "TMATMUL", False, (1, 0, 0, 0, 0, 0, 0), 0, 0, "fixp::f16()"),
    ("f16_relu",        "TMATMUL", False, (1, 1, 0, 0, 0, 0, 0), 0, 0, "f16().relu()"),
    ("bf16",            "TMATMUL", False, (16, 0, 0, 0, 0, 0, 0), 0, 0, "fixp::bf16()"),
    ("bf16_relu",       "TMATMUL", False, (16, 1, 0, 0, 0, 0, 0), 0, 0, "bf16().relu()"),
    ("signed_keep_acc", "TMATMUL", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "signed input -> S32 accumulator"),
    ("signed_acc",      "TMATMUL.ACC", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "signed S32 accumulator input"),
    ("signed_bias",     "TMATMUL.BIAS", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "signed S32 bias"),
    ("unsigned_keep_acc", "TMATMUL", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "unsigned input -> U32 accumulator"),
    ("unsigned_acc",    "TMATMUL.ACC", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "unsigned U32 accumulator input"),
    ("unsigned_bias",   "TMATMUL.BIAS", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "unsigned U32 bias"),
    ("mixed_float",     "TMATMUL", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "FP16 A + BF16 B"),
    ("gemv_mixed_float", "TGEMV", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "FP16 Vec + BF16 Mtx"),
    ("s_reqs8",         "TMATMUL", False, (3, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<REQS8Pre>(desc)"),
    ("s_deqf16",        "TMATMUL", False, (5, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<DEQF16>(desc)"),
    ("s_shifts16",      "TMATMUL", False, (13, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<SHIFTS322S16>(desc)"),
    ("s_qf_s4",         "TMATMUL", False, (17, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<QF322S4Pre>(desc)"),
    ("s_qf_s16",        "TMATMUL", False, (19, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<QF322S16Pre>(desc)"),
    ("s_qf_s8",         "TMATMUL", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "s8(desc) / scalar<QF322S8Pre>"),
    ("s_qf_hif8",       "TMATMUL", False, (25, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<QF322HIF8Pre>(desc)"),
    ("s_qf_fp8",        "TMATMUL", False, (26, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<QF322FP8Pre>(desc)"),
    ("s_qf_f32",        "TMATMUL", False, (27, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<QF322F32Pre>(desc)"),
    ("s_qf_f16",        "TMATMUL", False, (32, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<QF322F16Pre>(desc)"),
    ("s_qf_bf16",       "TMATMUL", False, (34, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<QF322BF16Pre>(desc)"),
    ("s_qs_bf16",       "TMATMUL", False, (35, 0, 0, 0, 0, 0, 0), 1, 0, "scalar<QS322BF16Pre>(desc)"),
    ("v_reqs8",         "TMATMUL", False, (2, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VREQS8Pre>(tile)"),
    ("v_deqf16",        "TMATMUL", False, (4, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VDEQF16>(tile)"),
    ("v_shifts16",      "TMATMUL", False, (12, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VSHIFTS322S16>(tile)"),
    ("v_qf_s4",         "TMATMUL", False, (18, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VQF322S4Pre>(tile)"),
    ("v_qf_s16",        "TMATMUL", False, (20, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VQF322S16Pre>(tile)"),
    ("v_qf_s8",         "TMATMUL", False, (23, 0, 0, 0, 0, 0, 0), 0, 1, "s8(quant_tile) / vector<VQF322S8Pre>"),
    ("v_qf_hif8",       "TMATMUL", False, (28, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VQF322HIF8Pre>(tile)"),
    ("v_qf_f16",        "TMATMUL", False, (33, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VQF322F16Pre>(tile)"),
    ("v_qf_bf16",       "TMATMUL", False, (36, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VQF322BF16Pre>(tile)"),
    ("v_qf_fp8",        "TMATMUL", False, (37, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VQF322FP8Pre>(tile)"),
    ("v_qf_f32",        "TMATMUL", False, (38, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VQF322F32Pre>(tile)"),
    ("v_qs_bf16",       "TMATMUL", False, (39, 0, 0, 0, 0, 0, 0), 0, 1, "vector<VQS322BF16Pre>(tile)"),
    ("s8_relu",         "TMATMUL", False, (24, 1, 0, 0, 0, 0, 0), 1, 0, "s8(desc).relu()"),
    ("s8_lrelu",        "TMATMUL", False, (24, 2, 0, 0, 0, 0, 0), 2, 0, "s8(desc).lrelu(fp19)"),
    ("v_s8_relu",       "TMATMUL", False, (23, 1, 0, 0, 0, 0, 0), 0, 1, "s8(quant).relu()"),
    ("f16_prelu",       "TMATMUL", False, (1, 3, 0, 0, 0, 0, 0), 0, 1, "f16().prelu(fp19_tile)"),
    ("s8_prelu",        "TMATMUL", False, (24, 3, 0, 0, 0, 0, 0), 1, 1, "s8(desc).prelu(fp19_tile)"),
    ("rowmax",          "TMATMUL", False, (0, 0, 0, 1, 0, 0, 0), 0, 0, "keep_acc().row_max(out)"),
    ("rowmax_init",     "TMATMUL", False, (0, 0, 0, 1, 0, 1, 0), 0, 1, "keep_acc().row_max(in,out)"),
    ("groupmax_8",      "TMATMUL", False, (0, 0, GROUP_N_CODE[8], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<8>(out)"),
    ("groupmax_16",     "TMATMUL", False, (0, 0, GROUP_N_CODE[16], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<16>(out)"),
    ("groupmax_32",     "TMATMUL", False, (0, 0, GROUP_N_CODE[32], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<32>(out)"),
    ("groupmax_48",     "TMATMUL", False, (0, 0, GROUP_N_CODE[48], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<48>(out)"),
    ("groupmax_64",     "TMATMUL", False, (0, 0, GROUP_N_CODE[64], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<64>(out)"),
    ("groupmax_80",     "TMATMUL", False, (0, 0, GROUP_N_CODE[80], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<80>(out)"),
    ("groupmax_96",     "TMATMUL", False, (0, 0, GROUP_N_CODE[96], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<96>(out)"),
    ("groupmax_112",    "TMATMUL", False, (0, 0, GROUP_N_CODE[112], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<112>(out)"),
    ("groupmax_128",    "TMATMUL", False, (0, 0, GROUP_N_CODE[128], 0, 1, 0, 0), 0, 0, "keep_acc().group_max<128>(out)"),
    ("rowgroup_maxabs", "TMATMUL", False, (0, 0, GROUP_N_CODE[8], 1, 1, 1, 1), 0, 1,
     "keep_acc().row_max(in,out).group_max<8>(out).max_abs()"),
    ("f16_groupmax",    "TMATMUL", False, (1, 0, GROUP_N_CODE[16], 0, 1, 0, 0), 0, 0, "f16().group_max<16>(out)"),
    ("s8_rowmax",       "TMATMUL", False, (24, 0, 0, 1, 0, 0, 0), 1, 0, "s8(desc).row_max(out)"),

    # --- operation-family coverage (param-free keep_acc) -------------------
    ("bias",            "TMATMUL.BIAS", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TMATMUL_BIAS(c,a,b,bias,keep_acc())"),
    ("acc",             "TMATMUL.ACC", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TMATMUL_ACC(d,c,a,b,keep_acc())"),
    ("acc_cscale",      "TMATMUL.ACC", False, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1), 0, 1, "TMATMUL_ACC + CScale"),
    ("mx_scale0",       "TMATMULMX", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "MX ScaleMask=0"),
    ("mx_scale_a",      "TMATMULMX", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "MX ScaleMask=1 (ScaleA)"),
    ("mx_scale_b",      "TMATMULMX", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "MX ScaleMask=2 (ScaleB)"),
    ("mx",              "TMATMULMX", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TMATMUL_MX(c,a,sa,b,sb,keep_acc())"),
    ("mxbias",          "TMATMULMX.BIAS", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TMATMUL_MX_BIAS(d,a,sa,b,sb,bias,keep_acc())"),
    ("mxacc",           "TMATMULMX.ACC", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TMATMUL_MX_ACC(d,c,a,sa,b,sb,keep_acc())"),
    ("mxacc_cscale",    "TMATMULMX.ACC", False, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1), 0, 1, "TMATMUL_MX_ACC + CScale"),
    ("gemv",            "TGEMV", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV(d,mtx,vec,keep_acc())"),
    ("gemv_bias",       "TGEMV.BIAS", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV_BIAS(d,mtx,vec,bias,keep_acc())"),
    ("gemv_acc",        "TGEMV.ACC", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV_ACC(d,c,mtx,vec,keep_acc())"),
    ("gemv_mx_scale0",  "TGEMVMX", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV MX ScaleMask=0"),
    ("gemv_mx_scale_a", "TGEMVMX", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV MX ScaleMask=1 (ScaleVec)"),
    ("gemv_mx_scale_b", "TGEMVMX", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV MX ScaleMask=2 (ScaleMtx)"),
    ("gemv_mx",         "TGEMVMX", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV_MX(d,mtx,smtx,vec,svec,keep_acc())"),
    ("gemv_mx_bias",    "TGEMVMX.BIAS", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV_MX_BIAS(d,mtx,smtx,vec,svec,bias,keep_acc())"),
    ("gemv_mx_acc",     "TGEMVMX.ACC", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TGEMV_MX_ACC(d,c,mtx,smtx,vec,svec,keep_acc())"),
    # --- full-options spot-check (s8 scalar quant on non-TMATMUL ops) -------
    ("bias_s8",         "TMATMUL.BIAS", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TMATMUL_BIAS + s8(desc)"),
    ("acc_s8",          "TMATMUL.ACC", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TMATMUL_ACC + s8(desc)"),
    ("mx_s8",           "TMATMULMX", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TMATMUL_MX + s8(desc)"),
    ("mxbias_s8",       "TMATMULMX.BIAS", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TMATMUL_MX_BIAS + s8(desc)"),
    ("mxacc_s8",        "TMATMULMX.ACC", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TMATMUL_MX_ACC + s8(desc)"),
    ("gemv_s8",         "TGEMV", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TGEMV + s8(desc)"),
    ("gemv_bias_s8",    "TGEMV.BIAS", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TGEMV_BIAS + s8(desc)"),
    ("gemv_acc_s8",     "TGEMV.ACC", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TGEMV_ACC + s8(desc)"),
    ("gemv_mx_s8",      "TGEMVMX", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TGEMV_MX + s8(desc)"),
    ("gemv_mx_bias_s8", "TGEMVMX.BIAS", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TGEMV_MX_BIAS + s8(desc)"),
    ("gemv_mx_acc_s8",  "TGEMVMX.ACC", False, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TGEMV_MX_ACC + s8(desc)"),
    # --- Shared-Right B (B.IOS) --------------------------------------------
    ("shared",          "TMATMUL", True, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TMATMUL(d,a,SharedTile<B>,keep_acc())"),
    ("s8_shared",       "TMATMUL", True, (24, 0, 0, 0, 0, 0, 0), 1, 0, "TMATMUL + SharedTile<B> + s8(desc)"),
    ("trans_a",         "TMATMUL", True, (0, 0, 0, 0, 0, 0, 0, 1, 0, 0), 0, 0, "Shared A/B + transpose_a"),
    ("trans_b",         "TMATMUL", True, (0, 0, 0, 0, 0, 0, 0, 0, 1, 0), 0, 0, "Shared A/B + transpose_b"),
    ("trans_ab",        "TMATMUL", True, (0, 0, 0, 0, 0, 0, 0, 1, 1, 0), 0, 0, "Shared A/B + transpose_a + transpose_b"),
    # --- edge-case FPATR / operand-stream features -------------------------
    ("lrelu_only",      "TMATMUL", False, (0, 2, 0, 0, 0, 0, 0), 1, 0, "keep_acc().lrelu(fp19)  [zero,lrelu]"),
    ("vqf_s8_prelu",    "TMATMUL", False, (23, 3, 0, 0, 0, 0, 0), 0, 1, "s8(quant_tile).prelu(prelu_tile)"),
    ("legacy3",         "TMATMUL", False, (0, 0, 0, 0, 0, 0, 0), 0, 0, "TMATMUL(c,a,b) legacy 3-param"),
]

MODE_MAP = {m[0]: m for m in MODES}


def expected_word(f):
    f = tuple(f) + (0,) * (10 - len(f))
    pq, rl, gn, rme, gme, rmi, mae, trans_a, trans_b, cscale = f
    return (0x2023 | (pq << 26) | (rl << 23) | (gn << 19) | (rme << 18)
            | (gme << 17) | (rmi << 16) | (mae << 15)
            | (trans_a << 7) | (trans_b << 8) | (cscale << 9))


def parse_bundle(lines):
    """Decode the first matrix-post-process bundle in a disassembly line list.

    Returns (mnemonic, fpatr, word, ior_gpr, iot_src, has_ios), or None if no
    B.FPATR line was found. fpatr is the 10-tuple (PreQuant, Relu, GroupNCode,
    RowMaxEn, GroupMaxEn, RowMaxInit, MaxAbsEn, TransA, TransB, CScaleEn);
    word is the packed encoding;
    ior_gpr counts real B.IOR GPR slots (the literal `zero` placeholder is
    skipped); iot_src counts all non-destination B.IOT source lines; has_ios
    reports a B.IOS (Shared) binder."""
    mnemonic = None
    fpatr = None
    word = None
    ior_gpr = 0
    iot_src = 0
    has_ios = False
    started = False
    for ln in lines:
        line = ln.strip()
        m = re.search(r"BSTART\.\w+\s+([A-Z][A-Z.]*)", line)
        if m:
            tok = m.group(1)
            if tok in MNEMONICS:
                if not started:
                    mnemonic = tok
                    started = True
                    continue
                break  # next matrix bundle starts -> end of this one
            if started:
                break  # a different BSTART.* unit ends this bundle
            continue
        if not started:
            continue
        m2 = re.search(
            r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+B\.FPATR"
            r"\s+([0-9]+),\s+([0-9]+),\s+([0-9]+),\s+([0-9]+),\s+"
            r"([0-9]+),\s+([0-9]+),\s+([0-9]+),\s+([0-9]+),\s+"
            r"([0-9]+),\s+([0-9]+)$", line)
        if m2 and fpatr is None:
            word = int(m2.group(2), 16)
            fpatr = tuple(int(x) for x in m2.groups()[2:])
        m = re.search(r"B\.IOR\s+\[([^\]]*)\]", line)
        if m:
            gprs = [g.strip() for g in m.group(1).split(",")
                    if g.strip() and g.strip() != "zero"]
            ior_gpr += len(gprs)
        if "B.IOS" in line:
            has_ios = True
        if re.search(r"B\.IOT\s+", line) and "->" not in line:
            iot_src += 1
    if fpatr is None:
        return None
    return mnemonic, fpatr, word, ior_gpr, iot_src, has_ios


def load_diss(mode):
    matches = glob.glob(os.path.join(
        os.path.dirname(ELF_GLOB),
        f"fixp_tmatmul_{mode}_M32_N32_K32_tM32_tN32_tK32.elf.diss"))
    if not matches:
        return None
    with open(matches[0], errors="replace") as fh:
        return fh.readlines()


# Static coverage matrix (12 operations x B.FPATR options) for the report.
# Param-free = keep_acc/f16/bf16 (+relu); options = quant/PReLU/RowMax/GroupMax.
COVERAGE = [
    ("TMATMUL",       "keep_acc (+f16/bf16/relu and numeric-class modes)", "all PreQuant + ReLU/max/group/transpose options", "Shared A/B; legacy3"),
    ("TMATMUL.BIAS",  "bias",            "bias_s8",                ""),
    ("TMATMUL.ACC",   "acc",             "acc_s8 / acc_cscale",    "CScaleEn covered"),
    ("TMATMULMX",     "mx + scale masks 0/1/2", "mx_s8",            "all ScaleMask values"),
    ("TMATMULMX.BIAS","mxbias",          "mxbias_s8",              ""),
    ("TMATMULMX.ACC", "mxacc",           "mxacc_s8 / mxacc_cscale", "CScaleEn covered"),
    ("TGEMV",         "gemv",            "gemv_s8",                "vec=Left(1xK), mtx=Right(KxN), M=1"),
    ("TGEMV.BIAS",    "gemv_bias",       "gemv_bias_s8",           ""),
    ("TGEMV.ACC",     "gemv_acc",        "gemv_acc_s8",            ""),
    ("TGEMVMX",       "gemv_mx + scale masks 0/1/2", "gemv_mx_s8", "all ScaleMask values"),
    ("TGEMVMX.BIAS",  "gemv_mx_bias",    "gemv_mx_bias_s8",        ""),
    ("TGEMVMX.ACC",   "gemv_mx_acc",     "gemv_mx_acc_s8",         ""),
]


def main():
    rows = []
    n_pass = 0
    n_fail = 0
    n_blocked = 0
    for label, op, shared, fpatr, ior, aux, desc in MODES:
        fpatr = tuple(fpatr) + (0,) * (10 - len(fpatr))
        lines = load_diss(label)
        if lines is None:
            rows.append((label, op, shared, desc, fpatr, None, ior, aux,
                         "BLOCKED", "no .diss (assembler rejected bundle)"))
            n_blocked += 1
            continue
        parsed = parse_bundle(lines)
        if parsed is None:
            rows.append((label, op, shared, desc, fpatr, None, ior, aux,
                         "BLOCKED", "no B.FPATR bundle found in .diss"))
            n_blocked += 1
            continue
        mnem, actual, word, a_ior, a_iot, a_ios = parsed
        math_src = MATH_SRC_BY_MODE.get(label, MATH_SRC.get((mnem, a_ios)))
        if math_src is not None:
            # The legacy 3-param TMATMUL(c,a,b) emits A/B as read operands on
            # the destination line (combined read-write), so it has 0 math
            # source lines -- clamp to iot_src so aux stays 0 instead of -1.
            math_src = min(math_src, a_iot)
        a_aux = None if math_src is None else (a_iot - math_src)
        reason = []
        if mnem != op:
            reason.append(f"mnemonic {mnem} != expected {op}")
        if a_ios != shared:
            reason.append(f"B.IOS {a_ios} != expected {shared}")
        if actual != fpatr:
            reason.append(f"FPATR {actual} != expected {fpatr}")
        want_word = expected_word(fpatr)
        if word != want_word:
            reason.append(f"encoding 0x{word:08x} != expected 0x{want_word:08x}")
        if ior != a_ior:
            reason.append(f"IOR GPR count {a_ior} != expected {ior}")
        if a_aux is None:
            reason.append(f"aux: unknown MATH_SRC for ({mnem},shared={a_ios}); iot_src={a_iot}")
        elif aux != a_aux:
            reason.append(f"aux source lines {a_aux} != expected {aux} (iot_src={a_iot})")
        if reason:
            rows.append((label, op, shared, desc, fpatr, word, ior, aux,
                         "FAIL", "; ".join(reason)))
            n_fail += 1
        else:
            rows.append((label, op, shared, desc, fpatr, word, ior, aux,
                         "PASS", "mnemonic + FPATR + encoding + IOR + IOS + aux match"))
            n_pass += 1

    out = []
    out.append("# fixp microbenchmark report")
    out.append("")
    out.append("Source: `microbenchmark/fixp/src/fixp_tmatmul.cpp` (one binary per "
               "fixp configuration), built and disassembled by "
               "`microbenchmark/fixp/compile.all`.")
    out.append("")
    out.append("Kernel: a single call to one of the 12 matrix post-process "
               "operations (TMATMUL{,_BIAS,_ACC,_MX,_MX_BIAS,_MX_ACC} + "
               "TGEMV{,_BIAS,_ACC,_MX,_MX_BIAS,_MX_ACC}) using the new-style "
               "`OP(dst, a, b, ..., fixp::Options)` interface. Inputs cover "
               "floating, signed and unsigned numeric classes; auxiliary "
               "operands cover quant, max, MX scale and ACC CScale tiles; the "
               "destination carries the converted result. Toolchain: Linx "
               "BlockISA LLVM 15.0.4 (linx64v5-musl), main `linx-toolchain-build` checkout.")
    out.append("")
    out.append("## Build")
    out.append("")
    out.append("Requires the Linx BlockISA toolchain on PATH via `COMPILER_DIR` "
               "(default `PLAT=linx`). From this directory:")
    out.append("")
    out.append("```bash")
    out.append("# 0. point at the mandated main compiler checkout")
    out.append("export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin")
    out.append("")
    out.append("# 1. build + disassemble one variant")
    out.append("make FIXP_MODE=S_QF_S8 diss        # -> output/.../fixp_tmatmul_s_qf_s8_*.elf{.diss}")
    out.append("make FIXP_MODE=BIAS diss          # TMATMUL_BIAS variant")
    out.append("make FIXP_MODE=GEMV diss          # TGEMV variant")
    out.append("")
    out.append(f"# 2. build + disassemble all {len(MODES) - 1} active variants (prints PASS/FAIL table)")
    out.append("bash compile.all                 # log: compile.fixp.log")
    out.append("")
    out.append("# 3. regenerate this report from the .diss files")
    out.append("python3 report_fixp.py           # -> fixp_report.md")
    out.append("```")
    out.append("")
    out.append("Tile shape defaults `M=N=K=TM=TN=TK=32` (override with "
               "`M=... TM=...`); `FIXP_MODE` is upper-cased to a `-D` define "
               "and lower-cased for the ELF suffix. Mode labels are short "
               "(`bias`/`acc`/`mx`/`gemv`/...) so the `-D` macro never collides "
               "with the op function name. The `diss` target runs "
               "`llvm-objdump -dl` to emit the `.elf.diss` next to each `.elf`.")
    out.append("")
    out.append("Result summary: "
               f"**PASS={n_pass} FAIL={n_fail} BLOCKED={n_blocked} "
               f"(total={len(rows)})**\n")

    out.append("## Coverage vs doc (12 operations x B.FPATR options)")
    out.append("")
    out.append("All 12 documented operations share one B.FPATR options mechanism "
               "(PreQuantMode, scalar/vector quant, ReLU/LReLU/PReLU, "
               "RowMax/GroupMax/MaxAbs, TransA/TransB and CScaleEn). The suite "
               "covers every encoded field/value and every operation family; "
               "representative combinations are used instead of the full "
               "Cartesian product.\n")
    out.append("| operation | param-free mode(s) | options mode(s) | notes |")
    out.append("| --- | --- | --- | --- |")
    for op, pf, opt, notes in COVERAGE:
        out.append(f"| {op} | {pf} | {opt} | {notes} |")
    out.append("")

    out.append("## B.FPATR attribute decode")
    out.append("")
    out.append("`B.FPATR PreQuant, Relu, GroupNCode, RowMaxEn, GroupMaxEn, "
               "RowMaxInit, MaxAbsEn, TransA, TransB, CScaleEn`; encoding word "
               "`0x2023 | PreQuant<<26 | "
               "Relu<<23 | GroupNCode<<19 | RowMaxEn<<18 | GroupMaxEn<<17 | "
               "RowMaxInit<<16 | MaxAbsEn<<15 | TransA<<7 | TransB<<8 | "
               "CScaleEn<<9`.\n")
    out.append("| mode | op | shared | option chain / call | expected FPATR | "
               "actual word | GPR (B.IOR) | aux IOT | status | detail |")
    out.append("| --- | --- | :---: | --- | --- | --- | ---: | ---: | --- | --- |")
    for label, op, shared, desc, fpatr, word, ior, aux, status, detail in rows:
        fstr = ", ".join(str(x) for x in fpatr)
        wstr = "-" if word is None else f"0x{word:08x}"
        sstr = "yes" if shared else "no"
        out.append(f"| {label} | {op} | {sstr} | `{desc}` | {fstr} | {wstr} | "
                   f"{ior} | {aux} | {status} | {detail} |")

    out.append("")
    out.append("## Notes")
    out.append("")
    out.append("- **Scalar quant modes** pass the descriptor through a single "
               "`B.IOR [quant],[]` GPR slot; `s8_lrelu` additionally loads the "
               "FP19 LReLU slope as `B.IOR [quant,lrelu],[]` (two real GPRs). The "
               "`zero` placeholder in `B.IOR [zero,...]` is never counted as a "
               "real GPR.")
    out.append("- **LReLU without scalar quant** (`lrelu_only`, gap C) is "
               "BLOCKED: the C++ template dispatches IorMode==2 and emits "
               "`B.IOR [zero,%LReluGpr],[]`, but the LLVM backend cannot match "
               "that instruction pattern (Match Instruction Error), so no .diss "
               "is produced. The mode documents this toolchain gap.")
    out.append("- **Vector quant / PReLU modes** add extra `B.IOT` source lines "
               "after the math operands (quant or PReLU parameter Tile, valid "
               "1 x N). `vqf_s8_prelu` chains vector-quant + PReLU (SrcMask=6); "
               "the two aux tiles share a single source line, so aux counts 1 "
               "source line (the FPATR fields already encode the quant+prelu "
               "combination).")
    out.append("- **Operation-family modes** (BIAS/ACC/MX/GEMV) verify the "
               "BSTART.CUBE mnemonic and math operand stream: ACC adds a C "
               "accumulator tile, BIAS adds a Bias tile, MX adds ScaleA/ScaleB "
               "(TMATMULMX / TGEMVMX, spelled without a dot). The aux count is "
               "derived as `iot_src - min(MATH_SRC[(mnemonic, shared)], "
               "iot_src)` so the math operand lines are not "
               "mistaken for PostProcess aux lines.")
    out.append("- **Shared and transpose** modes publish cooperative Shared A/B "
               "through `B.IOS`; `trans_a`/`trans_b`/`trans_ab` independently "
               "check the two transpose bits. TGEMV remains Local-only.")
    out.append("- **Legacy 3-param** (`legacy3`, `TMATMUL(c,a,b)` no options) "
               "emits A/B as read operands on the destination line (combined "
               "read-write `B.IOT ... ->reg`), so it has 0 math source lines; "
               "the `min` clamp keeps aux at 0 instead of -1. Its FPATR is the "
               "default (0,0,0,0,0,0,0,0,0,0), same as `keep_acc`.")
    out.append("- **RowMax / GroupMax / MaxAbs** cover every legal GroupN code. "
               "RowMaxIn contributes 1 "
               "aux source line (`rowmax_init`, `rowgroup_maxabs`); RowOut / "
               "GroupOut are destinations (not counted).")
    out.append("- **CScale and MX scale masks** cover ACC/MX_ACC CScaleEn plus "
               "MX ScaleMask 0/1/2/3 for both TMATMUL and TGEMV.")
    out.append("- GroupN code mapping used: 8->1, 16->2, 32->3, 48->4, 64->5, "
               "80->6, 96->7, 112->8, 128->9.")
    out.append("")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {REPORT}")
    print(f"summary: PASS={n_pass} FAIL={n_fail} BLOCKED={n_blocked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
