#!/usr/bin/env python3
"""Golden generator + checker for fa_lowp_logdomain (MXFP4 FA, algorithm B).

Pure-Python (no numpy).  Generates random Q/K/V, MXFP4-quantizes them
(E2M1x2 data + one E8M0 scale per 32-wide group along the reduction dim),
writes the kernel's input bins, and computes the reference output as exact FA
on the *dequantized* inputs.

Layout contract (matches fa_lowp_logdomain.cpp):
  srcq.bin        __fp4_e2m1x2 [Sq,  qD/2]   (K-dim packed, low nibble = 2j)
  srck.bin        __fp4_e2m1x2 [Skv, qD/2]
  srcv.bin        __fp4_e2m1x2 [Skv, vD/2]   (vD packed, PTO #348)
  srcq_scale.bin  __fp8_e8m0   [Sq,  qD/32]
  srck_scale.bin  __fp8_e8m0   [Skv, qD/32]
  srcv_scale.bin  __fp8_e8m0   [vD, Skv/32]   (PTO #343: ScaleB is [N,G_B])
  res.bin         __bf16       [Sq,  vD]
"""
import argparse
import math
import random
import struct
from pathlib import Path

SQ = SKV = 256
QD = VD = 128
GROUP = 32

E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def f32_to_bf16_bits(x):
    b = f32_bits(x)
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFFFFFF
    return (b >> 16) & 0xFFFF


def bf16_bits_to_f32(h):
    return struct.unpack("<f", struct.pack("<I", (h & 0xFFFF) << 16))[0]


def f32_to_e2m1(x):
    sign = 1 if (x < 0 or (x == 0 and math.copysign(1.0, x) < 0)) else 0
    ax = abs(x)
    if ax >= 6.0:
        code = 7
    else:
        best, code = 1e30, 0
        for c, m in enumerate(E2M1_MAG):
            d = abs(ax - m)
            if d < best or (d == best and c % 2 == 0):
                best, code = d, c
    return (sign << 3) | code


def e2m1_to_f32(code):
    sign = -1.0 if (code >> 3) & 1 else 1.0
    return sign * E2M1_MAG[code & 0x7]


def e8m0_to_f32(byte):
    return math.ldexp(1.0, (byte & 0xFF) - 127)


def quant_group(vals):
    amax = max((abs(float(v)) for v in vals), default=0.0)
    if amax == 0.0:
        return 0, [0] * len(vals)
    e = math.floor(math.log2(amax))
    byte = max(0, min(254, 127 + e - 2))          # scale = 2^(floor(log2(amax))-2)
    scale = e8m0_to_f32(byte)
    return byte, [f32_to_e2m1(float(v) / scale) for v in vals]


def main():
    global SQ, SKV
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--algorithm", choices=["B", "C"], default="C")
    ap.add_argument("--atol", type=float, default=5e-2)
    ap.add_argument("--rtol", type=float, default=5e-2)
    ap.add_argument("--positive", action="store_true")
    ap.add_argument("--sq", type=int, default=SQ)
    ap.add_argument("--skv", type=int, default=SKV)
    args = ap.parse_args()
    SQ, SKV = args.sq, args.skv
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(args.seed)
    q = [[rnd.uniform(-1.0, 1.0) for _ in range(QD)] for _ in range(SQ)]
    k = [[rnd.uniform(-1.0, 1.0) for _ in range(QD)] for _ in range(SKV)]
    v = [[rnd.uniform(-1.0, 1.0) for _ in range(VD)] for _ in range(SKV)]
    # 1/sqrt(head_dim) is folded into Q at quantization time (algorithm team);
    # the kernel applies no separate score scale.
    qscale = 1.0 / math.sqrt(QD)
    q = [[x * qscale for x in row] for row in q]
    if args.positive:
        q = [[abs(x) for x in row] for row in q]
        k = [[abs(x) for x in row] for row in k]
        v = [[abs(x) for x in row] for row in v]

    # quantize Q/K along QD (per row); V along SKV (per column)
    qd = [[0.0] * QD for _ in range(SQ)]
    qcode = [[0] * QD for _ in range(SQ)]
    qs = bytearray(SQ * (QD // GROUP))
    for r in range(SQ):
        for g in range(QD // GROUP):
            b, codes = quant_group(q[r][g * GROUP:(g + 1) * GROUP])
            qs[r * (QD // GROUP) + g] = b
            sc = e8m0_to_f32(b)
            for j, c in enumerate(codes):
                qd[r][g * GROUP + j] = e2m1_to_f32(c) * sc
                qcode[r][g * GROUP + j] = c
    kd = [[0.0] * QD for _ in range(SKV)]
    kcode = [[0] * QD for _ in range(SKV)]
    ks = bytearray(SKV * (QD // GROUP))
    for r in range(SKV):
        for g in range(QD // GROUP):
            b, codes = quant_group(k[r][g * GROUP:(g + 1) * GROUP])
            ks[r * (QD // GROUP) + g] = b
            sc = e8m0_to_f32(b)
            for j, c in enumerate(codes):
                kd[r][g * GROUP + j] = e2m1_to_f32(c) * sc
                kcode[r][g * GROUP + j] = c
    vd = [[0.0] * VD for _ in range(SKV)]
    vcode = [[0] * VD for _ in range(SKV)]
    vs = bytearray(VD * (SKV // GROUP))
    for g in range(SKV // GROUP):
        for n in range(VD):
            b, codes = quant_group([v[g * GROUP + j][n] for j in range(GROUP)])
            vs[n * (SKV // GROUP) + g] = b
            sc = e8m0_to_f32(b)
            for j, c in enumerate(codes):
                vd[g * GROUP + j][n] = e2m1_to_f32(c) * sc
                vcode[g * GROUP + j][n] = c

    # pack data from the quantized codes (scaled), not the raw values
    qbuf = bytearray(SQ * (QD // 2))
    for r in range(SQ):
        for j in range(QD // 2):
            lo = qcode[r][2 * j]; hi = qcode[r][2 * j + 1]
            qbuf[r * (QD // 2) + j] = (lo & 0xF) | ((hi & 0xF) << 4)
    kbuf = bytearray(SKV * (QD // 2))
    for r in range(SKV):
        for j in range(QD // 2):
            lo = kcode[r][2 * j]; hi = kcode[r][2 * j + 1]
            kbuf[r * (QD // 2) + j] = (lo & 0xF) | ((hi & 0xF) << 4)
    # fp4x2 packing follows the matrix's declared column axis (TLOAD "packed
    # four-bit columns"): V declares [K,N] with TransB, so pairs run along vD.
    vbuf = bytearray(SKV * (VD // 2))
    for k in range(SKV):
        for j in range(VD // 2):
            lo = vcode[k][2 * j]; hi = vcode[k][2 * j + 1]
            vbuf[k * (VD // 2) + j] = (lo & 0xF) | ((hi & 0xF) << 4)

    (out / "srcq.bin").write_bytes(bytes(qbuf))
    (out / "srck.bin").write_bytes(bytes(kbuf))
    (out / "srcv.bin").write_bytes(bytes(vbuf))
    (out / "srcq_scale.bin").write_bytes(bytes(qs))
    (out / "srck_scale.bin").write_bytes(bytes(ks))
    (out / "srcv_scale.bin").write_bytes(bytes(vs))

    # golden: exact FA on dequantized inputs
    inv = 1.0  # no 1/sqrt(head_dim): matches the kernel convention
    o = [[0.0] * VD for _ in range(SQ)]
    for r in range(SQ):
        srow = [0.0] * SKV
        for c in range(SKV):
            s = 0.0
            qr, kr = qd[r], kd[c]
            for d in range(QD):
                s += qr[d] * kr[d]
            srow[c] = s * inv
        m = max(srow)
        p = [math.exp(x - m) for x in srow]
        z = sum(p)
        orow = o[r]
        for c in range(SKV):
            pc = p[c] / z
            vc = vd[c]
            for n in range(VD):
                orow[n] += pc * vc[n]
    golden = bytearray(SQ * VD * 2)
    for i, x in enumerate([orow[n] for r in range(SQ) for n in range(VD)]):
        struct.pack_into("<H", golden, 2 * i, f32_to_bf16_bits(x))
    (out / "golden_bf16.bin").write_bytes(bytes(golden))

    # reference QK (first 32 rows x first 128 keys) from our own qd/kd
    refscore = bytearray(32 * 128 * 2)
    for r in range(32):
        for c in range(min(128, SKV)):
            s = 0.0
            for dd in range(QD):
                s += qd[r][dd] * kd[c][dd]
            struct.pack_into("<H", refscore, 2 * (r * 128 + c),
                             f32_to_bf16_bits(s))
    (out / "ref_score_bf16.bin").write_bytes(bytes(refscore))

    # goldB: faithful reimplementation of algorithm B (E2M1 P data + E8M0 group
    # scale with floor_pow2(m/4)), normalized by the *unquantized* row sum.
    def floor_pow2(x):
        if x <= 0.0:
            return 0.0
        return math.ldexp(1.0, math.floor(math.log2(x)))

    # B normalizes by the *unquantized* row sum zt=sum(t); C (self-normalized)
    # by the *quantized* sum zp=sum(phat).  Both share the same phat.
    oB = [[0.0] * VD for _ in range(SQ)]
    oC = [[0.0] * VD for _ in range(SQ)]
    for r in range(SQ):
        srow = [0.0] * SKV
        for c in range(SKV):
            s = 0.0
            qr, kr = qd[r], kd[c]
            for d in range(QD):
                s += qr[d] * kr[d]
            srow[c] = s * inv
        mx = max(srow)
        t = [math.exp(x - mx) for x in srow]
        zt = sum(t)
        phat = [0.0] * SKV
        for g in range(SKV // GROUP):
            tg = t[g * GROUP:(g + 1) * GROUP]
            mg = max(tg)
            sc = floor_pow2(mg * 0.25)
            for j, tv in enumerate(tg):
                code = f32_to_e2m1(4.0 * tv / mg) if mg > 0 else 0
                phat[g * GROUP + j] = e2m1_to_f32(code) * sc
        zp = sum(phat)
        for c in range(SKV):
            vc = vd[c]
            pcB = phat[c] / zt
            pcC = phat[c] / zp if zp > 0.0 else 0.0
            for n in range(VD):
                oB[r][n] += pcB * vc[n]
                oC[r][n] += pcC * vc[n]
    goldB = bytearray(SQ * VD * 2)
    for i, x in enumerate([oB[r][n] for r in range(SQ) for n in range(VD)]):
        struct.pack_into("<H", goldB, 2 * i, f32_to_bf16_bits(x))
    (out / "goldenB_bf16.bin").write_bytes(bytes(goldB))
    goldC = bytearray(SQ * VD * 2)
    for i, x in enumerate([oC[r][n] for r in range(SQ) for n in range(VD)]):
        struct.pack_into("<H", goldC, 2 * i, f32_to_bf16_bits(x))
    (out / "goldenC_bf16.bin").write_bytes(bytes(goldC))

    print(f"wrote bins + golden to {out}")

    if args.check:
        for name in ("srcq", "srck", "srcv", "srcq_scale", "srck_scale", "srcv_scale"):
            if (out / (name + ".bin")).read_bytes() != (out / (name + "_readback.bin")).read_bytes():
                raise SystemExit("FAIL input readback: " + name)
        res = (out / "res.bin").read_bytes()
        act = [bf16_bits_to_f32(struct.unpack_from("<H", res, 2 * i)[0])
               for i in range(len(res) // 2)]
        for name, blob in (("exactFA", golden), ("algoB", goldB),
                           ("algoC", goldC)):
            if name != "algo" + args.algorithm:
                continue
            ref = [bf16_bits_to_f32(struct.unpack_from("<H", blob, 2 * i)[0])
                   for i in range(len(blob) // 2)]
            if len(act) != len(ref):
                print(f"{name}: FAIL size act={len(act)} ref={len(ref)}")
                raise SystemExit(1)
            if not any(ref):
                raise SystemExit("FAIL vacuous golden")
            adiff = [abs(a - b) for a, b in zip(act, ref)]
            bad = sum(1 for a, b in zip(act, ref)
                      if not math.isfinite(a) or abs(a - b) > args.atol + args.rtol * abs(b))
            print(f"{name}: {'PASS' if bad == 0 else 'FAIL'}  "
                  f"max_abs={max(adiff):.3e} mean_abs={sum(adiff)/len(adiff):.3e} "
                  f"bad={bad}/{len(act)}")

            if bad:
                raise SystemExit(1)

if __name__ == "__main__":
    main()
