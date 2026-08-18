#!/usr/bin/python3

import argparse
import math
import os
import re
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


MAX_WORKERS = 8
DEFAULT_GFRUN = "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun"
DEFAULT_GFRUN_ARGS = " -t 1 -f "

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
CMP_ROOT = os.path.join(ROOT, "compare")

statics = {"pass": [], "fail": []}


def extract_int(name, elf_name, default=None):
    match = re.search(rf"{name}(\d+)", elf_name)
    if match:
        return int(match.group(1))
    if default is not None:
        return default
    raise ValueError(f"Cannot extract {name} from {elf_name}")


def parse_flashmla_shape(elf_name, args):
    shape = {
        "Sq": extract_int("Sq", elf_name),
        "QHeadPerHK": extract_int("QHeadPerHK", elf_name, 1),
        "NumBlocks": extract_int("NumBlocks", elf_name),
        "Dk": extract_int("Dk", elf_name),
        "Dv": extract_int("Dv", elf_name),
        "DChunk": extract_int("DChunk", elf_name, 128),
        "VChunk": extract_int("VChunk", elf_name, 128),
        "Tm": extract_int("Tm", elf_name),
        "Tk": extract_int("Tk", elf_name),
        "PageBlockSize": extract_int("PageBlockSize", elf_name, args.page_block_size),
    }
    shape["MaxBlocksPerSeq"] = extract_int("MaxBlocksPerSeq", elf_name, shape["NumBlocks"])
    return shape


def flashmla_reference(q_kernel, kv_flat, shape):
    """Reference for the tileop model in src/flashMLA.cpp.

    q_kernel: [Sq * QHeadPerHK, Dk], float16
    kv_flat : [NumBlocks * PageBlockSize, Dk], float16

    Returns:
      out_kernel: [Sq * QHeadPerHK, Dv], float16
      denom     : [Sq * QHeadPerHK], float32, matching current tileop lse_ptr
    """
    sq = shape["Sq"]
    q_head_per_hk = shape["QHeadPerHK"]
    dk = shape["Dk"]
    dv = shape["Dv"]
    num_blocks = shape["NumBlocks"]
    page_block_size = shape["PageBlockSize"]

    # Equivalent to FlashMLA reference_torch with:
    #   B=1, Hkv=1, block_table=[0, 1, ...], cache_seqlen=full length.
    # Shapes:
    #   query: [Hq, Sq, Dk]
    #   kv   : [1, Sk, Dk]
    query = q_kernel.reshape(sq, q_head_per_hk, dk).transpose(1, 0, 2).astype(np.float32)
    kv = kv_flat.reshape(num_blocks * page_block_size, 1, dk).transpose(1, 0, 2).astype(np.float32)

    attn_weight = np.matmul(query, np.swapaxes(kv, -2, -1))
    attn_weight /= math.sqrt(dk)
    row_max = np.max(attn_weight, axis=-1)
    exp_score = np.exp(attn_weight - row_max[..., None])
    denom = np.sum(exp_score, axis=-1)
    prob = exp_score / denom[..., None]
    out = np.matmul(prob, kv[..., :dv])

    # Tileop kernel output layout is [Sq * QHeadPerHK, Dv].
    out_kernel = out.transpose(1, 0, 2).reshape(sq * q_head_per_hk, dv).astype(np.float16)
    denom_kernel = denom.transpose(1, 0).reshape(sq * q_head_per_hk).astype(np.float32)
    return out_kernel, denom_kernel


def gen_input_and_golden(elf_name, path, args):
    print("Start to gen input data & golden data:", path, elf_name)
    shape = parse_flashmla_shape(elf_name, args)
    print("shape:", shape)

    q_seq_per_hk = shape["Sq"] * shape["QHeadPerHK"]
    kv_tokens = shape["NumBlocks"] * shape["PageBlockSize"]

    if getattr(args, "ones", False):
        # Deterministic all-ones smoke input: every Q/KV element == 1.0 (fp16).
        # With uniform inputs attention collapses to O = 1.0 and denom = Sk,
        # the documented flashMLA all-ones check (see
        # flashMLA_all_ones_validation_report.md).
        q_kernel = np.ones((q_seq_per_hk, shape["Dk"]), dtype=np.float16)
        kv_flat = np.ones((kv_tokens, shape["Dk"]), dtype=np.float16)
    else:
        rng = np.random.default_rng(args.seed)
        q_kernel = (rng.standard_normal((q_seq_per_hk, shape["Dk"]), dtype=np.float32) / 10.0)
        q_kernel = np.clip(q_kernel, -1.0, 1.0).astype(np.float16)
        kv_flat = (rng.standard_normal((kv_tokens, shape["Dk"]), dtype=np.float32) / 10.0)
        kv_flat = np.clip(kv_flat, -1.0, 1.0).astype(np.float16)
    golden_out, golden_denom = flashmla_reference(q_kernel, kv_flat, shape)

    q_kernel.tofile(os.path.join(path, "srcq.bin"))
    kv_flat.tofile(os.path.join(path, "srckv.bin"))
    golden_out.tofile(os.path.join(path, "golden.bin"))
    golden_denom.astype(np.float32).tofile(os.path.join(path, "golden_lse.bin"))

    # The local SuperScalarModel gfrun can open existing files but does not
    # reliably create missing output files through the guest O_CREAT path.
    np.zeros_like(golden_out, dtype=np.float16).tofile(os.path.join(path, "res.bin"))
    np.zeros_like(golden_denom, dtype=np.float32).tofile(os.path.join(path, "lse.bin"))

    # Optional kernel-side debug dumps. These files are harmless for normal
    # kernels and are populated only when flashMLA.cpp is built with
    # FLASHMLA_DEBUG_BIN.
    np.zeros((shape["Tm"], shape["Tk"]), dtype=np.float32).tofile(os.path.join(path, "dbg_score.bin"))
    np.zeros((shape["Tm"], shape["Tk"]), dtype=np.float32).tofile(os.path.join(path, "dbg_exp.bin"))
    np.zeros((shape["Tm"], 8), dtype=np.float32).tofile(os.path.join(path, "dbg_sum.bin"))
    np.zeros((shape["Tm"], shape["Tk"]), dtype=np.float32).tofile(os.path.join(path, "dbg_prob.bin"))
    np.zeros((shape["Tk"], shape["VChunk"]), dtype=np.float16).tofile(os.path.join(path, "dbg_v.bin"))
    np.zeros((shape["Tm"], shape["VChunk"]), dtype=np.float32).tofile(os.path.join(path, "dbg_pv.bin"))
    np.zeros((shape["Tm"], shape["VChunk"]), dtype=np.float32).tofile(os.path.join(path, "dbg_o.bin"))
    sub_blocks_per_page = shape["PageBlockSize"] // shape["Tk"]
    np.zeros((sub_blocks_per_page, shape["Tm"], shape["VChunk"]), dtype=np.float32).tofile(
        os.path.join(path, "dbg_pv_sub.bin")
    )
    np.zeros((sub_blocks_per_page, shape["Tm"], shape["VChunk"]), dtype=np.float32).tofile(
        os.path.join(path, "dbg_o_sub.bin")
    )
    np.zeros((sub_blocks_per_page, shape["Tk"], shape["VChunk"]), dtype=np.float16).tofile(
        os.path.join(path, "dbg_v_sub.bin")
    )
    np.zeros((sub_blocks_per_page, shape["Tm"], shape["Tk"]), dtype=np.float32).tofile(
        os.path.join(path, "dbg_prob_sub.bin")
    )
    np.zeros((sub_blocks_per_page, shape["Tm"], shape["Tk"]), dtype=np.float32).tofile(
        os.path.join(path, "dbg_score_sub.bin")
    )
    np.zeros((sub_blocks_per_page, shape["Tm"], shape["Tk"]), dtype=np.float32).tofile(
        os.path.join(path, "dbg_raw_score_sub.bin")
    )
    np.zeros((sub_blocks_per_page, shape["Dk"] // shape["DChunk"], shape["DChunk"], shape["Tk"]), dtype=np.float16).tofile(
        os.path.join(path, "dbg_k_sub.bin")
    )


def run_qemu(elf, args):
    print("Start to run gfrun----------")
    try:
        if not os.path.exists(elf):
            print("elf not exist")
            return elf, "not exist", ""

        if args.plat == "cpu":
            cmd = elf
        else:
            cmd = args.gfrun + args.gfrun_args + elf
        proc = subprocess.Popen(
            [cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
            preexec_fn=os.setsid,
        )
        stdout, stderr = proc.communicate(timeout=args.timeout)
        output = stdout + stderr
        status = "pass" if proc.returncode == 0 else "fail"
        print("gfrun status:", status)
        return elf, status, output
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return elf, "timeout", "Timeout expired"
    except Exception as exc:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return elf, "error", str(exc)


def compare_array(cmp_data, golden_data, dtype, atol, rtol):
    res = np.fromfile(cmp_data, dtype=dtype).astype(np.float32)
    ref = np.fromfile(golden_data, dtype=dtype).astype(np.float32)
    if res.shape != ref.shape:
        return "fail", {"reason": f"shape mismatch: got {res.shape}, ref {ref.shape}"}
    diff = res - ref
    mse = float(np.mean(diff * diff)) if diff.size else 0.0
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    close = np.allclose(res, ref, atol=atol, rtol=rtol)
    return ("pass" if close else "fail"), {"mse": mse, "max_abs": max_abs}


def result_compare(cmp_path, args):
    out_status, out_metric = compare_array(
        os.path.join(cmp_path, "res.bin"),
        os.path.join(cmp_path, "golden.bin"),
        np.float16,
        args.out_atol,
        args.out_rtol,
    )

    lse_status = "skip"
    lse_metric = {"reason": "lse.bin not found"}
    lse_path = os.path.join(cmp_path, "lse.bin")
    if os.path.exists(lse_path):
        lse_status, lse_metric = compare_array(
            lse_path,
            os.path.join(cmp_path, "golden_lse.bin"),
            np.float32,
            args.lse_atol,
            args.lse_rtol,
        )

    status = "pass" if out_status == "pass" and lse_status in ("pass", "skip") else "fail"
    return status, {"out": out_metric, "lse": lse_metric}


def _format_matrix_head(name, data, rows=4, cols=8):
    out = [f"\n[{name}] shape={data.shape}"]
    out.append(
        "  min/max/mean/nonzero: "
        f"{float(np.min(data)):.8g} / {float(np.max(data)):.8g} / "
        f"{float(np.mean(data)):.8g} / {int(np.count_nonzero(data))}"
    )
    out.append("  head:")
    head = data[:rows, :cols] if data.ndim == 2 else data[:rows]
    out.append(np.array2string(head, precision=8, suppress_small=False))
    return "\n".join(out)


def _format_diff(name, got, ref, rows=4, cols=8):
    diff = got - ref
    max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    out = [_format_matrix_head(f"{name}/got", got, rows, cols)]
    out.append(_format_matrix_head(f"{name}/ref", ref, rows, cols))
    out.append(
        f"  diff max_abs={float(np.max(np.abs(diff))):.8g} "
        f"mse={float(np.mean(diff * diff)):.8g} "
        f"idx={max_idx} got={float(got[max_idx]):.8g} ref={float(ref[max_idx]):.8g}"
    )
    return "\n".join(out)


def dump_debug_report(cmp_path, elf_name, args):
    dbg_score_path = os.path.join(cmp_path, "dbg_score.bin")
    if not os.path.exists(dbg_score_path):
        return None

    shape = parse_flashmla_shape(elf_name, args)
    sq = shape["Sq"] * shape["QHeadPerHK"]
    dk = shape["Dk"]
    dv = shape["Dv"]
    tm = shape["Tm"]
    tk = shape["Tk"]
    vchunk = shape["VChunk"]
    kv_tokens = shape["NumBlocks"] * shape["PageBlockSize"]

    q = np.fromfile(os.path.join(cmp_path, "srcq.bin"), dtype=np.float16).reshape(sq, dk).astype(np.float32)
    kv = np.fromfile(os.path.join(cmp_path, "srckv.bin"), dtype=np.float16).reshape(kv_tokens, dk).astype(np.float32)
    res = np.fromfile(os.path.join(cmp_path, "res.bin"), dtype=np.float16).reshape(sq, dv).astype(np.float32)
    golden = np.fromfile(os.path.join(cmp_path, "golden.bin"), dtype=np.float16).reshape(sq, dv).astype(np.float32)
    lse = np.fromfile(os.path.join(cmp_path, "lse.bin"), dtype=np.float32)
    golden_lse = np.fromfile(os.path.join(cmp_path, "golden_lse.bin"), dtype=np.float32)

    dbg_score = np.fromfile(dbg_score_path, dtype=np.float32).reshape(tm, tk)
    dbg_exp = np.fromfile(os.path.join(cmp_path, "dbg_exp.bin"), dtype=np.float32).reshape(tm, tk)
    dbg_sum = np.fromfile(os.path.join(cmp_path, "dbg_sum.bin"), dtype=np.float32).reshape(tm, 8)
    dbg_prob = np.fromfile(os.path.join(cmp_path, "dbg_prob.bin"), dtype=np.float32).reshape(tm, tk)
    dbg_v = np.fromfile(os.path.join(cmp_path, "dbg_v.bin"), dtype=np.float16).reshape(tk, vchunk).astype(np.float32)
    dbg_pv = np.fromfile(os.path.join(cmp_path, "dbg_pv.bin"), dtype=np.float32).reshape(tm, vchunk)
    dbg_o = np.fromfile(os.path.join(cmp_path, "dbg_o.bin"), dtype=np.float32).reshape(tm, vchunk)
    sub_blocks_per_page = shape["PageBlockSize"] // tk
    dbg_pv_sub = np.fromfile(os.path.join(cmp_path, "dbg_pv_sub.bin"), dtype=np.float32).reshape(
        sub_blocks_per_page, tm, vchunk
    )
    dbg_o_sub = np.fromfile(os.path.join(cmp_path, "dbg_o_sub.bin"), dtype=np.float32).reshape(
        sub_blocks_per_page, tm, vchunk
    )
    dbg_v_sub = np.fromfile(os.path.join(cmp_path, "dbg_v_sub.bin"), dtype=np.float16).reshape(
        sub_blocks_per_page, tk, vchunk
    ).astype(np.float32)
    dbg_prob_sub = np.fromfile(os.path.join(cmp_path, "dbg_prob_sub.bin"), dtype=np.float32).reshape(
        sub_blocks_per_page, tm, tk
    )
    dbg_score_sub = np.fromfile(os.path.join(cmp_path, "dbg_score_sub.bin"), dtype=np.float32).reshape(
        sub_blocks_per_page, tm, tk
    )
    dbg_raw_score_sub = np.fromfile(os.path.join(cmp_path, "dbg_raw_score_sub.bin"), dtype=np.float32).reshape(
        sub_blocks_per_page, tm, tk
    )
    dbg_k_sub = np.fromfile(os.path.join(cmp_path, "dbg_k_sub.bin"), dtype=np.float16).reshape(
        sub_blocks_per_page, dk // shape["DChunk"], shape["DChunk"], tk
    ).astype(np.float32)

    scale = 1.0 / math.sqrt(dk)
    score_all = np.matmul(q[:tm], kv.T) * scale
    rowmax_all = np.max(score_all, axis=1)
    exp_all = np.exp(score_all - rowmax_all[:, None])
    denom = np.sum(exp_all, axis=1)
    prob_all = exp_all / denom[:, None]
    out_ref = np.matmul(prob_all, kv[:, :dv]).astype(np.float16).astype(np.float32)

    score_ref = score_all[:, :tk]
    exp_first_sub_ref = np.exp(score_ref - np.max(score_ref, axis=1, keepdims=True))
    prob_ref = prob_all[:, :tk]
    v_ref = kv[:tk, :vchunk]
    pv_first_sub_ref = np.matmul(prob_ref, v_ref)
    pv_sub_ref = []
    o_sub_ref = []
    running_o = np.zeros((tm, vchunk), dtype=np.float32)
    for sub in range(sub_blocks_per_page):
        col0 = sub * tk
        col1 = col0 + tk
        pv = np.matmul(prob_all[:, col0:col1], kv[col0:col1, :vchunk])
        running_o = running_o + pv
        pv_sub_ref.append(pv)
        o_sub_ref.append(running_o.copy())
    pv_sub_ref = np.stack(pv_sub_ref)
    o_sub_ref = np.stack(o_sub_ref)

    dchunk = shape["DChunk"]
    dchunk_notes = []
    for d_block in range(dk // dchunk):
        ref = np.matmul(
            q[:tm, d_block * dchunk:(d_block + 1) * dchunk],
            kv[:tk, d_block * dchunk:(d_block + 1) * dchunk].T,
        ) * scale
        diff = dbg_score - ref
        dchunk_notes.append(
            f"  DChunk {d_block}: max_abs={float(np.max(np.abs(diff))):.8g}, "
            f"mse={float(np.mean(diff * diff)):.8g}, ref_norm={float(np.linalg.norm(ref)):.8g}"
        )

    lines = [
        f"flashMLA debug compare report",
        f"elf: {elf_name}",
        f"compare dir: {cmp_path}",
        f"shape: {shape}",
        "",
        "QK score difference summary:",
        _format_diff("QK score first sub: dbg_score vs q@k^T/sqrt(Dk)", dbg_score, score_ref),
        "",
        "Check whether dbg_score looks like one DChunk contribution:",
        *dchunk_notes,
        "",
        _format_diff("exp first sub", dbg_exp, exp_first_sub_ref),
        "",
        _format_matrix_head("dbg_sum physical [Tm,8]", dbg_sum, 8, 8),
        "  dbg_sum[:,0] vs full denominator:",
        f"  max_abs={float(np.max(np.abs(dbg_sum[:, 0] - denom))):.8g}",
        f"  got={np.array2string(dbg_sum[:, 0], precision=8)}",
        f"  ref={np.array2string(denom, precision=8)}",
        "",
        _format_diff("prob first sub", dbg_prob, prob_ref),
        "",
        _format_diff("V first sub", dbg_v, v_ref),
        "",
        _format_diff("PV first sub", dbg_pv, pv_first_sub_ref),
        "",
        "PV/O per sub-block:",
        *[
            (
                f"  sub {sub}: "
                f"RawScore min/max/mean={float(np.min(dbg_raw_score_sub[sub])):.8g}/"
                f"{float(np.max(dbg_raw_score_sub[sub])):.8g}/{float(np.mean(dbg_raw_score_sub[sub])):.8g}; "
                f"Score min/max/mean={float(np.min(dbg_score_sub[sub])):.8g}/"
                f"{float(np.max(dbg_score_sub[sub])):.8g}/{float(np.mean(dbg_score_sub[sub])):.8g}; "
                f"K dchunk means="
                f"{[float(np.mean(dbg_k_sub[sub, d])) for d in range(dbg_k_sub.shape[1])]}; "
                f"Prob min/max/mean={float(np.min(dbg_prob_sub[sub])):.8g}/"
                f"{float(np.max(dbg_prob_sub[sub])):.8g}/{float(np.mean(dbg_prob_sub[sub])):.8g}; "
                f"V min/max/mean={float(np.min(dbg_v_sub[sub])):.8g}/"
                f"{float(np.max(dbg_v_sub[sub])):.8g}/{float(np.mean(dbg_v_sub[sub])):.8g}; "
                f"PV max_abs={float(np.max(np.abs(dbg_pv_sub[sub] - pv_sub_ref[sub]))):.8g}, "
                f"PV mean={float(np.mean(dbg_pv_sub[sub])):.8g}, ref={float(np.mean(pv_sub_ref[sub])):.8g}; "
                f"O max_abs={float(np.max(np.abs(dbg_o_sub[sub] - o_sub_ref[sub]))):.8g}, "
                f"O mean={float(np.mean(dbg_o_sub[sub])):.8g}, ref={float(np.mean(o_sub_ref[sub])):.8g}"
            )
            for sub in range(sub_blocks_per_page)
        ],
        "",
        _format_diff("final O tile", dbg_o, out_ref[:, :vchunk]),
        "",
        _format_diff("final res.bin", res[:tm, :vchunk], golden[:tm, :vchunk]),
        "",
        "final lse:",
        f"  got={np.array2string(lse[:tm], precision=8)}",
        f"  ref={np.array2string(golden_lse[:tm], precision=8)}",
        f"  diff={np.array2string(lse[:tm] - golden_lse[:tm], precision=8)}",
    ]

    log_path = os.path.join(cmp_path, "debug_compare.log")
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return log_path


def check_elf(elf, args):
    print("Start to check elf----------------")
    elf_name = os.path.basename(elf).replace(".elf", "").strip()
    cmp_data_path = os.path.join(CMP_ROOT, elf_name)

    os.system(f"rm -rf {cmp_data_path}; mkdir -p {cmp_data_path}")
    os.makedirs(cmp_data_path, exist_ok=True)

    gen_input_and_golden(elf_name, cmp_data_path, args)
    _, status, output = run_qemu(elf, args)
    if status != "pass":
        return elf_name, status, "not chk", output

    cmp_data = os.path.join(cmp_data_path, "res.bin")
    if not os.path.exists(cmp_data):
        return elf_name, status, "output not exist", "NaN"

    chk_status, metric = result_compare(cmp_data_path, args)
    debug_log = dump_debug_report(cmp_data_path, elf_name, args)
    if debug_log is not None:
        print("debug compare log:", debug_log)
    return elf_name, status, chk_status, metric


def log_result(elf, status, chk_status, metric):
    item = f"{elf} -> run_status: {status.upper()} chk_status: {chk_status.upper()} metric:{metric}"
    if status == "pass" and chk_status == "pass":
        statics["pass"].append(item)
    else:
        statics["fail"].append(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run flashMLA gfrun result check")
    parser.add_argument("-l", dest="elf_list", default="./tmp.list", type=str)
    parser.add_argument("-d", dest="dbg_elf", default=None, type=str)
    parser.add_argument("-o", dest="res_log", default="flashMLA_result_check.log", type=str)
    parser.add_argument("-plat", dest="plat", default="linx", type=str)
    parser.add_argument("--gfrun", default=DEFAULT_GFRUN, type=str)
    parser.add_argument("--gfrun-args", default=DEFAULT_GFRUN_ARGS, type=str)
    parser.add_argument("--timeout", default=1200, type=int)
    parser.add_argument("--ones", action="store_true",
                        help="use all-ones fp16 input instead of random (deterministic smoke check)")
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--page-block-size", default=64, type=int)
    parser.add_argument("--out-atol", default=2e-2, type=float)
    parser.add_argument("--out-rtol", default=2e-2, type=float)
    parser.add_argument("--lse-atol", default=5e-2, type=float)
    parser.add_argument("--lse-rtol", default=5e-2, type=float)
    parser.add_argument("--workers", default=MAX_WORKERS, type=int)
    args = parser.parse_args()

    if args.dbg_elf is not None:
        elf_paths = [args.dbg_elf]
    else:
        with open(args.elf_list, "r") as f:
            elf_paths = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_elf, elf, args): elf for elf in elf_paths}
        for future in as_completed(futures):
            elf, status, chk_status, metric = future.result()
            print(f"{elf} -> run_status: {status.upper()} chk_status: {chk_status.upper()} metric:{metric}")
            log_result(elf, status, chk_status, metric)

    if args.dbg_elf is None:
        os.makedirs(CMP_ROOT, exist_ok=True)
        with open(os.path.join(CMP_ROOT, args.res_log), "w") as f:
            f.write("\nResult_Check Summary:\n")
            f.write(f"\npass : {len(statics['pass'])}\n")
            f.write(f"\nfail : {len(statics['fail'])}\n")
            f.write("\npass list:\n")
            for item in statics["pass"]:
                f.write(f"{item}\n")
            f.write("\n\nfail list:\n")
            for item in statics["fail"]:
                f.write(f"{item}\n")
