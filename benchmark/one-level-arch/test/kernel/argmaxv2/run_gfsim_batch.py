#!/usr/bin/env python3
"""ArgMaxV2 50-case gfsim performance batch (real L2, random SoC seed 2, 4PE).

Per the supernpu-gfsim-validation skill: timing ELFs are built WITHOUT res_check
(_start.s broadcasts main to all four PEs); each case runs through the skill's
run_gfsim_matrix.sh (explicit --conf fourpe --pto-v02 true, real L2, random SoC
seed 2, 900s bound); PASS criteria and log parsing come from that script. This
orchestrator only selects cases, builds ELFs, parallelizes runs, and aggregates.

Selection: 50 default-tile cases from the 200-case accuracy suite (report.tsv),
covering dtype x path x size x PE-partition (SPMD vs PE0-fallback) x idtype.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent.parent.parent                # .../one-level-arch
REPO = ARCH.parents[1]
# Optional CLI arg: output validation dir (default keeps the original run path).
# Pass a new dir to preserve a previous run for A/B comparison.
VALIDATION = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1
              else REPO / ".validation/argmaxv2_gfsim_50")
OUTPUT = ARCH / "output/kernel/argmaxv2/elf"
REPORT200 = REPO / ".validation/argmaxv2_fuzz_200/report.tsv"


def _req_env(name: str, hint: str) -> Path:
    """仓库约定（compile.all）：外部工具路径经环境变量注入，禁止硬编码。"""
    v = os.environ.get(name)
    if not v:
        sys.exit(f"error: 环境变量 {name} 未设置（{hint}）")
    return Path(v)

MODEL_DIR = _req_env("SUPERSCALAR_MODEL_DIR", "SuperScalarModel 仓库根目录（含 bin/gfsim、bin/gfrun）")
COMPILER_DIR = _req_env("COMPILER_DIR", "Linx 编译器 bin 目录（linx_blockisa_llvm_musl/bin）")
SKILL_SCRIPT = _req_env("GFSIM_MATRIX_SCRIPT", "gfsim 矩阵执行脚本 run_gfsim_matrix.sh 路径（supernpu-gfsim-validation 技能）")

TIMEOUT_S = 900
BUILD_JOBS = 8
RUN_JOBS = 4

# 50-case selection: fp32 x15, fp16 x13, bf16 x8, i32 x14
SELECTION = [
    # fp32 rows (11): tiny / SPMD small / rmd both / exact / rmd_M / fb / red1 / 3D / 4D / 5D / 524K stress
    "f000", "f006", "f009", "f010", "f011", "f012", "f015", "f020", "f021", "f024", "f028",
    # fp32 cols (4): 200/1000/2048 rows, 4D
    "f034", "f037", "f042", "f047",
    # fp16 rows (9)
    "f053", "f056", "f059", "f060", "f061", "f065", "f070", "f071", "f074",
    # fp16 cols (4)
    "f084", "f087", "f092", "f093",
    # bf16 rows (5)
    "f103", "f109", "f110", "f120", "f124",
    # bf16 cols (3)
    "f134", "f142", "f143",
    # i32 rows (10)
    "f150", "f153", "f156", "f159", "f160", "f161", "f162", "f165", "f170", "f174",
    # i32 cols (4)
    "f184", "f187", "f192", "f197",
]


def load_case_meta() -> dict:
    import csv
    meta = {}
    with open(REPORT200) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            meta[r["cid"]] = r
    return meta


def build_one(cid: str, log_dir: Path) -> tuple[str, str]:
    log = log_dir / f"{cid}.build.log"
    cmd = ["make", f"TESTCASE={cid}", "SUFFIX=_t", f"COMPILER_DIR={COMPILER_DIR}"]
    try:
        proc = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=600)
        log.write_text(proc.stdout, encoding="utf-8")
        elf = OUTPUT / f"kernel_argmaxv2_{cid}_PE4_t.elf"
        if proc.returncode != 0 or not elf.is_file():
            return "BUILD_FAIL", f"rc={proc.returncode}; log={log}"
        return "PASS", "built"
    except subprocess.TimeoutExpired:
        log.write_text("build timeout", encoding="utf-8")
        return "BUILD_FAIL", "build timeout"


def run_one(cid: str, meta: dict) -> dict:
    elf = OUTPUT / f"kernel_argmaxv2_{cid}_PE4_t.elf"
    out_dir = VALIDATION / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(SKILL_SCRIPT), "--model-dir", str(MODEL_DIR),
           "--elf", str(elf), "--case", cid, "--out-dir", str(out_dir),
           "--mode", "real", "--soc", "random", "--soc-seed", "2",
           "--timeout", str(TIMEOUT_S)]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    wall = time.time() - t0

    row = dict(cid=cid, wall_s=round(wall, 1), status="TEST_INFRA_FAIL",
               total_cycles="NA", waw="NA", exit_code="NA",
               log="", pe_active="", vec_cycles="", tlsu_cycles="")
    # parse the skill script's summary.tsv
    summary = out_dir / "summary.tsv"
    if summary.is_file():
        import csv
        rows = list(csv.DictReader(open(summary), delimiter="\t"))
        if rows:
            r = rows[-1]
            row.update(status=r["status"], total_cycles=r["total_cycles"],
                       waw=r["scb_waw_violation"], exit_code=r["exit_code"],
                       log=r["log"])
    # per-PE Vector activity + engine cycle breakdown from the run log
    if row["log"] and Path(row["log"]).is_file():
        text = Path(row["log"]).read_text(errors="replace")
        active = []
        for pe in range(4):
            import re
            m = re.search(rf"Vector PMU Stats PE{pe}.*?TileopNum\.+:\s*(\d+)",
                          text, re.S)
            if m and int(m.group(1)) > 0:
                active.append(pe)
        row["pe_active"] = ",".join(map(str, active))
        m = re.search(r"Vector Tileop Total Cycles\.+:\s*(\d+)", text)
        row["vec_cycles"] = m.group(1) if m else ""
        m = re.search(r"TLSU Tileop Total Cycles\.+:\s*(\d+)", text)
        row["tlsu_cycles"] = m.group(1) if m else ""
        if row["status"] == "PASS" and len(active) not in (0, 4):
            # PE partitioning: SPMD/col cases should use all 4 PEs; PE0-fallback
            # cases legitimately use only PE0. Flag only mismatches.
            dims = [int(d) for d in meta[cid]["dims"].split(",")]
            ax = int(meta[cid]["axis"]) % len(dims)
            outer = 1
            for d in dims[:ax]:
                outer *= d
            if outer % 4 == 0 and len(active) != 4:
                row["status"] = "PE_ACTIVITY_FAIL"
    return row


def main() -> int:
    assert len(SELECTION) == 50, f"expected 50 cases, got {len(SELECTION)}"
    meta = load_case_meta()
    unknown = [c for c in SELECTION if c not in meta]
    if unknown:
        sys.exit(f"unknown cids: {unknown}")
    VALIDATION.mkdir(parents=True, exist_ok=True)
    (VALIDATION / "build").mkdir(exist_ok=True)

    print(f"[SELECT] {len(SELECTION)} cases", flush=True)
    with ThreadPoolExecutor(max_workers=BUILD_JOBS) as pool:
        builds = dict(zip(SELECTION, pool.map(
            lambda c: build_one(c, VALIDATION / "build"), SELECTION)))
    failed = [c for c, (s, _) in builds.items() if s != "PASS"]
    print(f"[BUILD] {len(SELECTION) - len(failed)}/{len(SELECTION)} timing ELFs built"
          + (f"; failed: {failed}" if failed else ""), flush=True)
    runnable = [c for c in SELECTION if builds[c][0] == "PASS"]

    print(f"[RUN] {len(runnable)} cases, {RUN_JOBS} gfsim workers, "
          f"timeout {TIMEOUT_S}s each", flush=True)
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=RUN_JOBS) as pool:
        for i, row in enumerate(pool.map(lambda c: run_one(c, meta), runnable), 1):
            results.append(row)
            flag = "" if row["status"] == "PASS" else f" <<<< {row['status']}"
            print(f"[RUN {i:02d}/{len(runnable)}] {row['cid']} "
                  f"cycles={row['total_cycles']} waw={row['waw']} "
                  f"PE[{row['pe_active']}] {row['wall_s']}s{flag}", flush=True)

    # ---- aggregate report ----
    tsv = VALIDATION / "report.tsv"
    with tsv.open("w") as f:
        f.write("cid\tdtype\tidtype\taxis\tdims\telems\tpattern\tstatus\t"
                "total_cycles\tvec_cycles\ttlsu_cycles\twaw\tpe_active\t"
                "exit_code\twall_s\tlog\n")
        for r in sorted(results, key=lambda r: SELECTION.index(r["cid"])):
            m = meta[r["cid"]]
            f.write("\t".join(str(x) for x in [
                r["cid"], m["dtype"], m["idtype"], m["axis"], m["dims"],
                m["elems"], m["pattern"], r["status"], r["total_cycles"],
                r["vec_cycles"], r["tlsu_cycles"], r["waw"], r["pe_active"],
                r["exit_code"], r["wall_s"], r["log"]]) + "\n")

    from collections import Counter
    counts = Counter(r["status"] for r in results)
    print(f"\n===== SUMMARY ({time.time()-t0:.0f}s) =====")
    for k in sorted(counts):
        print(f"{k}: {counts[k]}")
    print(f"report: {tsv}")
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
