#!/usr/bin/env python3
"""Run gfsim on an ELF and emit a Perfetto SwimLane trace (counters stripped).

Every perf run should keep:
  <outdir>/<name>_swim.log              raw gfsim stdout (cycle summary)
  <outdir>/<name>.json             full SwimLane JSON
  <outdir>/<name>_nocounters.json  same trace with ph:"C" PMU counters removed
                                        (open this one at https://ui.perfetto.dev/)

Usage:
  python3 test/common/run_swimlane.py \
      --gfsim /path/to/model/bin/gfsim \
      --elf   /path/to/output/.../kernel_....elf \
      --outdir /path/to/perf_runs/<run_id> \
      --name  scheme10
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


from trace_stream import strip_counters
import resource

def limit_memory():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = min(6 * 1024**3, hard) if hard != resource.RLIM_INFINITY else 6 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfsim", required=True, type=Path)
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--conf", default=None,
                        help="optional gfsim config bundle (e.g. fourpe)")
    args = parser.parse_args()
    limit_memory()

    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.outdir / args.name
    command = [str(args.gfsim), "-f", str(args.elf),
               "--swimlane", "1", "--swimfile", str(prefix)]
    if args.conf:
        command += ["--conf", args.conf]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        (args.outdir / f"{args.name}_swim.log").write_text(
            (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout) or "", encoding="utf-8")
        print(f"TIMEOUT after {args.timeout}s", file=sys.stderr)
        return 1
    log = args.outdir / f"{args.name}_swim.log"
    log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        print(f"gfsim rc={proc.returncode}: {log}", file=sys.stderr)
        return proc.returncode

    full = Path(str(prefix) + ".json")
    if not full.exists():
        print(f"missing {full}", file=sys.stderr)
        return 1
    noc = Path(str(prefix) + "_nocounters.json")
    total, kept = strip_counters(full, noc)

    cycles = ""
    for line in proc.stdout.splitlines():
        if "Total Cycles" in line:
            cycles = line.strip()
            break
    print(f"{args.name}: {cycles or 'cycles n/a'}")
    print(f"  full       {full} ({full.stat().st_size} B)")
    print(f"  nocounters {noc} ({total} -> {kept} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
