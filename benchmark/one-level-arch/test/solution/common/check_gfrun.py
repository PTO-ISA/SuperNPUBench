#!/usr/bin/env python3
"""Build the six solution cases and check four-PE gfrun against host goldens."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess


ARCH = Path(__file__).resolve().parents[3]


def pack(dtype, values):
    code = {"int32_t": "i", "float": "f", "__half": "e"}[dtype]
    return b"".join(struct.pack("<" + code, int(v) if code == "i" else v)
                    for v in values)


def golden(elf):
    """Independent scalar coordinate enumeration, including output guards."""
    name = elf.name
    # __half begins with underscores, so use the full known dtype spellings.
    dtype = next(t for t in ("int32_t", "float", "__half")
                 if "_DType" + t + "_" in name)
    if "view_copy" in name:
        match = re.search(r"_SHAPE([\d_]+)_TILE\d+_IN_STRIDE([\d_]+)"
                          r"_OUT_STRIDE([\d_]+)\.elf", name)
        shape, istride, ostride = [tuple(map(int, x.split("_")))
                                  for x in match.groups()]
        # Offsets are from the three shipped compile.all fixtures.
        output_offset = {(2, 4, 8): 64, (4, 4, 8): 128, (4, 4, 6): 96}[shape]
        itemsize = 4 if dtype == "int32_t" else 2
        prefix = output_offset // itemsize
        values = [-7] * (prefix + math.prod(shape))
        for coord in itertools.product(*(range(n) for n in shape)):
            src = sum(c * s for c, s in zip(coord, istride))
            dst = sum(c * s for c, s in zip(coord, ostride))
            value = (src * 17 + 11) % 101 - 50
            values[prefix + dst] = value if dtype == "int32_t" else value * 0.125
    else:
        match = re.search(r"_DIM(\d+)_IN([\d_]+)_OUT([\d_]+)_TILE(\d+)", name)
        axis = int(match[1])
        ishape, oshape = [tuple(map(int, match[i].split("_"))) for i in (2, 3)]
        values = []
        for coord in itertools.product(*(range(n) for n in oshape)):
            selected = list(coord)
            selected[axis] = (selected[axis] * 7 + 3) % ishape[axis]
            src = 0
            for c, extent in zip(selected, ishape):
                src = src * extent + c
            value = (src * 29 + 13) % 211 - 105
            values.append(value if dtype == "int32_t" else value * 0.0625)
        values.extend([-17] * int(match[4]))
    return pack(dtype, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-dir", required=True, type=Path)
    parser.add_argument("--gfrun", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=ARCH / "output/solution/fixed_validation")
    args = parser.parse_args()
    compiler = args.compiler_dir.resolve()
    gfrun = args.gfrun.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, COMPILER_DIR=str(compiler), baremetal="off")
    rows = []
    for op in ("view_copy", "gather_v2"):
        with (out / (op + "_compile.log")).open("w") as log:
            subprocess.run(["bash", "compile.all"], env=env,
                           cwd=ARCH / "test/solution" / op,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        elfs = sorted((ARCH / "output/solution" / op / "elf").glob("*.elf"))
        if len(elfs) != 3:
            raise RuntimeError(f"Expected 3 shipped {op} ELFs, found {len(elfs)}")
        for elf in elfs:
            symbols = subprocess.check_output([str(compiler / "llvm-nm"),
                                               "-S", str(elf)], text=True)
            symbol = next(line.split() for line in symbols.splitlines()
                          if line.endswith(" _ZZ4mainE6output"))
            address, size = int(symbol[0], 16), int(symbol[1], 16)
            expected = golden(elf)
            if len(expected) != size:
                raise RuntimeError(f"Unexpected output buffer size in {elf.name}")
            dump = out / (elf.name + ".bin")
            (out / (elf.name + ".golden.bin")).write_bytes(expected)
            dump.unlink(missing_ok=True)
            command = [str(gfrun), "-t", "1", "-s", "softcore.multiThreadNum=4",
                       "-f", str(elf), "--dump-memory",
                       f"{address:#x}:{size}:{dump}"]
            with (out / (elf.name + ".log")).open("w") as log:
                try:
                    proc = subprocess.run(command, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=30)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    rc = 124
            logtext = (out / (elf.name + ".log")).read_text(errors="replace")
            numeric = dump.is_file() and dump.read_bytes() == expected
            passed = (rc == 0 and "PASS: " + op in logtext and
                      re.search(r"Reach the End of Benchmark! R2 = 0\s", logtext)
                      and numeric)
            rows.append(dict(case=elf.name, exit_code=rc,
                             host_golden=numeric, passed=bool(passed), command=command))
            print(f"{'PASS' if passed else 'FAIL'} {elf.name} host_golden={numeric}")
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
