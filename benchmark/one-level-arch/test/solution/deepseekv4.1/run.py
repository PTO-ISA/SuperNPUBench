#!/usr/bin/env python3
"""Run one hosted Linx ELF per layer and compare its guest memory to the decode oracle."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from cases import build_case
import layout as L


def compare(actual, expected, name=""):
    if expected.dtype.names:
        for field in expected.dtype.names:
            compare(actual[field], expected[field], f"{name}.{field}" if name else field)
    elif expected.dtype.kind == "f":
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-5, equal_nan=False, err_msg=name)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=name)


def symbols(elf, bin_dir):
    listing = subprocess.check_output([bin_dir / "llvm-nm", "-S", elf], text=True)
    table = {
        parts[3]: (int(parts[0], 16), int(parts[1], 16))
        for line in listing.splitlines()
        if len(parts := line.split()) == 4
    }
    address, size = table["v41_result"]
    if table["v41_input"][1] != L.case.itemsize or size != L.result.itemsize:
        raise ValueError("fixture layout does not match the Linx ELF")
    return address


def check(layer, *, args, root, address):
    data, expected = build_case(args.seed, layer, args.shards, args.start, args.empty_shard)
    with tempfile.TemporaryDirectory(prefix=f"layer-{layer:02}-", dir=root) as tmp:
        path = Path(tmp)
        image, case, dump = path / "decode.elf", path / "input.bin", path / "result.bin"
        case.write_bytes(data.tobytes())
        subprocess.run(
            [args.bin / "llvm-objcopy", "--update-section", f".v41.input={case}", args.elf, image], check=True
        )
        command = [
            args.gfrun,
            "-s",
            "softcore.multiThreadNum=4",
            "-f",
            image,
            "--dump-memory",
            f"{address:#x}:{L.result.itemsize}:{dump}",
        ]
        log = subprocess.run(command, cwd=path, capture_output=True, text=True, timeout=args.timeout)
        output = log.stdout + log.stderr
        if log.returncode or not re.search(r"formal_End=1 formal_Abort=0 status=0\b", output):
            raise RuntimeError(f"gfrun layer {layer} failed (exit {log.returncode})\n{output[-4000:]}")
        actual = np.frombuffer(dump.read_bytes(), dtype=L.result).reshape(())
        for field in ("steps", "published", "final_cache"):
            compare(actual[field], expected[field], field)
        check_transport(actual, layer, args)
        error = np.max(abs(actual["steps"]["next"]["x"] - expected["steps"]["next"]["x"]))
        return f"PASS layer={layer:02} positions={args.start},{args.start + 1} shards={args.shards} max_hidden_abs={error:.3g}"


def check_transport(actual, layer, args):
    for field in ("dynamic_puts", "dynamic_gets", "sdma_transfers", "sdma_bytes"):
        if actual[field] != 0:
            raise AssertionError(f"unexpected {field} in GM case")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--gfrun", required=True, type=Path)
    parser.add_argument("--bin", type=Path, default=os.environ.get("COMPILER_DIR"), help="Linx toolchain bin")
    layers = parser.add_mutually_exclusive_group()
    layers.add_argument("--layers", type=int, nargs="+")
    layers.add_argument("--all-layers", action="store_true")
    parser.add_argument("--shards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--start", type=int, default=22)
    parser.add_argument("--empty-shard", action="store_true", help="bias routing to the first expert shard")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.bin is None or args.jobs < 1:
        parser.error("set COMPILER_DIR or --bin and choose a positive --jobs")
    args.bin, args.elf, args.gfrun = args.bin.resolve(), args.elf.resolve(), args.gfrun.resolve()
    cases = range(40) if args.all_layers else args.layers or (0, 1, 2, 3, 14, 20, 24, 39)
    address = symbols(args.elf, args.bin)
    with tempfile.TemporaryDirectory() as root, ThreadPoolExecutor(args.jobs) as pool:
        for status in pool.map(partial(check, args=args, address=address, root=root), cases):
            print(status, flush=True)


if __name__ == "__main__":
    main()
