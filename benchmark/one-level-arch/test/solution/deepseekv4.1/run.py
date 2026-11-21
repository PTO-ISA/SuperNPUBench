#!/usr/bin/env python3
"""Run a Linx layer on gfrun and compare its guest memory to the decode oracle."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Program:
    elf: Path
    address: int
    jobs: str


def inspect_program(elf, bin_dir):
    listing = subprocess.check_output([bin_dir / "llvm-nm", "-S", elf], text=True)
    table = {
        parts[3]: (int(parts[0], 16), int(parts[1], 16))
        for line in listing.splitlines()
        if len(parts := line.split()) == 4
    }
    address, size = table["v41_result"]
    if table["v41_input"][1] != L.case.itemsize or size != L.result.itemsize:
        raise ValueError("fixture layout does not match the Linx ELF")
    jobs = ""
    if "v41_layer_client" in table:
        # PE0 computes and PE1 serves SDMA. ACR0 endpoints on this profile are 16 * PE.
        jobs = f"0 1 {table['v41_layer_client'][0]:#x} 16\n1 2 {table['v41_copy_service'][0]:#x} 0\n"
    return Program(elf, address, jobs)


def execute(args, path, program, model, data):
    path.mkdir()
    image, case, dump = path / "decode.elf", path / "input.bin", path / "result.bin"
    case.write_bytes(data.tobytes())
    subprocess.run(
        [args.bin / "llvm-objcopy", "--update-section", f".v41.input={case}", program.elf, image], check=True
    )
    command = [
        model,
        "-s",
        "softcore.multiThreadNum=4",
        "-f",
        image,
        "--dump-memory",
        f"{program.address:#x}:{L.result.itemsize}:{dump}",
    ]
    if program.jobs:
        manifest = path / "jobs.txt"
        manifest.write_text(program.jobs)
        command.extend(["--msgb", "--hscb-jobs", manifest, "-m", "30000000"])
    log = subprocess.run(command, cwd=path, capture_output=True, text=True, timeout=args.timeout)
    output = log.stdout + log.stderr
    if program.jobs:
        required = {f"HSCB DONE worker={i} pe={i} slot=0 id={i + 1} result=0" for i in (0, 1)}
        required.add("HSCB IDLE workers=4 completed=2")
        finished = required.issubset(output.splitlines())
    else:
        finished = re.search(r"formal_End=1 formal_Abort=0 status=0\b", output)
    if log.returncode or not finished:
        raise RuntimeError(
            f"gfrun {program.elf.name} layer {data['layer_idx']} failed (exit {log.returncode})\n{output[-4000:]}"
        )
    return np.frombuffer(dump.read_bytes(), dtype=L.result).reshape(())


def check(layer, *, args, root, program, baseline_program):
    data, expected = build_case(args.seed, layer, args.shards, args.start, args.empty_shard)
    with tempfile.TemporaryDirectory(prefix=f"layer-{layer:02}-", dir=root) as tmp:
        actual = execute(args, Path(tmp) / "kernel", program, args.gfrun, data)
        fields = ("steps", "published", "final_cache")
        for field in fields:
            compare(actual[field], expected[field], field)
        if args.empty_shard and np.any(actual["steps"]["routes"]["expert"] >= L.EXPERTS // 2):
            raise AssertionError("empty-shard fixture routed a token to the second shard")
        check_transport(actual, data, args)
        if baseline_program:
            data["shards"] = 1
            baseline = execute(args, Path(tmp) / "baseline", baseline_program, args.baseline_gfrun, data)
            for field in fields:
                if actual[field].tobytes() != baseline[field].tobytes():
                    raise AssertionError(f"layer {layer} differs from the one-shard baseline in {field}")
        error = np.max(abs(actual["steps"]["next"]["x"] - expected["steps"]["next"]["x"]))
        profile = "msgb" if program.jobs else "hosted"
        paired = " baseline=bitwise" if baseline_program else ""
        return (
            f"PASS layer={layer:02} positions={args.start},{args.start + 1} shards={args.shards} "
            f"profile={profile} max_hidden_abs={error:.3g}{paired}"
        )


def check_transport(actual, data, args):
    counters = ("dynamic_puts", "dynamic_gets", "sdma_transfers", "sdma_bytes")
    expected = [0, 0, 0, 0]
    if args.wse:
        layer = int(data["layer_idx"])
        writes, reads = 0, 0
        for token, out in zip(data["steps"]["token"], actual["steps"]):
            # One dynamic dispatch and one return per route, plus a count per shard.
            writes += 2 * L.K + args.shards
            reads += np.count_nonzero(out["selected"] >= 0)
            pos = int(token["position"])
            rows = pos + 1 if layer >= 20 else (pos + 1) // 2
            if layer in (2, 8, 14, 20):
                writes += 1 + 2 * int(layer == 20 or pos % 2 == 1)
                reads += rows
                writes += int(layer == 20)
            elif layer in (24, 28, 32, 36):
                reads += 1 + np.count_nonzero(np.isin(np.arange(rows) // L.PAGE, out["candidates"]))
                writes += 1
            elif layer >= 2:
                reads += 1
        copies = args.shards + len(data["steps"])
        size = data["weights"]["routed"].nbytes + len(data["steps"]) * L.hidden.itemsize
        expected = [writes, reads, copies, size]
    np.testing.assert_array_equal([int(actual[k]) for k in counters], expected, err_msg="transport counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument(
        "--baseline-elf", type=Path, help="also compare byte-for-byte with a one-shard baseline"
    )
    parser.add_argument("--gfrun", required=True, type=Path)
    parser.add_argument(
        "--baseline-gfrun", type=Path, help="gfrun for the baseline ELF (defaults to --gfrun)"
    )
    parser.add_argument("--bin", type=Path, default=os.environ.get("COMPILER_DIR"), help="Linx toolchain bin")
    layers = parser.add_mutually_exclusive_group()
    layers.add_argument("--layers", type=int, nargs="+")
    layers.add_argument("--all-layers", action="store_true")
    parser.add_argument("--shards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--wse", action="store_true", help="check PGAS operations and completed SDMA bytes")
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
    program = inspect_program(args.elf, args.bin)
    args.wse = args.wse or bool(program.jobs)
    baseline_program = None
    if args.baseline_elf:
        baseline_program = inspect_program(args.baseline_elf.resolve(), args.bin)
        args.baseline_gfrun = (args.baseline_gfrun or args.gfrun).resolve()
    with tempfile.TemporaryDirectory() as root, ThreadPoolExecutor(args.jobs) as pool:
        run = partial(check, args=args, program=program, baseline_program=baseline_program, root=root)
        for status in pool.map(run, cases):
            print(status, flush=True)


if __name__ == "__main__":
    main()
