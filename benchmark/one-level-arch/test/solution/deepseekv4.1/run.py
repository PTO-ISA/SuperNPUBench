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


def execute(args, path, program, model, data, expected_status=0):
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
        command.extend(["--msgb", "--hscb-jobs", manifest, "-m", "250000" if expected_status else "30000000"])
    log = subprocess.run(command, cwd=path, capture_output=True, text=True, timeout=args.timeout)
    output = log.stdout + log.stderr
    if program.jobs:
        required = {
            f"HSCB DONE worker=0 pe=0 slot=0 id=1 result={expected_status}",
            "HSCB DONE worker=1 pe=1 slot=0 id=2 result=0",
            "HSCB IDLE workers=4 completed=2",
        }
        finished = required.issubset(output.splitlines())
        expected_exit = 0  # The resident scheduler reports guest status per worker.
    else:
        expected_exit = int(expected_status != 0)
        terminal = rf"formal_End={1 - expected_exit} formal_Abort={expected_exit} status={expected_status}\b"
        finished = re.search(terminal, output)
    if log.returncode != expected_exit or not finished:
        raise RuntimeError(
            f"gfrun {program.elf.name} layer {data['layer_idx']} failed (exit {log.returncode})\n{output[-4000:]}"
        )
    return np.frombuffer(dump.read_bytes(), dtype=L.result).reshape(())


def check(case, *, args, root, program, baseline_program):
    seed, layer, start = case
    data, expected = build_case(seed, layer, args.shards, start, args.empty_shard)
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
            f"PASS layer={layer:02} positions={start},{start + 1} shards={args.shards} "
            f"profile={profile} max_hidden_abs={error:.3g}{paired}"
        )


def check_rejections(args, root, program):
    # A Full layer writes its own history. A Reindex fixture first imports an upstream row.
    for layer, status in ((20, 3), (24, 2)):
        data, _ = build_case(41, layer, args.shards, start=7)
        data["initial"]["pages"] = [0, 0, 2, 3]
        actual = execute(args, root / f"duplicate-{layer}", program, args.gfrun, data, status)
        if actual["final_cache"].tobytes() != data["initial"].tobytes():
            raise AssertionError(f"layer {layer} changed the cache before rejecting duplicate pages")
        print(f"PASS duplicate pages layer={layer} status={status} cache=bitwise")

    def reject_selection(data, name, reads):
        actual = execute(args, root / name, program, args.gfrun, data, 3)
        np.testing.assert_array_equal(actual["final_cache"]["window"], data["initial"]["window"])
        if args.wse:
            np.testing.assert_equal(actual["dynamic_gets"], reads)
        print(f"PASS {name} layer={data['layer_idx']} status=3")

    # Each cold Reuse publication has one live physical slot and three padding slots.
    for layer, start, bank in ((3, 1, 0), (21, 0, 3)):
        data, _ = build_case(41, layer, args.shards, start)
        rows = data["steps"][0]["published"]["indices"][bank]
        assert rows[0] >= 0 and np.all(rows[1:] == -1)
        live = int(rows[0])
        rows[-1] = live
        reject_selection(data, f"duplicate-row-{layer}", 1)
        rows[-1] = -1
        rows[0] = data["initial"]["pages"][1] * L.PAGE
        reject_selection(data, f"unpublished-page-{layer}", 1)
        rows[0] = live + 1
        reject_selection(data, f"unpublished-lane-{layer}", 1)

    data, _ = build_case(41, 0, args.shards, start=0)
    token = data["steps"][0]["token"]
    token["window_indices"][0] = token["window_slot"]
    reject_selection(data, "duplicate-window", 0)
    data, _ = build_case(41, 24, args.shards, start=8)
    blocks = data["steps"][0]["published"]["candidates"]
    np.testing.assert_array_equal(blocks, [0, 1])
    blocks[1] = blocks[0]
    reject_selection(data, "duplicate-block", 1)

    if program.jobs:
        data, _ = build_case(41, 24, args.shards)
        invalid_shards = data.copy()
        invalid_shards["shards"] = 0
        execute(args, root / "invalid-shards", program, args.gfrun, invalid_shards, 1)
        for step in range(len(data["steps"])):
            invalid_window = data.copy()
            invalid_window["steps"][step]["token"]["window_slot"] = L.WIN
            actual = execute(args, root / f"invalid-window-{step}", program, args.gfrun, invalid_window, 3)
            np.testing.assert_equal(actual["sdma_transfers"], args.shards + step)
        print("PASS resident shutdown before staging and on either decode step")


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
    layers.add_argument(
        "--regressions", action="store_true", help="run the mHC, page table and shutdown regressions"
    )
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
    if args.regressions:
        cases = [(7, 2, 22), (41, 20, 7), (41, 3, 1), (41, 21, 0), (41, 0, 0), (41, 24, 8)]
    else:
        layers_to_run = range(40) if args.all_layers else args.layers or (0, 1, 2, 3, 14, 20, 24, 39)
        cases = [(args.seed, layer, args.start) for layer in layers_to_run]
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
        if args.regressions:
            check_rejections(args, Path(root), program)


if __name__ == "__main__":
    main()
