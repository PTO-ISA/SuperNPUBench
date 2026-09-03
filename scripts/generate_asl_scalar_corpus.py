#!/usr/bin/env python3
"""Generate verified ASL sidecars and independent host goldens for scalar cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import struct
import tempfile


RESULT_BYTES = 8192
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_commit(value: str, label: str) -> str:
    if COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact lowercase 40-hex commit")
    return value


def git_identity(repo: pathlib.Path, label: str = "SuperNPUBench") -> tuple[str, str]:
    for command in (["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--"],
                    ["git", "-C", str(repo), "diff", "--cached", "--quiet", "HEAD", "--"]):
        if subprocess.run(command).returncode != 0:
            raise ValueError(f"{label} checkout must be clean")
    def resolve(expression: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", expression], check=True,
            text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
    return (checked_commit(resolve("HEAD"), f"{label} commit"),
            checked_commit(resolve("HEAD^{tree}"), f"{label} tree"))


def checked_checkout(path: pathlib.Path, expected_commit: str,
                     label: str) -> tuple[str, str]:
    checkout = path.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"{label} is not a Git checkout: {checkout}")
    commit, tree = git_identity(checkout, label)
    if commit != checked_commit(expected_commit, f"{label} expected commit"):
        raise ValueError(f"{label} checkout is at {commit}, expected {expected_commit}")
    return commit, tree


def checked_tool(path: pathlib.Path, expected_hash: str,
                 label: str) -> dict[str, str]:
    tool = path.resolve()
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise ValueError(f"missing executable {label}: {tool}")
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError(f"{label} expected hash must be lowercase SHA-256")
    actual_hash = sha256(tool)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} hash mismatch")
    return {"path": str(tool), "sha256": actual_hash}


def add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llvm-readelf", required=True, type=pathlib.Path)
    parser.add_argument("--llvm-readelf-sha256", required=True)
    parser.add_argument("--clang", required=True, type=pathlib.Path)
    parser.add_argument("--clang-sha256", required=True)
    parser.add_argument("--lld", required=True, type=pathlib.Path)
    parser.add_argument("--lld-sha256", required=True)
    parser.add_argument("--host-cxx", required=True, type=pathlib.Path)
    parser.add_argument("--host-cxx-sha256", required=True)
    for name in ("toolchain", "tileop", "pto", "model"):
        parser.add_argument(f"--{name}-checkout", required=True, type=pathlib.Path)
        parser.add_argument(f"--{name}-commit", required=True)


def resolve_identity(arguments: argparse.Namespace,
                     repo: pathlib.Path) -> dict[str, object]:
    producer_commit, producer_tree = git_identity(repo)
    identity: dict[str, object] = {
        "producer_commit": producer_commit,
        "producer_tree": producer_tree,
    }
    for name, label in (("toolchain", "LLVM toolchain"), ("tileop", "TileOp"),
                        ("pto", "PTO"), ("model", "ASL model")):
        commit, tree = checked_checkout(getattr(arguments, f"{name}_checkout"),
                                        getattr(arguments, f"{name}_commit"), label)
        identity[f"{name}_commit"] = commit
        identity[f"{name}_tree"] = tree
    toolchain = arguments.toolchain_checkout.resolve()
    target_tools = {
        "clang": checked_tool(arguments.clang, arguments.clang_sha256, "clang"),
        "lld": checked_tool(arguments.lld, arguments.lld_sha256, "lld"),
        "llvm_readelf": checked_tool(arguments.llvm_readelf,
                                      arguments.llvm_readelf_sha256,
                                      "llvm-readelf"),
    }
    if any(not pathlib.Path(row["path"]).is_relative_to(toolchain)
           for row in target_tools.values()):
        raise ValueError("target tool is outside the verified toolchain checkout")
    identity["tools"] = {
        **target_tools,
        "host_cxx": checked_tool(arguments.host_cxx,
                                 arguments.host_cxx_sha256, "host C++"),
    }
    return identity


def discover_elfs(root: pathlib.Path) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for elf in root.rglob("*.elf"):
        if elf.stem in result:
            raise ValueError(f"duplicate ELF basename: {elf.stem}")
        result[elf.stem] = elf.resolve()
    return result


def validate_raw_elf_header(elf: pathlib.Path) -> None:
    header = elf.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise ValueError(f"not an ELF file: {elf}")
    if header[4] != 2 or header[5] != 1:
        raise ValueError(f"ELF must be 64-bit little-endian: {elf}")
    if struct.unpack_from("<H", header, 18)[0] != 0xE9:
        raise ValueError(f"wrong ELF machine: {elf}")


def llvm_metadata(readelf: pathlib.Path, elf: pathlib.Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(readelf),
            "--elf-output-style=JSON",
            "--file-header",
            "--program-headers",
            "--symbols",
            str(elf),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    outer = json.loads(completed.stdout)[0]
    return next(iter(outer.values()))


def symbol_map(metadata: dict[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in metadata["Symbols"]:
        symbol = row["Symbol"]
        name = symbol["Name"]["Value"]
        if name and symbol["Binding"]["RawValue"] == 1:
            if name in result:
                raise ValueError(f"duplicate ELF symbol: {name}")
            result[name] = symbol
    return result


def load_segments(metadata: dict[str, object]) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    for row in metadata["ProgramHeaders"]:
        header = row["ProgramHeader"]
        if header["Type"]["RawValue"] != 1:
            continue
        if header["VirtualAddress"] != header["PhysicalAddress"]:
            raise ValueError("PT_LOAD virtual and physical addresses differ")
        segments.append({
            "address": header["VirtualAddress"],
            "filesz": header["FileSize"],
            "memsz": header["MemSize"],
            "flags": header["Flags"]["RawFlags"],
        })
    return segments


def validate_elf_contract(readelf: pathlib.Path, elf: pathlib.Path) -> tuple[
    dict[str, object], dict[str, dict[str, object]], list[dict[str, int]]
]:
    validate_raw_elf_header(elf)
    metadata = llvm_metadata(readelf, elf)
    if not all(key in metadata for key in ("ElfHeader", "ProgramHeaders", "Symbols")):
        raise ValueError(f"incomplete llvm-readelf metadata: {elf}")
    header = metadata["ElfHeader"]
    if header.get("Machine", {}).get("RawValue") != 0xE9:
        raise ValueError(f"wrong ELF machine in metadata: {elf}")
    symbols = symbol_map(metadata)
    required = {"main", "_end", "cross_model_stop", "cross_model_result",
                "cross_model_result_size"}
    if not required.issubset(symbols):
        raise ValueError(f"missing cross-model symbol in {elf}")
    result = symbols["cross_model_result"]
    size = symbols["cross_model_result_size"]
    if result.get("Size") != RESULT_BYTES:
        raise ValueError(f"cross_model_result is not {RESULT_BYTES} bytes: {elf}")
    if size.get("Section", {}).get("RawValue") != 0xFFF1 or size.get("Value") != RESULT_BYTES:
        raise ValueError(f"cross_model_result_size is not ABS {RESULT_BYTES}: {elf}")
    if symbols["cross_model_stop"].get("Value") == 0:
        raise ValueError(f"invalid cross_model_stop alias: {elf}")
    if symbols["cross_model_stop"].get("Value") != symbols["_end"].get("Value"):
        raise ValueError(f"cross_model_stop does not alias _end: {elf}")
    segments = load_segments(metadata)
    if not segments:
        raise ValueError(f"ELF has no PT_LOAD segment: {elf}")
    start = result["Value"]
    end = start + RESULT_BYTES
    if not any(segment["address"] <= start
               and end <= segment["address"] + segment["memsz"]
               and segment["flags"] & 2 for segment in segments):
        raise ValueError(f"result carrier is not in one writable PT_LOAD: {elf}")
    return header, symbols, segments


def next_power_of_two(value: int) -> int:
    return 1 << max(1, value - 1).bit_length()


def scalar_dtype(case_id: str) -> tuple[str, int, list[int]]:
    tokens = case_id.split("_")
    dtype = next(
        (token for token in tokens if token in {"i32", "i64", "fp32", "f64"}),
        None,
    )
    if dtype is None:
        raise ValueError(f"cannot resolve scalar dtype: {case_id}")
    width = {"i32": 4, "i64": 8, "fp32": 4, "f64": 8}[dtype]
    shape = [16] if case_id.startswith("st_") else [1]
    return dtype, width * shape[0], shape


def generate_golden(repo: pathlib.Path, cxx: str, source: pathlib.Path,
                    output: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pto-scalar-golden-") as directory:
        root = pathlib.Path(directory)
        wrapper = root / "golden.cpp"
        executable = root / "golden"
        wrapper.write_text(
            "#define CROSS_MODEL_GOLDEN_HOST 1\n"
            "#define CROSS_MODEL_CORPUS 1\n"
            "#define main pto_benchmark_main\n"
            f'#include "{source}"\n'
            "#undef main\n"
            "#include <fstream>\n"
            "int main(int argc, char **argv) {\n"
            "  if (argc != 2 || pto_benchmark_main() != 0) return 2;\n"
            "  std::ofstream out(argv[1], std::ios::binary);\n"
            "  out.write(reinterpret_cast<const char *>(cross_model_result), "
            "kCrossModelResultBytes);\n"
            "  return out ? 0 : 3;\n"
            "}\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                cxx,
                "-std=c++20",
                "-O2",
                "-I" + str(repo / "microbenchmark" / "scalar"),
                "-I" + str(repo / "microbenchmark" / "common"),
                "-I" + str(repo / "benchmark" / "one-level-arch" / "test" / "common" / "src"),
                str(wrapper),
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable), str(output)], check=True)
    if output.stat().st_size != RESULT_BYTES:
        raise ValueError(f"golden has wrong size: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path(__file__).parents[1])
    parser.add_argument("--elf-root", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    add_identity_arguments(parser)
    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    elf_root = (arguments.elf_root or repo / "output" / "asl_corpus" / "microbenchmark").resolve()
    identities = resolve_identity(arguments, repo)
    coverage = json.loads((repo / "microbenchmark" / "coverage.json").read_text())
    active = [row for row in coverage["active"] if row["family"] == "scalar"]
    elf_by_name = discover_elfs(elf_root)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    index_rows: list[str] = []
    for row in active:
        case_id = row["name"]
        elf = elf_by_name.get(case_id)
        if elf is None:
            raise ValueError(f"missing active scalar ELF: {case_id}")
        source = repo / "microbenchmark" / "scalar" / "src" / f"{case_id}.cpp"
        if not source.is_file():
            raise ValueError(f"missing active scalar source: {case_id}")
        golden = output / f"{case_id}.golden.bin"
        sidecar = output / f"{case_id}.sidecar.json"
        generate_golden(repo, arguments.host_cxx, source, golden)
        header, symbols, segments = validate_elf_contract(arguments.llvm_readelf, elf)
        result_symbol = symbols["cross_model_result"]
        high_address = max(segment["address"] + segment["memsz"] for segment in segments)
        memory_bytes = next_power_of_two(max(0x20000, high_address + 0x2000))
        dtype, logical_size, shape = scalar_dtype(case_id)
        document = {
            "schema": "pto-asl-elf-sidecar-v1",
            "case_id": f"scalar.{case_id}",
            "identity": identities,
            "source": {
                "path": source.relative_to(repo).as_posix(),
                "sha256": sha256(source),
            },
            "elf": {
                "path": elf.name,
                "sha256": sha256(elf),
                "machine": 0xE9,
                "entry": header["Entry"],
                "segments": segments,
            },
            "model": {
                "profile": "bounded-reference-v1",
                "pe_count": 1,
                "memory_bytes": memory_bytes,
                "tile_elements": 1,
                "runtime_typecheck": "minimal",
            },
            "start": {
                "symbol": "main",
                "pc": symbols["main"]["Value"],
                "return_symbol": "cross_model_stop",
                "return_pc": symbols["cross_model_stop"]["Value"],
            },
            "execution": {
                "classification": "scalar",
                "stop_symbol": "cross_model_stop",
                "stop_pc": symbols["cross_model_stop"]["Value"],
                "stop_after_hits": 1,
                "max_steps": 100000,
                "stack_top": memory_bytes - 0x1000,
            },
            "result": {
                "symbol": "cross_model_result",
                "size_symbol": "cross_model_result_size",
                "address": result_symbol["Value"],
                "size": RESULT_BYTES,
                "segments": [
                    {
                        "offset": 0,
                        "size": logical_size,
                        "dtype": dtype,
                        "shape": shape,
                        "comparison": "exact",
                    },
                    {
                        "offset": logical_size,
                        "size": RESULT_BYTES - logical_size,
                        "dtype": "u8",
                        "shape": [RESULT_BYTES - logical_size],
                        "comparison": "exact-zero-padding",
                    },
                ],
                "golden": {
                    "path": golden.name,
                    "sha256": sha256(golden),
                    "derivation": "host-cxx-source-reference",
                },
            },
        }
        sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        index_rows.append("|".join([
            document["case_id"], str(elf), str(sidecar), str(golden)
        ]))
    (output / "cases.index").write_text("\n".join(index_rows) + "\n")
    print(f"generated scalar ASL corpus: {len(index_rows)} cases in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
