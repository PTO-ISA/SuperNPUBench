#!/usr/bin/env python3
"""Merge family indexes only after independently verifying every artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
from collections import Counter

from generate_asl_scalar_corpus import (
    RESULT_BYTES,
    checked_checkout,
    checked_tool,
    validate_elf_contract,
)


SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
EXPECTED_FAMILIES = {"scalar": 124, "memory": 14, "cube": 11,
                     "fixp": 63, "vector": 121}
EXPECTED_UNSUPPORTED = {
    "fixp.fixp_tmatmul_lrelu_only",
    "cube.tmatmul_fp16_64x64x64",
    "vector.tsel_fp16_16x16", "vector.tsel_fp32_16x16",
    "vector.tabs_bf16_16x16",
    *{f"vector.{op}_{dtype}_16x16" for op, dtypes in (
        ("tcmp", ("fp16", "fp32", "i32")),
        ("tcmps", ("fp16", "fp32")),
        ("thistogram", ("i16", "i32")),
    ) for dtype in dtypes},
    *{f"memory.tmov_{dtype}_{shape}" for dtype, shape in (
        ("fp16", "16x16"), ("fp32", "16x16"), ("i32", "16x16"),
        ("fp16", "32x32"), ("fp32", "32x32"),
    )},
    *{f"memory.{op}_{dtype}_16x16" for op in ("mgather_mask", "mscatter_mask")
      for dtype in ("fp16", "fp32")},
    *{f"vector.{op}_{dtype}_16x16"
      for op in ("tpartadd", "tpartmul", "tpartmax", "tpartmin")
      for dtype in ("fp16", "fp32")},
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label}: invalid sha256")
    return value


def checked_file(path_text: str, label: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label}: missing absolute artifact {path_text}")
    return path.resolve()


def claim_case(seen: set[str], case_id: str) -> None:
    if case_id in seen:
        raise ValueError(f"duplicate corpus case id: {case_id}")
    seen.add(case_id)


def require_matching_hashes(case_id: str, actual: dict[str, str],
                            expected: dict[str, str]) -> None:
    if actual != expected:
        raise ValueError(f"{case_id}: artifact hash mismatch")


def committed_source_sha256(repo: pathlib.Path, commit: str, path: str) -> str:
    content = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"], check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def validate_coverage(coverage: object) -> set[str]:
    if not isinstance(coverage, dict) or coverage.get("schema_version") != 1:
        raise ValueError("invalid coverage schema")
    active, unsupported = coverage.get("active"), coverage.get("unsupported")
    if not isinstance(active, list) or not isinstance(unsupported, list):
        raise ValueError("coverage active/unsupported must be lists")
    active_ids: list[str] = []
    for row in active:
        if not isinstance(row, dict) or row.get("status") != "active":
            raise ValueError("invalid active coverage row")
        if not isinstance(row.get("family"), str) or not isinstance(row.get("name"), str):
            raise ValueError("active coverage row lacks identity")
        active_ids.append(f"{row['family']}.{row['name']}")
    unsupported_ids: list[str] = []
    for row in unsupported:
        if not isinstance(row, dict) or row.get("status") != "unsupported":
            raise ValueError("invalid unsupported coverage row")
        if not isinstance(row.get("reason"), str) or not row["reason"]:
            raise ValueError("unsupported coverage row lacks reason")
        unsupported_ids.append(f"{row.get('family')}.{row.get('name')}")
    if len(active_ids) != len(set(active_ids)):
        raise ValueError("duplicate active coverage id")
    if len(unsupported_ids) != len(set(unsupported_ids)):
        raise ValueError("duplicate unsupported coverage id")
    if set(active_ids) & set(unsupported_ids):
        raise ValueError("active and unsupported coverage overlap")
    counts = Counter(row["family"] for row in active)
    if dict(counts) != EXPECTED_FAMILIES or len(active_ids) != 333:
        raise ValueError(f"coverage is not the locked 333-case corpus: {dict(counts)}")
    if set(unsupported_ids) != EXPECTED_UNSUPPORTED:
        raise ValueError("unsupported coverage is not the locked 29-case set")
    return set(active_ids)


def validate_identity(identity: object, repo: pathlib.Path,
                      checkouts: dict[str, pathlib.Path]) -> dict[str, object]:
    required = {"producer_commit", "producer_tree", "toolchain_commit", "toolchain_tree",
                "tileop_commit", "tileop_tree", "pto_commit", "pto_tree",
                "model_commit", "model_tree", "tools"}
    if not isinstance(identity, dict) or set(identity) != required:
        raise ValueError("invalid producer identity schema")
    for key in required - {"tools"}:
        if not isinstance(identity[key], str) or COMMIT_RE.fullmatch(identity[key]) is None:
            raise ValueError(f"invalid identity field: {key}")
    producer_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        text=True, stdout=subprocess.PIPE).stdout.strip()
    producer_tree = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], check=True,
        text=True, stdout=subprocess.PIPE).stdout.strip()
    if (identity["producer_commit"], identity["producer_tree"]) != (producer_commit, producer_tree):
        raise ValueError("producer identity does not match repository")
    for name, label in (("toolchain", "LLVM toolchain"), ("tileop", "TileOp"),
                        ("pto", "PTO"), ("model", "ASL model")):
        commit, tree = checked_checkout(checkouts[name], str(identity[f"{name}_commit"]), label)
        if tree != identity[f"{name}_tree"]:
            raise ValueError(f"{label} tree identity mismatch")
    tools = identity["tools"]
    if not isinstance(tools, dict) or set(tools) != {"clang", "lld", "llvm_readelf", "host_cxx"}:
        raise ValueError("invalid tool identity schema")
    for name, row in tools.items():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"invalid {name} identity")
        checked_tool(pathlib.Path(str(row["path"])),
                     checked_hash(row["sha256"], name), name)
    toolchain = checkouts["toolchain"].resolve()
    if any(not pathlib.Path(str(tools[name]["path"])).resolve().is_relative_to(toolchain)
           for name in ("clang", "lld", "llvm_readelf")):
        raise ValueError("target tool is outside verified toolchain checkout")
    return identity


def validate_sidecar(case_id: str, document: object, repo: pathlib.Path,
                     elf: pathlib.Path, golden: pathlib.Path,
                     identity: dict[str, object]) -> tuple[pathlib.Path, str, str, str]:
    top_keys = {"schema", "case_id", "identity", "source", "elf", "model",
                "start", "execution", "result"}
    if not isinstance(document, dict) or set(document) != top_keys:
        raise ValueError(f"{case_id}: invalid sidecar fields")
    if document["schema"] != "pto-asl-elf-sidecar-v1" or document["case_id"] != case_id:
        raise ValueError(f"{case_id}: invalid sidecar schema or id")
    if document["identity"] != identity:
        raise ValueError(f"{case_id}: mixed or incorrect identity")
    family = case_id.split(".", 1)[0]
    case_name = case_id.split(".", 1)[1]
    if elf.stem != case_name:
        raise ValueError(f"{case_id}: ELF basename does not match case id")

    source_doc = document["source"]
    if not isinstance(source_doc, dict) or set(source_doc) != {"path", "sha256"}:
        raise ValueError(f"{case_id}: invalid source commitment")
    source = (repo / str(source_doc["path"])).resolve()
    if not source.is_relative_to(repo) or not source.is_file():
        raise ValueError(f"{case_id}: invalid source path")
    expected_source = (repo / "microbenchmark" / family / "src" /
                       ("fixp_tmatmul.cpp" if family == "fixp" else f"{case_name}.cpp")).resolve()
    if source != expected_source:
        raise ValueError(f"{case_id}: source path does not match case id")
    source_hash = checked_hash(source_doc["sha256"], f"{case_id} source")
    if source_hash != committed_source_sha256(repo, str(identity["producer_commit"]),
                                               str(source_doc["path"])):
        raise ValueError(f"{case_id}: source is not committed by producer")

    readelf = pathlib.Path(str(identity["tools"]["llvm_readelf"]["path"]))
    header, symbols, segments = validate_elf_contract(readelf, elf)
    elf_doc = document["elf"]
    if not isinstance(elf_doc, dict) or set(elf_doc) != {"path", "sha256", "machine", "entry", "segments"}:
        raise ValueError(f"{case_id}: invalid ELF fields")
    elf_hash = checked_hash(elf_doc["sha256"], f"{case_id} ELF")
    if (elf_doc["path"] != elf.name or elf_doc["machine"] != 0xE9
            or elf_doc["entry"] != header["Entry"] or elf_doc["segments"] != segments):
        raise ValueError(f"{case_id}: ELF metadata/path mismatch")

    model = document["model"]
    if not isinstance(model, dict) or set(model) != {"profile", "pe_count", "memory_bytes",
                                                    "tile_elements", "runtime_typecheck"}:
        raise ValueError(f"{case_id}: invalid model fields")
    if (model["profile"] != "bounded-reference-v1" or model["runtime_typecheck"] != "minimal"
            or not all(isinstance(model[key], int) and model[key] > 0
                       for key in ("pe_count", "memory_bytes", "tile_elements"))):
        raise ValueError(f"{case_id}: invalid model contract")

    start = document["start"]
    if not isinstance(start, dict) or set(start) != {"symbol", "pc", "return_symbol", "return_pc"}:
        raise ValueError(f"{case_id}: invalid start fields")
    if start != {"symbol": "main", "pc": symbols["main"]["Value"],
                 "return_symbol": "cross_model_stop",
                 "return_pc": symbols["cross_model_stop"]["Value"]}:
        raise ValueError(f"{case_id}: start/return symbol mismatch")

    execution = document["execution"]
    required_execution = {"classification", "stop_symbol", "stop_pc", "stop_after_hits",
                          "max_steps", "stack_top"}
    if not isinstance(execution, dict) or set(execution) != required_execution:
        raise ValueError(f"{case_id}: invalid execution fields")
    if (execution["classification"] != family or execution["stop_symbol"] != "cross_model_stop"
            or execution["stop_pc"] != symbols["cross_model_stop"]["Value"]
            or execution["stop_after_hits"] != 1
            or not isinstance(execution["max_steps"], int) or execution["max_steps"] <= 0
            or not isinstance(execution["stack_top"], int) or execution["stack_top"] <= 0):
        raise ValueError(f"{case_id}: invalid execution contract")
    if execution["stack_top"] >= model["memory_bytes"]:
        raise ValueError(f"{case_id}: stack is outside model memory")

    result = document["result"]
    required_result = {"symbol", "size_symbol", "address", "size", "segments", "golden"}
    if not isinstance(result, dict) or set(result) != required_result:
        raise ValueError(f"{case_id}: invalid result fields")
    if (result["symbol"] != "cross_model_result" or result["size_symbol"] != "cross_model_result_size"
            or result["address"] != symbols["cross_model_result"]["Value"]
            or result["size"] != RESULT_BYTES or not isinstance(result["segments"], list)
            or len(result["segments"]) != 2):
        raise ValueError(f"{case_id}: invalid result contract")
    payload, padding = result["segments"]
    payload_keys = {"offset", "size", "dtype", "shape", "comparison"}
    if (not isinstance(payload, dict) or set(payload) != payload_keys
            or not isinstance(payload["size"], int) or payload["size"] <= 0
            or payload["offset"] != 0 or not isinstance(payload["dtype"], str)
            or not isinstance(payload["shape"], list) or not payload["shape"]
            or payload["comparison"] != "exact"):
        raise ValueError(f"{case_id}: invalid result payload segment")
    if (not isinstance(padding, dict) or set(padding) != payload_keys
            or padding["offset"] != payload["size"]
            or padding["size"] != RESULT_BYTES - payload["size"]
            or padding["dtype"] != "u8"
            or padding["shape"] != [padding["size"]]
            or padding["comparison"] != "exact-zero-padding"):
        raise ValueError(f"{case_id}: invalid result padding segment")
    golden_doc = result["golden"]
    if not isinstance(golden_doc, dict) or set(golden_doc) != {"path", "sha256", "derivation"}:
        raise ValueError(f"{case_id}: invalid golden fields")
    golden_hash = checked_hash(golden_doc["sha256"], f"{case_id} golden")
    if golden_doc["path"] != golden.name or not isinstance(golden_doc["derivation"], str):
        raise ValueError(f"{case_id}: golden path/derivation mismatch")
    require_matching_hashes(case_id,
        {"source": sha256(source), "elf": sha256(elf), "golden": sha256(golden)},
        {"source": source_hash, "elf": elf_hash, "golden": golden_hash})
    return source, source_hash, elf_hash, golden_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--repo", type=pathlib.Path,
                        default=pathlib.Path(__file__).parents[1])
    parser.add_argument("--coverage", type=pathlib.Path)
    for name in ("toolchain", "tileop", "pto", "model"):
        parser.add_argument(f"--{name}-checkout", required=True, type=pathlib.Path)
    parser.add_argument("indexes", nargs="+", type=pathlib.Path)
    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    coverage_path = (arguments.coverage or repo / "microbenchmark" / "coverage.json").resolve()
    coverage = json.loads(coverage_path.read_text())
    expected_ids = validate_coverage(coverage)
    checkouts = {name: getattr(arguments, f"{name}_checkout")
                 for name in ("toolchain", "tileop", "pto", "model")}

    seen: set[str] = set()
    elf_basenames: set[str] = set()
    rows: list[str] = []
    cases: list[dict[str, object]] = []
    common_identity: dict[str, object] | None = None
    for index in arguments.indexes:
        for line_number, raw in enumerate(index.read_text().splitlines(), start=1):
            if not raw or raw.startswith("#"):
                continue
            fields = raw.split("|")
            if len(fields) != 4 or not fields[0]:
                raise ValueError(f"{index}:{line_number}: invalid corpus row")
            case_id, elf_text, sidecar_text, golden_text = fields
            claim_case(seen, case_id)
            elf = checked_file(elf_text, f"{case_id} ELF")
            if elf.name in elf_basenames:
                raise ValueError(f"duplicate ELF basename: {elf.name}")
            elf_basenames.add(elf.name)
            sidecar = checked_file(sidecar_text, f"{case_id} sidecar")
            golden = checked_file(golden_text, f"{case_id} golden")
            case_name = case_id.split(".", 1)[-1]
            if sidecar.name != f"{case_name}.sidecar.json":
                raise ValueError(f"{case_id}: sidecar path does not match case id")
            if golden.name != f"{case_name}.golden.bin":
                raise ValueError(f"{case_id}: golden path does not match case id")
            document = json.loads(sidecar.read_text())
            if common_identity is None:
                common_identity = validate_identity(document.get("identity"), repo, checkouts)
            source, source_hash, elf_hash, golden_hash = validate_sidecar(
                case_id, document, repo, elf, golden, common_identity)
            cases.append({
                "case_id": case_id,
                "execution_classification": document["execution"]["classification"],
                "source": {"path": source.relative_to(repo).as_posix(), "sha256": source_hash},
                "elf": {"path": elf.name, "sha256": elf_hash},
                "sidecar": {"path": sidecar.name, "sha256": sha256(sidecar)},
                "independent_golden": {"path": golden.name, "sha256": golden_hash},
            })
            rows.append(raw)
    if seen != expected_ids:
        raise ValueError(f"corpus coverage mismatch: missing={sorted(expected_ids-seen)} extra={sorted(seen-expected_ids)}")
    if common_identity is None:
        raise ValueError("empty corpus")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(rows) + "\n")
    manifest = arguments.manifest or arguments.output.with_suffix(".manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema": "pto-asl-artifact-manifest-v1",
        "identity": common_identity,
        "coverage": {"active": 333, "explicitly_unsupported": 29,
                     "families": EXPECTED_FAMILIES},
        "cases": cases,
    }, indent=2, sort_keys=True) + "\n")
    print(f"merged ASL corpus: {len(cases)} cases; manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
