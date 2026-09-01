#!/usr/bin/env python3
"""Fetch the content-locked compiler-workload sources into a local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "sources.lock.json").read_text())
RECEIPT = ".supernpubench-source.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, directory: Path) -> Path:
    directory.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as bundle:
        root = directory.resolve()
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(directory)
    roots = [path for path in directory.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"archive has {len(roots)} top-level directories")
    return roots[0]


def receipt_matches(destination: Path, expected: dict[str, object]) -> bool:
    receipt = destination / RECEIPT
    if not destination.is_dir() or not receipt.is_file():
        return False
    try:
        return json.loads(receipt.read_text()) == expected
    except (OSError, json.JSONDecodeError):
        return False


def install_archive(*, url: str, destination: Path,
                    expected: dict[str, object], archive_sha256: str | None = None) -> None:
    if receipt_matches(destination, expected):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="supernpu-workloads-") as tmp_name:
        tmp = Path(tmp_name)
        archive = tmp / "source.tar.gz"
        urllib.request.urlretrieve(url, archive)
        observed = sha256(archive)
        if archive_sha256 and observed != archive_sha256:
            raise RuntimeError(
                f"archive digest mismatch for {url}: expected {archive_sha256}, got {observed}"
            )
        extracted = extract_archive(archive, tmp / "extract")
        staged = destination.parent / f".{destination.name}.staged"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(extracted, staged, symlinks=True)
        (staged / RECEIPT).write_text(json.dumps(expected, indent=2) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=HERE / ".cache")
    args = parser.parse_args()
    cache = args.cache.resolve()

    linx = LOCK["linx_isa"]
    commit = linx["commit"]
    linx_root = cache / f"linx-isa-{commit}"
    install_archive(
        url=f"https://codeload.github.com/LinxISA/linx-isa/tar.gz/{commit}",
        destination=linx_root,
        expected={"repository": linx["repository"], "commit": commit},
    )

    third_party = linx_root / "workloads" / "third_party"
    for name, destination_name in (("polybench", "PolyBenchC"),
                                   ("ctuning_codelets", "ctuning-programs")):
        source = LOCK[name]
        ref = source["ref"]
        repository = source["repository"].removesuffix(".git")
        owner_repo = repository.removeprefix("https://github.com/")
        expected = {
            "repository": source["repository"],
            "ref": ref,
            "archive_sha256": source["archive_sha256"],
        }
        install_archive(
            url=f"https://codeload.github.com/{owner_repo}/tar.gz/{ref}",
            destination=third_party / destination_name,
            expected=expected,
            archive_sha256=source["archive_sha256"],
        )

    # TSVC is already vendored at the pinned commit by the locked LinxISA tree.
    tsvc_src = linx_root / "workloads" / "tsvc" / "upstream" / "TSVC_2" / "src"
    if not (tsvc_src / "tsvc.c").is_file():
        raise RuntimeError(f"locked TSVC source is missing: {tsvc_src}")

    print(linx_root / "workloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
