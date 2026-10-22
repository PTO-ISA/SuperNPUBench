#!/usr/bin/env python3
"""Focused tests for pre/post execution input provenance verification."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from unittest import mock
import sys
import subprocess
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import write_provenance
from write_provenance import verify_immutable_inputs


def sample_identity() -> dict[str, object]:
    return {
        "created_utc": "before",
        "target": {"triple": "linx64v5", "resource_dir": "/resource",
                   "selected_models": ["gfrun", "gfsim"], "pe_count": 1},
        "compiler_version": "clang version test",
        "artifacts": {
            "elf": {"path": "/run/test.elf", "sha256": "elf"},
            "gfrun": "gfrun",
            "gfsim": "gfsim",
            "goldens": {"result.bin": "golden"},
            "installed_tileop_headers": {"include/api.hpp": "api"},
        },
        "repositories": {
            "benchmark": {"head": "bench", "tracked_diff_sha256": "source"},
            "compiler": {"head": "compiler", "tracked_diff_sha256": "clean"},
        },
        "commands": {
            "gfrun": ["gfrun", "-f", "/run/test.elf"],
            "gfsim": ["gfsim", "-f", "/run/test.elf"],
        },
        "run_exit_status": 125,
        "evidence": {"build.log": "initial"},
        "content_id": "before-content",
    }


class InputProvenanceTest(unittest.TestCase):
    def test_hashes_invoked_compiler_driver_without_versioned_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / "bin"
            compiler.mkdir()
            (compiler / "clang").write_bytes(b"actual compiler")
            (compiler / "clang++").symlink_to("clang")
            (compiler / "lld").write_bytes(b"linker")
            (compiler / "ld.lld").symlink_to("lld")
            (compiler / "llvm-objdump").write_bytes(b"objdump")
            for name in ("libc.a", "libc++.a", "libc++abi.a", "libunwind.a"):
                p = root / "sysroot/usr/lib" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"runtime")
            # A documented SYSROOT override must win over the runtime root.
            decoy_lib = root / "different-runtime/sysroot/usr/lib/libc.a"
            decoy_lib.parent.mkdir(parents=True)
            decoy_lib.write_bytes(b"unused runtime")
            for name in ("api/common/pto_tileop.hpp",
                         "resource/include/tileop-api/common/pto_tileop.hpp",
                         "golden/expected.bin", "test.elf", "gfrun"):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fixture")
            artifact = root / "artifacts"
            artifact.mkdir()
            output = root / "provenance.json"
            argv = ["write_provenance.py", "--out", str(output),
                    "--elf", str(root / "test.elf"), "--compiler-bin", str(compiler),
                    "--model-root", str(root), "--runtime-root", str(root / "different-runtime"),
                    "--api-include", str(root / "api"), "--bench-root", str(root),
                    "--golden-dir", str(root / "golden"), "--target-triple", "linx64v5",
                    "--sysroot", str(root / "sysroot"),
                    "--resource-dir", str(root / "resource"),
                    "--artifact-dir", str(artifact), "--dump-range", "0x1000:4",
                    "--run-exit-status", "125", "--models", "gfrun",
                    "--gfrun", str(root / "gfrun")]
            for decoy, pe_count in ((False, 1), (True, 1), (True, 4)):
                with self.subTest(versioned_decoy=decoy, pe_count=pe_count):
                    if decoy:
                        (compiler / "clang-15").write_bytes(b"different unused compiler")
                    with mock.patch.object(sys, "argv", argv + ["--pe-count", str(pe_count)]), \
                            mock.patch.object(write_provenance, "run", return_value=str(root)), \
                            mock.patch.object(write_provenance, "git_identity", return_value={}):
                        self.assertEqual(write_provenance.main(), 0)
                    data = json.loads(output.read_text())
                    self.assertEqual(data["artifacts"]["clang"],
                                     hashlib.sha256(b"actual compiler").hexdigest())
                    self.assertEqual(set(data["artifacts"]["runtime"]),
                                     {str(root / "sysroot/usr/lib" / name) for name in
                                      ("libc.a", "libc++.a", "libc++abi.a", "libunwind.a")})
                    self.assertEqual(data["target"]["pe_count"], pe_count)
                    self.assertIn(f"softcore.multiThreadNum={pe_count}",
                                  data["commands"]["gfrun"])

    def test_unchanged_inputs_accept_growing_evidence_and_status(self) -> None:
        before = sample_identity()
        after = copy.deepcopy(before)
        after["created_utc"] = "after"
        after["run_exit_status"] = 0
        after["evidence"]["gfrun.log"] = "new-log"  # type: ignore[index]
        after["content_id"] = "after-content"
        verify_immutable_inputs(before, after)

    def test_rejects_changed_elf_binary_and_source(self) -> None:
        mutations = {
            "PE count": lambda value: value["target"].update(pe_count=4),
            "selected models": lambda value: value["target"].update(
                selected_models=["gfrun"]
            ),
            "ELF": lambda value: value["artifacts"]["elf"].update(sha256="new"),
            "model binary": lambda value: value["artifacts"].update(gfrun="new"),
            "source": lambda value: value["repositories"]["benchmark"].update(
                tracked_diff_sha256="new"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                before = sample_identity()
                after = copy.deepcopy(before)
                mutate(after)
                with self.assertRaisesRegex(
                    RuntimeError, "validation inputs changed after execution"
                ):
                    verify_immutable_inputs(before, after)


class FailureAndExternalBuildTest(unittest.TestCase):
    def make_arguments(self, root: Path, complete: bool, status: int = 0):
        artifact = root / "artifacts"
        artifact.mkdir(exist_ok=True)
        if complete:
            for name in ("bin/clang++", "bin/ld.lld", "bin/llvm-objdump", "gfrun",
                         "test.elf", "golden/expected.bin", "api/common/pto_tileop.hpp",
                         "resource/include/tileop-api/common/pto_tileop.hpp",
                         *["sysroot/usr/lib/" + name for name in
                           ("libc.a", "libc++.a", "libc++abi.a", "libunwind.a")]):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            compiler = root / "bin/clang++"
            compiler.write_text("#!/bin/sh\nprintf 'clang version fixture\\n'\n")
            compiler.chmod(0o755)
        return ["write_provenance.py", "--out", str(artifact / "provenance.json"),
                "--elf", str(root / "test.elf"), "--compiler-bin", str(root / "bin"),
                "--model-root", str(root / "model-build"), "--runtime-root", str(root),
                "--api-include", str(root / "api"), "--bench-root", str(root),
                "--golden-dir", str(root / "golden"), "--target-triple", "linx64v5",
                "--sysroot", str(root / "sysroot"), "--resource-dir", str(root / "resource"),
                "--artifact-dir", str(artifact), "--run-exit-status", str(status),
                "--models", "gfrun", "--gfrun", str(root / "gfrun")]

    def run_writer(self, args):
        with mock.patch.object(sys, "argv", args):
            status = write_provenance.main()
        return status, json.loads(Path(args[args.index("--out") + 1]).read_text())

    def test_external_builds_without_git_keep_binary_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_arguments(root, True) + ["--dump-range", "0x1000:4"]
            status, data = self.run_writer(args)
            self.assertEqual(status, 0)
            for repo in data["repositories"].values():
                self.assertFalse(repo["identity_available"])
                self.assertIsNone(repo["head"])
            self.assertEqual(data["artifacts"]["clang"],
                             write_provenance.sha256_file(root / "bin/clang++"))

    def test_explicit_source_checkout_for_external_binaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_arguments(root, True) + ["--dump-range", "0x1000:4"]
            source = root / "source"
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "user.name=Fixture",
                            "-c", "user.email=fixture@example.invalid", "commit", "-qm",
                            "fixture", "--allow-empty"], cwd=source, check=True)
            for name in ("compiler", "model", "api"):
                args += ["--" + name + "-source", str(source)]
            status, data = self.run_writer(args)
            self.assertEqual(status, 0)
            for name in ("compiler", "model", "tileop_api"):
                self.assertTrue(data["repositories"][name]["identity_available"])
                self.assertEqual(data["repositories"][name]["root"], str(source.resolve()))

    def test_early_failure_writes_partial_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status, data = self.run_writer(
                self.make_arguments(root, False, 37) + ["--allow-incomplete"])
            self.assertEqual(status, 0)
            self.assertEqual(data["run_exit_status"], 37)
            self.assertEqual(data["provenance_status"], "incomplete")
            self.assertIsNone(data["artifacts"]["elf"]["sha256"])
            self.assertEqual(data["commands"], {})
            self.assertEqual(len(data["content_id"]), 64)
            self.assertFalse((root / "artifacts/gfrun.mem").exists())

    def test_missing_inputs_never_become_success(self):
        with tempfile.TemporaryDirectory() as directory:
            status, data = self.run_writer(
                self.make_arguments(Path(directory), False) + ["--allow-incomplete"])
            self.assertEqual(status, 1)
            self.assertEqual(data["run_exit_status"], 1)
            self.assertEqual(data["execution_exit_status"], 0)

    def test_mutated_input_still_leaves_failure_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_arguments(root, True) + ["--dump-range", "0x1000:4"]
            before = root / "input.json"
            initial = args[:]
            initial[initial.index("--out") + 1] = str(before)
            self.assertEqual(self.run_writer(initial)[0], 0)
            (root / "test.elf").write_bytes(b"changed")
            status, data = self.run_writer(args + ["--verify-inputs-from", str(before)])
            self.assertEqual(status, 1)
            self.assertEqual(data["run_exit_status"], 1)
            self.assertTrue(any("artifacts.elf.sha256" in p for p in data["provenance_errors"]))


if __name__ == "__main__":
    unittest.main()
