#!/usr/bin/env python3
"""Focused tests for pre/post execution input provenance verification."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_provenance import verify_immutable_inputs


def sample_identity() -> dict[str, object]:
    return {
        "created_utc": "before",
        "target": {"triple": "linx64v5", "resource_dir": "/resource"},
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


if __name__ == "__main__":
    unittest.main()
