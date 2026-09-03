#!/usr/bin/env python3

import copy
import json
import pathlib
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import merge_asl_corpus_indexes as manifest
import generate_asl_scalar_corpus as producer


class ManifestFailClosedTest(unittest.TestCase):
    @staticmethod
    def elf_metadata(include_stop=True):
        def symbol(name, value, size=0, section=1):
            return {"Symbol": {"Name": {"Value": name}, "Binding": {"RawValue": 1},
                               "Value": value, "Size": size,
                               "Section": {"RawValue": section}}}
        symbols = [symbol("main", 0x1000),
                   symbol("_end", 0x1800),
                   symbol("cross_model_result", 0x2000, producer.RESULT_BYTES),
                   symbol("cross_model_result_size", producer.RESULT_BYTES, 0, 0xFFF1)]
        if include_stop:
            symbols.append(symbol("cross_model_stop", 0x1800))
        return {
            "ElfHeader": {"Machine": {"RawValue": 0xE9}, "Entry": 0x1000},
            "ProgramHeaders": [{"ProgramHeader": {
                "Type": {"RawValue": 1}, "VirtualAddress": 0,
                "PhysicalAddress": 0, "FileSize": 0x4000,
                "MemSize": 0x5000, "Flags": {"RawFlags": 2}}}],
            "Symbols": symbols,
        }

    @staticmethod
    def write_elf_header(path):
        data = bytearray(64)
        data[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", data, 18, 0xE9)
        path.write_bytes(data)

    def test_missing_artifact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing absolute artifact"):
            manifest.checked_file("/definitely/missing/corpus.elf", "case ELF")

    def test_duplicate_case_is_rejected(self):
        seen = set()
        manifest.claim_case(seen, "scalar.case")
        with self.assertRaisesRegex(ValueError, "duplicate corpus case id"):
            manifest.claim_case(seen, "scalar.case")

    def test_invalid_hash_is_rejected(self):
        for value in ("", "0" * 63, "g" * 64, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "invalid sha256"):
                    manifest.checked_hash(value, "artifact")

    def test_mismatched_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            manifest.require_matching_hashes(
                "scalar.case", {"elf": "a" * 64}, {"elf": "b" * 64}
            )

    def test_relative_artifact_is_rejected_even_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "artifact"
            path.write_bytes(b"data")
            with self.assertRaisesRegex(ValueError, "missing absolute artifact"):
                manifest.checked_file(path.name, "artifact")

    def test_non_elf_is_rejected_before_metadata_tool_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            elf = pathlib.Path(directory) / "fake.elf"
            elf.write_bytes(b"not an ELF")
            with self.assertRaisesRegex(ValueError, "not an ELF"):
                producer.validate_elf_contract(pathlib.Path("/missing/readelf"), elf)

    def test_shared_elf_contract_accepts_complete_abi(self):
        with tempfile.TemporaryDirectory() as directory:
            elf = pathlib.Path(directory) / "valid.elf"
            self.write_elf_header(elf)
            with mock.patch.object(producer, "llvm_metadata", return_value=self.elf_metadata()):
                header, symbols, segments = producer.validate_elf_contract(pathlib.Path("readelf"), elf)
            self.assertEqual(header["Machine"]["RawValue"], 0xE9)
            self.assertIn("cross_model_stop", symbols)
            self.assertTrue(segments[0]["flags"] & 2)

    def test_shared_elf_contract_rejects_missing_stop_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            elf = pathlib.Path(directory) / "missing-stop.elf"
            self.write_elf_header(elf)
            with mock.patch.object(producer, "llvm_metadata",
                                   return_value=self.elf_metadata(include_stop=False)):
                with self.assertRaisesRegex(ValueError, "missing cross-model symbol"):
                    producer.validate_elf_contract(pathlib.Path("readelf"), elf)

    def test_duplicate_elf_basename_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "same.elf").write_bytes(b"a")
            (root / "b" / "same.elf").write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "duplicate ELF basename"):
                producer.discover_elfs(root)

    def test_duplicate_unsupported_is_rejected(self):
        root = pathlib.Path(__file__).parents[1]
        coverage = json.loads((root / "microbenchmark" / "coverage.json").read_text())
        coverage["unsupported"].append(copy.deepcopy(coverage["unsupported"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate unsupported"):
            manifest.validate_coverage(coverage)

    def test_wrong_unsupported_set_is_rejected(self):
        root = pathlib.Path(__file__).parents[1]
        coverage = json.loads((root / "microbenchmark" / "coverage.json").read_text())
        coverage["unsupported"][0]["name"] = "invented"
        with self.assertRaisesRegex(ValueError, "locked 29-case set"):
            manifest.validate_coverage(coverage)

    def test_active_unsupported_overlap_is_rejected(self):
        root = pathlib.Path(__file__).parents[1]
        coverage = json.loads((root / "microbenchmark" / "coverage.json").read_text())
        coverage["unsupported"][0]["family"] = coverage["active"][0]["family"]
        coverage["unsupported"][0]["name"] = coverage["active"][0]["name"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            manifest.validate_coverage(coverage)

    def test_mixed_identity_is_rejected_before_elf_processing(self):
        document = {key: {} for key in (
            "identity", "source", "elf", "model", "start", "execution", "result"
        )}
        document.update({"schema": "pto-asl-elf-sidecar-v1",
                         "case_id": "scalar.case"})
        document["identity"] = {"unexpected": "identity"}
        with self.assertRaisesRegex(ValueError, "mixed or incorrect identity"):
            manifest.validate_sidecar(
                "scalar.case", document, pathlib.Path.cwd(),
                pathlib.Path("fake.elf"), pathlib.Path("fake.golden"), {}
            )

    def test_missing_sidecar_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid sidecar fields"):
            manifest.validate_sidecar(
                "scalar.case", {}, pathlib.Path.cwd(),
                pathlib.Path("fake.elf"), pathlib.Path("fake.golden"), {}
            )

    def test_wrong_identity_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity schema"):
            manifest.validate_identity({}, pathlib.Path.cwd(), {})

    def test_tool_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = pathlib.Path(directory) / "tool"
            tool.write_bytes(b"tool")
            tool.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                producer.checked_tool(tool, "0" * 64, "tool")

    def test_checkout_commit_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = pathlib.Path(directory)
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "config", "user.name", "test"], check=True)
            subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True)
            marker = checkout / "marker"
            marker.write_text("x")
            subprocess.run(["git", "-C", str(checkout), "add", "marker"], check=True)
            subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "initial"], check=True)
            with self.assertRaisesRegex(ValueError, "expected"):
                producer.checked_checkout(checkout, "0" * 40, "checkout")


if __name__ == "__main__":
    unittest.main()
