#!/usr/bin/env python3
"""Host regressions: python3 -m unittest discover -s benchmark/one-level-arch/test/solution/common -p 'test_*.py'."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

runner = load_module("solution_res_check", HERE / "res_check_all.py")
generator = load_module("group_norm_generator", HERE.parent / "normalization/group_norm_grad/src/gen_group_norm_grad_data.py")

class HalfOracleTests(unittest.TestCase):
    def test_all_half_encodings_match_standard_library(self):
        for bits in range(65536):
            expected = struct.unpack("<e", struct.pack("<H", bits))[0]
            actual = generator.f16_bits_to_f32(bits)
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual), hex(bits))
            else:
                self.assertEqual(actual, expected, hex(bits))
                if expected == 0:
                    self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))

    def test_golden_dbeta_matches_serialized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generator.gen_all(output, 2, 4, 2, 8, 8, 1e-5, 123)
            dy = struct.unpack("<64e", (output / "dy.bin").read_bytes())
            golden = struct.unpack("<4e", (output / "golden_dbeta.bin").read_bytes())
            expected = [sum(dy[(n * 4 + c) * 8 + h] for n in range(2) for h in range(8))
                        for c in range(4)]
            expected = struct.unpack("<4e", struct.pack("<4e", *expected))
            self.assertEqual(golden, expected)

class FailurePropagationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.unit = self.root / "test/solution/fake"
        self.unit.mkdir(parents=True)
        self.output = self.root / "output/solution"
        self.case = runner.Case("fake", "fake/elf/fake.elf")
        self.elf = self.output / self.case.elf
        self.elf.parent.mkdir(parents=True)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("ROOT", self.root), ("OUTPUT", self.output),
                            ("COMPARE", self.root / "compare"), ("CASES", [self.case])):
            self.stack.enter_context(patch.object(runner, name, value))

    def main(self, compile=True):
        args = ["res_check_all.py", "--gfrun", "/must-not-run", "fake"]
        if compile:
            args += ["--compiler-dir", "/missing/compiler"]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.main()

    def test_compile_failure_stops_even_with_stale_elf(self):
        self.elf.write_bytes(b"old ELF")
        (self.unit / "compile.all").write_text("echo compiler-failed >&2\nexit 7\n")
        with patch.object(runner, "run_case") as run:
            self.assertEqual(self.main(), 1)
            run.assert_not_called()

    def test_missing_compile_script_fails(self):
        self.assertEqual(self.main(), 1)

    def test_successful_script_with_missing_elf_fails(self):
        (self.unit / "compile.all").write_text("exit 0\n")
        self.assertEqual(self.main(), 1)

    def test_successful_compilation_runs_case(self):
        (self.unit / "compile.all").write_text("touch " + str(self.elf) + "\n")
        with patch.object(runner, "run_case", return_value=("PASS", "ok")) as run:
            self.assertEqual(self.main(), 0)
            run.assert_called_once()

    def test_external_missing_elf_keeps_skip_behavior(self):
        self.assertEqual(self.main(compile=False), 0)

    def test_failed_generator_does_not_run_or_check_stale_data(self):
        self.elf.write_bytes(b"old ELF")
        (self.unit / "gen.py").write_text("import sys\nprint('generator failed', file=sys.stderr)\nsys.exit(9)\n")
        self.case.prepare = runner.make_prep_gen("fake/gen.py")
        self.case.verify = lambda *_: self.fail("checker must not run after preparation failure")
        status, detail = runner.run_case(self.case, Path("/must-not-run"), 1)
        self.assertEqual(status, "FAIL")
        self.assertIn("rc=9", detail)
        self.assertIn("generator failed", (self.root / "compare/fake/prepare.log").read_text())

if __name__ == "__main__":
    unittest.main()
