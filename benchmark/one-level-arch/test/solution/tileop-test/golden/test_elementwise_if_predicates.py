#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("golden.py")
SPEC = importlib.util.spec_from_file_location("tileop_golden", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def expected_outputs() -> np.ndarray:
    lanes = np.arange(128)
    lhs = np.where(
        lanes % 3 == 0,
        np.float32(1.0),
        np.where(lanes % 3 == 1, np.float32(-1.0), np.float32(0.0)),
    ).astype(np.float32)
    rhs = np.full(128, np.float32(0.5), dtype=np.float32)
    predicates = (
        lhs > 0.0,
        lhs < 0.0,
        lhs >= 0.0,
        lhs <= 0.0,
        lhs == 0.0,
        lhs != 0.0,
        np.logical_not(lhs > 0.0),
    )
    return np.stack(
        [np.where(predicate, lhs + rhs, lhs - rhs) for predicate in predicates]
    ).astype(np.float32)


class PredicateGoldenTest(unittest.TestCase):
    def test_accepts_all_seven_independent_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected_outputs().tofile(Path(directory) / "out.bin")
            self.assertEqual(
                MODULE.check("elementwise_if_predicates", directory), 0
            )

    def test_rejects_missing_or_partial_word_outputs(self) -> None:
        for filename in ['out.bin']:
            for mode in ('missing', 'truncated', 'extra-byte'):
                with self.subTest(filename=filename, mode=mode), tempfile.TemporaryDirectory() as directory:
                    expected_outputs().tofile(Path(directory) / "out.bin")
                    path = Path(directory) / filename
                    if mode == 'missing':
                        path.unlink()
                    elif mode == 'truncated':
                        path.write_bytes(path.read_bytes()[:-1])
                    else:
                        path.write_bytes(path.read_bytes() + b'\x00')
                    self.assertEqual(MODULE.check('elementwise_if_predicates', directory), 1)

    def test_rejects_corruption_in_each_result(self) -> None:
        for predicate in range(7):
            with self.subTest(predicate=predicate), tempfile.TemporaryDirectory() as directory:
                actual = expected_outputs()
                actual[predicate, predicate] += np.float32(1.0)
                actual.tofile(Path(directory) / "out.bin")
                self.assertEqual(
                    MODULE.check("elementwise_if_predicates", directory), 1
                )


if __name__ == "__main__":
    unittest.main()
