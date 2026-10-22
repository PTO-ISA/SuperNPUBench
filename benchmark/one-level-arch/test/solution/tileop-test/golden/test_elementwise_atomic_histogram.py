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


def write_outputs(directory: str) -> None:
    root = Path(directory)
    np.full(8, 16, dtype=np.int32).tofile(root / "out_bins.bin")
    np.full(128, 16, dtype=np.uint32).tofile(root / "out_observed.bin")
    np.array(
        [128, 8, 0, 128, 2048, 128, 4, 0x41544853], dtype=np.uint32
    ).tofile(root / "out_status.bin")


class AtomicHistogramGoldenTest(unittest.TestCase):
    def test_accepts_complete_independent_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_outputs(directory)
            self.assertEqual(
                MODULE.check("elementwise_atomic_histogram", directory), 0
            )

    def test_rejects_missing_or_partial_word_outputs(self) -> None:
        for filename in ['out_bins.bin', 'out_observed.bin', 'out_status.bin']:
            for mode in ('missing', 'truncated', 'extra-byte'):
                with self.subTest(filename=filename, mode=mode), tempfile.TemporaryDirectory() as directory:
                    write_outputs(directory)
                    path = Path(directory) / filename
                    if mode == 'missing':
                        path.unlink()
                    elif mode == 'truncated':
                        path.write_bytes(path.read_bytes()[:-1])
                    else:
                        path.write_bytes(path.read_bytes() + b'\x00')
                    self.assertEqual(MODULE.check('elementwise_atomic_histogram', directory), 1)

    def test_rejects_each_corrupted_output(self) -> None:
        files = ("out_bins.bin", "out_observed.bin", "out_status.bin")
        for filename in files:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                write_outputs(directory)
                path = Path(directory) / filename
                values = np.fromfile(path, dtype=np.uint32)
                values[len(values) // 2] ^= np.uint32(1)
                values.tofile(path)
                self.assertEqual(
                    MODULE.check("elementwise_atomic_histogram", directory), 1
                )


if __name__ == "__main__":
    unittest.main()
