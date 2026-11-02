"""Regression tests: incomplete or corrupted model results must never pass."""

import struct
import tempfile
import unittest
from pathlib import Path

from run_payload_check import input_and_golden, verify_exports


class ExportCheckTests(unittest.TestCase):
    def test_complete_exports_and_corruptions(self):
        _, _, golden = input_and_golden()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output = directory / "outputs.bin"
            status = directory / "statuses.bin"
            output.write_bytes(golden)
            status.write_bytes(bytes(48))
            verify_exports(directory, golden)
            for corrupt in (b"", golden[:-4], golden + bytes(4), bytes(len(golden))):
                with self.subTest(length=len(corrupt)):
                    output.write_bytes(corrupt)
                    with self.assertRaises(ValueError):
                        verify_exports(directory, golden)
            output.write_bytes(golden)
            status.write_bytes(struct.pack("<12i", 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
            with self.assertRaises(ValueError):
                verify_exports(directory, golden)
            status.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_exports(directory, golden)

    def test_tail_overwrite_is_a_failure(self):
        _, _, golden = input_and_golden()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            corrupted = bytearray(golden)
            # First case is 32 live elements in a 96-element destination.
            struct.pack_into("<f", corrupted, 32 * 4, 0.0)
            (directory / "outputs.bin").write_bytes(corrupted)
            (directory / "statuses.bin").write_bytes(bytes(48))
            with self.assertRaises(ValueError):
                verify_exports(directory, golden)


if __name__ == "__main__":
    unittest.main()
