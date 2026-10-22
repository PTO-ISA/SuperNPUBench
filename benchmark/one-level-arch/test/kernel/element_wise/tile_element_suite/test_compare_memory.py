#!/usr/bin/env python3
"""Corruption tests for paired-atomic and exact FP32 observation contracts."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MASK32 = 0xFFFFFFFF


class AtomicPairTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.golden = self.root / "golden"
        subprocess.run(
            ["python3", str(HERE / "generate_case.py"), "--case",
             "generic_atomic_cfg_i32", "--out", str(self.golden)], check=True,
        )
        manifest = json.loads((self.golden / "manifest.json").read_text())
        names_and_sizes = [
            ("generic_atomic_cfg_i32_indices", 512),
            ("generic_atomic_cfg_i32_deltas", 512),
            ("generic_atomic_cfg_i32_gates", 512),
            ("generic_atomic_cfg_i32_histogram", 128),
            ("generic_atomic_cfg_i32_first", 512),
            ("generic_atomic_cfg_i32_second", 512),
            ("generic_atomic_cfg_i32_combined", 512),
            ("generic_atomic_cfg_i32_poison", 512),
            ("generic_atomic_cfg_i32_status", 32),
        ]
        address = 0x1000
        self.symbols: dict[str, tuple[int, int]] = {}
        for name, size in names_and_sizes:
            self.symbols[name] = (address, size)
            address += (size + 31) & ~31
        self.memory = bytearray(address - 0x1000)
        for item in manifest["segments"]:
            if item["kind"] == "exact":
                self._store(item["symbol"], (self.golden / item["file"]).read_bytes())

        indices = list(struct.unpack("<128i", (self.golden / "indices_s32.bin").read_bytes()))
        first_adds = self._read_u32("first_adds_u32.bin")
        second_adds = self._read_u32("second_adds_u32.bin")
        initial = self._read_u32("initial_histogram_u32.bin")
        first, second, combined = [11] * 128, [13] * 128, [17] * 128
        for bin_index in range(32):
            current = initial[bin_index]
            elements = [
                element for element in reversed(range(125))
                if indices[element] + 16 == bin_index
            ]
            # Reverse source element order and interleave two elements as
            # first(A), first(B), second(B), second(A). This is deliberately
            # different from serialized element order while retaining the
            # required source order inside each element.
            for begin in range(0, len(elements), 2):
                pair = elements[begin:begin + 2]
                old_first: dict[int, int] = {}
                for element in pair:
                    old_first[element] = current
                    current = (current + first_adds[element]) & MASK32
                old_second: dict[int, int] = {}
                for element in reversed(pair):
                    old_second[element] = current
                    current = (current + second_adds[element]) & MASK32
                for element in pair:
                    first[element] = (old_first[element] + 11) & MASK32
                    second[element] = (old_second[element] + 13) & MASK32
                    combined[element] = (
                        (old_first[element] ^
                         ((old_second[element] + 0x13579BDF) & MASK32)) + 17
                    ) & MASK32
        self._store("generic_atomic_cfg_i32_first", struct.pack("<128I", *first))
        self._store("generic_atomic_cfg_i32_second", struct.pack("<128I", *second))
        self._store("generic_atomic_cfg_i32_combined", struct.pack("<128I", *combined))
        self.symbol_file = self.root / "symbols.txt"
        self.symbol_file.write_text("".join(
            f"{address:016x} g O .data {size:08x} {name}\n"
            for name, (address, size) in self.symbols.items()
        ))
        self.dump = self.root / "memory.bin"
        self.dump.write_bytes(self.memory)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _read_u32(self, filename: str) -> list[int]:
        data = (self.golden / filename).read_bytes()
        return list(struct.unpack(f"<{len(data) // 4}I", data))

    def _store(self, symbol: str, data: bytes) -> None:
        address, size = self.symbols[symbol]
        self.assertEqual(len(data), size)
        offset = address - 0x1000
        self.memory[offset:offset + size] = data

    def _run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(HERE / "compare_memory.py"),
             "--symbols", str(self.symbol_file), "--golden", str(self.golden),
             "--dump", str(self.dump), "--dump-base", "0x1000"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_accepts_source_order_without_global_element_order(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("order-independent paired atomic trace", result.stdout)

    def test_rejects_first_before_second_inversion(self) -> None:
        first_address, _ = self.symbols["generic_atomic_cfg_i32_first"]
        second_address, _ = self.symbols["generic_atomic_cfg_i32_second"]
        combined_address, _ = self.symbols["generic_atomic_cfg_i32_combined"]
        element = 0
        first_stored = struct.unpack_from(
            "<I", self.memory, first_address - 0x1000 + element * 4
        )[0]
        second_offset = second_address - 0x1000 + element * 4
        second_stored = struct.unpack_from("<I", self.memory, second_offset)[0]
        old_first = (first_stored - 11) & MASK32
        old_second = (second_stored - 13) & MASK32
        struct.pack_into(
            "<I", self.memory, first_address - 0x1000 + element * 4,
            (old_second + 11) & MASK32,
        )
        struct.pack_into(
            "<I", self.memory, second_offset, (old_first + 13) & MASK32
        )
        old_first, old_second = old_second, old_first
        combined = (
            (old_first ^ ((old_second + 0x13579BDF) & MASK32)) + 17
        ) & MASK32
        struct.pack_into(
            "<I", self.memory, combined_address - 0x1000 + element * 4, combined
        )
        self.dump.write_bytes(self.memory)
        result = self._run()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("cannot form a modulo-2^32 atomic order", result.stdout)

    def test_rejects_inactive_seed_corruption(self) -> None:
        first_address, _ = self.symbols["generic_atomic_cfg_i32_first"]
        struct.pack_into(
            "<I", self.memory, first_address - 0x1000 + 127 * 4, 12
        )
        self.dump.write_bytes(self.memory)
        result = self._run()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("inactive element 127 did not retain", result.stdout)


class OriginalF32BitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fixture(self, strong: bool) -> tuple[Path, Path, bytearray]:
        golden = self.root / ("strong" if strong else "original")
        subprocess.run(["python3", str(HERE / "generate_case.py"), "--case",
                        "elementwise_if_for_strong" if strong else "elementwise_if_for",
                        "--out", str(golden)], check=True)
        manifest = json.loads((golden / "manifest.json").read_text())
        memory = bytearray()
        symbols = self.root / "symbols.txt"
        symbol_lines = []
        for item in manifest["segments"]:
            data = (golden / item["file"]).read_bytes()
            symbol_lines.append(f"{0x1000 + len(memory):016x} g O .data {len(data):08x} {item['symbol']}\n")
            memory.extend(data)
        symbols.write_text("".join(symbol_lines))
        return golden, symbols, memory

    def _compare(self, golden: Path, symbols: Path, memory: bytearray) -> subprocess.CompletedProcess[str]:
        dump = self.root / "memory.bin"
        dump.write_bytes(memory)
        return subprocess.run(["python3", str(HERE / "compare_memory.py"),
                               "--symbols", str(symbols), "--golden", str(golden),
                               "--dump", str(dump), "--dump-base", "0x1000"],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def test_original_retains_three_512_byte_zero_buffers(self) -> None:
        golden, symbols, memory = self._fixture(False)
        self.assertEqual(len(memory), 1536)
        self.assertEqual(memory, bytes(1536))
        result = self._compare(golden, symbols, memory)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_strong_rawbits_and_guards_pass(self) -> None:
        golden, symbols, memory = self._fixture(True)
        self.assertEqual(len(memory), 1664)
        self.assertEqual(memory[1024:1088], bytes([0x5A]) * 64)
        self.assertEqual(memory[-64:], bytes([0x5A]) * 64)
        self.assertEqual(struct.unpack_from("<I", memory, 1088 + 4)[0], 0x80000000)
        result = self._compare(golden, symbols, memory)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_rejects_signed_zero_normalization(self) -> None:
        golden, symbols, memory = self._fixture(True)
        struct.pack_into("<I", memory, 1088 + 4, 0)
        result = self._compare(golden, symbols, memory)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("expected=0x80000000", result.stdout)

    def test_rejects_input_or_output_guard_corruption(self) -> None:
        golden, symbols, memory = self._fixture(True)
        for offset, symbol in ((0, "lhs_storage"), (512, "rhs_storage"),
                               (1024, "out_storage"), (len(memory) - 1, "out_storage")):
            with self.subTest(offset=offset):
                corrupted = memory.copy()
                corrupted[offset] ^= 1
                result = self._compare(golden, symbols, corrupted)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(symbol + ": mismatch", result.stdout)

    def test_rejects_short_output_extent(self) -> None:
        golden, symbols, memory = self._fixture(True)
        result = self._compare(golden, symbols, memory[:-1])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("dump does not cover out_storage", result.stdout)


class MaskedGatherScatterF32BitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fixture(self, strong: bool) -> tuple[Path, Path, bytearray, dict[str, int]]:
        case = ("elementwise_masked_gather_scatter_strong" if strong else
                "elementwise_masked_gather_scatter")
        golden = self.root / case
        subprocess.run(["python3", str(HERE / "generate_case.py"), "--case",
                        case, "--out", str(golden)], check=True)
        manifest = json.loads((golden / "manifest.json").read_text())
        memory = bytearray()
        offsets: dict[str, int] = {}
        symbol_lines = []
        for item in manifest["segments"]:
            data = (golden / item["file"]).read_bytes()
            offsets[item["symbol"]] = len(memory)
            symbol_lines.append(
                f"{0x1000 + len(memory):016x} g O .data "
                f"{len(data):08x} {item['symbol']}\n"
            )
            memory.extend(data)
        symbols = self.root / (case + ".symbols")
        symbols.write_text("".join(symbol_lines))
        return golden, symbols, memory, offsets

    def _compare(self, golden: Path, symbols: Path,
                 memory: bytearray) -> subprocess.CompletedProcess[str]:
        dump = self.root / "masked-memory.bin"
        dump.write_bytes(memory)
        return subprocess.run(
            ["python3", str(HERE / "compare_memory.py"),
             "--symbols", str(symbols), "--golden", str(golden),
             "--dump", str(dump), "--dump-base", "0x1000"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_original_preserves_five_128_element_arrays(self) -> None:
        golden, symbols, memory, offsets = self._fixture(False)
        self.assertEqual(len(memory), 512 + 512 + 128 + 512 + 512)
        self.assertEqual(memory[offsets["gathered_storage"]:
                                offsets["gathered_storage"] + 512], bytes(512))
        output = struct.unpack_from("<128I", memory, offsets["output_storage"])
        self.assertEqual(output[:8], (0x41200000, 0, 0x41F00000, 0,
                                      0x42480000, 0, 0x428C0000, 0))
        self.assertEqual(self._compare(golden, symbols, memory).returncode, 0)

    def test_strong_has_raw_payloads_poison_masks_and_guards(self) -> None:
        golden, symbols, memory, offsets = self._fixture(True)
        source = struct.unpack_from("<128I", memory, offsets["source_storage"])
        indices = struct.unpack_from("<128i", memory, offsets["index_storage"])
        masks = memory[offsets["mask_storage"]:offsets["mask_storage"] + 128]
        output = struct.unpack_from("<128I", memory, offsets["output_storage"])
        self.assertIn(0x80000000, source)
        self.assertIn(0x7F800000, source)
        self.assertIn(0x7FC12345, source)
        self.assertIn(0x7F800123, source)
        self.assertIn(0xFF800321, output)
        self.assertIn(-0x80000000, indices)
        self.assertIn(0x7FFFFFFF, indices)
        self.assertTrue({2, 3, 0x80, 0xFF}.issubset(set(masks)))
        self.assertEqual(memory[offsets["output_guard_before"]:
                                offsets["output_guard_before"] + 64],
                         bytes([0x5A]) * 64)
        self.assertEqual(memory[offsets["output_guard_after"]:
                                offsets["output_guard_after"] + 64],
                         bytes([0xA5]) * 64)
        # Each repeated active destination has one source-derived raw payload.
        for element, active in enumerate(masks):
            if active:
                index = indices[element]
                self.assertEqual(output[64 + index], source[64 + index])
        result = self._compare(golden, symbols, memory)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_rejects_nan_payload_sentinel_or_guard_corruption(self) -> None:
        golden, symbols, memory, offsets = self._fixture(True)
        corruptions = (
            (offsets["output_storage"] + (64 - 60) * 4, "output_storage"),
            (offsets["output_storage"] + (64 - 62) * 4, "output_storage"),
            (offsets["output_guard_before"], "output_guard_before"),
            (offsets["output_guard_after"] + 63, "output_guard_after"),
        )
        for offset, symbol in corruptions:
            with self.subTest(symbol=symbol, offset=offset):
                corrupted = memory.copy()
                corrupted[offset] ^= 1
                result = self._compare(golden, symbols, corrupted)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(symbol + ": mismatch", result.stdout)

    def test_rejects_nonadjacent_output_guards(self) -> None:
        golden, symbols, memory, _ = self._fixture(True)
        altered = []
        for line in symbols.read_text().splitlines():
            if line.endswith(" output_storage"):
                fields = line.split()
                fields[0] = f"{int(fields[0], 16) + 1:016x}"
                line = " ".join(fields)
            altered.append(line)
        symbols.write_text("\n".join(altered) + "\n")
        result = self._compare(golden, symbols, memory)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("guarded segments are not adjacent", result.stdout)


class ConcatGatherF16BitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _source_element(global_index: int) -> int:
        row = (global_index // 8000) % 256
        column = global_index % 8000
        return (column // 8) * (256 * 8) + row * 8 + column % 8

    @staticmethod
    def _input_bits(element: int) -> int:
        return (element * 40503 + 0x35A7) & 0xFFFF

    def _fixture(
        self, alias: bool
    ) -> tuple[Path, Path, bytearray, dict[str, int]]:
        case = ("concat_gather_f16_alias_probe" if alias else
                "concat_gather_f16_original")
        golden = self.root / case
        subprocess.run(
            ["python3", str(HERE / "generate_case.py"), "--case", case,
             "--out", str(golden)],
            check=True,
        )
        manifest = json.loads((golden / "manifest.json").read_text())
        memory = bytearray()
        offsets: dict[str, int] = {}
        symbol_lines = []
        guard_order = {
            "concat_gather_f16_original_output_guard_before": 0,
            "concat_gather_f16_original_output": 1,
            "concat_gather_f16_original_output_guard_after": 2,
        }
        for item in sorted(manifest["segments"],
                           key=lambda item: guard_order.get(item["symbol"], 3)):
            data = (golden / item["file"]).read_bytes()
            offsets[item["symbol"]] = len(memory)
            symbol_lines.append(
                f"{0x1000 + len(memory):016x} g O .data "
                f"{len(data):08x} {item['symbol']}\n"
            )
            memory.extend(data)
        symbols = self.root / (case + ".symbols")
        symbols.write_text("".join(symbol_lines))
        return golden, symbols, memory, offsets

    def _compare(self, golden: Path, symbols: Path,
                 memory: bytearray) -> subprocess.CompletedProcess[str]:
        dump = self.root / "concat-f16-memory.bin"
        dump.write_bytes(memory)
        return subprocess.run(
            ["python3", str(HERE / "compare_memory.py"),
             "--symbols", str(symbols), "--golden", str(golden),
             "--dump", str(dump), "--dump-base", "0x1000"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_original_preserves_full_raw_input_output_and_guards(self) -> None:
        golden, symbols, memory, offsets = self._fixture(False)
        input_offset = offsets["concat_gather_f16_original_input"]
        output_offset = offsets["concat_gather_f16_original_output"]
        before_offset = offsets[
            "concat_gather_f16_original_output_guard_before"
        ]
        after_offset = offsets[
            "concat_gather_f16_original_output_guard_after"
        ]
        self.assertEqual(
            memory[before_offset:before_offset + 64],
            struct.pack("<32H", *([0x5A5A] * 32)),
        )
        self.assertEqual(
            memory[after_offset:after_offset + 64],
            struct.pack("<32H", *([0x5A5A] * 32)),
        )
        nan_global = next(
            global_index
            for global_index in range(2048000)
            if ((self._input_bits(self._source_element(global_index)) &
                 0x7C00) == 0x7C00 and
                (self._input_bits(self._source_element(global_index)) &
                 0x03FF) != 0)
        )
        raw_nan = struct.unpack_from(
            "<H", memory, output_offset + nan_global * 2
        )[0]
        self.assertEqual(raw_nan & 0x7C00, 0x7C00)
        self.assertNotEqual(raw_nan & 0x03FF, 0)
        result = self._compare(golden, symbols, memory)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_alias_probe_captures_one_complete_512_element_snapshot(self) -> None:
        golden, symbols, memory, offsets = self._fixture(True)
        input_offset = offsets["concat_gather_f16_original_input"]
        output_offset = offsets["concat_gather_f16_original_output"]
        before_offset = offsets[
            "concat_gather_f16_original_output_guard_before"
        ]
        after_offset = offsets[
            "concat_gather_f16_original_output_guard_after"
        ]
        alias_bits = struct.unpack_from(
            "<512H", memory, input_offset + 2049 * 2
        )
        for global_index, actual in enumerate(alias_bits):
            expected = self._input_bits(self._source_element(global_index))
            self.assertEqual(actual, expected)
        self.assertEqual(
            memory[output_offset:output_offset + 512 * 2],
            struct.pack("<512H", *([0x5A5A] * 512)),
        )
        self.assertEqual(memory[before_offset:before_offset + 64],
                         struct.pack("<32H", *([0x5A5A] * 32)))
        self.assertEqual(memory[after_offset:after_offset + 64],
                         struct.pack("<32H", *([0x5A5A] * 32)))
        result = self._compare(golden, symbols, memory)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_rejects_raw_payload_guard_and_unused_tail_corruption(self) -> None:
        golden, symbols, memory, offsets = self._fixture(False)
        input_offset = offsets["concat_gather_f16_original_input"]
        output_offset = offsets["concat_gather_f16_original_output"]
        guard_offset = offsets[
            "concat_gather_f16_original_output_guard_before"
        ]
        corruptions = (
            (input_offset + (8192000 - 1) * 2,
             "concat_gather_f16_original_input"),
            (guard_offset, "concat_gather_f16_original_output_guard_before"),
            (output_offset + 0x7C01 * 2,
             "concat_gather_f16_original_output"),
        )
        for offset, symbol in corruptions:
            with self.subTest(symbol=symbol, offset=offset):
                corrupted = memory.copy()
                corrupted[offset] ^= 1
                result = self._compare(golden, symbols, corrupted)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(symbol + ": mismatch", result.stdout)

    def test_rejects_nonadjacent_output_guards(self) -> None:
        golden, symbols, memory, _ = self._fixture(False)
        altered = []
        for line in symbols.read_text().splitlines():
            if line.endswith(" concat_gather_f16_original_output"):
                fields = line.split()
                fields[0] = f"{int(fields[0], 16) + 2:016x}"
                line = " ".join(fields)
            altered.append(line)
        symbols.write_text("\n".join(altered) + "\n")
        result = self._compare(golden, symbols, memory)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("guarded segments are not adjacent", result.stdout)



class ConcatScatterReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        for case in ("concat_scatter_s32_original", "concat_scatter_s32_alias_probe",
                     "concat_scatter_f16_alias_probe"):
            subprocess.run(["python3", str(HERE / "generate_case.py"), "--case", case,
                            "--out", str(cls.root / case)], check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_original_moves_all_tensors_not_just_tensor_zero(self) -> None:
        data = (self.root / "concat_scatter_s32_original/output.bin").read_bytes()
        self.assertEqual(len(data), 128000 * 4)
        # Last output row contains row63 from tensor0 and tensor999.
        for output_index, input_index in ((126000, 126), (127999, 127999),
                                           (2, 128), (2000, 2)):
            actual = struct.unpack_from("<I", data, output_index * 4)[0]
            self.assertEqual(actual, (input_index * 104729 + 0x80003039) & MASK32)

    def test_s32_alias_needs_evolving_batch_snapshots(self) -> None:
        data = (self.root / "concat_scatter_s32_alias_probe/input.bin").read_bytes()
        self.assertEqual(len(data), 128000 * 4)
        # First batch must retain its source509 despite earlier writes to it.
        # Second batch must observe source512 overwritten by source129 in batch1.
        for destination, source in ((124516, 509), (517, 129)):
            self.assertEqual(struct.unpack_from("<I", data, destination * 4)[0],
                             (source * 104729 + 0x80003039) & MASK32)
        self.assertNotEqual(struct.unpack_from("<I", data, 517 * 4)[0],
                            (512 * 104729 + 0x80003039) & MASK32)

    def test_f16_alias_preserves_full_capacity_and_batch_order(self) -> None:
        case = self.root / "concat_scatter_f16_alias_probe"
        data = (case / "input.bin").read_bytes()
        self.assertEqual(len(data), 8192000 * 2)
        for destination, source in ((504514, 509), (512509, 3),
                                    (8191999, 8191999)):
            self.assertEqual(struct.unpack_from("<H", data, destination * 2)[0],
                             (source * 40503 + 0x35A7) & 0xFFFF)
        self.assertNotEqual(struct.unpack_from("<H", data, 512509 * 2)[0],
                            (512 * 40503 + 0x35A7) & 0xFFFF)
        output = (case / "output.bin").read_bytes()
        self.assertEqual(len(output), 2048000 * 2)
        self.assertEqual(output, b"\x5a\x5a" * 2048000)


if __name__ == "__main__":
    unittest.main()
