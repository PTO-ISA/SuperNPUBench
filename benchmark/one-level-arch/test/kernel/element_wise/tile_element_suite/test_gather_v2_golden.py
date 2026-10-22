#!/usr/bin/env python3
"""Independent raw-bit oracle for the three original gather_v2 profiles."""

import argparse
import itertools
from math import prod
from pathlib import Path
import struct
import subprocess
import unittest


PROFILES = {
    0: dict(bits=32, index_signed=False, rank=1, axis=0,
            input_shape=(19,), output_shape=(37,), input_elements=19,
            output_elements=37, index_elements=37, tile=32),
    1: dict(bits=32, index_signed=True, rank=2, axis=1,
            input_shape=(5, 11), output_shape=(5, 7), input_elements=55,
            output_elements=35, index_elements=7, tile=32),
    2: dict(bits=16, index_signed=False, rank=2, axis=0,
            input_shape=(11, 17), output_shape=(8, 17), input_elements=187,
            output_elements=136, index_elements=8, tile=64),
}


def value_sentinel(profile):
    return 0x5A5A if profile["bits"] == 16 else 0x5A5A5A5A


def input_bits(profile_id, index):
    profile = PROFILES[profile_id]
    if profile["bits"] == 16:
        return (index * 40503 + 0x35A7) & 0xFFFF
    if profile_id == 0:
        edge = (0, 0x80000000, 0x7F800000, 0xFF800000,
                0x7FC12345, 0xFFC54321, 1, 0x80000001)
        if index < len(edge):
            return edge[index]
    return (index * 104729 + 0x80003039) & 0xFFFFFFFF


def index_values(profile_id):
    profile = PROFILES[profile_id]
    if profile_id == 1:
        return [4, 5, 6, 7, 8, 2, 2]
    axis_extent = profile["input_shape"][profile["axis"]]
    values = [(index * 7 + 3) % axis_extent
              for index in range(profile["index_elements"])]
    if profile_id == 2:
        values[-1] = values[2]
    return values


def row_major_strides(shape):
    return tuple(prod(shape[dimension + 1:])
                 for dimension in range(len(shape)))


def logical_coordinates(shape):
    return list(itertools.product(*(range(extent) for extent in shape)))


def strided_index(coordinates, strides):
    return sum(coordinate * stride
               for coordinate, stride in zip(coordinates, strides))


def capacities(profile):
    index_padding = 128 // 4
    return (profile["input_elements"] + profile["tile"],
            profile["index_elements"] + index_padding,
            profile["output_elements"] + profile["tile"])


def initial_buffers(profile_id):
    profile = PROFILES[profile_id]
    input_count, index_count, output_count = capacities(profile)
    source = [value_sentinel(profile)] * input_count
    indices = [0x5A5A5A5A] * index_count
    output = [value_sentinel(profile)] * output_count
    alias = [value_sentinel(profile)] * input_count
    for position in range(profile["input_elements"]):
        bits = input_bits(profile_id, position)
        source[position] = bits
        alias[position] = bits
    selected = index_values(profile_id)
    indices[:len(selected)] = selected
    return source, indices, output, alias


def input_offsets(profile_id):
    profile = PROFILES[profile_id]
    selected = index_values(profile_id)
    input_strides = row_major_strides(profile["input_shape"])
    result = []
    for coordinates in logical_coordinates(profile["output_shape"]):
        source_coordinates = list(coordinates)
        source_coordinates[profile["axis"]] = selected[coordinates[profile["axis"]]]
        result.append(strided_index(source_coordinates, input_strides))
    return result


def run_oracle(profile_id, alias=False, tile_order=None, snapshot=True):
    profile = PROFILES[profile_id]
    source, indices, output, alias_buffer = initial_buffers(profile_id)
    offsets = input_offsets(profile_id)
    if alias:
        if profile_id != 1:
            raise ValueError("the race-free alias probe is profile 1")
        source = alias_buffer
        output = alias_buffer
    tile_begins = list(range(0, profile["output_elements"], profile["tile"]))
    if tile_order is not None:
        tile_begins = list(tile_order)
    for begin in tile_begins:
        end = min(begin + profile["tile"], profile["output_elements"])
        values = [source[offsets[linear]] for linear in range(begin, end)] \
            if snapshot else None
        for ordinal, linear in enumerate(range(begin, end)):
            value = values[ordinal] if snapshot else source[offsets[linear]]
            output[linear] = value
    return source, indices, output, alias_buffer


def pack_values(profile, values):
    code = "H" if profile["bits"] == 16 else "I"
    return struct.pack(f"<{len(values)}{code}", *values)


def expected_symbols(profile_id, alias):
    profile = PROFILES[profile_id]
    source, indices, output, alias_buffer = run_oracle(profile_id, alias=alias)
    initial_source, initial_indices, initial_output, initial_alias = \
        initial_buffers(profile_id)
    value_width = profile["bits"] // 8
    value_before = pack_values(
        profile, [value_sentinel(profile)] * (4096 // value_width))
    value_after = pack_values(
        profile, [value_sentinel(profile)] * (64 // value_width))
    index_before = struct.pack("<1024I", *([0x5A5A5A5A] * 1024))
    index_after = struct.pack("<16I", *([0x5A5A5A5A] * 16))
    input_count, index_count, output_count = capacities(profile)
    input_shape = (*profile["input_shape"], 1)[:2]
    output_shape = (*profile["output_shape"], 1)[:2]
    status = (profile_id, profile["rank"], profile["axis"],
              profile["input_elements"], profile["output_elements"],
              profile["index_elements"], profile["tile"],
              input_shape[0], input_shape[1], output_shape[0], output_shape[1],
              input_count, index_count, output_count, int(alias), value_width,
              4, 4, 0x47563231, 0)
    return {
        "gather_v2_input_guard_before": value_before,
        "gather_v2_input": pack_values(profile, initial_source),
        "gather_v2_input_guard_after": value_after,
        "gather_v2_index_guard_before": index_before,
        "gather_v2_index": struct.pack(f"<{len(initial_indices)}I", *initial_indices),
        "gather_v2_index_guard_after": index_after,
        "gather_v2_output_guard_before": value_before,
        "gather_v2_output": pack_values(profile, initial_output if alias else output),
        "gather_v2_output_guard_after": value_after,
        "gather_v2_alias_guard_before": value_before,
        "gather_v2_alias": pack_values(profile, alias_buffer if alias else initial_alias),
        "gather_v2_alias_guard_after": value_after,
        "gather_v2_seen": struct.pack("<4I", *(0x47560000 | pe for pe in range(4))),
        "gather_v2_status": struct.pack("<20I", *status),
    }


def read_symbols(llvm_objdump, elf):
    completed = subprocess.run(
        [str(llvm_objdump), "-t", str(elf)], check=True,
        text=True, capture_output=True)
    symbols = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "g" and fields[2] == "O":
            try:
                symbols[fields[5]] = (int(fields[0], 16), int(fields[4], 16))
            except ValueError:
                pass
    return symbols


def validate_guard_adjacency(symbols):
    for stem in ("input", "index", "output", "alias"):
        before = symbols[f"gather_v2_{stem}_guard_before"]
        body = symbols[f"gather_v2_{stem}"]
        after = symbols[f"gather_v2_{stem}_guard_after"]
        if before[0] + before[1] != body[0]:
            raise AssertionError(f"{stem}: before guard is not adjacent to body")
        if body[0] + body[1] != after[0]:
            raise AssertionError(f"{stem}: after guard is not adjacent to body")


def selected_range(symbols, names):
    start = min(symbols[name][0] for name in names)
    end = max(symbols[name][0] + symbols[name][1] for name in names)
    return start, end


def compare_dump(llvm_objdump, elf, dump, dump_base, profile_id, alias):
    expected = expected_symbols(profile_id, alias)
    symbols = read_symbols(llvm_objdump, elf)
    missing = sorted(set(expected) - set(symbols))
    if missing:
        raise AssertionError(f"ELF is missing gather_v2 symbols: {missing}")
    validate_guard_adjacency(symbols)
    memory = Path(dump).read_bytes()
    for name, golden in expected.items():
        address, size = symbols[name]
        if size != len(golden):
            raise AssertionError(
                f"{name}: ELF size {size} != golden size {len(golden)}")
        offset = address - dump_base
        observed = memory[offset:offset + size]
        if len(observed) != size:
            raise AssertionError(
                f"{name}: dump has {len(observed)} of {size} required bytes")
        if observed != golden:
            mismatch = next(i for i, pair in enumerate(zip(observed, golden))
                            if pair[0] != pair[1])
            raise AssertionError(
                f"{name}: first mismatch at byte {mismatch}: "
                f"0x{observed[mismatch]:02x} != 0x{golden[mismatch]:02x}")


class GatherV2GoldenTest(unittest.TestCase):
    def test_exact_original_capacities_repeated_indices_and_tails(self):
        for profile_id, profile in PROFILES.items():
            with self.subTest(profile=profile_id):
                source, indices, output, _ = initial_buffers(profile_id)
                counts = capacities(profile)
                self.assertEqual((len(source), len(indices), len(output)), counts)
                self.assertEqual(len(logical_coordinates(profile["output_shape"])),
                                 profile["output_elements"])
                self.assertLess(len(set(index_values(profile_id))),
                                profile["index_elements"])
                expected = run_oracle(profile_id)[2]
                self.assertTrue(all(value == value_sentinel(profile)
                                    for value in expected[profile["output_elements"]:]))
                self.assertEqual(profile["output_elements"] % profile["tile"],
                                 (5, 3, 8)[profile_id])

    def test_alias_read_write_sets_are_cross_tile_disjoint(self):
        profile = PROFILES[1]
        offsets = input_offsets(1)
        first_reads = set(offsets[:32])
        second_reads = set(offsets[32:35])
        first_writes = set(range(0, 32))
        second_writes = set(range(32, 35))
        self.assertEqual(first_reads,
                         {2} | set(range(4, 9)) |
                         {13} | set(range(15, 20)) |
                         {24} | set(range(26, 31)) |
                         {35} | set(range(37, 42)) |
                         set(range(48, 52)))
        self.assertEqual(second_reads, {46, 52})
        self.assertTrue(first_reads.isdisjoint(second_writes))
        self.assertTrue(second_reads.isdisjoint(first_writes))

    def test_alias_requires_per_tile_snapshot_and_is_pe_order_independent(self):
        begins = (0, 32)
        expected = run_oracle(1, alias=True)[3]
        for order in itertools.permutations(begins):
            self.assertEqual(run_oracle(1, alias=True, tile_order=order)[3],
                             expected)
        sequential = run_oracle(1, alias=True, snapshot=False)[3]
        self.assertNotEqual(expected, sequential)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm-objdump", type=Path)
    parser.add_argument("--elf", type=Path)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--dump-base", type=lambda value: int(value, 0))
    parser.add_argument("--profile", type=int, choices=PROFILES)
    parser.add_argument("--alias", action="store_true")
    parser.add_argument("--print-range", action="store_true")
    args, remaining = parser.parse_known_args()
    requested = (args.llvm_objdump, args.elf, args.profile)
    if any(value is not None for value in requested):
        if any(value is None for value in requested):
            parser.error("--llvm-objdump, --elf, and --profile are required together")
        expected = expected_symbols(args.profile, args.alias)
        symbols = read_symbols(args.llvm_objdump, args.elf)
        missing = sorted(set(expected) - set(symbols))
        if missing:
            parser.error(f"ELF is missing symbols: {missing}")
        validate_guard_adjacency(symbols)
        if args.print_range:
            start, end = selected_range(symbols, expected)
            print(f"0x{start:x}:0x{end - start:x}")
            return
        if args.dump is None or args.dump_base is None:
            parser.error("comparison requires --dump and --dump-base")
        compare_dump(args.llvm_objdump, args.elf, args.dump, args.dump_base,
                     args.profile, args.alias)
        print("PASS: gather_v2 raw memory matches independent golden")
        return
    unittest.main(argv=[__file__, *remaining])


if __name__ == "__main__":
    main()
