#!/usr/bin/env python3
"""Independent raw-bit oracle for the three original view_copy profiles."""

import argparse
import itertools
from pathlib import Path
import struct
import subprocess
import unittest


PROFILES = {
    0: dict(bits=32, elements=64, tile=32, shape=(2, 4, 8),
            input_stride=(32, 1, 4), output_stride=(32, 8, 1),
            input_offset=32, output_offset=64),
    1: dict(bits=16, elements=128, tile=64, shape=(4, 4, 8),
            input_stride=(32, 8, 1), output_stride=(32, 1, 4),
            input_offset=64, output_offset=128),
    2: dict(bits=16, elements=96, tile=64, shape=(4, 4, 6),
            input_stride=(24, 1, 4), output_stride=(24, 6, 1),
            input_offset=64, output_offset=96),
}


def sentinel(profile):
    return 0x5A5A if profile["bits"] == 16 else 0x5A5A5A5A


def fixture_bits(profile, index):
    if profile["bits"] == 16:
        return (index * 40503 + 0x35A7) & 0xFFFF
    return (index * 104729 + 0x80003039) & 0xFFFFFFFF


def logical_coordinates(profile):
    return list(itertools.product(*(range(extent)
                                    for extent in profile["shape"])))


def strided_index(coordinates, strides):
    return sum(coordinate * stride
               for coordinate, stride in zip(coordinates, strides))


def initial_buffers(profile_id):
    profile = PROFILES[profile_id]
    width = profile["bits"] // 8
    input_prefix = profile["input_offset"] // width
    output_prefix = profile["output_offset"] // width
    source = [sentinel(profile)] * (input_prefix + profile["elements"])
    output = [sentinel(profile)] * (output_prefix + profile["elements"])
    for index in range(profile["elements"]):
        source[input_prefix + index] = fixture_bits(profile, index)
    alias = [sentinel(profile)] * (output_prefix + profile["elements"])
    for index in range(profile["elements"]):
        alias[output_prefix + index] = fixture_bits(profile, index)
    return source, output, alias


def run_oracle(profile_id, alias=False, tile_order=None, snapshot=True):
    profile = PROFILES[profile_id]
    width = profile["bits"] // 8
    input_prefix = profile["input_offset"] // width
    output_prefix = profile["output_offset"] // width
    source, output, alias_buffer = initial_buffers(profile_id)
    if alias:
        if profile_id != 0:
            raise ValueError("the deterministic alias probe is profile 0")
        source = alias_buffer
        output = alias_buffer
        input_prefix = output_prefix

    tile_begins = list(range(0, profile["elements"], profile["tile"]))
    if tile_order is not None:
        tile_begins = list(tile_order)
    coordinates = logical_coordinates(profile)
    for begin in tile_begins:
        end = min(begin + profile["tile"], profile["elements"])
        values = []
        if snapshot:
            for linear in range(begin, end):
                index = strided_index(
                    coordinates[linear], profile["input_stride"])
                values.append(source[input_prefix + index])
        for ordinal, linear in enumerate(range(begin, end)):
            input_index = strided_index(
                coordinates[linear], profile["input_stride"])
            output_index = strided_index(
                coordinates[linear], profile["output_stride"])
            value = values[ordinal] if snapshot else source[input_prefix + input_index]
            output[output_prefix + output_index] = value
    return source, output, alias_buffer


def pack_values(profile, values):
    code = "H" if profile["bits"] == 16 else "I"
    return struct.pack(f"<{len(values)}{code}", *values)


def expected_symbols(profile_id, alias):
    profile = PROFILES[profile_id]
    source, output, alias_buffer = run_oracle(profile_id, alias=alias)
    initial_source, initial_output, initial_alias = initial_buffers(profile_id)
    before_guard_count = 4096 // (profile["bits"] // 8)
    after_guard_count = 64 // (profile["bits"] // 8)
    before_guard = pack_values(
        profile, [sentinel(profile)] * before_guard_count)
    after_guard = pack_values(
        profile, [sentinel(profile)] * after_guard_count)
    status = (profile_id, profile["elements"], profile["tile"],
              *profile["shape"], *profile["input_stride"],
              *profile["output_stride"], profile["input_offset"],
              profile["output_offset"], int(alias), 0x56435031)
    return {
        "view_copy_input_guard_before": before_guard,
        "view_copy_input": pack_values(profile, initial_source),
        "view_copy_input_guard_after": after_guard,
        "view_copy_output_guard_before": before_guard,
        "view_copy_output": pack_values(profile, initial_output if alias else output),
        "view_copy_output_guard_after": after_guard,
        "view_copy_alias_guard_before": before_guard,
        "view_copy_alias": pack_values(profile, alias_buffer if alias else initial_alias),
        "view_copy_alias_guard_after": after_guard,
        "view_copy_seen": struct.pack("<4I", *(0x56430000 | pe for pe in range(4))),
        "view_copy_status": struct.pack("<16I", *status),
    }


def read_symbols(llvm_objdump, elf):
    completed = subprocess.run(
        [str(llvm_objdump), "-t", str(elf)],
        check=True, text=True, capture_output=True)
    symbols = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "g" and fields[2] == "O":
            try:
                symbols[fields[5]] = (int(fields[0], 16), int(fields[4], 16))
            except ValueError:
                pass
    return symbols


def selected_range(symbols, names):
    start = min(symbols[name][0] for name in names)
    end = max(symbols[name][0] + symbols[name][1] for name in names)
    return start, end


def validate_guard_adjacency(symbols):
    for stem in ("input", "output", "alias"):
        before = symbols[f"view_copy_{stem}_guard_before"]
        body = symbols[f"view_copy_{stem}"]
        after = symbols[f"view_copy_{stem}_guard_after"]
        if before[0] + before[1] != body[0]:
            raise AssertionError(f"{stem}: before guard is not adjacent to body")
        if body[0] + body[1] != after[0]:
            raise AssertionError(f"{stem}: after guard is not adjacent to body")


def compare_dump(llvm_objdump, elf, dump, dump_base, profile_id, alias):
    expected = expected_symbols(profile_id, alias)
    symbols = read_symbols(llvm_objdump, elf)
    missing = sorted(set(expected) - set(symbols))
    if missing:
        raise AssertionError(f"ELF is missing view_copy symbols: {missing}")
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


class ViewCopyGoldenTest(unittest.TestCase):
    def test_exact_original_profiles_and_extents(self):
        for profile_id, profile in PROFILES.items():
            with self.subTest(profile=profile_id):
                expected = expected_symbols(profile_id, False)
                width = profile["bits"] // 8
                self.assertEqual(
                    len(expected["view_copy_input"]),
                    profile["input_offset"] + width * profile["elements"])
                self.assertEqual(
                    len(expected["view_copy_output"]),
                    profile["output_offset"] + width * profile["elements"])
                self.assertNotEqual(
                    expected["view_copy_output"],
                    pack_values(profile, [sentinel(profile)] *
                                (profile["output_offset"] // width +
                                 profile["elements"])))

    def test_alias_is_per_tile_snapshot_and_pe_order_independent(self):
        profile = PROFILES[0]
        begins = list(range(0, profile["elements"], profile["tile"]))
        coordinates = logical_coordinates(profile)
        read_sets = []
        write_sets = []
        for begin in begins:
            end = begin + profile["tile"]
            read_sets.append({strided_index(coordinates[linear],
                                            profile["input_stride"])
                              for linear in range(begin, end)})
            write_sets.append({strided_index(coordinates[linear],
                                             profile["output_stride"])
                               for linear in range(begin, end)})
            self.assertEqual(read_sets[-1], set(range(begin, end)))
            self.assertEqual(write_sets[-1], set(range(begin, end)))
        self.assertTrue(read_sets[0].isdisjoint(write_sets[1]))
        self.assertTrue(read_sets[1].isdisjoint(write_sets[0]))
        expected = run_oracle(0, alias=True)[2]
        for order in itertools.permutations(begins):
            self.assertEqual(run_oracle(0, alias=True, tile_order=order)[2], expected)
        sequential = run_oracle(0, alias=True, snapshot=False)[2]
        self.assertNotEqual(expected, sequential)

    def test_tail_writes_exactly_96_logical_elements(self):
        profile = PROFILES[2]
        _, output, _ = run_oracle(2)
        prefix = profile["output_offset"] // 2
        self.assertTrue(all(value == sentinel(profile)
                            for value in output[:prefix]))
        self.assertTrue(all(value != sentinel(profile)
                            for value in output[prefix:]))
        self.assertEqual(len(output) - prefix, 96)


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
        print("PASS: view_copy raw memory matches independent golden")
        return
    unittest.main(argv=[__file__, *remaining])


if __name__ == "__main__":
    main()
