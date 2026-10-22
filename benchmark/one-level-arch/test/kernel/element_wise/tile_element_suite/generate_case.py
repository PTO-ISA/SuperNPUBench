#!/usr/bin/env python3
"""Generate independent exact goldens and observation contracts."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from array import array
from pathlib import Path

COUNT = 263
PADDED_COUNT = 384
RADIX = 256
TOPK_CASES = [
    (0, 0, 0), (0, 7, 0), (129, 0, 0), (129, 1, 0),
    (129, 129, 0), (129, 222, 0), (1, 1, 0), (31, 17, 0),
    (32, 17, 0), (33, 17, 0), (127, 37, 0), (128, 37, 0),
    (129, 37, 0), (10, 3, 2), (129, 1, 1), (129, 65, 1),
    (129, 129, 1),
]


def histogram_input() -> list[int]:
    result: list[int] = []
    for element in range(COUNT):
        value = (element * 73 + (element // 9) * 11) & 0xFF
        if (element & 3) < 2:
            value = 0x2A
        if element in (0, 128):
            value = 0
        if element in (1, 262):
            value = 255
        result.append(value)
    return result


def selected_input() -> list[int]:
    result: list[int] = []
    for element in range(COUNT):
        high = (element * 29 + 3) & 0xFF
        if element % 7 < 3:
            high = 0xA5
        if high == 0xA5 and element % 7 >= 3:
            high = 0x5A
        low = (element * 73 + (element // 9) * 11) & 0xFF
        if element % 5 < 3:
            low = 0x2A
        if element == 0:
            low = 0
        if element == 1:
            low = 255
        result.append((high << 8) | low)
    return result


def boundary_input(count: int, pattern: int) -> list[int]:
    ties = [
        0xFFFF, 0xF100, 0xF100, 0xF100, 0xF100,
        0xE999, 0xE999, 0x8000, 0x0100, 0x0000,
    ]
    result: list[int] = []
    for element in range(count):
        if pattern == 1:
            value = 0x9A7C
        elif pattern == 2:
            value = ties[element]
        else:
            value = (
                element * 40503
                + (element // 17) * 257
                + (element % 11) * 4099
            ) & 0xFFFF
            if element == 0:
                value = 0xFFFF
            if element == 1:
                value = 0
        result.append(value)
    return result


def write_u32(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}I", *values))


def write_u16_array(path: Path, values: array) -> None:
    if values.typecode != "H" or values.itemsize != 2:
        raise ValueError("u16 golden requires a two-byte unsigned-short array")
    if sys.byteorder == "little":
        with path.open("wb") as stream:
            values.tofile(stream)
        return
    little_endian = array("H", values)
    little_endian.byteswap()
    with path.open("wb") as stream:
        little_endian.tofile(stream)


def write_linked_fixture(out: Path, prefix: str, include: str,
                         buffers: list[tuple[str, int, str]]) -> None:
    assembly = ['.pushsection .data,"aw",@progbits']
    for name, alignment, filename in buffers:
        path = (out / filename).resolve()
        if '"' in str(path) or '\n' in str(path):
            raise ValueError('unsupported linked fixture path')
        symbol = prefix + name
        assembly += [f'.balign {alignment}', f'.global {symbol}',
                     f'.type {symbol},@object', symbol + ':',
                     f'.incbin "{path}"', f'.size {symbol},.-{symbol}']
    assembly += ['.popsection']
    (out / include).write_text('asm(R"pto(\n' + '\n'.join(assembly) + '\n)pto");\n')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=(
        "broadcast_inline_s32_original", "broadcast_inline_s32_alias_probe",
        'gather_v2_f32_0_original', 'gather_v2_s32_1_original', 'gather_v2_s32_1_alias_probe', 'gather_v2_f16_2_original', 'view_copy_s32_0_original', 'view_copy_s32_0_alias_probe', 'view_copy_f16_1_original', 'view_copy_f16_2_tail', 'gather_f32_3_index_alias_probe', 'gather_f32_0_original', 'gather_f32_0_alias_probe', 'gather_f32_1_original', 'gather_f32_1_alias_probe', 'gather_f32_2_original', 'gather_f32_2_alias_probe', 'gather_f32_3_original', 'gather_f32_3_alias_probe',
        'transpose_f16_0_original', 'transpose_f16_0_alias_probe', 'transpose_f16_1_original', 'transpose_f16_1_alias_probe', 'transpose_f16_2_original', 'transpose_f16_2_alias_probe', 'transpose_f16_3_original', 'transpose_f16_3_alias_probe',
        'broadcast_f16_2d_original', 'broadcast_f16_2d_alias_probe', 'broadcast_f16_3d_original', 'broadcast_f16_3d_alias_probe', 'broadcast_f16_5d_original', 'broadcast_f16_5d_alias_probe',
        "histogram_tile_element", "selected_radix_tile_element",
        "element_expression_chain", "signed_element_expression", "indexed_gather_tile_element",
        "generic_predicated_cfg_i32", "generic_typed_tile_cfg_i32",
        "generic_atomic_cfg_i32", "elementwise_atomic_histogram", "concat_gather_s32_original",
        "concat_gather_s32_alias_probe", "concat_gather_f16_original",
        "concat_gather_f16_alias_probe",
        "broadcast_fourpe", "gather_fourpe", "concat_gather_fourpe", "concat_scatter_fourpe", "concat_scatter_s32_original", "concat_scatter_s32_alias_probe",
        "concat_scatter_f16_original", "concat_scatter_f16_alias_probe",
        "group_count_s32_original", "group_count_s32_strong",
        "elementwise_if_for", "elementwise_if_for_strong",
        "elementwise_masked_gather_scatter",
        "elementwise_masked_gather_scatter_strong",
        "topk_boundaries", "topk_boundaries_0", "topk_boundaries_1",
        "topk_boundaries_2",
    ))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.case == "broadcast_fourpe":
        edge = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f800123, 1, 0x80000001]
        source = [edge[i % 32] if i % 32 < 8 else 0x3f000000 + i * 8191 for i in range(128)]
        arrays = {"input": source, "output": source * 4,
                  "guard_before": [0x5a5a5a5a] * 896, "guard_after": [0x5a5a5a5a] * 16,
                  "seen": [0xc04e0000 | pe for pe in range(4)]}
        segments = []
        for name, values in arrays.items():
            write_u32(args.out / (name + ".bin"), values)
            segments.append({"symbol": "broadcast_fourpe_" + name, "kind": "exact", "file": name + ".bin"})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "pe_count": 4, "dtype": "raw_f32_bits",
            "input_shape": [1, 128], "output_shape": [4, 128], "tile_elements": 128,
            "path": "existing generic leading-prefix repetition copy",
            "segments": segments,
            "adjacent_segments": [{"before": "broadcast_fourpe_guard_before", "before_size": 3584,
                                   "body": "broadcast_fourpe_output", "body_size": 2048,
                                   "after": "broadcast_fourpe_guard_after", "after_size": 64}],
        }, indent=2) + "\n")
        return 0

    if args.case == "gather_fourpe":
        edge = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f800123, 1, 0x80000001]
        source = [edge[i % 32] if i % 32 < 8 else 0x3f000000 + i * 8191 for i in range(8192)]
        indices = [(i * 37 + 13) % 128 for i in range(128)]
        output = [value for row in indices for value in source[row * 64:(row + 1) * 64]]
        arrays = {"input": source, "indices": indices, "output": output,
                  "guard_before": [0x5a5a5a5a] * 896, "guard_after": [0x5a5a5a5a] * 16,
                  "seen": [0xc04e0000 | pe for pe in range(4)]}
        segments = []
        for name, values in arrays.items():
            write_u32(args.out / (name + ".bin"), values)
            segments.append({"symbol": "gather_fourpe_" + name, "kind": "exact", "file": name + ".bin"})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "pe_count": 4, "dtype": "raw_f32_bits", "index_dtype": "S32",
            "table_shape": [128, 64], "output_shape": [128, 64], "tile_shape": [32, 64],
            "segments": segments,
            "adjacent_segments": [{"before": "gather_fourpe_guard_before", "before_size": 3584,
                                   "body": "gather_fourpe_output", "body_size": 32768,
                                   "after": "gather_fourpe_guard_after", "after_size": 64}],
        }, indent=2) + "\n")
        return 0

    if args.case.startswith(("view_copy_", "gather_v2_")):
        if args.case.startswith("view_copy_"):
            from test_view_copy_golden import PROFILES, expected_symbols
            prefix, buffers = "view_copy_", ("input", "output", "alias")
        else:
            from test_gather_v2_golden import PROFILES, expected_symbols
            prefix, buffers = "gather_v2_", ("input", "index", "output", "alias")
        profile_id = int(args.case.split("_")[3])
        alias = args.case.endswith("alias_probe")
        arrays = expected_symbols(profile_id, alias)
        segments = []
        for symbol, data in arrays.items():
            filename = symbol + ".bin"
            (args.out / filename).write_bytes(data)
            segments.append({"symbol": symbol, "kind": "exact", "file": filename})
        adjacent = []
        for name in buffers:
            before = prefix + name + "_guard_before"
            after = prefix + name + "_guard_after"
            body = prefix + name
            adjacent.append({"before": before, "before_size": len(arrays[before]),
                             "body": body, "body_size": len(arrays[body]),
                             "after": after, "after_size": len(arrays[after])})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "pe_count": 4, "profile": PROFILES[profile_id],
            "alias": alias, "segments": segments, "adjacent_segments": adjacent,
        }, indent=2) + "\n")
        return 0

    if args.case.startswith("gather_f32_"):
        profile = int(args.case.split("_")[2])
        index_alias = args.case.endswith("index_alias_probe")
        alias = args.case.endswith("alias_probe") and not index_alias
        rows, selected_rows, columns, tile_rows, tile_columns = [
            (200000, 202, 97, 128, 128), (875000, 8192, 1, 128, 128),
            (754, 640, 17, 128, 128), (131072, 32, 256, 32, 64)][profile]
        source = array("I", [0]) * (rows * columns + 11)
        source[:11] = array("I", [0x5a5a5a5a]) * 11
        indices = [rows - 1 if i % 17 == 0 else (i * 7 + 3) % 64 for i in range(selected_rows)]
        edge = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f800123, 1, 0x80000001]
        for row in indices:
            for column in range(columns):
                index = row * columns + column
                source[11 + index] = ((index * 13 + 5) % rows if index_alias else
                                      (edge[index % 32] if index % 32 < 8 else (index * 104729 + 0x80003039) & 0xffffffff))
        output = array("I", [0x5a5a5a5a]) * (selected_rows * columns + 17)
        index_storage = array("I", [0x5a5a5a5a]) * (selected_rows * columns + 17 if index_alias else selected_rows + 11)
        index_storage[11:11 + selected_rows] = array("I", indices)
        for m in range(0, selected_rows, tile_rows):
            for n in range(0, columns, tile_columns):
                # Each original MxN iteration snapshots its indices and then
                # all selected values before publishing any output element.
                picked = index_storage[11 + m:11 + min(m + tile_rows, selected_rows)]
                width = min(tile_columns, columns - n)
                snapshot = [source[11 + row * columns + n:11 + row * columns + n + width]
                            for row in picked]
                destination = index_storage if index_alias else (source if alias else output)
                for row, values in enumerate(snapshot):
                    start = 17 + (m + row) * columns + n
                    destination[start:start + width] = values
        arrays = {"input": source, "indices": index_storage, "output": output}
        for name in ("input_guard_before", "input_guard_after", "output_guard_before", "output_guard_after"):
            arrays[name] = array("I", [0x5a5a5a5a] * 16)
        arrays["status"] = array("I", [rows, selected_rows, columns, tile_rows, tile_columns, 11, 11, 17, (2 if index_alias else int(alias)), profile, rows * columns, 0x47415431])
        segments = []
        for name, values in arrays.items():
            if values.itemsize != 4:
                raise ValueError("requires 32-bit array elements")
            if sys.byteorder != "little":
                values.byteswap()
            (args.out / (name + ".bin")).write_bytes(values.tobytes())
            segments.append({"symbol": "gather_" + name, "kind": "exact", "file": name + ".bin"})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "dtype": "raw_f32_bits", "index_dtype": "U32",
            "table_rows": rows, "output_rows": selected_rows, "columns": columns,
            "tile_shape": [tile_rows, tile_columns], "alias": alias, "index_output_alias": index_alias,
            "input_offset": 11, "index_offset": 11, "output_offset": 17,
            "snapshot_scope": "indices and values per original MxN tile",
            "adjacent_segments": [{"before": "gather_" + name + "_guard_before", "before_size": 64,
                                   "body": "gather_" + name, "body_size": len(arrays[name]) * 4,
                                   "after": "gather_" + name + "_guard_after", "after_size": 64}
                                  for name in ("input", "output")],
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case.startswith("transpose_f16_"):
        import itertools
        import math
        profile = int(args.case.split("_")[2])
        alias = args.case.endswith("alias_probe")
        shape, axis0, axis1 = [((1476, 32), 0, 1), ((1, 8, 4096, 3), 2, 3),
                               ((1, 32, 8, 32), 1, 2), ((8, 64, 4, 16, 7), 1, 3)][profile]
        output_shape = list(shape)
        output_shape[axis0], output_shape[axis1] = output_shape[axis1], output_shape[axis0]
        count = math.prod(shape)
        source = array("H", ((i * 40503 + 0x35a7) & 65535 for i in range(count)))
        output = array("H", [0x5a5a] * count)
        # Build the inverse permutation by visiting INPUT coordinates. Kernel
        # instead decomposes each output index to locate its source.
        source_for_output = [0] * count
        for index, coordinate in enumerate(itertools.product(*(range(n) for n in shape))):
            permuted = list(coordinate)
            permuted[axis0], permuted[axis1] = permuted[axis1], permuted[axis0]
            destination = 0
            for extent, part in zip(output_shape, permuted):
                destination = destination * extent + part
            source_for_output[destination] = index
        for begin in range(0, count, 512):
            end = min(count, begin + 512)
            snapshot = array("H", (source[source_for_output[i]] for i in range(begin, end)))
            if alias:
                source[begin:end] = snapshot
            else:
                output[begin:end] = snapshot
        arrays = {"input": source, "output": output}
        for name in ("input_guard_before", "input_guard_after", "output_guard_before", "output_guard_after"):
            arrays[name] = array("H", [0x5a5a] * 32)
        segments = []
        for name, data in arrays.items():
            write_u16_array(args.out / (name + ".bin"), data)
            segments.append({"symbol": "transpose_" + name, "kind": "exact", "file": name + ".bin"})
        for name, data in (("in_shape", list(shape) + [0] * (128 - len(shape))),
                           ("out_shape", output_shape + [0] * (128 - len(shape))),
                           ("status", [count, 512, len(shape), axis0, axis1, int(alias), profile, 0x54525031])):
            write_u32(args.out / (name + ".bin"), data)
            segments.append({"symbol": "transpose_" + name, "kind": "exact", "file": name + ".bin"})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "dtype": "raw_f16_bits", "input_shape": shape,
            "output_shape": output_shape, "tile_elements": 512, "alias": alias,
            "snapshot_scope": "each original output tile before any store",
            "adjacent_segments": [{"before": "transpose_" + name + "_guard_before", "before_size": 64,
                                   "body": "transpose_" + name, "body_size": 2 * count,
                                   "after": "transpose_" + name + "_guard_after", "after_size": 64}
                                  for name in ("input", "output")],
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case.startswith("broadcast_inline_s32_"):
        alias = args.case.endswith("alias_probe")
        count_in, count_out, tile = 64 * 49, 64 * 8 * 49, 512
        source = [(i * 104729 + 0x80003039) & 0xffffffff for i in range(count_in)]
        output = [0x5a5a5a5a] * (count_out + 17)
        if alias:
            output[11:11 + count_in] = source
        for batch in range(0, count_out, tile):
            # Independent ABA broadcast identity: repeat each49-element row8times.
            snapshot = []
            for i in range(batch, batch + tile):
                index = (i // (8 * 49)) * 49 + i % 49
                snapshot.append(output[11 + index] if alias else source[index])
            output[17 + batch:17 + batch + tile] = snapshot
        arrays = {"input": [0x5a5a5a5a] * 11 + source, "output": output,
                  "output_guard_before": [0x5a5a5a5a] * 16,
                  "output_guard_after": [0x5a5a5a5a] * 16,
                  "status": [count_in, count_out, 11, 17, tile, 3, int(alias), 0x42524331]}
        segments = []
        for name, values in arrays.items():
            write_u32(args.out / (name + ".bin"), values)
            segments.append({"symbol": "broadcast_" + name, "kind": "exact", "file": name + ".bin"})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "dtype": "raw_s32_bits", "input_shape": [64, 1, 49],
            "output_shape": [64, 8, 49], "tile_elements": tile, "alias": alias,
            "input_offset": 11, "output_offset": 17,
            "snapshot_scope": "each original output tile before any store",
            "adjacent_segments": [{"before": "broadcast_output_guard_before", "before_size": 64,
                                   "body": "broadcast_output", "body_size": 4 * (count_out + 17),
                                   "after": "broadcast_output_guard_after", "after_size": 64}],
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case.startswith("broadcast_f16_"):
        rank = int(args.case.split("_")[2][0])
        alias = args.case.endswith("alias_probe")
        input_shape, output_shape = {
            2: ((1042, 1), (1042, 129)),
            3: ((1280, 1, 49), (1280, 8, 49)),
            5: ((1, 1, 1, 65, 128), (1, 1, 7, 65, 128)),
        }[rank]
        # Cartesian-coordinate oracle, independent of kernel stride lowering.
        import itertools
        import math
        count_in, count_out = math.prod(input_shape), math.prod(output_shape)
        source = array("H", ((i * 40503 + 0x35a7) & 0xffff for i in range(count_in)))
        observed_input = array("H", [0x5a5a] * 11) + source
        output = array("H", [0x5a5a] * (count_out + 17))
        if alias:
            output[11:11 + count_in] = source
        coordinates = itertools.product(*(range(n) for n in output_shape))
        for begin in range(0, count_out, 2048):
            gathered = array("H")
            for coord in itertools.islice(coordinates, min(2048, count_out - begin)):
                source_index = 0
                for extent, index in zip(input_shape, coord):
                    source_index = source_index * extent + (0 if extent == 1 else index)
                gathered.append(output[11 + source_index] if alias else source[source_index])
            output[17 + begin:17 + begin + len(gathered)] = gathered
        arrays = {
            "input": observed_input,
            "output_guard_before": array("H", [0x5a5a] * 32),
            "output": output,
            "output_guard_after": array("H", [0x5a5a] * 32),
        }
        segments = []
        for name, values in arrays.items():
            filename = name + ".bin"
            write_u16_array(args.out / filename, values)
            segments.append({"symbol": "broadcast_" + name, "kind": "exact", "file": filename})
        write_u32(args.out / "status.bin", [count_in, count_out, 11, 17, 2048, rank, int(alias), 0x42524331])
        segments.append({"symbol": "broadcast_status", "kind": "exact", "file": "status.bin"})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "dtype": "raw_f16_bits", "input_shape": input_shape,
            "output_shape": output_shape, "tile_elements": 2048,
            "input_offset": 11, "output_offset": 17, "alias": alias,
            "snapshot_scope": "each original output tile before any store",
            "adjacent_segments": [{"before": "broadcast_output_guard_before", "before_size": 64,
                                   "body": "broadcast_output", "body_size": 2 * (count_out + 17),
                                   "after": "broadcast_output_guard_after", "after_size": 64}],
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case in ("elementwise_masked_gather_scatter",
                     "elementwise_masked_gather_scatter_strong"):
        strong = args.case.endswith("_strong")
        if strong:
            source_pattern = [
                0x00000000, 0x80000000, 0x7F800000, 0xFF800000,
                0x7FC12345, 0xFFC54321, 0x3F800000, 0xBF800000,
                0x7F800123, 0xFF800321, 0x00000001, 0x80000001,
                0x7F7FFFFF, 0xFF7FFFFF, 0x3EAAAAAB, 0xBEAAAAAB,
            ]
            active_indices = [
                -64, -63, -62, -61, -60, -59, -58, -57,
                8, 9, 10, 11, 12, 13, 14, 15,
            ]
            active_masks = [2, 3, 0x80, 0xFF]
            poison_indices = [-0x80000000, 0x7FFFFFFF,
                              -0x40000000, 0x40000000]
            source = source_pattern * 8
            indices: list[int] = []
            mask: list[int] = []
            for element in range(128):
                if (element & 3) < 2:
                    indices.append(active_indices[(element // 4) % 16])
                    mask.append(active_masks[(element // 4) % 4])
                else:
                    indices.append(poison_indices[(element // 4) % 4])
                    mask.append(0)
            sentinel = 0x5A5A6B6B
            output = [sentinel] * 128
            for index, active in zip(indices, mask):
                if active:
                    output[64 + index] = source[64 + index]
        else:
            source = [
                0x41200000, 0x41A00000, 0x41F00000, 0x42200000,
                0x42480000, 0x42700000, 0x428C0000, 0x42A00000,
            ] + [0] * 120
            indices = list(range(8)) + [0] * 120
            mask = [1, 0, 1, 0, 1, 0, 1, 0] + [0] * 120
            output = [0] * 128
            for index, active in zip(indices, mask):
                if active:
                    output[index] = source[index]

        files: dict[str, tuple[str, bytes]] = {
            "source_storage": ("source_f32_bits.bin",
                               struct.pack("<128I", *source)),
            "index_storage": ("indices_s32.bin",
                              struct.pack("<128i", *indices)),
            "mask_storage": ("mask_u8.bin", bytes(mask)),
            "gathered_storage": ("unused_gathered_f32_bits.bin",
                                 bytes(128 * 4)),
        }
        if strong:
            files["output_guard_before"] = ("output_guard_before.bin",
                                             bytes([0x5A]) * 64)
        files["output_storage"] = ("output_f32_bits.bin",
                                   struct.pack("<128I", *output))
        if strong:
            files["output_guard_after"] = ("output_guard_after.bin",
                                            bytes([0xA5]) * 64)
        segments = []
        for symbol, (filename, data) in files.items():
            (args.out / filename).write_bytes(data)
            segments.append({"symbol": symbol, "kind": "exact",
                             "file": filename})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case,
            "count": 128,
            "dtype": "raw_f32_bits",
            "center": 64 if strong else 0,
            "inactive_poison_indices": strong,
            "duplicate_active_destinations": strong,
            **({"adjacent_segments": [{
                "before": "output_guard_before", "before_size": 64,
                "body": "output_storage", "body_size": 512,
                "after": "output_guard_after", "after_size": 64,
            }]} if strong else {}),
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case in ("elementwise_if_for", "elementwise_if_for_strong"):
        strong = args.case == "elementwise_if_for_strong"
        # Independent raw-bit oracle: every operation below is exact in FP32.
        # In particular, -0 - +0 retains -0, whereas -0 - -0 yields +0.
        lhs_pattern = [
            0x00000000, 0x80000000, 0x00000000, 0x80000000,
            0x3F800000, 0xBF800000, 0x3F800000, 0xBF800000,
            0x3F000000, 0xBF000000, 0x40000000, 0xC0000000,
            0x3E800000, 0xBE800000, 0x41800000, 0xC1800000,
        ]
        rhs_pattern = [
            0x00000000, 0x00000000, 0x80000000, 0x80000000,
            0xBF800000, 0xBF800000, 0xC0000000, 0x40000000,
            0x3E800000, 0x3E800000, 0x3F800000, 0xBF800000,
            0xBE000000, 0xBE000000, 0xC1000000, 0x41000000,
        ]
        # Values: +0,-0,+0,+0,+0,+0,-1,-3,.75,-.75,3,-1,
        # .125,-.125,8,-24. No host-float tolerance or NaN policy is used.
        result_pattern = [
            0x00000000, 0x80000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0xBF800000, 0xC0400000,
            0x3F400000, 0xBF400000, 0x40400000, 0xBF800000,
            0x3E000000, 0xBE000000, 0x41000000, 0xC1C00000,
        ]
        lhs = lhs_pattern * 8 if strong else [0] * 128
        rhs = rhs_pattern * 8 if strong else [0] * 128
        result = result_pattern * 8 if strong else [0] * 128
        output = [0x5A5A5A5A] * 16 + result + [0x5A5A5A5A] * 16 if strong else result
        arrays = {"lhs_storage": ("lhs_f32_bits.bin", lhs),
                  "rhs_storage": ("rhs_f32_bits.bin", rhs),
                  "out_storage": ("output_f32_bits.bin", output)}
        segments = []
        for symbol, (filename, values) in arrays.items():
            write_u32(args.out / filename, values)
            segments.append({"symbol": symbol, "kind": "exact", "file": filename})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "count": 128, "dtype": "f32_bits",
            "output_guard_bytes": 64 if strong else 0,
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case in ("group_count_s32_original", "group_count_s32_strong"):
        input_count = 256
        used_count = 128
        output_count = 64
        bins = 32
        magic = 0x47435332
        strong = args.case == "group_count_s32_strong"

        if strong:
            state = 1
            input_values = []
            for element in range(input_count):
                if element < used_count:
                    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                    value = (state >> 16) % bins
                    if element == 0:
                        value = -1
                    elif element == 1:
                        value = 32
                    elif element == 2:
                        value = 99
                    elif element == 63:
                        value = -7
                    elif element == 127:
                        value = 0x7FFFFFFF
                else:
                    value = 0x6B6B0000 + element
                input_values.append(value & 0xFFFFFFFF)
            counts = [0] * bins
            for value in input_values[:used_count]:
                signed = value if value < 0x80000000 else value - 0x100000000
                if 0 <= signed < bins:
                    counts[signed] += 1
            ignored = used_count - sum(counts)
            output = counts + [0x5A5A6B6B] * (output_count - bins)
            alias = counts + input_values[bins:]
            status = [0, 0, 0, 0, 0, used_count, bins, 2,
                      sum(counts), ignored, sum(counts), magic]
            arrays = {
                "group_count_s32_strong_input": ("input_s32.bin", input_values),
                "group_count_s32_strong_output": ("output_s32.bin", output),
                "group_count_s32_strong_alias": ("alias_s32.bin", alias),
                "group_count_s32_strong_status": ("status_u32.bin", status),
            }
        else:
            input_values = [0] * input_count
            output = [used_count] + [0] * (output_count - 1)
            status = [0, 0, 0, 0, input_count, used_count, output_count,
                      bins, 1, used_count, 0, magic]
            arrays = {
                "group_count_s32_original_input": ("input_s32.bin", input_values),
                "group_count_s32_original_output": ("output_s32.bin", output),
                "group_count_s32_original_status": ("status_u32.bin", status),
            }

        segments = []
        for symbol, (filename, values) in arrays.items():
            write_u32(args.out / filename, values)
            segments.append({"symbol": symbol, "kind": "exact",
                             "file": filename})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case,
            "input_count": input_count,
            "used_count": used_count,
            "output_count": output_count,
            "bins": bins,
            "repeated_calls": 2 if strong else 1,
            "alias": strong,
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case in ("concat_gather_fourpe", "concat_scatter_fourpe"):
        edge = [0, 0x80000000, 0x7F800000, 0xFF800000,
                0x7FC12345, 0x7F800123, 1, 0x80000001]
        source = [edge[i % 32] if i % 32 < 8 else 0x3F000000 + i * 8191
                  for i in range(512)]
        gather = args.case == "concat_gather_fourpe"
        output = [0x5A5A5A5A] * (512 if gather else 1024)
        if gather:
            for row in range(16):
                for tensor in range(2):
                    output[row * 32 + tensor * 16:row * 32 + (tensor + 1) * 16] = source[tensor * 256 + row * 16:tensor * 256 + (row + 1) * 16]
        else:
            for row in range(32):
                output[row * 32:row * 32 + 16] = source[row * 16:row * 16 + 16]
        arrays = {"input": source, "output": output,
                  "guard_before": [0x5A5A5A5A] * 512,
                  "guard_after": [0x5A5A5A5A] * 16,
                  "seen": [0xC04E0000 | pe for pe in range(4)]}
        segments = []
        for name, values in arrays.items():
            filename = name + ".bin"
            write_u32(args.out / filename, values)
            segments.append({"symbol": args.case + "_" + name,
                             "kind": "exact", "file": filename})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "pe_count": 4, "input_count": 512,
            "output_count": len(output), "logical_batch": 128,
            "input_shape": [16 if gather else 32, 16], "output_shape": [16 if gather else 32, 32],
            "segments": segments,
            "adjacent_segments": [{
                "before": args.case + "_guard_before",
                "body": args.case + "_output",
                "after": args.case + "_guard_after",
                "before_size": 2048, "body_size": len(output) * 4, "after_size": 64,
            }],
        }, indent=2) + "\n")
        return 0

    if args.case.startswith("concat_scatter_"):
        fp16 = "_f16_" in args.case
        alias = args.case.endswith("_alias_probe")
        rows, columns, tensors = (256, 8, 1000) if fp16 else (64, 2, 1000)
        input_count = 8192000 if fp16 else 128000
        output_count = rows * columns * tensors
        processed = 1024 if alias else output_count
        alias_offset = 509
        code, bits = ("H", 16) if fp16 else ("I", 32)
        guard = 0x5A5A if fp16 else 0x5A5A5A5A
        mask = (1 << bits) - 1
        multiplier, bias = (40503, 0x35A7) if fp16 else (104729, 0x80003039)
        values = array(code, ((i * multiplier + bias) & mask
                              for i in range(input_count)))
        output = array(code, [guard]) * output_count
        if fp16:
            write_u16_array(args.out / 'input_seed.bin', values)
            write_u16_array(args.out / 'output_seed.bin', output)
        if alias:
            # Independent [N,rows,cols] -> [rows,N,cols] enumeration. Each
            # original512 batch snapshots evolving memory before its stores.
            destinations = []
            for tensor in range(tensors):
                for row in range(rows):
                    destinations.extend(range((row * tensors + tensor) * columns,
                                              (row * tensors + tensor + 1) * columns))
                    if len(destinations) >= processed:
                        break
                if len(destinations) >= processed:
                    break
            for begin in range(0, processed, 512):
                snapshot = values[begin:begin + 512]
                for lane, value in enumerate(snapshot):
                    values[alias_offset + destinations[begin + lane]] = value
        else:
            output = array(code)
            for row in range(rows):
                for tensor in range(tensors):
                    begin = (tensor * rows + row) * columns
                    output.extend(values[begin:begin + columns])
        segments = []
        for name, data in (("input", values), ("output", output),
                           ("output_guard_before", array(code, [guard]) * (64 // (bits // 8))),
                           ("output_guard_after", array(code, [guard]) * (64 // (bits // 8)))):
            filename = name + ".bin"
            if sys.byteorder != "little":
                data = array(code, data)
                data.byteswap()
            (args.out / filename).write_bytes(data.tobytes())
            segments.append({"symbol": "concat_scatter_" + name,
                             "kind": "exact", "file": filename})
        if fp16:
            write_linked_fixture(args.out, 'concat_scatter_',
                                 'concat_scatter_f16_input.inc', [
                ('input', 4096, 'input_seed.bin'),
                ('output_guard_before', 64, 'output_guard_before.bin'),
                ('output', 64, 'output_seed.bin'),
                ('output_guard_after', 64, 'output_guard_after.bin'),
                ('in_shape', 32, 'in_shape.bin'),
                ('out_shape', 32, 'out_shape.bin')])
        status = [0, input_count, output_count, processed, 512, rows, columns,
                  tensors * columns, tensors, int(alias),
                  alias_offset if alias else 0, 0x43534331]
        write_u32(args.out / "status.bin", status)
        segments.append({"symbol": "concat_scatter_status", "kind": "exact",
                         "file": "status.bin"})
        for name, shape in (("in_shape", [rows, columns] + [0] * 6),
                            ("out_shape", [rows, tensors * columns] + [0] * 6)):
            filename = name + ".bin"
            (args.out / filename).write_bytes(struct.pack("<8Q", *shape))
            segments.append({"symbol": "concat_scatter_" + name,
                             "kind": "exact", "file": filename})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "input_count": input_count,
            "output_count": output_count, "processed_count": processed,
            "logical_batch": 512, "alias_offset": alias_offset if alias else None,
            "reference": "tensor[N,rows,cols] -> tensor[rows,N,cols]; evolving512 snapshots for alias",
            "original_processing_count": output_count,
            "adjacent_segments": [{
                "before": "concat_scatter_output_guard_before",
                "body": "concat_scatter_output",
                "after": "concat_scatter_output_guard_after",
                "before_size": 64, "body_size": output_count * (bits // 8),
                "after_size": 64,
            }], "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case in ("concat_gather_s32_original", "concat_gather_s32_alias_probe"):
        input_count, rows, columns, tensors = 128000, 64, 2, 1000
        alias = args.case == "concat_gather_s32_alias_probe"
        output_count = 512 if alias else 128000
        guard = 0x5A5A6B6B
        original = [(element * 104729 + 12345) & 0x7FFFFFFF
                    for element in range(input_count)]
        # Independent reference concatenates each tensor's two-value row;
        # it does not reuse the kernel's flattened inverse-index expression.
        concatenated = []
        for row in range(rows):
            for tensor in range(tensors):
                start = tensor * rows * columns + row * columns
                concatenated.extend(original[start:start + columns])
        actual_input = original.copy()
        if alias:
            # One full512 snapshot is taken before any overlapping output.
            actual_input[2049:2049 + 512] = concatenated[:512]
            output = [guard] * (output_count + 32)
        else:
            output = [guard] * 16 + concatenated + [guard] * 16
        prefix = "concat_gather_s32_original_"
        data = {
            "input": actual_input,
            "output": output,
            "status": [0, 0, 0, 0, 0, input_count, output_count,
                       512, 64, 2, 2000, 0x43475332],
        }
        segments = []
        for name, values in data.items():
            filename = name + ".bin"
            write_u32(args.out / filename, values)
            segments.append({"symbol": prefix + name, "kind": "exact",
                             "file": filename})
        for name, shape in (("in_shape", [64, 2] + [0] * 6),
                            ("out_shape", [64, 2000] + [0] * 6)):
            filename = name + ".bin"
            (args.out / filename).write_bytes(struct.pack("<8Q", *shape))
            segments.append({"symbol": prefix + name, "kind": "exact",
                             "file": filename})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "input_count": input_count,
            "output_count": output_count, "logical_batch": 512,
            "alias_offset": 2049 if alias else None, "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case in ("concat_gather_f16_original", "concat_gather_f16_alias_probe"):
        input_count = 4000 * 256 * 8
        used_input_count = 1000 * 256 * 8
        rows, columns, tensors = 256, 8, 1000
        alias = args.case == "concat_gather_f16_alias_probe"
        output_count = 512 if alias else rows * tensors * columns
        alias_offset = 2049
        guard = 0x5A5A

        # Keep the 16 MiB input and 4 MiB output compact. The odd multiplier
        # walks every raw binary16 encoding, including signed zero, infinities,
        # and NaN payloads, without constructing millions of Python integers.
        input_values = array(
            "H",
            ((element * 40503 + 0x35A7) & 0xFFFF
             for element in range(input_count)),
        )
        # Seed the actual ELF independently of post-execution alias goldens.
        # The host oracle below still observes every input and output byte.
        write_u16_array(args.out / "input_seed_u16.bin", input_values)
        write_u16_array(args.out / "output_seed_u16.bin",
                        array("H", [guard]) * output_count)
        concatenated = array("H")
        for row in range(rows):
            for tensor in range(tensors):
                start = tensor * rows * columns + row * columns
                concatenated.extend(input_values[start:start + columns])

        if alias:
            input_values[alias_offset:alias_offset + output_count] = \
                concatenated[:output_count]
            output_values = array("H", [guard]) * output_count
        else:
            output_values = concatenated

        prefix = "concat_gather_f16_original_"
        segments = []
        for name, values in (("input", input_values),
                             ("output", output_values)):
            filename = name + "_u16.bin"
            write_u16_array(args.out / filename, values)
            segments.append({"symbol": prefix + name, "kind": "exact",
                             "file": filename})
        guard_values = array("H", [guard]) * 32
        for name in ("output_guard_before", "output_guard_after"):
            filename = name + "_u16.bin"
            write_u16_array(args.out / filename, guard_values)
            segments.append({"symbol": prefix + name, "kind": "exact",
                             "file": filename})

        status = [0, 0, 0, 0, 0, input_count, output_count, 512,
                  rows, columns, tensors * columns, tensors,
                  used_input_count, 0x43474631]
        write_u32(args.out / "status_u32.bin", status)
        segments.append({"symbol": prefix + "status", "kind": "exact",
                         "file": "status_u32.bin"})
        for name, shape in (("in_shape", [rows, columns] + [0] * 6),
                            ("out_shape", [rows, tensors * columns] + [0] * 6)):
            filename = name + ".bin"
            (args.out / filename).write_bytes(struct.pack("<8Q", *shape))
            segments.append({"symbol": prefix + name, "kind": "exact",
                             "file": filename})

        write_linked_fixture(args.out, prefix, 'concat_gather_f16_input.inc', [
            ("input", 4096, 'input_seed_u16.bin'),
            ("output_guard_before", 64, 'output_guard_before_u16.bin'),
            ("output", 64, 'output_seed_u16.bin'),
            ("output_guard_after", 64, 'output_guard_after_u16.bin'),
            ("in_shape", 32, 'in_shape.bin'),
            ("out_shape", 32, 'out_shape.bin')])

        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case,
            "input_count": input_count,
            "used_input_count": used_input_count,
            "output_count": output_count,
            "logical_batch": 512,
            "logical_part": 64,
            "alias_offset": alias_offset if alias else None,
            "raw_binary16": True,
            "input_fixture": "deterministic raw binary16; legacy driver had uninitialized buffers and no enabled file fixture",
            "adjacent_segments": [{
                "before": prefix + "output_guard_before",
                "body": prefix + "output",
                "after": prefix + "output_guard_after",
                "before_size": 64, "body_size": output_count * 2,
                "after_size": 64,
            }],
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case == "elementwise_atomic_histogram":
        # Original smoke:128 signed values of1, eight repeated bins,
        # byte masks all1. All atomic parts complete before readonly gathers.
        prefix = "elementwise_atomic_histogram_"
        data = {
            "values": [1] * 128,
            "indices": [element % 8 for element in range(128)],
            "bins": [16] * 8,
            "observed": [16] * 128,
            "status": [128, 8, 0, 128, 2048, 128, 4, 0x41544853],
        }
        segments = []
        for name, values in data.items():
            filename = name + ".bin"
            write_u32(args.out / filename, values)
            segments.append({"symbol": prefix + name, "kind": "exact",
                             "file": filename})
        (args.out / "active.bin").write_bytes(bytes([1]) * 128)
        segments.append({"symbol": prefix + "active", "kind": "exact",
                         "file": "active.bin"})
        (args.out / "manifest.json").write_text(json.dumps({
            "case": args.case, "count": 128, "bins": 8,
            "segments": segments,
        }, indent=2) + "\n")
        return 0

    if args.case == "generic_atomic_cfg_i32":
        elements = 128
        valid_elements = 125
        bins = 32
        center = 16
        mask32 = 0xFFFFFFFF
        poison_seed = 0x6C8E9CF5

        indices: list[int] = []
        deltas: list[int] = []
        first_adds: list[int] = []
        second_adds: list[int] = []
        for element in range(elements):
            if element >= valid_elements:
                indices.append(-0x80000000)
                deltas.append(-0x80000000)
                first_adds.append(0)
                second_adds.append(0)
                continue
            index = (element * 5 + 3) % bins - center
            delta = (element * 7 + 2) % 17 - 8
            indices.append(index)
            deltas.append(delta)
            if index < 0:
                if delta < 0:
                    first_add, second_add = delta & mask32, 5
                else:
                    first_add, second_add = (delta + 2) & mask32, (-3) & mask32
            elif index & 1 == 0:
                first_add, second_add = (delta - 4) & mask32, 9
            else:
                first_add, second_add = (delta + 6) & mask32, (-7) & mask32
            first_adds.append(first_add)
            second_adds.append(second_add)

        initial_histogram = [
            (0x10203040 + bin_index * 0x101) & mask32
            for bin_index in range(bins)
        ]
        histogram = initial_histogram.copy()
        for element in range(valid_elements):
            bin_index = indices[element] + center
            histogram[bin_index] = (
                histogram[bin_index] + first_adds[element]
                + second_adds[element]
            ) & mask32
        poison = [poison_seed + 1] * elements
        status = [
            elements, valid_elements, 0, sum(histogram) & mask32,
            sum(poison) & mask32, bins, 2, 0x41544D43,
        ]

        signed_files = {
            "indices_s32.bin": indices,
            "deltas_s32.bin": deltas,
        }
        for filename, values in signed_files.items():
            (args.out / filename).write_bytes(
                struct.pack(f"<{len(values)}i", *values)
            )
        unsigned_files = {
            "gates_u32.bin": [0] * elements,
            "initial_histogram_u32.bin": initial_histogram,
            "histogram_u32.bin": histogram,
            "first_adds_u32.bin": first_adds,
            "second_adds_u32.bin": second_adds,
            "poison_u32.bin": poison,
            "status_u32.bin": status,
        }
        for filename, values in unsigned_files.items():
            write_u32(args.out / filename, values)

        manifest = {
            "case": args.case,
            "count": elements,
            "valid_count": valid_elements,
            "center": center,
            "segments": [
                {"symbol": "generic_atomic_cfg_i32_indices", "kind": "exact",
                 "file": "indices_s32.bin"},
                {"symbol": "generic_atomic_cfg_i32_deltas", "kind": "exact",
                 "file": "deltas_s32.bin"},
                {"symbol": "generic_atomic_cfg_i32_gates", "kind": "exact",
                 "file": "gates_u32.bin"},
                {"symbol": "generic_atomic_cfg_i32_histogram", "kind": "exact",
                 "file": "histogram_u32.bin"},
                {"symbol": "generic_atomic_cfg_i32_poison", "kind": "exact",
                 "file": "poison_u32.bin"},
                {"symbol": "generic_atomic_cfg_i32_status", "kind": "exact",
                 "file": "status_u32.bin"},
                {
                    "symbol": "generic_atomic_cfg_i32_first",
                    "kind": "atomic_pair_trace_mod32",
                    "second_symbol": "generic_atomic_cfg_i32_second",
                    "combined_symbol": "generic_atomic_cfg_i32_combined",
                    "index_symbol": "generic_atomic_cfg_i32_indices",
                    "histogram_symbol": "generic_atomic_cfg_i32_histogram",
                    "initial_histogram_file": "initial_histogram_u32.bin",
                    "first_add_file": "first_adds_u32.bin",
                    "second_add_file": "second_adds_u32.bin",
                    "first_post_add": 11,
                    "second_post_add": 13,
                    "combined_post_add": 17,
                    "combined_bias": 0x13579BDF,
                },
            ],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    if args.case == "generic_typed_tile_cfg_i32":
        element_count = 263
        padded_count = 384
        guard_elements = 16
        guard_value = 0x5A5A6B6B
        mask32 = 0xFFFFFFFF

        def signed32(value: int) -> int:
            value &= mask32
            return value if value < 0x80000000 else value - 0x100000000

        def trunc_div(dividend: int, divisor: int) -> int:
            quotient = abs(dividend) // abs(divisor)
            return -quotient if dividend < 0 else quotient

        input_values = []
        for element in range(padded_count):
            value = 0x1234567
            if element < element_count:
                value = (element * 7919 + 12345) % 200001 - 100000
            input_values.append(value)
        input_values[0:4] = [-3, -4, 0, 100]

        output = []
        retained = 0
        for element in range(padded_count):
            source = input_values[element] if element < element_count else 0
            prepared = source + 3
            if prepared < 0:
                selected = trunc_div(prepared, 3)
            elif prepared == 0:
                selected = 7
            else:
                selected = signed32(prepared ^ 5)
            result = selected + 11
            assert result != 0
            if result < -100:
                result = signed32(result ^ 0x55)
            elif result > 100:
                result -= 4
            else:
                retained += 1
            output.append(signed32(result))

        guarded_output = ([guard_value] * guard_elements + output +
                          [guard_value] * guard_elements)
        checksum = sum(value & mask32 for value in output) & mask32
        status = [
            element_count, padded_count, 0, 0, checksum,
            output[0] & mask32, output[1] & mask32, output[2] & mask32,
            retained, 2, 0, 0x54594346,
        ]
        (args.out / "input_s32.bin").write_bytes(
            struct.pack(f"<{len(input_values)}i", *input_values)
        )
        (args.out / "output_s32.bin").write_bytes(
            struct.pack(f"<{len(guarded_output)}i", *guarded_output)
        )
        write_u32(args.out / "status_u32.bin", status)
        manifest = {
            "case": args.case,
            "count": element_count,
            "padded_count": padded_count,
            "segments": [
                {"symbol": "generic_typed_tile_cfg_i32_input",
                 "kind": "exact", "file": "input_s32.bin"},
                {"symbol": "generic_typed_tile_cfg_i32_output",
                 "kind": "exact", "file": "output_s32.bin"},
                {"symbol": "generic_typed_tile_cfg_i32_status",
                 "kind": "exact", "file": "status_u32.bin"},
            ],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    if args.case == "generic_predicated_cfg_i32":
        elements = 32
        valid_elements = 29
        guard_elements = 8
        guard_value = 0x5A5A6B6B
        mask32 = 0xFFFFFFFF

        def signed32(value: int) -> int:
            value &= mask32
            return value if value < 0x80000000 else value - 0x100000000

        def trunc_div(dividend: int, divisor: int) -> int:
            quotient = abs(dividend) // abs(divisor)
            return -quotient if (dividend < 0) != (divisor < 0) else quotient

        input_values = [element * 7919 + 12345 - 131071
                        for element in range(elements)]
        input_values[0] = -100
        input_values[3] = 100
        control = [element % 3 - 1 for element in range(elements)]
        divisor = [
            (-5 if element & 1 else 3) if control[element] < 0 else 0
            for element in range(elements)
        ]
        output = []
        for element in range(elements):
            if element >= valid_elements:
                output.append(guard_value)
                continue
            value = input_values[element]
            selector = control[element]
            if selector < 0:
                result = trunc_div(value + 21, divisor[element])
            elif selector == 0:
                result = value * 3 - 17
            else:
                result = signed32((value & mask32) ^ 0x13579BDF) + 9
            output.append(signed32(result))
        guarded_output = ([guard_value] * guard_elements + output +
                          [guard_value] * guard_elements)
        direct = []
        for element in range(elements):
            if element < valid_elements and control[element] < 0:
                direct.append(trunc_div(
                    input_values[element] + 21, divisor[element]
                ))
            else:
                direct.append(guard_value)
        guarded_direct = ([guard_value] * guard_elements + direct +
                          [guard_value] * guard_elements)
        checksum = sum(value & mask32 for value in output) & mask32
        direct_checksum = sum(value & mask32 for value in direct) & mask32
        status = [
            elements, valid_elements, 0, 0, checksum, checksum,
            direct_checksum, output[0] & mask32, output[3] & mask32,
            direct[0] & mask32, 3, 3,
            sum(1 for element in range(valid_elements)
                if control[element] < 0),
            0, 0, 0x47434647,
        ]
        signed_files = {
            "generic_predicated_cfg_i32_input": ("input_s32.bin", input_values),
            "generic_predicated_cfg_i32_control": ("control_s32.bin", control),
            "generic_predicated_cfg_i32_divisor": ("divisor_s32.bin", divisor),
            "generic_predicated_cfg_i32_nested": (
                "nested_s32.bin", guarded_output
            ),
            "generic_predicated_cfg_i32_reversed": (
                "reversed_s32.bin", guarded_output
            ),
            "generic_predicated_cfg_i32_direct": (
                "direct_s32.bin", guarded_direct
            ),
        }
        for filename, values in signed_files.values():
            (args.out / filename).write_bytes(
                struct.pack(f"<{len(values)}i", *values)
            )
        write_u32(args.out / "status_u32.bin", status)
        manifest = {
            "case": args.case,
            "count": elements,
            "valid_count": valid_elements,
            "segments": [
                {"symbol": symbol, "kind": "exact", "file": filename}
                for symbol, (filename, _values) in signed_files.items()
            ] + [{
                "symbol": "generic_predicated_cfg_i32_status",
                "kind": "exact",
                "file": "status_u32.bin",
            }],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    if args.case == "signed_element_expression":
        active = [(element * 7919 + 12345) % 65535 - 32767
                  for element in range(COUNT)]
        active[0:2] = [-7, 7]
        input_values = active + [-559038737] * (PADDED_COUNT - COUNT)
        loaded = active + [0] * (PADDED_COUNT - COUNT)
        guard = [-1234567] * 16
        # C++ truncates division toward zero. The remainder keeps the
        # dividend sign; Python's signed % would implement the wrong rule.
        quotient = []
        remainder = []
        for value in loaded:
            numerator = (value + 19) * 3 - 7
            quotient.append((1 if numerator >= 0 else -1) * (abs(numerator) // 3))
            remainder.append((1 if value >= 0 else -1) * (abs(value) % 3))
        assert remainder[0:2] == [-1, 1]
        shifted = [-(((value >> 2) ^ 0x555) & 0xFF) for value in loaded]
        arrays = {
            "signed_element_input": ("input_s32.bin", input_values),
            "signed_element_quotient": ("quotient_s32.bin", guard + quotient + guard),
            "signed_element_remainder": ("remainder_s32.bin", guard + remainder + guard),
            "signed_element_shift": ("shift_s32.bin", guard + shifted + guard),
        }
        for filename, values in arrays.values():
            (args.out / filename).write_bytes(struct.pack(f"<{len(values)}i", *values))
        status = [COUNT, 0, 0, PADDED_COUNT, 0xFFFFFFFF, 1, 3, 0x53333245]
        write_u32(args.out / "status_u32.bin", status)
        manifest = {"case": args.case, "count": COUNT,
                    "padded_count": PADDED_COUNT, "segments": [
            {"symbol": symbol, "kind": "exact", "file": filename}
            for symbol, (filename, _) in arrays.items()]
                    + [{"symbol": "signed_element_status", "kind": "exact",
                        "file": "status_u32.bin"}]}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return 0

    if args.case == "indexed_gather_tile_element":
        mask32 = 0xFFFFFFFF
        table = [
            0x80000000 | ((element * 0x01020305 + 0x10203040) & 0x7FFFFFFF)
            for element in range(257)
        ]
        indices = []
        for element in range(PADDED_COUNT):
            if element >= COUNT:
                indices.append(mask32)
                continue
            index = (element * 73 + (element // 5) * 19) % 257
            if element % 11 < 4:
                index = 42
            if element == 0:
                index = 0
            if element == COUNT - 1:
                index = 256
            indices.append(index)

        gathered = [
            (table[indices[element]] + 3) & mask32
            if element < COUNT else 3
            for element in range(PADDED_COUNT)
        ]
        guard = [0x6A09E667] * 16
        output = guard + gathered + guard
        status = [
            COUNT,
            0,
            sum(gathered[:COUNT]) & mask32,
            sum(gathered) & mask32,
            0,
            len(table),
            PADDED_COUNT,
            0x47415448,
            257,
            0,
            0,
            0,
        ]
        files = {
            "indexed_gather_tile_element_table": ("table_u32.bin", table),
            "indexed_gather_tile_element_indices": (
                "indices_u32.bin", indices
            ),
            "indexed_gather_tile_element_output": ("output_u32.bin", output),
            "indexed_gather_tile_element_status": ("status_u32.bin", status),
        }
        for filename, values in files.values():
            write_u32(args.out / filename, values)
        manifest = {
            "case": args.case,
            "count": COUNT,
            "padded_count": PADDED_COUNT,
            "segments": [
                {"symbol": symbol, "kind": "exact", "file": filename}
                for symbol, (filename, _values) in files.items()
            ],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    if args.case == "element_expression_chain":
        mask32 = 0xFFFFFFFF
        active = []
        for element in range(COUNT):
            value = (element * 2654435761 + 0x01234567) & mask32
            value ^= ((element // 7) * 0x00010101) & mask32
            active.append(value)
        input_values = active + [0xDEADBEEF] * (PADDED_COUNT - COUNT)
        loaded_values = active + [0] * (PADDED_COUNT - COUNT)

        arithmetic = []
        bitwise = []
        unary = []
        for value in loaded_values:
            biased = (value + 19) & mask32
            multiplied = (biased * 13) & mask32
            reduced = (multiplied - 7) & mask32
            divided = reduced // 5
            arithmetic.append(divided % 251)

            shifted_left = (value << 3) & mask32
            shifted_right = shifted_left >> 2
            masked = shifted_right & 0xFFFFFFF0
            merged = masked | 0x00000105
            mixed = (merged ^ 0x0055AA11) & mask32
            bitwise.append((mixed ^ shifted_left) & mask32)

            negated = (-value) & mask32
            unary.append((~negated) & mask32)

        status = [
            COUNT,
            0,
            sum(arithmetic) & mask32,
            sum(bitwise) & mask32,
            sum(unary) & mask32,
            PADDED_COUNT,
            10,
            0x45585052,
        ]
        files = {
            "element_expression_chain_input": ("input_u32.bin", input_values),
            "element_expression_chain_arithmetic": (
                "arithmetic_u32.bin", arithmetic
            ),
            "element_expression_chain_bitwise": ("bitwise_u32.bin", bitwise),
            "element_expression_chain_unary": ("unary_u32.bin", unary),
            "element_expression_chain_status": ("status_u32.bin", status),
        }
        for filename, values in files.values():
            write_u32(args.out / filename, values)
        manifest = {
            "case": args.case,
            "count": COUNT,
            "padded_count": PADDED_COUNT,
            "segments": [
                {"symbol": symbol, "kind": "exact", "file": filename}
                for symbol, (filename, _values) in files.items()
            ],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    if args.case == "histogram_tile_element":
        active = histogram_input()
        selected_high = None
        addend = 1
        input_symbol = "histogram_tile_element_input"
        histogram_symbol = "histogram_tile_element_histogram"
        old_symbol = "histogram_tile_element_old_plus_one"
        status_symbol = "histogram_tile_element_status"
        marker = 0x48495354
    elif args.case == "selected_radix_tile_element":
        active = selected_input()
        selected_high = 0xA5
        addend = 7
        input_symbol = "selected_radix_tile_element_input"
        histogram_symbol = "selected_radix_tile_element_histogram"
        old_symbol = "selected_radix_tile_element_old_plus_seven"
        status_symbol = "selected_radix_tile_element_status"
        marker = addend

    else:
        if args.case == "topk_boundaries":
            first, last = 0, len(TOPK_CASES)
        else:
            batch = int(args.case.rsplit("_", 1)[1])
            first, last = ((0, 6), (6, 13), (13, 17))[batch]
        count, requested_k, pattern = TOPK_CASES[last - 1]
        active = boundary_input(count, pattern)
        actual_k = min(count, requested_k)
        sorted_values = sorted(active, reverse=True)
        high_histogram = [0] * RADIX
        low_histogram = [0] * RADIX
        for value in active:
            high_histogram[value >> 8] += 1
        selected_high = None
        if actual_k:
            selected_high = sorted_values[actual_k - 1] >> 8
            for value in active:
                if value >> 8 == selected_high:
                    low_histogram[value & 0xFF] += 1
        sentinel = 0xDEADBEEF
        files = {
            "topk_boundaries_input": "input_u32.bin",
            "topk_boundaries_high_hist": "high_hist_u32.bin",
            "topk_boundaries_low_hist": "low_hist_u32.bin",
            "topk_boundaries_output": "output_u32.bin",
        }
        write_u32(
            args.out / files["topk_boundaries_input"],
            active + [sentinel] * (129 - count),
        )
        write_u32(args.out / files["topk_boundaries_high_hist"], high_histogram)
        write_u32(args.out / files["topk_boundaries_low_hist"], low_histogram)
        write_u32(
            args.out / files["topk_boundaries_output"],
            sorted_values[:actual_k] + [sentinel] * (133 - actual_k),
        )
        manifest = {
            "case": args.case,
            "count": count,
            "padded_count": ((count + 127) // 128) * 128,
            "segments": [
                {"symbol": symbol, "kind": "exact", "file": filename}
                for symbol, filename in files.items()
            ] + [
                {
                    "symbol": "topk_boundaries_status",
                    "kind": "zero_status_range",
                    "first": first,
                    "last": last,
                },
                {
                    "symbol": "topk_boundaries_old_high",
                    "kind": "old_value_permutation",
                    "input_symbol": "topk_boundaries_input",
                    "histogram_symbol": "topk_boundaries_high_hist",
                    "bucket_shift": 8,
                    "selected_high": None,
                    "addend": 0,
                },
                {
                    "symbol": "topk_boundaries_old_low",
                    "kind": "old_value_permutation",
                    "input_symbol": "topk_boundaries_input",
                    "histogram_symbol": "topk_boundaries_low_hist",
                    "bucket_shift": 0,
                    "selected_high": 256 if selected_high is None
                    else selected_high,
                    "addend": 0,
                },
            ],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    histogram = [0] * RADIX
    selected_count = 0
    for value in active:
        if selected_high is None or value >> 8 == selected_high:
            histogram[value & 0xFF] += 1
            selected_count += 1

    padded = active + [0xFFFFFFFF] * (PADDED_COUNT - COUNT)
    checksum = sum(bin_index * count for bin_index, count in enumerate(histogram))
    status = [
        COUNT if selected_high is None else selected_count,
        0,
        selected_count,
        histogram[0],
        histogram[255],
        checksum,
        addend if selected_high is None else selected_high,
        marker,
    ]

    input_file = "input_u32.bin"
    histogram_file = "histogram_u32.bin"
    status_file = "status_u32.bin"
    write_u32(args.out / input_file, padded)
    write_u32(args.out / histogram_file, histogram)
    write_u32(args.out / status_file, status)

    manifest = {
        "case": args.case,
        "count": COUNT,
        "padded_count": PADDED_COUNT,
        "selected_high": selected_high,
        "old_addend": addend,
        "segments": [
            {"symbol": input_symbol, "kind": "exact", "file": input_file},
            {
                "symbol": histogram_symbol,
                "kind": "exact",
                "file": histogram_file,
            },
            {
                "symbol": old_symbol,
                "kind": "old_value_permutation",
                "input_symbol": input_symbol,
                "histogram_symbol": histogram_symbol,
            },
            {"symbol": status_symbol, "kind": "exact", "file": status_file},
        ],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
