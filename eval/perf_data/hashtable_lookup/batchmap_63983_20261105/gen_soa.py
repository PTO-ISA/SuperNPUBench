#!/usr/bin/env python3
"""Regenerate the SoA key/value arrays from the archived AoS table.

Source: ../hash_6144_PE4.elf, first PT_LOAD, file offset 0x1000,
2,550,000 x 16 B entries (key i64, value i32, pad i32).
Outputs: data/keys32.bin (u32 low keys + 128-entry guard) and
data/values.bin (i32 values + 128-entry guard).
"""
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "hash_6144_PE4.elf"
OUT = HERE / "data"
CAP = 2550000

with REF.open("rb") as f:
    f.seek(0x1000)
    data = f.read(CAP * 16)

keys = bytearray(CAP * 4)
vals = bytearray(CAP * 4)
for i in range(CAP):
    off = i * 16
    key = struct.unpack_from("<q", data, off)[0]
    val = struct.unpack_from("<i", data, off + 8)[0]
    struct.pack_into("<I", keys, i * 4, key & 0xFFFFFFFF)
    struct.pack_into("<i", vals, i * 4, val)

keys += keys[:128 * 4]   # guard: mirror slots 0..127
vals += vals[:128 * 4]
(OUT / "keys32.bin").write_bytes(keys)
(OUT / "values.bin").write_bytes(vals)
print("wrote", OUT / "keys32.bin", len(keys), "bytes")
print("wrote", OUT / "values.bin", len(vals), "bytes")
