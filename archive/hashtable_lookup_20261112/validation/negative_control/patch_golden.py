#!/usr/bin/env python3
"""Build the negative-control ELF: flip one byte of the embedded golden array.

The r1 check fixture embeds `golden` as .incbin from the reference ELF at
0x26f6000, length 0x6000. The same 24 KiB blob occurs exactly once in
check.elf, so patching that occurrence corrupts golden[0] and the in-ELF
comparator must report a mismatch.
"""
import sys
ref = open(sys.argv[1], 'rb').read()[0x26f6000:0x26f6000 + 0x6000]
src = sys.argv[2]
bad = sys.argv[3]
b = bytearray(open(src, 'rb').read())
i = b.find(ref)
assert i >= 0 and b.count(ref) == 1, (i, b.count(ref))
b[i] ^= 0xFF
open(bad, 'wb').write(b)
print(f'patched {src} at 0x{i:x} -> {bad}')
