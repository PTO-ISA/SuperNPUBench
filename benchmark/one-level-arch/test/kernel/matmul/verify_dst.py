#!/usr/bin/env python3
# verify_dst.py
#
# Verify a gfrun --dump-memory dump of a matmul C matrix against the analytic
# golden for all-1 inputs: C[i][j] = K everywhere (bit-exact, since every
# addend is 1.0f and partial sums 1..K are all exactly representable).
#
# Usage:
#   ./verify_dst.py <dump.bin> <golden> [M] [N]
#
# Example (M=N=K=64, dst dumped from g_dst @ 0x180e8):
#   ./verify_dst.py /tmp/g_dst_dump.bin 64 64 64
#
# Exit code: 0 = all cells match golden, 1 = mismatch found.

import struct
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: verify_dst.py <dump.bin> <golden> [M] [N]", file=sys.stderr)
        return 2

    path = sys.argv[1]
    golden = float(sys.argv[2])
    m = int(sys.argv[3]) if len(sys.argv) > 3 else None
    n = int(sys.argv[4]) if len(sys.argv) > 4 else None

    data = open(path, "rb").read()
    count = len(data) // 4
    vals = struct.unpack("<%df" % count, data)

    bad = [(i, v) for i, v in enumerate(vals) if v != golden]

    if not bad:
        print("PASS: all %d floats == %g (0x%s)  <- matches golden K=%g"
              % (count, golden, struct.pack("<f", golden).hex(), golden))
        return 0

    print("FAIL: %d/%d cells != %g" % (len(bad), count, golden))
    for i, v in bad[:8]:
        row, col = (i // n, i % n) if (m and n) else (None, None)
        loc = "row=%s col=%s" % (row, col) if row is not None else "idx=%d" % i
        print("  %s val=%r raw=%s"
              % (loc, v, struct.pack("<f", v).hex()))
    return 1


if __name__ == "__main__":
    sys.exit(main())
