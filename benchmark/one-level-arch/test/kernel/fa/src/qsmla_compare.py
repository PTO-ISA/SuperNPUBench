#!/usr/bin/env python3

import math
import struct

path = "SuperNPUBench/benchmark/one-level-arch/test/kernel/fa/src/"

with open(path + "qsmla_onepass_npu_out.bin", "rb") as f:
    actual = struct.unpack("<32768e", f.read())

with open(path + "qsmla_golden.bin", "rb") as f:
    golden = struct.unpack("<32768f", f.read())

errors = [abs(float(a) - float(g)) for a, g in zip(actual, golden)]
passed = [e <= 1e-3 + 1e-3 * abs(g) for e, g in zip(errors, golden)]

print(f"passed  = {sum(passed)}/32768 ({100*sum(passed)/32768:.6f}%)")
print(f"failed  = {32768-sum(passed)}")
print(f"max_abs = {max(errors):.9f}")
print(f"mean_abs= {sum(errors)/len(errors):.9f}")
print(f"nan_npu = {sum(math.isnan(float(x)) for x in actual)}")
