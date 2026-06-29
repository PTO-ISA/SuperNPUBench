# Probe Distance Statistics

## Dataset

- Source files: `inserted_slot.data`, `lookup_keys.data`, `lookup_values.data`
- Generator: `gen_data.py` (seed_table=42, seed_query=211, deterministic)
- Table capacity: 2,550,000
- Occupied slots: 2,116,500
- Load factor: 83.0%
- Query keys: 6,144 (all found in table)
- Matches code constant `kCapacity = 2550000` in both `hashtable_lookup_simd.cpp` and `hashtable_lookup_simt.cpp`

## Per-key probe distance

| Metric | Value |
|--------|-------|
| Min | 0 |
| Max | 402 |
| Mean | 2.42 |

> Max probe (402) is within `MAX_PROBE=512` used by the build configs.

## Per-warp probe distance (warp=64, 96 warps)

Warp probe count = max probe distance across the 64 threads (SIMT lockstep — the warp iterates until the slowest thread finds its key).

| Metric | Value | Target |
|--------|-------|--------|
| Mean | 41.12 | ~40 |
| Median | 32.5 | 32 |
| Max | 315 | ~300 |
| Min | 11 | — |

## Per-warp probe distance (warp=32, 192 warps)

| Metric | Value |
|--------|-------|
| Mean | 28.64 |
| Max | 315 |

## Key takeaway

Per-key mean is only 2.42, but per-warp(64) mean max is **41.12** — the warp is bottlenecked by the worst thread in each 64-lane group (SIMT inefficiency). The 83% load factor produces a heavy-tailed probe distribution: most keys probe 0-2 times, but a few probe 100-300+ times, dragging the per-warp max up.
