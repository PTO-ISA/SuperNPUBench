# 256-lane hash: bubble analysis and optimization journey (2026-11-04)

Baseline: `outputs/hash256_s4g_20261104` (256-lane, four slots per round issued
one after another, 48 B guard, deferred value gather).
Kernel 130,577 / whole fixture 193,010.

## Final result

| metric | value |
|---|---:|
| **kernel cycles** (first input TLOAD -> last result TSTORE) | **109,193** |
| whole fixture | 171,646 |
| numerical check | PASS (6144 outputs + guards, 4 finishers 0x5555) |
| vs s4g baseline | -16.4% kernel |
| vs PR 256-lane stride-1 (187,851) | -41.9% |

## Optimization journey

| step | whole fixture | mechanism |
|---|---:|---|
| s4g baseline | 193,010 | four serial slot round trips per round |
| + 2-slot parallel loads | 185,191 | round-start mask for all loads; slot-gather spacing 350-430 -> 79 cycles |
| + slot indices instead of byte offsets | 182,739 | removes the `>>4`/`x16` address round trip (~17 TCI per four loads) |
| + merged decision, drop always-true `element < valid` | 176,416 | fewer compares/selects per slot |
| + **empty-first branch order** | 174,016 | equality-first was 176,416 |
| + **no-wrap 2 KB guard** | **171,646** | guard mirrors slots 0..127; longest probe is 63 slots, so no per-round wrap |

`empty-first` is the branch order `found == -1` before `found == key`; it is
faster in the stride-2 form but slower in the stride-3 form, so the two were
measured separately.

Rejected variants (measured, not compiled away):

| variant | whole fixture |
|---|---:|
| stride-3 + no-wrap | 174,588 |
| stride-4 + no-wrap | 183,681 |
| 512 lanes, optimized structure | 197,408 |
| 128 lanes, stride-2 | 212,360 |

## Bubble analysis (swimlane)

s4g, four slots per round:

| metric | value |
|---|---:|
| rounds per PE | 42/41/48/45 (176 total) |
| round period (median) | 2,200-2,600 cycles |
| spacing between the four slot gathers | 350-430 cycles |
| single gather duration | 110-155 cycles |
| TLSU busy share of a round | ~24% |
| VEC decision chain per round | ~1,900 cycles (~475 per slot) |

The four gathers are separated by the previous slot's compare/select chain, so
each round pays four serial TLSU round trips.

Final stride-2 variant: intra-round gather spacing 79 cycles (both ~170 in
flight), so the TLSU latency overlaps; the remaining cost is the VEC decision
chain (~365 cycles per slot in the pre-merge build, less now).

## Why wider lanes do not help

Doubling the lanes halves the rounds but doubles the lanes per round, and the
per-batch maximum probe (the tail) grows with the batch size. Lane-slot work
for the fixed 13,039 useful probes grows 137k (128) -> 173k (256) -> 207k
(512), and each tile op moves more bytes. Measured optimum: 256 lanes.

## Blocked attempts

- `anyActive` every two rounds / merging tests into one condition: the
  element-region lowering rejects multi-block or duplicated regions.
- 4-slot parallel (pre-optimization): model stalls with 16 wide gathers in
  flight; two to three is the supported depth.
- PE load balancing (rounds 80/78/94/86): no scalar GM atomic add.
- Double buffer for hash(FMA)/probe(VEC) overlap: three upstream blockers
  (element-region sentinel, model TCI hang on prefetch+TMOV, compiler
  tile-descriptor PHI join for hand-written wide tiles).

## Open items

1. Per-slot VEC chain (the dominant remaining cost): the lowered body still
   spends ~5-6 TAND/TOR/TXOR/TSEL mask ops per slot on the if/else-if
   resolution chain.
2. `anyActive` TROWMAX reduction: ~7-10% of a round.
3. Hash cost (~5.5%): the TileOP hash uses two TREMs for the modulo.

## Artifacts

- Final source: `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash256_s4g_par_20261104/src/kernel.hpp`
- Final fixture (2 KB guard): `.../src/fixture.cpp`
- Final ELF: `.../src/hash256.elf`
- Final result: `.../final-result.json`, `.../final-metrics.txt`
- Final swimlane: `.../swimlane_final.json.gz`
- PipeView: `.../pipeview.out.gz`
- Disassembly: `.../src/kernel.expanded.diss`
