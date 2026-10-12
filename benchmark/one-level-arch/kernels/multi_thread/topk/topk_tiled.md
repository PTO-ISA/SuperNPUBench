# topk_tiled — tiled top-k radix-select

`topk_tiled.hpp` (shared core) + `topk_tail_fp32.hpp` / `topk_tail_fp16.hpp`
(boundary refinement, selected by `kFp32Refine`).

Pipeline:

1. **Stage 1** — one 256-bin histogram over each element's **FP16 sortable-key
   high byte** (`key16 >> 8`), suffix-cumsum, threshold `thr`. Elements with
   `bin > thr` are written to `out`; `bin == thr` become the candidate list.
2. **Tail** — refine the candidate bucket into an exact top-k:
   - `kFp32Refine == true`  → four FP32 key-byte rounds (exact FP32 top-k).
   - `kFp32Refine == false` → one FP16 key low-byte round (exact top-k of the
     FP16-rounded values), reusing `round_write`'s final pass; ties in the full
     16-bit FP16 key are broken in candidate order.

Outer shape is runtime (`TopkTilingData {batch, cols, topk}`) validated against
the compile-time maxima `kBatchMax=4`, `kColsMax=131072`, `kTopKMax=1024`,
`kCandCap=65536`.

## The 131072 -> 1024 fix

`Scratch.hist[256]` was immediately followed by `num[0]`. Two things need
`hist[256]` to be a real, per-round-cleared word:

- `round_write` allocates the FP32 top-byte (`0xFF`) slot through
  `H[bin+1]`, i.e. `H[256]`;
- `suffix_cumsum` reads `hist[256:384]` as its zero boundary.

Because `hist[256]` aliased `num[0]` — the same counter the round uses to append
its `== thr` candidates (`nr == 0` for round 1) — `0xFF` slots and the append
stream shared one word, producing duplicate slots and skipped (`-1`) slots.

Why only 131072: it is the first size whose `[127.5, 128)` mass makes FP32
`byte1 == 0xFF` a GT lane in round 1 (65536 and below pass). Fixed by inserting
a per-round-cleared `hist_pad[256]` between `hist` and `num`.

## GM atom add

`mgather_add_s32_m32` (histogram bumps and slot allocation) goes through the
TileOP `MGATHER_ADD` wrapper.  The wrapper emits `B.DATR CUBE_M32` from the M32
index / value tiles (LinxISA/Linx-TileOP-API#185, PR #209) and the LinxV5
backend folds its default `lb0`; gfrun resolves an omitted `LB0` to one since
LinxISA/SuperScalarModel#818 (`linx-isa#202`).  `topk_tiled.hpp` /
`topk_tail_fp32.hpp` / `topk_tail_fp16.hpp` contain no hand-written asm.

## Dependencies

- **LinxISA/Linx-TileOP-API** `#207` (CUBE_M32 layout for the indexed
  `MGATHER` / `MSCATTER.MASK` wrappers), `#208` (TCMPS CUBE GPR predicate
  output) and `#209` (GM atom/red `B.DATR.Layout`, used by `MGATHER_ADD`).
- **LinxISA/SuperScalarModel** PR `#785` (CUBE predicate carriers), the
  canonical `B.IOR` RegDst fix (`#806`), and PR `#818` (honor omitted bundle
  dimension defaults: an omitted `LB0` has effective value one, per
  `linx-isa#202`).  #818 closes the `SuperScalarModel#816` follow-up.

Note: the operator also uses `histogram_cumsum_m32.hpp` (the grouped
`suffix_cumsum` and `hist_clear`), which is where the remaining hand-written
asm lives — those regional assemble / `B.SUBVIEW` patterns have no TileOP
wrapper yet.  `find_threshold` reads the TCMPS CUBE GPR carrier into a GPR and
does `ctz` on it (the intended mechanism, not a scalar workaround).

## Build / run

The outer shape is runtime (`TopkTilingData`), so different batch/cols/topk
need **no recompile** as long as they stay within the compile-time maxima
(`kBatchMax` / `kColsMax` / `kTopKMax`; exceeding them means editing those
constants and `./compile.all` again).

```sh
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
cd test/kernel/multi_thread/topk_tiled && ./compile.all

# shipped case: batch 4 x 8192, topk 512, varied per-batch ranges
python3 src/run_topk_tiled_check.py --gfrun <gfrun>

# any runtime shape within the maxima, e.g. batch 1 x 131072 -> 1024
python3 src/run_topk_tiled_check.py --gfrun <gfrun> \
    --batch 1 --total_len 131072 --topk 1024 --mode fp32

# fp16 build: same flags, --mode fp16
python3 src/run_topk_tiled_check.py --gfrun <gfrun> \
    --batch 1 --total_len 131072 --topk 1024 --mode fp16
```

`--batch` is the outer batch count (each entry is an independent length
`total_len` input slice, i.e. one top-k problem). `--batch/--total_len/--topk`
override the shipped shape and run each entry over the full `[0, total_len)`
range. `--mode` must match the compiled `kFp32Refine`: fp16 mode compares the
output's FP16 sortable-key multiset against the golden top-K keys (arbitrary
tie-break), fp32 mode compares the exact index set.

## Runtime limits

- Runtime shape: `batch <= kBatchMax` (4), `total_len <= kColsMax` (131072),
  `topk <= kTopKMax` (1024). `batch 1 x 131072 -> 1024` (the DeepSeek-v4 style
  case) is exactly the current maximum; larger shapes need editing the maxima
  (and `kCandCap`) and rebuilding.
- The fp32/fp16 precision is a **compile-time** switch (`kFp32Refine`): one
  build produces one of the two, not a per-call runtime choice.
- Candidate overflow: candidates are the `bin16 == thr` elements of the window.
  `stage1_collect` gates the append at `kCandCap`, so the buffer is never
  overrun; `run()` then reports `errors[bx] = 1` and the host check fails loudly
  rather than returning a wrong top-k.  With `kCandCap = 65536` a window would
  need more than 65536 elements in one FP16 high-byte bin to trigger it.  The
  guard was verified by temporarily setting `kCandCap = 64`
  (`device error flags set: (1, 0, 0, 1)`).

## Validation status

gfrun `-s softcore.multiThreadNum=4`, LinxV5 toolchain (TileOP API #209).

| mode | shape | seeds | result |
| --- | --- | --- | --- |
| fp32 | `4x8192x512` | 12345 | PASS |
| fp32 | `1x131072x1024` | 0..20 (21 seeds) | 21/21 PASS |
| fp32 | `2x131072x1024` | 3 | PASS |
| fp16 | `4x8192x512` | 12345 | PASS |
| fp16 | `1x131072x1024` | 0..20 (21 seeds) | 21/21 PASS |
| fp16 | `2x131072x1024` | 3 | PASS |

Also: `test/kernel/multi_thread/gm_atom_dup` PASS (same-address `MGATHER.ADD`
serialization, which the slot allocator relies on); `histogram_cumsum_m32`
PASS (4 rows).
