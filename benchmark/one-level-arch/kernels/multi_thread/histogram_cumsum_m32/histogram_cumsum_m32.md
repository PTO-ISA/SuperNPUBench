# HISTOGRAM_CUMSUM_M32 — 256-bin suffix cumsum

`histogram_cumsum_m32::suffix_cumsum(hist)` computes, in place on a per-PE GM
histogram,

```text
hist[i] = sum_{j = i .. 255} hist_orig[j]      (i = 0..255)
```

`hist[256:384]` is a zero pad the grouped load reads past the sentinel and is
left untouched.

## Data layout

The 256 S32 bins are one 1KB S32 CUBE_M32 grouped tile. A single grouped
`ND2M32` TLOAD (`ValidCol=8`, `ValidRow=32`) packs the dense stream so that

```text
cell_q[r] = bin[8r+q]        r = 0..31, q = 0..7
```

i.e. the eight physical CELLs sit along the logical columns and are
individually addressable with `B.SUBVIEW`.

## Dataflow

1. **Load** — one 1KB grouped `ND2M32` `TLOAD` (`T(r,q) = bin[8r+q]`), TileOP
   API.
2. **Within-group suffix** — a parallel pair tree of binary TADDs over
   `B.SUBVIEW` CELL reads (`c[q]` reads `cell_q` straight out of the parent).
   Depth 3, no TMULS materialize for cells 0..6; only `c[7]` is a copy:
   ```text
   p67=X6+X7  p45=X4+X5  p23=X2+X3  p01=X0+X1
   S5 = X5+p67           S1 = X1+p23
   s47 = p45+p67         s03 = p01+p23
   S3 = X3+s47  S2 = p23+s47  S1 += s47  S0 = s03+s47
   S4 = s47  S6 = p67  S7 = X7
   ```
   After this `c[q][r] = sum_{j>=q} bin[8r+j]` and `c[0][r] = G(r)`, the row
   (group) total.
3. **Across-group suffix** — a 32-lane Hillis-Steele scan of `c[0]` using
   mode-1 `TSHUF` row shifts (TileOP API; segment width 32, zero boundary),
   then the shifted-by-one scan `S` is broadcast back:
   `c[q][r] += S(r)`, where `S(r) = sum_{r'>r} G(r')`.
4. **Store** — eight strided `[32,1]` TSTOREs scatter `c[q][r] -> hist[8r+q]`.

Result: `hist[8r+q] = sum_{i>=8r+q} bin[i]` over all 256 bins.

## Cost and future work

The grouped load / store / zero-fill (`TLOAD` / `TSTORE` / `TEXPANDS`) and the
`TSHUF` scan go through the TileOP API. Two hand-written blocks remain in
`histogram_cumsum_m32.hpp`, both blocked on TileOP API gaps
(<https://github.com/LinxISA/Linx-TileOP-API/issues/211>):

- **Step-2 CELL reads (`B.SUBVIEW`)** — TEPL elementwise does not accept subview
  sources (only `TOR_ASS` does), so the pair tree still reads cells with
  hand-written `B.SUBVIEW`.
- **Step 4 (strided stores)** — the intended collapse is one grouped 1KB write
  via a destination-side `B.ASSEMBLE`: `TADD_ASS` the eight computed cells into
  a `[32,8]` parent, then one grouped `TSTORE`. The backend crash
  ([llvm-project#103](https://github.com/LinxISA/llvm-project/issues/103)) is
  fixed and already in the toolchain, so `TADD_ASS` MIDDLE/LAST work — but
  `TEPL _ASS` cannot write the INIT slot ("must use a plain allocating
  producer", in practice `TLOAD`), and no plain producer can allocate the INIT
  slot for computed data. Until the API exposes an INIT-capable computed
  producer, step 4 keeps the eight strided `[32,1]` stores, which dominate the
  kernel wall time.
- **TSTORE microarchitecture coalescing** — merge the eight strided `[32,1]`
  stores into one large-packet write in the TLSU.

Until one of these lands the strided stores are kept.

## Test / perf

```bash
# Functional (res_check): one pass per PE, compared to the scalar suffix sum.
cd benchmark/one-level-arch/test/kernel/multi_thread/histogram_cumsum_m32 && ./compile.all
cd benchmark/one-level-arch && python3 test/kernel/multi_thread/histogram_cumsum_m32/src/run_histogram_cumsum_m32_check.py \
    --gfrun <model>/bin/gfrun

# Cycle timing + SwimLane. One call already computes the complete cumsum, so
# the perf entry defaults to a single pass; CUMSUM_ITERS may repeat it to
# lengthen the trace.
python3 test/common/run_swimlane.py \
    --gfsim <model>/bin/gfsim --elf <perf-elf> \
    --outdir test/kernel/multi_thread/suffix_cumsum_perf/perf_runs/<run_id> --name cumsum
```
