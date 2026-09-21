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

1. **Load** — one 1KB `group_tload_832` (`TLOAD`, `T(r,q) = bin[8r+q]`).
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
   mode-1 `TSHUF` row shifts (segment width 32, zero boundary), then the
   shifted-by-one scan `S` is broadcast back:
   `c[q][r] += S(r)`, where `S(r) = sum_{r'>r} G(r')`.
4. **Store** — eight strided `[32,1]` TSTOREs scatter `c[q][r] -> hist[8r+q]`.

Result: `hist[8r+q] = sum_{i>=8r+q} bin[i]` over all 256 bins.

## Cost and future work

Step 4 dominates the kernel wall time: it is a stride-8 scatter written as
eight `[32,1]` strided TSTOREs, which is several times more expensive per byte
than a grouped store. Two ways to collapse it into one large-packet 1KB write:

- **TADD + assemble** — let the final broadcast TADD write the eight CELLs
  into a `[32,8]` parent through a destination-side `B.ASSEMBLE`, then issue a
  single grouped TSTORE. Not available today: the TEPL `_ASS` producer path
  (e.g. `TADD_ASS`) crashes the LinxV5 backend
  (<https://github.com/LinxISA/llvm-project/issues/103>), and the working
  region-assemble route only accepts a RowMajor destination with CUBE subview
  sources, so a CUBE parent cannot be built from computed CELLs.
- **TSTORE microarchitecture coalescing** — merge the eight strided `[32,1]`
  stores into one large-packet write in the TLSU.

Until one of these lands the strided stores are kept.

## Test / perf

```bash
# Functional (res_check): one pass per PE, compared to the scalar suffix sum.
cd benchmark/one-level-arch/test/kernel/multi_thread/histogram_cumsum_m32 && ./compile.all
cd benchmark/one-level-arch && python3 test/kernel/multi_thread/histogram_cumsum_m32/src/run_histogram_cumsum_m32_check.py \
    --gfrun <model>/bin/gfrun

# Cycle timing + SwimLane (one per-PE hist, CUMSUM_ITERS calls).
python3 test/common/run_swimlane.py \
    --gfsim <model>/bin/gfsim --elf <perf-elf> \
    --outdir test/kernel/multi_thread/suffix_cumsum_perf/perf_runs/<run_id> --name cumsum
```
