# RMS Norm CubeM32 layout

This change covers RMS Norm dynamic/static, split-R dynamic/static, and
dynamic R-tree/R-simt only. GroupNormGrad changes are in a separate PR.
Local Tiles use CubeM32; GM tensors remain row-major.

## Reduction paths

- Ordinary RMS and split-R full-width strips use native TROWSUM followed
  by TCOLSUM, with wide physical carriers and one valid output column.
- Static RMS and split-R no longer include m32_utils.hpp.
- Dynamic short strips still use m32_utils.hpp to encode runtime columns
  in LB2. The native TROWSUM wrapper currently encodes allocation columns.
- The split-R GM workspace still stores one float per row/cache level.
- R-tree/R-simt retain their existing multi-row reduction paths.

## Validation

Model PR #764, commit 3e91790ba7ec9ec8f8f6e3de7878da8b92a6c4da:
all six default FP16/4PE cases run successfully and pass golden comparison.
RMS and R-tree/R-simt use [128,8192]; split-R uses [16,16384].
Maximum absolute error is 0.001953125 for all six.
The four ordinary RMS/split-R cases were rebuilt and rechecked after the
native reduction refactor (badd494). Tree/SIMT were unchanged.

Compiler: dev-llvm15_56 at 3a9b70d00aacb1739122c460a989e06af3412b42.
TileOP: linx at b1545cef9cf001c9bbf30d35a880dda57793029d.

This is not an all-shape or gfsim pass claim. Dynamic [5,3333] tests still
stop at the row-expansion descriptor check; the pre-refactor ordinary RMS
also fails that check. R-tree/R-simt gfsim remains tracked in model #760.
