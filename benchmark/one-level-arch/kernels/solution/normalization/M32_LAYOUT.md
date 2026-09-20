# Normalization Local Tile layout

All ten normalization kernel implementations use CubeM32 for Local Tiles.
GM tensors remain row-major; TLOAD/TSTORE perform the layout conversion.
Logical shape and physical capacity are distinct: M32 always stores 32 rows.

- A logical one-row FP32 strip with 512 physical columns allocates 64 KiB,
  not the 2 KiB used by its former RowMajor representation.
- GroupNormGrad data strips are limited to 256 physical columns (32 KiB
  FP32 / 16 KiB FP16). Larger spatial/channel axes are processed in loops.
  The static H=2024 path uses seven 256-column blocks and one 232-column tail.
- One-row reductions use m32_utils.hpp: a wide carrier preserves the source
  physical columns, then TREDUCEPREFIXVIEW and TMULS(1) materialize a single
  FP32 CELL for accumulation/broadcast/GM cache reload. This is tile-only
  computation, not a scalar fallback. The helper requires one logical row.
- Split-R GM cache remains one float per row/level; M32 physical padding is
  not part of the externally allocated workspace.
- R-tree and R-simt retain their explicit wide multi-row reduction carriers.

## Validation at this change

All ten default FP16/4PE RES_CHECK builds compile with LLVM
3a9b70d00aacb1739122c460a989e06af3412b42 (dev-llvm15_56) and TileOP
b1545cef9cf001c9bbf30d35a880dda57793029d.

Runtime validation is NOT passing with SuperScalarModel
8d7d3cdb19d4b895aef168b4eba4bc13a80ec1c9:
gfrun stops at CUBE descriptor, row-reduction or broadcast checks before
precision comparison. The newly encountered descriptor/reduction failures
still need field-level diagnosis; they must not all be presumed model bugs.
R-tree/R-simt gfsim also abort before producing valid performance results.

Known tracking:
- https://github.com/LinxISA/SuperScalarModel/issues/762 (gfrun TCOLEXPAND)
- https://github.com/LinxISA/SuperScalarModel/issues/760 (gfsim CUBE elementwise)

Compilation alone is not an accuracy pass. Re-run every affected case and
its golden comparator after runtime blockers are resolved.
