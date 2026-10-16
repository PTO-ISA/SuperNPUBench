# Element-wise masked gather/scatter

`src/elementwise_masked_gather_scatter.cpp` keeps the original five aligned
128-element arrays and its byte-backed mask. `main` widens the byte mask into
the public U32 element transport. The kernel loads official 128-element index
and mask Tiles, borrows logical 32-element parts, runs an ordinary masked FP32
indexed gather, carries the raw result through official `TMOV`, then runs a
second ordinary marked loop containing the masked indexed scatter. The unused
`gathered_storage` allocation remains zero and is not a new output.

`elementwise_masked_gather_scatter_strong` is a separate 128-element case. It
uses a valid centered base with signed negative active indices, inactive extreme
poison indices, noncanonical nonzero byte masks, duplicate active destinations
whose writes all carry the same payload, raw signed-zero/infinity/NaN payloads,
output sentinels, and two 64-byte guards. Exact raw-bit goldens independently
check all five arrays and both guards; no floating tolerance or NaN arithmetic
policy is involved.

Both cases use the Tile element suite and the FP32 functional model lane:

```sh
MODELS=gfrun bash benchmark/one-level-arch/verification/run_tile_element_suite.sh \
  --case elementwise_masked_gather_scatter
MODELS=gfrun bash benchmark/one-level-arch/verification/run_tile_element_suite.sh \
  --case elementwise_masked_gather_scatter_strong
```

The runner rejects a timing-model selection for these cases. Existing default
integer cases retain their normal dual-model selection.
