# Unsorted segment sum (INT32, four PE)

The kernel consumes INT32 `data[Rows, Features]` and signed 32-bit
`segment_ids[Rows]`, and writes INT32 `output[Segments, Features]`:

`output[s, j] = sum(data[i, j] for i if segment_ids[i] == s)`.

Input and output share the same dtype, matching the ops-nn
`UnsortedSegmentSum` contract (fp16/fp32/bf16/int32/int64/uint32/uint64, with
`output.dtype == input.dtype`). IDs outside `[0, Segments)` are ignored. Empty
segments produce zero. Features must be a positive multiple of 32.

The checked fixture uses `Rows=64`, `Features=64`, `Segments=12`, with repeated
IDs, invalid IDs, and five empty segments. Each PE owns output segments in a
stride-four partition, so repeated IDs cannot cause concurrent writes.

```sh
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
cd benchmark/one-level-arch/test/kernel/reduction/unsorted_segment_sum_int32
python3 src/check.py --self-test
make res_check=on diss
python3 src/check.py --gfrun /path/to/SuperScalarModel/bin/gfrun
```

The checker creates INT32 input and golden files under the architecture's
`compare/` directory, runs the four-PE model, and compares every output value
exactly. A regular `compile.all` builds the benchmark ELF without binary result
checking.
