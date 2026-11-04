# Unsorted segment sum (four PE baseline)

The kernel consumes FP32 `data[Rows, Features]` and signed 32-bit
`segment_ids[Rows]`, and writes FP32 `output[Segments, Features]`:

`output[s, j] = sum(data[i, j] for i if segment_ids[i] == s)`.

IDs outside `[0, Segments)` are ignored. Empty segments produce zero. Features
must be a positive multiple of 32. Each PE owns output segments in a stride-four
partition, so repeated IDs cannot cause concurrent writes. This deterministic
baseline scans all input rows for each output segment; its work is proportional
to `Rows * Segments * Features`. It is intended as a correctness baseline before
adding a faster algorithm.

The checked fixture uses `Rows=64`, `Features=64`, `Segments=12`, with repeated
IDs, invalid IDs, and five empty segments. Input values are small exact FP32
integers, so the expected result is independent of summation rounding order.

```sh
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
cd benchmark/one-level-arch/test/kernel/reduction/unsorted_segment_sum
python3 src/check.py --self-test
make res_check=on diss
python3 src/check.py --gfrun /path/to/SuperScalarModel/bin/gfrun
```

The checker creates input and golden files under the architecture's `compare/`
directory, runs the four-PE model, and compares every output value. A regular
`compile.all` builds the benchmark ELF without binary result checking.
