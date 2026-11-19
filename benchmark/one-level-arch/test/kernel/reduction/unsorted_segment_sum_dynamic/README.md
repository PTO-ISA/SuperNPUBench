# Unsorted segment sum (dynamic shape, four PE)

Exercises the runtime-shape entry:

```cpp
unsorted_segment_sum(data, segment_ids, output, rows, features, segments);
```

The fixture uses `Rows=37`, `Features=45`, `Segments=70`, so it covers:

- a partial feature tile (`45 = 64 + 1` column);
- more than one segment group (`70 > 12`);
- repeated IDs, invalid IDs (`-1`, `70`) and empty segments.

The kernel keeps the physical tile width compile-time, processes segments in
groups of 12, and writes empty segments as zero.

```sh
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
cd benchmark/one-level-arch/test/kernel/reduction/unsorted_segment_sum_dynamic
python3 src/check.py --self-test
make res_check=on diss
python3 src/check.py --gfrun /path/to/SuperScalarModel/bin/gfrun
```
