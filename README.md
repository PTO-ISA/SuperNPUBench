# SuperNPUBench

SuperNPUBench is a C++ tile-kernel benchmark suite and programming manual for a
superscalar NPU. It combines buildable kernels, source-backed benchmark pages,
an intrinsic reference, and pinned translations of production model kernels.

## Layout

```text
SuperNPUBench/
├── benchmark/            # buildable operators and translated model kernels
├── microbenchmark/       # focused scalar, memory, vector, and matrix cases
├── docs/                 # programmer manual and generated references
├── scripts/              # catalog generators and publication gates
└── compile_all.sh        # repository build driver
```

Build outputs under `output/` are ignored.

## Toolchain setup

Build the compiler and C library from the parent toolchain checkout, then export
their locations:

```bash
cmake --build compiler/llvm/build-linxisa-clang \
  --target clang lld llvm-objdump llvm-objcopy -j8

export COMPILER_DIR="$PWD/compiler/llvm/build-linxisa-clang/bin"
export LINX_SYSROOT="$PWD/out/libc/musl/install/phase-b"
```

The Makefiles add:

```text
-target linx64-linx-none-elf
--sysroot=$LINX_SYSROOT
-fenable-matrix
-O2
-std=c++20
```

`COMPILER_DIR` must contain `clang`, `clang++`, `llvm-objdump`, and
`llvm-objcopy`. The workload does not download or select an external compiler.

## Build example

Compile a matrix multiplication benchmark:

```bash
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=HIF4_HIF4 \
  M=256 N=2048 K=2048 tM=128 tN=128 tK=128
```

Run the active benchmark batch:

```bash
./compile_all.sh one-level
```

Artifacts are written below the selected benchmark tree's `output/` directory.

## Focused benchmarks

The microbenchmark tree covers:

| Family | Representative operations |
| --- | --- |
| scalar | integer, floating-point, conversion, and memory latency/throughput |
| memory | TLOAD, TSTORE, TMOV, gather, and scatter |
| vector | named TEPL elementwise, scalar-tile, reduction, and expansion ops |
| cube | TMATMUL, bias/accumulate variants, and ACCCVT |

```bash
cd microbenchmark
make TESTCASE=tmatmul_fp16_64x64x64
bash compile_all.sh all
```

See [microbenchmark/README.md](microbenchmark/README.md) for case-generation
details. The public website focuses on the unified C++ programming model and
active kernel benchmark reference.

<!-- BENCHMARK-CATALOG:START -->
## Benchmark catalog

The active manifests contain **62 build variants**. Every name below
has a source-backed page with its build command and tile intrinsic surface in
the website's **Benchmark Reference** section.

<details><summary><strong>broadcast</strong> (4 names, 6 variants)</summary>

`broadcast`, `broadcast_vec_019`, `broadcast_vec_039`, `broadcast_vec_07`

</details>

<details><summary><strong>concat</strong> (2 names, 4 variants)</summary>

`concat_gather`, `concat_scatter`

</details>

<details><summary><strong>control</strong> (1 name, 6 variants)</summary>

`hashtable_lookup_simd`

</details>

<details><summary><strong>element_wise/gelu</strong> (1 name, 1 variant)</summary>

`gelu`

</details>

<details><summary><strong>fa</strong> (2 names, 9 variants)</summary>

`fa_2d_unroll`, `fa_HIF4_HIF4`

</details>

<details><summary><strong>gather</strong> (1 name, 1 variant)</summary>

`gather`

</details>

<details><summary><strong>matmul</strong> (1 name, 16 variants)</summary>

`matmul`

</details>

<details><summary><strong>pto_kernels</strong> (9 names, 9 variants)</summary>

`pto_add`, `pto_flash_attention`, `pto_gemm`, `pto_gemm_basic`, `pto_gemm_demo`, `pto_gemm_performance`, `pto_mamulb`, `pto_tload_store`, `pto_tmatmul_acc`

</details>

<details><summary><strong>reduction/reducemax_col</strong> (1 name, 1 variant)</summary>

`reducemax_col`

</details>

<details><summary><strong>reduction/reducemax_row</strong> (1 name, 1 variant)</summary>

`reducemax_row`

</details>

<details><summary><strong>reduction/reducesum_col</strong> (1 name, 2 variants)</summary>

`reducesum_col`

</details>

<details><summary><strong>reduction/reducesum_row</strong> (1 name, 1 variant)</summary>

`reducesum_row`

</details>

<details><summary><strong>sort</strong> (1 name, 1 variant)</summary>

`topk`

</details>

<details><summary><strong>transpose</strong> (1 name, 4 variants)</summary>

`transpose`

</details>

<!-- BENCHMARK-CATALOG:END -->

## Verification

Regenerate and verify the publication artifacts before submitting changes:

```bash
python3 scripts/sync_golden_manual.py --check
python3 scripts/generate_benchmark_manual.py
python3 scripts/generate_deepseek_manual.py
python3 scripts/verify_deepseek_migration.py
python3 -m mkdocs build --strict --clean
python3 scripts/verify_golden_manual.py --site site
```

## Documentation

- [Superscalar NPU programming guide](https://pto-isa.github.io/SuperNPUBench/)
- Install site dependencies with `python3 -m pip install -r requirements-docs.txt`
- Build the site locally with `python3 -m mkdocs serve`

## License

See `LICENSE`.
