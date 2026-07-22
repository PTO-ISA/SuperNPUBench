# Linx Direct-Boot TileOP Cases

This directory is the canonical two-level-architecture home for the bounded
TileOP cases consumed by the LinxISA superproject AI workload flow. The active
case set is defined by `compile.all`; every row has one matching source file in
`src/`.

The sources were migrated from the last active direct-boot integration at
`8e2e1b54b0f0633231729fbcf9973cfd7e0a4823` into the rewritten `origin/main`
architecture layout based at `c9cfac34b353bfc1fd978ff7e2e15ce177670f0c`.
Retired top-level `test/`, `tests/`, and `benchmarks/` compatibility trees are
intentionally not restored.

Use the in-repo Linx compiler explicitly:

```sh
export COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin
make TESTCASE=TAdd PLAT=linx
```

`TLoad` and `TStore` are the active memory-operation names. Legacy
`TLoad`/`TStore` aliases are not part of this manifest.
