# Single-PE Kernels

This directory contains the original one-level-architecture operator
implementations that execute on one PE.

Four-PE-specific implementations are kept separately under
[`../multi_thread/`](../multi_thread/README.md).

## Run with gfrun

Run an ELF built from a single-thread operator test with one simulated PE:

```bash
gfrun -t 1 -f /path/to/operator.elf
```

The run passes when `gfrun` exits with status 0 and its output contains both
`Reach the End of Benchmark` and `R2 = 0`.
