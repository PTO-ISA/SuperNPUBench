# Tests

This tree keeps correctness material that is not the primary Linx benchmark
navigation surface.

| Path | Purpose |
| --- | --- |
| [`py_api`](py_api) | Active Python-facing TileOP correctness and golden-comparison flow. |
| [`tileop_layout`](tileop_layout) | TileOP layout and behavior checks that are not cataloged as primary benchmark suites. |

These directories still use the shared benchmark harness through
`benchmarks/common/Makefile.common`, but active benchmark entrypoints should be
added under `benchmarks/`.
