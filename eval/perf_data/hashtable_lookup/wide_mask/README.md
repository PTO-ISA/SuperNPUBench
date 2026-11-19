# Hashtable wide PredicateCell fixture

This standalone source fixture builds the same 6,144-query, 70%-load table
workload at genuine ElementTile widths of 32, 64, 128, 256, and 512 lanes. Each PE
owns one contiguous 1,536-query slice, which is divisible by every width.

`build.py` builds the selected freestanding ELFs (128/256/512 by default) and both normal and expanded
disassemblies with pinned compiler/API/sysroot inputs. For example:

```bash
python3 eval/perf_data/hashtable_lookup/wide_mask/build.py \
  --llvm /path/to/llvm \
  --api /path/to/Linx-TileOP-API \
  --sysroot /path/to/musl/sysroot \
  --reference eval/perf_data/hashtable_lookup/hash_6144_PE4.elf \
  --output /tmp/hashtable-wide
```

Use `--lanes 128 256 512` for the wide comparison; 32 and 64 remain available. The
script records compiler/linker hashes, version text, commands, reference SHA,
and output ELF hashes in `build-provenance.json`.

The fixture reuses the exact table/query/golden byte ranges embedded in
`../hash_6144_PE4.elf`; it does not regenerate random data. Exported symbols
include `table`, `queries`, `golden`, `output`, `sync`, `errors`, `checked`,
`wide_mask_run`, `wide_mask_check_results`, `wide_mask_checker_load_begin`,
`wide_mask_checker_load_end`, and `wide_mask_finisher_store`.

`fixture.cpp` exports the common `_start` entry itself. It calls the same SPMD
fixture on every PE and, after each PE publishes its finisher value, remains in
a side-effect barrier loop. No libc startup, PT_INTERP, or worker runtime is
required.

The kernel retains ordinary `#pragma pto element for` and scalar `if`
statements. Wide liveness is numeric U32 state: `TROWMAX` reduces physical
32xC to valid 32x1, then `TCMPS<NE>(..., 0)` supplies the scalar loop test.
No kernel is implemented as repeated 32-lane host loops.

This archived fixture is deliberately full-count-only. Every PE receives
1,536 queries, divisible by every admitted width. `lookup` traps when
`valid != Lanes`; this source does not claim partial U64 ElementTile transport
or arbitrary tail support. A separate tail test is required before removing
that guard.

Validation must cover both ends of every paired U64 column: construct keys
whose high 32-bit words differ while low words match, and vice versa. The
64/128/256/512 disassembly must show genuine `32x2`/`32x4`/`32x8`/`32x16` carriers and not repeated
32-lane host loops. Numerical acceptance requires all 6,144 golden values,
both output guards, zero `errors`, `checked == 1`, four `0x5555` finishers, and
the exported checker-load witness under fixed and random model memory timing.

The 256/512-lane predicates occupy two/four 128-byte M32 CELLs. The frontend
keeps one element loop over the complete Tile. Internally, the timing model
returns larger indexed results in windows of at most eight physical beats,
with global beat IDs and acknowledgement-based credit release. The 8-entry
RWDB and per-PE 256 KiB Tile pool are unchanged. The larger profile requires
pre-bound local destinations; lazy destinations spanning more than one
window are explicitly unsupported.

`--query-repeats 4` expands the workload to 24,576 queries by repeating the
archived query/golden byte sequences four times against the same table. This
is a larger repeated-input stress case, not a newly sampled independent data
set; comparisons must keep the same repeat count. It still checks every output
and both guards. Build outputs include a `.witness.json` with disassembly-derived
checker PCs bound to the ELF hash and query count. Pass it to the model checker
with `--witness-json`, and use the same `--query-repeats` value.

Current LLVM15 branch versions enforce complete macro coverage in the default
objdump view and return nonzero when any bundle cannot be folded. This builder
preserves that failure. Use `--expanded-only` explicitly when validating the ELF
through physical disassembly while macro coverage is incomplete; that mode is
recorded in provenance and does not claim a complete macro view. The generated
witness metadata always comes from the physical disassembly.
