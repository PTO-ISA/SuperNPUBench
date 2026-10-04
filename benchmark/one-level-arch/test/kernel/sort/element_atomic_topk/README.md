# Element-wise atomic Top-K histogram

This test covers the standard-C lowering path requested for TLEA. Input values
are carried as `uint32_t` but are restricted to the unsigned 16-bit key domain.
The representation avoids an unrelated narrow-integer conversion dependency
while keeping the two-level radix algorithm identical to a `uint16_t` Top-K.

Each 128-element input chunk is loaded as one `CUBE_M32` Tile and split with
`TPARTVIEW` into four 32-element views. `TSHRS` and `TANDS` consume the views to
form the high and low radix digits. A canonical `#pragma linx elementwise` loop
then performs a normal C conditional and
`__atomic_fetch_add(&histogram[index], 1u, __ATOMIC_RELAXED)`. The compiler is
responsible for lowering the logical U32 index through TLEA to a U64 byte
offset before the indexed atomic memory operation.

The target test validates all 256 bins at both radix levels, exactly 37 output
elements, the complete Top-K multiset across a tied cutoff, and the returned
old value from every atomic. The old values for each bin must be exactly the
permutation `0..count-1`; unused and padded tail lanes must remain zero. The
777-element input crosses six full 128-element parent Tiles
and ends in a partial parent Tile whose inactive storage
contains values outside the documented key domain.

The same ELF also runs a dedicated coherence probe. Four additional exported
golden segments contain its input, final histogram, old-value array, and scalar
observations `[initial, after_atomic, after_scalar, failures] = [0, 9, 10, 0]`. A volatile scalar load first caches a zero histogram
line, a volatile scalar store makes bin zero dirty with value 7, and a
two-element call to `histogram_high8` performs two native masked atomic adds.
Their old-value Tile stores must contain 7 at column-buffer offset 0 and 8 at
offset 32, every inactive old-value lane must remain zero, and the histogram
must become 9. A later volatile scalar store/load pair must observe 10, proving
that the later scalar writer supersedes the queued tile commit. Any mismatch
sets status failure bit `0x20`.

The portable reference test also covers `K=0`, `K=1`, `K=N`, duplicate hot
bins, cutoff ties, an all-equal input, and partial-tile tails:

```sh
benchmark/one-level-arch/test/kernel/sort/element_atomic_topk/check_reference.sh
```

`generate_case.py --out <dir>` independently emits little-endian U32 input,
both histogram levels, sorted Top-K output, status, the four coherence probe
goldens, and a manifest mapping all nine segments to stable ELF symbols. Atomic old-value arrays remain runtime-order
dependent and are verified inside the ELF as exact per-bin permutations.

The end-to-end harness combines a freshly built compiler/backend with an
installed Linx musl runtime and current TileOP headers. It rejects an ELF
whose three histogram invocations do not each contain B.SUBVIEW, TSHRS/TANDS,
TCI, TCMPS, scalar predicate AND, exactly one TLEA, MGATHER.ADD, and the native
execution-mask binder. Both gfrun and gfsim must then run as one logical
thread/PE and match the independently generated input, histogram, Top-K, and
status and coherence memory segments. UART text is diagnostic only and is not used as the
correctness oracle:

```sh
COMPILER_DIR=/path/to/fresh-llvm-build/bin \
LINX_RUNTIME_ROOT=/path/to/linx_blockisa_llvm_musl \
API_INCLUDE=/path/to/Linx-TileOP-API/include \
SSM=/path/to/SuperScalarModel \
benchmark/one-level-arch/verification/run_element_atomic_topk.sh
```

`TARGET_TRIPLE`, `SYSROOT`, and `RESOURCE_DIR` can be overridden. Their
defaults are respectively `linx64v5-unknown-linux-musl`,
`$LINX_RUNTIME_ROOT/sysroot`, and
`$LINX_RUNTIME_ROOT/lib/clang/15.0.4`.

Each run writes a unique evidence directory with a `content_id` under
`output/verification/element_atomic_topk/`. It contains the frozen ELF, disassembly,
symbol table, compiler/model logs, both memory dumps, independent goldens, and
`provenance.json` with commands, configs, tool and ELF hashes, repository
revisions/diff hashes, runtime-library hashes, and the TileOP header hash.

The general sort `compile.all` and repository compile smoke keep their legacy
behavior. Set `ELEMENT_ATOMIC_TOPK=on` together with `LINX_RUNTIME_ROOT` and
`API_INCLUDE` to opt this draft PTO 0.59 case into those broader scripts.
