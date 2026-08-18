## Summary

The `fixp` front-end emits `B.IOR [zero,%[LReluGpr]],[]` (macro `PTO_FIXP_IOR_2`) for the **LReLU-without-scalar-quant** case (`IorMode==2`). The LLVM MC instruction matcher has no pattern for a `B.IOR` operand list whose first slot is the `zero` token, so the inline-asm bundle fails to match with `error: Match Instruction Error!`. This makes the PTO-ISA-defined `IorMode==2` operand-stream form unbuildable, and the SuperNPUBench `lrelu_only` fixp microbench mode stays BLOCKED.

Only the `zero`-first-slot form fails; the sibling IorMode forms (1 = `[QuantGpr]`, 3 = `[QuantGpr, LReluGpr]`) match fine.

## Environment

- Toolchain: `linx_blockisa_llvm_musl`, clang 15.0.4
- `llvm-project` commit: `86959776b` `[LinxV5] Fix regclass spill/reload asymmetry and B.IOT %Z non-immediate handling` (latest worktree)
- `COMPILER_DIR`: `<linx_blockisa_llvm_musl>/bin`
- Reproducer: SuperNPUBench `microbenchmark/fixp/src/fixp_tmatmul.cpp`, mode `LRELU_ONLY`

## Reproduce

```bash
cd microbenchmark/fixp
make TESTCASE=fixp_tmatmul FIXP_MODE=LRELU_ONLY \
  COMPILER_DIR=<linx_blockisa_llvm_musl>/bin diss
```

Minimal C++ (the mode body):

```cpp
// LRELU_ONLY — LReLU without scalar quant -> B.IOR [zero, lrelu_gpr]
buf_t<__half, float> buf;
run_matmul<__half, float>(buf.a, buf.b, buf.d,
    [&](auto &tD, auto &tA, auto &tB) {
      TMATMUL(tD, tA, tB, fixp::keep_acc().lrelu(1));
    });
```

## Expected

`B.IOR [zero, LReluGpr],[]` is a valid PTO v0.58 `B.IOR` operand-stream form (IorMode==2: no scalar quant descriptor, LReLU leak descriptor in the second slot). The bundle should assemble.

## Actual

```
tileop-api/jcore/template_asm.hpp:3406:23: error: Match Instruction Error!
<inline asm>:9:1: note: instantiated into assembly here
B.IOR [zero,a1],[]
```

The matcher rejects the `B.IOR [zero,a1],[]` line. The rest of the bundle (`BSTART.CUBE TMATMUL`, `B.DATR`, `B.FPATR`, `B.IOT`) matches. `a1` is the GPR bound to `[LReluGpr]` (the leak descriptor).

## Root cause (front-end emit → backend match)

1. `fixp::keep_acc().lrelu(1)` ⇒ FPATR: `PreQuant=keep_acc` (no scalar/vector quant), `Relu=LReLU`.
   In `emit_fixp` the IorMode is computed as:
   ```cpp
   constexpr int IorMode = (HasScalarQuant ? 1 : 0)
                        | (Attr.Relu == FixpReluMode::LRelu ? 2 : 0);
   // lrelu_only: 0 | 2 = 2
   ```
2. Dispatch `PTO_FIXP_DISPATCH(PTO_FIXP_EMIT_LOCAL)` (`template_asm.hpp:3406`) → `PTO_FIXP_EMIT_LOCAL(0,0,2)` → bundle string includes macro `PTO_FIXP_IOR_2` (`template_asm.hpp:3043`):
   ```c
   #define PTO_FIXP_IOR_2 "B.IOR [zero,%[LReluGpr]],[]\n"
   ```
   `zero` is a literal token (no `%`-substitution); `[LReluGpr]` binds a real GPR (e.g. `a1`).
3. The MC instruction matcher has patterns for the other IorMode forms but **none for `B.IOR` with a `zero` first operand** → `Match Instruction Error!`.

## Contrast (proves the gap is specifically the `zero` first-slot form)

| IorMode | B.IOR form | bench mode | result |
|---|---|---|---|
| 1 | `B.IOR [a0],[]` | s_qf_s8 (scalar quant only) | PASS |
| 2 | `B.IOR [zero,a1],[]` | **lrelu_only** | **FAIL — Match Instruction Error** |
| 3 | `B.IOR [a0,a1],[]` | s8_lrelu (quant + LReLU) | PASS |

Verified with the latest worktree; full fixp microbench suite is `PASS=63 FAIL=1`, the single failure being this mode.

## Suggested fix location

The MC instruction-selector needs a pattern (or canonical acceptance) for `B.IOR` whose first operand is the `zero` token/register, consistent with the `B.IOR` canonicalization tracked in #38. If the bare `zero` token spelling is itself to be deprecated per #38's "numeric `0` placeholders are not canonical assembly operands", then the front-end macro `PTO_FIXP_IOR_2` (in `tileop-api/jcore/template_asm.hpp`) and the backend matcher should be changed together so the LReLU-without-quant path emits a form the backend matches.

## Related

- #38 — [PTO v0.58][MC] Canonicalize B.IOS/B.IOT/B.IOR assembly and disassembly (B.IOR scope; defines `R0=zero`).
