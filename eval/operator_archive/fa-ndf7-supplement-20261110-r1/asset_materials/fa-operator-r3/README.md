# FA timing and recovery supplement r3

Linux x86_64, Python >=3.9. The operator and runtime archives are separate. Replay needs no compiler, historical trace or external input: these four freestanding ELFs use zero-initialized static buffers. This is timing qualification only. The unresolved independent numerical comparison has 6756/32768 mismatches; rechecking its captured data does not validate these timing ELFs.

From an unrelated working directory:

```sh
python3 OPERATOR/replay.py --operator-root OPERATOR --model-root RUNTIME --case fa-8k-scoped-d2-20519-portable-r1 --outdir FRESH_OUTPUT
```

Case IDs and expected Total Cycles:

- `fa-8k-algod-21488-portable-r1`: 21488.
- `fa-8k-scoped-d2-20519-portable-r1`: 20519.
- `fa-64k-algod-155269-portable-r1`: 155269.
- `fa-64k-unscoped-d2-258099-portable-r2`: 258099. This early unscoped case is a separate algorithm variant, not final scoped D2 at 64K.

The entry verifies all declared files and configs, rejects symlink dependencies and output inside either payload, clears inherited model/loader/Python environment variables, and verifies simulator exit, cycles and the exact uncompressed full trace hash. Both full and nocounters trace are gzip compressed while capturing. `--preflight` checks files only. Linux host address-space cap defaults to 10GiB; it does not reserve that memory or change model timing. Insufficient RAM/swap is a failure, not a passing result.

## Optional operator rebuild

```sh
python3 OPERATOR/build.py --case fa-8k-scoped-d2-20519 --toolchain-root TOOLCHAIN --outdir FRESH_BUILD
python3 OPERATOR/replay.py --operator-root OPERATOR --model-root RUNTIME --case fa-8k-scoped-d2-20519-portable-r1 --qualification-mode --rebuilt-elf FRESH_BUILD/fa-8k-scoped-d2-20519.elf --outdir FRESH_QUALIFICATION
```

`TOOLCHAIN/build/bin` must contain exactly the pinned clang++, ld.lld and llvm-objdump; see dependencies/identity.json. Historical API files, resource headers, harness and the 606 actually consumed sysroot headers are included, with source commits and file hashes. No external sysroot, CRT or libc is used in these freestanding recipes. The optional compiler source build recipe is not rerun here and cannot guarantee those tool binary hashes. Exact compiler binary acquisition remains an explicit gap; a source commit alone is not a qualified binary. Normal ELF replay does not require downloading a toolchain.

`build_recipes.json` preserves four actual timing commands; `build.py` emits ELF and matching diss plus an identity receipt. Prefix maps remove personal build paths. Builds alone do not establish numerical validity. See the Git archive supplement for actual final build and replay records.

## Captured numerical failure

```sh
python3 OPERATOR/recheck_failure.py
```

This standard-library checker verifies six inputs/readbacks, output/golden identities, tolerances, the historical 6756/32768 failure, and zero-output/NaN negative controls. The separately identified RES_CHECK ELF is not one of the four timing ELFs. No fresh numerical model execution is claimed. Source model recovery and optional rebuilt-runtime qualification live in the separate runtime payload. Prior r1/r2 assets and historical results remain immutable.
