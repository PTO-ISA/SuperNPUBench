# FA fixed runtime and optional recovery r2

This payload reuses the immutable published gfsim f288a7053ddeeafabf42df45b292d04b47954022c6c5c1d990bde633ce40447a, loader, host libraries and configurations. Its source is SuperScalarModel 89b2366ea704fd75fd702440496439dadf5496be, recovered from base dd86f8a20b9161bdb1f836dc234c86d187d561e4 plus source/portable-model.patch. Repository access is private and requires collaborator permissions; fix/gfsim-tpack-issue1010 is a reachable branch, not a public access guarantee.

```sh
python3 RUNTIME/rebuild_model.py --model-repo MODEL_REPOSITORY --source-out NEW_SOURCE --build-out NEW_BUILD --recover-only
python3 RUNTIME/rebuild_model.py --model-repo MODEL_REPOSITORY --source-out OTHER_NEW_SOURCE --build-out OTHER_NEW_BUILD --cxx g++-11 --cc gcc-11
python3 RUNTIME/qualify_rebuilt.py --operator-root OPERATOR --runtime OTHER_NEW_BUILD/bin/gfsim --outdir FRESH_QUALIFICATION
```

Source recovery uses POSIX patch (including inside a parent Git worktree), verifies the patch and every postimage, and refuses reused output directories. The optional model build requires GCC/libstdc++; GCC 11.4.0 on Ubuntu22.04.5 is the historical qualified family. Clang/libc++ is not qualified. Build uses Release/O3/NDEBUG, prefix maps including softfloat, and -j4. Build identity is recorded separately; any new binary needs all four exact cycle/full-trace qualifications. A different compiler or host does not inherit the old binary's qualification.

This supplement tests patch/postimage recovery, including corruption rejection, but does not perform a new complete model source build. The old portable runtime is reused for actual timing replay. --data-only on verify.py checks payload bytes without executing its Linux loader. Model fixes/issues and limits remain recorded in the accompanying Model Git supplement; no single-fix attribution or mainline same-ELF comparison is inferred from timing reproduction.

## Mandatory guard overlay r1

Apply fa-runtime-r2-guard-overlay-r1-20261110.tar.gz after extracting the base r2 runtime, from its parent directory. The overlay replaces only README, two recovery/qualification scripts and the matching manifest; runtime/config/library bytes are unchanged. It prevents empty-case qualification, rejects output inside either payload before writing, verifies base files before recovery, and clears build environment overrides. Full new model build remains not_checked.
