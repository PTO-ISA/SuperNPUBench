# Transpose U16: full refresh 2026-11-19

All15 gfrun RES_CHECK file exports match independent NumPy transpose golden byte-for-byte. All15 gfsim performance runs complete; numerical and performance are separate ELFs/runs. gfsim numerical file export remains blocked by SuperScalarModel#1306. No claim of gfsim numerical validation.

Compiler13b51d9f7d7f71dcaa83b011e3dda68df1ca5e5a; exact compiler/model hashes in the downloadable archive identity.json. FourPE,1.8GHz,latency=cycles/1800; frozen profile in the downloadable archive configs/fourpe.conf. Numeric input U16(((i*1009)^(i>>16)^32768)&65535); performance zeroBSS as before. No target-side scalar numeric check or initialization loop. No current guard/alias/multiple-pattern coverage claimed.

|Input → Output|Fixed shape cycles / μs|Gather-plane cycles / μs|TLOAD cycles / μs|gfrun numeric(all3)|Target μs|
|---|---:|---:|---:|---|---:|
|129×22×120 → 120×22×129|483909 / 268.838|88280 / 49.044|470147 / 261.193|PASS|—|
|8×4096×3 → 8×3×4096|63365 / 35.203|12310 / 6.839|51659 / 28.699|PASS|9.63|
|1476×32 → 32×1476|45361 / 25.201|7088 / 3.938|34468 / 19.149|PASS|20.29|
|8×64×4×16×7 → 8×16×4×64×7|271510 / 150.839|49933 / 27.741|133065 / 73.925|PASS|28.55|
|8×8×64×64 → 8×64×8×64|87492 / 48.607|24296 / 13.498|15542 / 8.634|PASS|17.07|

All15 cycle deltas vs r4 are0. Targets are historical user-provided goals, not simulator results.

|Case|Method|Logical tile|Physical tile|Layout|
|---|---|---|---|---|
|gather_plane1|gather_plane|32x32 full rectangles|32x32|M32 full; M32 TLOAD24-row tail, M16 not used at24|
|gather_plane2|gather_plane|32x16 flattened plane|32x16|M32|
|gather_plane3|gather_plane|32x32 full rectangles|32x32|M32|
|gather_plane4|gather_plane|32x14 flattened plane|32x14|M32|
|gather_plane5|gather_plane|32x16 flattened plane|32x16|M32|
|tload1|tload|32x1|32x2|M32; M16 row tail|
|tload2|tload|32x1|32x2|M32|
|tload3|tload|32x1|32x2|M32; M16 row tail|
|tload4|tload|32x7 logical; transfers32x4 +32x2 +32x1|32x4 +32x2 +32x2|M32|
|tload5|tload|8x64|16x64|M16|
|u16_shapes1|u16_shapes|32x1|32x2|M32|
|u16_shapes2|u16_shapes|32x1|32x2|M32|
|u16_shapes3|u16_shapes|32x1|32x2|M32|
|u16_shapes4|u16_shapes|8x7 logical; transfers8x4 +8x2 +8x1|32x4 +32x2 +32x2|M32|
|u16_shapes5|u16_shapes|1x64|32x64|M32|

## GitHub delivery index

This page is the post-freeze upload index, not the frozen package README. Full sources, binaries, logs, configuration, references and reproduction tools are in the package; paths mentioned above refer to package members.

[Release](https://github.com/PTO-ISA/SuperNPUBench/releases/tag/transpose-u16-20261119-r5) · [Complete archive](https://github.com/PTO-ISA/SuperNPUBench/releases/download/transpose-u16-20261119-r5/transpose-r5-frozen-local-20261119.tar.gz) · [SHA256 file](https://github.com/PTO-ISA/SuperNPUBench/releases/download/transpose-u16-20261119-r5/transpose-r5-frozen-local-20261119.tar.gz.sha256)

Package size: 104325923 bytes. SHA256: `e09e9be1ee45d1584bf713a5fd84475c07c12a303f3ab504656ecb0be62046b6`.

Delivery status: uploaded; remote download/replay not performed. Independent review deferred by user. NDF draft.8 acceptance remains incomplete. Local extraction and representative numeric/performance/byte-identical rebuild checks passed; these are not shared or independent replay.

The package contains 471 inventoried files and excludes compiler/model runtime binaries. Exact installed tools remain required by the runner. The previously prepared portable model has different build flags (including NDEBUG) and is NOT used or bundled. See package NDF_REMAINING.md and validation/model_runtime_difference.json for limitations. No requirement to download/replay before this upload has been imposed; successful upload does not itself establish reproducibility.

Operator evidence is published here. Model logs/traces are preserved within the unchanged evidence package; this is not a separate authoritative model runtime release. Cross-repository dependency delivery remains pending. The Release tag identifies the preexisting repository base; the frozen operator sources are identified by the package source snapshot and checksums, not by that tag alone.
