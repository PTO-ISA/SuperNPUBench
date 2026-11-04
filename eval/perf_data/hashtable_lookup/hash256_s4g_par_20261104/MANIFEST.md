# 109,193-cycle build: manifest

| file | description | SHA256 |
|---|---|---|
| `src/kernel.hpp` | pipeline kernel (256-lane, 2 slots/parallel, slot indices, empty-first, no-wrap) | `5699c3d585e375d567d0456745f8554964dc4247a59e317fc9e26637addbdc65` |
| `src/fixture.cpp` | test fixture; guards the table with 128 mirrored entries (2 KB) | `4b1fb90e084b3bbfe5a1f48ad26328877e5bb794812227dc1a77f0d34dd94733` |
| `src/link.ld` | link script | — |
| `src/hash256.elf` | compiled ELF (PR stack) | `eb4c3bae829e44db631e7892069561b67c05d6bdc2654aaefdb28334f3a3d563` |
| `src/kernel.expanded.diss` | physical disassembly (`llvm-objdump -d --disassembler-options=no-tile-macros`) | `555f3f4bc5609e3b9eee0e286ff0a4c814af3de357889422d8555b99ec791484` |
| `swimlane_final.json.gz` | Perfetto trace of this ELF (gunzip, open at ui.perfetto.dev) | `027701b4c3c7cc9bf82579b4c250dd11ebb2ee66cda28a86ed6b77d3150c4280` |
| `final-result.json` | checker result | `769e9ab85db6d10526646b6ef4a1be69a2a1336c01ee4c0ab9a0e0b18732cb04` |
| `final-metrics.txt` | kernel 109,193 / first TLOAD 290 / last TSTORE 109,483 / whole fixture 171,646 | — |

## Reproduce

Build (PR compiler + API, guarded fixture):
```sh
CLANG=/tmp/llvm-pr146/build-hash/bin/clang++
SRC=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash256_s4g_par_20261104/src
REF=/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/eval/perf_data/hashtable_lookup/hash_6144_PE4.elf
$CLANG --target=linx64v5-unknown-linux-musl -mlxbc -fenable-matrix -O2 -std=c++20 \
  -D__linx -DENABLE_TENSOR_INSTR -ffreestanding -fno-exceptions -fno-rtti \
  -fno-vectorize -fno-slp-vectorize \
  -mllvm -linxv5-enable-generic-element=true \
  -mllvm -enable-all-vector-as-tilereg=true \
  --sysroot=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot \
  -nostdinc++ -isystem /tmp/lane128-prcfg -isystem /tmp/llvm-pr146/libcxx/include \
  -I/tmp/api-pr303/include -I/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench \
  -DHASH_DATA_ELF_PATH="\"$REF\"" -DHASH_QUERY_REPEATS=1 -DHASH_WIDE_LANES=256 \
  -mllvm -linxv5-force-element-tile-mask=true -c $SRC/fixture.cpp -o /tmp/hash.o
/tmp/llvm-pr146/build-hash/bin/ld.lld -T $SRC/link.ld /tmp/hash.o -o /tmp/hash.elf
```

Run (PR model):
```sh
cd /tmp/ssm-pr1177
./bin/gfsim --test-finisher 1 -f /tmp/hash.elf --conf fourpe \
  -s softcore.multiThreadNum=4 -s cube.enable_internal_acc=true \
  -s bctrl.vec_cell_sched_enable=true -s core.soc_random=false \
  -s core.soc_enable=false -s core.read_bandwidth_limit=128 \
  -s core.write_bandwidth_limit=128 -s core.sharetreg_size_bytes=262144 \
  -s core.sharetreg_ssb_count=64 -t 1
```

Numerical check: `python3 /tmp/ssm-pr1177/scripts/check_hashtable_fixture.py
--elf /tmp/hash.elf --reference-elf "$REF" --model gfsim --output /tmp/check`.
