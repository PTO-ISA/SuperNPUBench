# FAG standalone source archive and Release replay

Required assets are publicly published; actual downloaded bytes are verified. 3/3 actual downloaded zero-input timing cases have completed receipts. Each reproduced rc0, historical cycles and complete annotation-normalized full-trace fields/order/metadata (compact6239 fag_direct labels, slots7682160 fag labels;89646 unannotated); nocounters event/field/metadata equivalence is checked separately from raw serialization. Source files remain unchanged and outside default integration.

Requires Linux x86_64 and Python>=3.9; no compiler or historical trace is needed. SuperNPUBench owns the two unique ELF/diss pairs (three case configs); Model owns two fixed runtime variants with a single loader/library closure, configs and source recovery recipe. Exact hashes, IDs and current publication state are in release_index.json / DELIVERY_STATUS.

[Operator asset](https://github.com/PTO-ISA/SuperNPUBench/releases/download/work-archive-fag-20261109/fag-operator-r2.tar.gz); [Model runtime/config asset](https://github.com/LinxISA/SuperScalarModel/releases/download/work-archive-fag-20261109/fag-runtime-r1.tar.gz).

```sh
curl -fL https://github.com/PTO-ISA/SuperNPUBench/releases/download/work-archive-fag-20261109/fag-operator-r2.tar.gz -o fag-operator-r2.tar.gz
curl -fL https://github.com/LinxISA/SuperScalarModel/releases/download/work-archive-fag-20261109/fag-runtime-r1.tar.gz -o fag-runtime-r1.tar.gz
tar -xzf fag-operator-r2.tar.gz
tar -xzf fag-runtime-r1.tar.gz
python3 fag-operator-r2/replay.py --operator-root fag-operator-r2 --model-root fag-runtime-r1 --case fag-compact29028-portable-r2 --outdir replay-fag-compact
```

Use a new output directory for each case. Runtime/config hashes are cross-pinned by the operator manifest. The runner clears inherited diagnostic settings, fixes per-case environment, verifies config override echoes, rc/cycles and full-trace canonical identities after only exact documented historical annotation inverse, and compares all nocounters events/metadata with the filtered full trace plus historical semantic digest. Different raw nocounters hashes are reported as different serialization, never as byte equality. Every run retains cycle log, full gzip, lossless nocounters gzip and identities. Optional historical trace assets are separately available, not run dependencies; their downloads are not verified.

| portable case | historical expected cycles | actual downloaded observation |
|---|---:|---|
| `fag-compact29028-portable-r2` | 29028 | 29028, rc0 |
| `fag-prefetch89646-portable-r2` | 89646 | 89646, rc0 |
| `fag-sl2_768_88050-portable-r2` | 88050 | 88050, rc0 |

All timing ELFs use static zero input. Sq1024 separate historical exact checks and Sq4096 separate frozen-baseline equivalence do not constitute same-run TimingSim numeric acceptance; independent Sq4096 dQ/dK host-golden differences remain unresolved. Same-ELF clean-main model comparison, original operator compiler/API/ISA closure and independent S/numerical acceptance remain gaps. Slots768 adds32KiB SL2 data capacity: a hardware assumption, not a kernel or issue fix.
