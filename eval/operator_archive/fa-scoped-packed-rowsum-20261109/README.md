# FA source archive and diagnostic timing replay

This FA-only archive preserves the final8K scoped packed-rowsum source byte-for-byte and leaves default kernel integration unchanged. D2 compresses denominator rowsum using packed BF16 carriers and pair-local scope; the snapshot also moves V load after softmax. It approximates unscaled attention using BF16 log references and MXFP4/E8M0 probabilities, with FP32 PV accumulation. It is not exact FP32 softmax. The archived API patch belongs to Linx-TileOP-API; complete original ISA/compiler/API and operator commit/dirty closure remain unresolved.

Packed rowsum smoke and its negative control passed, but full attention failed6756/32768 outputs at atol1e-3/rtol2e-2. All six historical input readbacks matched and old/new compiler full outputs were byte-identical; BroadcastByteOffset issue history is background, not an established root cause. Historical8K AlgoD21488/scopedD220519 and64K AlgoD155269/early-unscopedD2258099 are diagnostic zero-input results. No final scoped64K result or clean-main model same-ELF/config/input A/B exists. Streamed JSON changes host memory/observability, not a claimed cycle optimization.

Source hashes are listed in identity.json. Matching four ELF/diss/source snapshots are separate operator Release assets; fixed model/config/libraries/recovery belong to the Model Release. This Git directory does not promise a complete operator/compiler rebuild. Source verification checks bytes only, not functional correctness.

## Download and run

Requires Linux x86_64 and Python>=3.9. Current operator-r2 defaults to a10GiB host virtual address-space limit (RLIMIT_AS), configurable with --memory-limit-gib; this is not a resident-RAM allocation. Provide enough RAM plus swap for the simulator, and a compatible host hard limit. No compiler or historical trace is needed. The operator asset owns four ELF/diss pairs and matching source snapshots; the Model runtime asset owns the fixed model, loader/libraries, native configs and source recovery recipe. Current publication/download status and exact hashes are in release_index.json / DELIVERY_STATUS; Required assets are publicly published and their actual downloaded bytes were verified.

- [Operator ELF/diss asset](https://github.com/PTO-ISA/SuperNPUBench/releases/download/work-archive-fa-20261109/fa-operator-r2.tar.gz) (SHA256 `6157ff881ed00ebf95a64196ba6725f57d8fcb1931f7516950c9ff13390a82b6`).
- [Model runtime/config asset](https://github.com/LinxISA/SuperScalarModel/releases/download/work-archive-fa-20261109/fa-runtime-r1.tar.gz) (SHA256 `7c6080abe2076038d5657f171e708118b4aa898b3b39d945635278aeaa6ab47f`).

```sh
curl -fL https://github.com/PTO-ISA/SuperNPUBench/releases/download/work-archive-fa-20261109/fa-operator-r2.tar.gz -o fa-operator-r2.tar.gz
curl -fL https://github.com/LinxISA/SuperScalarModel/releases/download/work-archive-fa-20261109/fa-runtime-r1.tar.gz -o fa-runtime-r1.tar.gz
tar -xzf fa-operator-r2.tar.gz
tar -xzf fa-runtime-r1.tar.gz
python3 fa-operator-r2/replay.py --operator-root fa-operator-r2 --model-root fa-runtime-r1 --case fa-8k-algod-21488-portable-r1 --outdir replay-fa-8k-algod --memory-limit-gib 10
```

Use a new output directory for each case. The runner verifies asset identities, copies configs, clears diagnostic environment contamination and records full compressed SwimLane, nocounters and cycle log. Optional historical trace assets are on the Model Release and are not replay dependencies.

| portable case | expected Total Cycles | actual downloaded replay in this snapshot |
|---|---:|---|
| `fa-8k-algod-21488-portable-r1` | 21488 | 21488, rc0 |
| `fa-8k-scoped-d2-20519-portable-r1` | 20519 | 20519, rc0 |
| `fa-64k-algod-155269-portable-r1` | 155269 | 155269, rc0 |
| `fa-64k-unscoped-d2-258099-portable-r2` | 258099 | 258099, rc0 |

Full attention numerical validation remains failed:6756/32768 out of tolerance. All four timing ELFs use static zero inputs; timing replay cannot promote correctness. The258099-cycle64K D2 is early unscoped source, not final scoped candidate. Final scoped64K and clean-main model same-ELF/config/input comparison are absent. Historical compiler/API/ISA closure and operator original commit/dirty gaps remain.
