# FA NDF draft.7 archive supplement — 2026-11-10

This additive revision supplements PR284/1222. It preserves four historical zero-input timing cases and unresolved numerical failure. It does not change the benchmark mainline kernel or claim a clean-mainline model speedup. The specification is local unpublished v1.0-draft.7 (SHA25666462847e345c6fe9a23a0b9f356d5fb35d7902a62d6aa6f1d9ab218e0e5f394); its original text is not published by this supplement.

The archive owner for this supplement is Codex on behalf of the workspace author; long-term numerical/model ownership remains unassigned. Task identity: fa-ndf7-supplement-20261110. Each historical case and new output directory has a separate result/run identity. The complete executable files remain in Release; asset_materials/ is a review-only subset and cannot be run as a standalone payload.

## Get the fixed payloads

Linux x86_64, Python>=3.9, GitHub CLI authenticated with private SuperScalarModel read access. Download operator and runtime plus mandatory small guard overlay:

```sh
gh release download work-archive-fa-20261109 -R PTO-ISA/SuperNPUBench -p fa-operator-r3-ndf7-20261110.tar.gz
gh release download work-archive-fa-20261109 -R LinxISA/SuperScalarModel -p fa-runtime-r2-ndf7-20261110.tar.gz -p fa-runtime-r2-guard-overlay-r1-20261110.tar.gz
```

Verify bytes/SHA256 against evidence/new_release_assets.json and evidence/overlay_release.json before extraction. Extract operator, base runtime, then the guard overlay, into the same parent directory. The overlay replaces only recovery/qualification scripts, README and the matching manifest. It preserves gfsim/config/library bytes and prevents empty-case qualification and output inside the payload. Old r1/r2 published files remain immutable.

```sh
tar -xzf fa-operator-r3-ndf7-20261110.tar.gz
tar -xzf fa-runtime-r2-ndf7-20261110.tar.gz
tar -xzf fa-runtime-r2-guard-overlay-r1-20261110.tar.gz
python3 fa-operator-r3/replay.py --operator-root fa-operator-r3 --model-root fa-runtime-r2 --case fa-8k-scoped-d2-20519-portable-r1 --outdir NEW_RUN
```

No compiler or historical trace download is needed for ELF replay. The runner copies its 22 pinned configurations into NEW_RUN/configs; model --conf perf.conf resolves there. Manual launches without that cwd/config convention are unqualified. The helper explicitly supports Linux; it refuses unsupported RLIMIT_AS hosts instead of silently discarding a resource limit. Both full and nocounters SwimLane outputs are gzip compressed. The default10GiB address-space cap is configurable and does not reserve resident memory. Insufficient host RAM/swap is a failure.

## Operations and evidence

Read capabilities.json before choosing an operation. checks.json contains individual applicability, observation, evidence identities and unresolved review fields; it is not an overall NDF conformance certificate. inventory.json distinguishes shared assets, local tools, unknown historical identities and genuine gaps. gaps.json retains the reason, impact, owner and next step for incomplete work.

Four timing ELFs were locally rebuilt from the bundled API/resource/harness/606sysroot headers with explicit hash-pinned compiler tools; all are byte-identical to their historical ELF. The compiler binaries are caller-supplied, not an included SDK; a fixed source commit and untested source-build recipe do not guarantee exact tool hashes. Newly emitted diss and build receipts are retained separately. Model patch tests cover seven changed files in ordinary and enclosing-Git directories, not a new full model build. GCC/libstdc++ is required; historical evidence identifies GCC11.4.0, correcting the review's unsupported g++12 attribution. The build's cmake environment is cleared, but discovery and recovery subprocesses still inherit some host environment: no hermetic model-build claim is made.

Run the host checker with a new output record (the frozen operator README example omits required --out):

```sh
python3 fa-operator-r3/recheck_failure.py --out NEW_NUMERICAL_AUDIT.json
```

The packaged host checker recounts the historical6756/32768 numerical failure with independent golden and rejects zero/NaN controls. Its hosted RES_CHECK ELF is distinct from these four timing ELFs. It is not fresh numerical execution. No timing result grants numerical acceptance.

## Model changes and issues

The exact previous composition is preserved under previous_archive/ for historical review, including its stale access wording; this page supersedes that wording. Model access is private. Current issue states are in evidence/issue_status_current.json. Raw TPACK/local-parent lifecycle fixes relate to issue1010/PR1057; retire drain to issue1053/PR1054. Both PRs merged and issues closed. Full JSON streaming changes host observability/memory use; it is not a new simulated-cycle optimization. API TPACK emit is an interface change, separate from model fixes. BroadcastByteOffset issues926/979 and PR957 are numerical background, not proven causes or repairs for the remaining FA failure.

The four cases bind FA-M01 (TPACK), FA-M02 (retire drain), FA-M03 (streaming) and FA-M04 (API emission); evidence/composition_and_scope.json records each enabled set and lifecycle limits.

Clean-mainline same-ELF/config/input and per-mechanism causal A/B remain missing. Reproducing historical cycles does not isolate a model gain. 8K final scoped D2 and 64K early unscoped D2 are different variants; no final scoped64K result is offered. Only frozen Sq256/Skv8192 or65536/D128/Dv128 shapes, four PE, tileQ128/tileKV256 are described; generic tails/layout/alias/NaN contracts remain unaccepted.

## Read this metadata

asset_materials/ contains exact review copies of selected Release scripts/manifests. previous_archive/ contains selected old archive records. evidence/ holds actual receipts and derived logs; provenance.json records raw versus published hashes when personal paths were replaced with tokens. A token denotes provenance, not a required local directory. Complete binary/header/input closures are in the fixed Release assets, never inferred from this small Git subset. Optional historical large traces remain separate attachments and are not replay dependencies. New traces are retained in the producer workspace; their hashes and historical equivalence are in new run receipts.

## Validation in this revision

All three new assets were actually downloaded and SHA/size checked. A sibling reviewer executed the new8K scoped runner from an unrelated cwd with polluted host environment:20519cycles/rc0, full trace101301625B exactly a330ed9b... . This is the only new accepted gfsim case in this supplement. The prior four shared runs remain valid against unchanged current ELF/runtime/all22config identities; see evidence/historical_current_identity_binding.json. Do not read this as four new-runner executions.

The two extra64K attempts were deliberately interrupted after reviewing the larger Linux resource requirement; they have no Total Cycles and are excluded from acceptance. The revised gzip-nocounters path was instead tested offline on the two complete historical full traces (859090794/1020145527B): both produced byte-identical uncompressed nocounters versus their historical counterparts. This checks compression/filtering, not a new model run.

New8K scoped peak RSS was2354732KiB. Historical Linux64K unscoped sampled RSS reached5579308KiB and ran about24.5minutes; that sampled value is a lower bound on true peak, not a recommended memory allocation. Independent model runs should be scheduled with enough RAM/swap. Resource measurements describe the publisher host, not simulated performance.
