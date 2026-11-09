# NDF operator source snapshots, 2026-11-09

Experimental source handoff accompanying SuperScalarModel `archive/workshop-20261109` at `2547509024c1ed8ff00b0a89a27d8107f637fa9a`. That branch contains model changes and an incomplete archive scaffold; it does not contain these operator changes. Base SuperNPUBench main is `c1d9a4f982d218bd14605376fa8c72eb105887c2`.

These are source snapshots for review, separate from the default build/test entries. This is not the planned `ops-20261109` release tag, a complete NDF result bundle, or numerical/performance acceptance. Each operator records its own failures and evidence limits. No model source change is transplanted into this repository.

Run `python3 verify_sources.py` from any directory to check the snapshot byte identities. This checks files, not compiler admission, functional correctness, performance, or independent reproduction. Dependencies and large historical artifacts are explicitly incomplete.
