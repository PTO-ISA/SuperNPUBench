# FAG supplement revision4

Adds exact-file-set/symlink preflight and explicit failure receipts to the fixed ELF replay runner, without replacing the default operator or old archive bytes. Use make_payload.py with the downloaded operator-r2 to assemble a hash-checked operator-r3; the Release also supplies the assembled package. The required Model runtime-r1 remains unchanged. release_assets.json fixes new asset identities and download receipts.

New local compact build produced the identical5ebaef32531ba25cb91389a31c1ad0184e54f9ee15151885909469d4b881409b ELF;631 actual include identities are in local_build_evidence.json. This is a local build result only. Shared toolchain/API/sysroot/ISA closure, nonzero same-run correctness and clean-main control are not accepted.

Model companion archive archives/ndf-fag-20261109-r4 owns run/config/model-source/issue evidence and223-plus atomic checks. capabilities.json separates local3-case replay, actual shared-download replay, model recovery/build, local operator build and old-output host checking. The optional Model nonzero-review-r1 package is prior-output rechecking against a model-derived predecessor, not fresh execution or independent golden correctness.

NDF v1.0-draft.7 is a local unpublished draft. Only version/hash references are shared; its text is deliberately excluded. All new assets use new names; historical Release and operator snapshots remain unchanged.
