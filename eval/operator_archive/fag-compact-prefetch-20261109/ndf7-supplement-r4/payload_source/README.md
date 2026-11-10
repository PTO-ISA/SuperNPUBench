# FAG operator replay revision 3

Pairs with the unchanged FAG runtime-r1 Release. The three cases are zero-input timing diagnostics. No same-run numerical acceptance or clean-main speedup is claimed. New runner adds an exact file-set/symlink guard and --preflight; preflight verifies identities only and never counts as replay. All actual runs require a new external output directory and generate log/full gzip/nocounters trace.

Run `python3 replay.py --operator-root . --model-root <runtime-r1> --case fag-compact29028-portable-r3 --outdir <new-directory>`; replace the case with fag-prefetch89646-portable-r3 or fag-sl2_768_88050-portable-r3. Add --preflight for identity-only verification. Resource/config changes are rejected before simulation by the cross-bound configuration hashes; inherited feature variables are removed.

Historical operator-r2, runtime-r1 and all original bytes remain unchanged. New payload still requires actual replay before release acceptance. Source reconstruction and nonzero numerical verification are separate operations.
