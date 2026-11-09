# Standalone FA source archive

This package contains only FA; it is split from mixed PR281. See README.md for historical observations and known correctness limits.

This draft provides source snapshots and an integrity index only. No ELF, disassembly, runtime, input/golden, effective configuration, trace, or portable shared replay is delivered here. Release delivery and shared replay remain pending. No new build or simulation was performed. This is an incomplete archive, not NDF-complete acceptance.

Run `python3 verify_sources.py` from any working directory to verify the original archived material only. The verifier does not build or execute kernels and is not a correctness test.
