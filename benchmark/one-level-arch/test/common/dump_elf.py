#!/usr/bin/env python3
"""Preserve macro diagnostics; fall back only for unsupported TileOp macros.

A failed physical disassembly or unrelated objdump failure remains fatal.
The .diss.mode sidecar distinguishes macro output from physical fallback.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def dump(objdump, elf):
    output = Path(str(elf) + '.diss')
    mode = Path(str(output) + '.mode')
    errors = Path(str(output) + '.macro-errors')
    mode.unlink(missing_ok=True)
    errors.unlink(missing_ok=True)
    with output.open('wb') as stream:
        result = subprocess.run([objdump, '-dl', str(elf)], stdout=stream, stderr=subprocess.PIPE)
    if result.returncode == 0:
        mode.write_text('macro\n')
        if result.stderr: sys.stderr.buffer.write(result.stderr)
        return 0
    errors.write_bytes(result.stderr)
    if b'no complete macro instruction form' not in result.stderr:
        sys.stderr.buffer.write(result.stderr)
        return result.returncode
    print(f'WARNING: macro rendering unavailable for {elf}; using physical disassembly. '
          f'Diagnostics: {errors}', file=sys.stderr)
    with output.open('wb') as stream:
        result = subprocess.run([objdump, '-dl', '--disassembler-options=no-tile-macros', str(elf)],
                                stdout=stream, stderr=subprocess.PIPE)
    if result.stderr: sys.stderr.buffer.write(result.stderr)
    if result.returncode == 0: mode.write_text('physical-fallback\n')
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objdump', required=True)
    parser.add_argument('--elf', required=True, type=Path)
    args = parser.parse_args()
    sys.exit(dump(args.objdump, args.elf.resolve()))
