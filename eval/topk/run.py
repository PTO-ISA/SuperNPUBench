#!/usr/bin/env python3
"""Run the portable FP32 topk fixture in an isolated output directory."""
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
ELF_NAME = 'kernel_multi_thread_topk_tiled_topk_tiled_PE4.elf'
INPUTS = ('shape.bin', 'input.bin', 'starts.bin', 'ends.bin', 'golden.bin')
OUTPUTS = ('input_readback.bin', 'output.bin', 'errors.bin')


def verify_inputs(run):
    batch, cols, topk = struct.unpack('<3q', (run / 'shape.bin').read_bytes())
    if not (1 <= batch <= 4 and 1 <= cols <= 131072 and 1 <= topk <= min(cols, 1024)):
        raise ValueError('invalid shape')
    values = struct.unpack(f'<{batch * cols + 32}f', (run / 'input.bin').read_bytes())
    starts = struct.unpack(f'<{batch}i', (run / 'starts.bin').read_bytes())
    ends = struct.unpack(f'<{batch}i', (run / 'ends.bin').read_bytes())
    expected = []
    for row, (start, end) in enumerate(zip(starts, ends)):
        if not 0 <= start < end <= cols or end - start < topk:
            raise ValueError('invalid row range')
        indices = sorted(range(start, end), key=lambda i: (-values[row * cols + i], i))[:topk]
        expected.extend(sorted(indices))
    golden = struct.pack(f'<{len(expected)}i', *expected)
    if golden != (run / 'golden.bin').read_bytes() or not any(expected):
        raise ValueError('independent golden mismatch or vacuous fixture')
    return batch, topk, expected


def verify_outputs(run, batch, topk, expected):
    if (run / 'input_readback.bin').read_bytes() != (run / 'input.bin').read_bytes():
        raise ValueError('input readback mismatch')
    errors = struct.unpack(f'<{batch}i', (run / 'errors.bin').read_bytes())
    if any(errors):
        raise ValueError(f'device error flags: {errors}')
    output = struct.unpack(f'<{batch * topk}i', (run / 'output.bin').read_bytes())
    for row in range(batch):
        lo, hi = row * topk, (row + 1) * topk
        if sorted(output[lo:hi]) != expected[lo:hi]:
            raise ValueError(f'row {row}: output index set mismatch')
    print(f'PASS readback {(run / "input.bin").stat().st_size} bytes; '
          f'output {len(output) * 4}/{len(output) * 4} bytes exact; '
          f'errors {batch * 4} bytes zero')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gfsim', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    model = args.gfsim.resolve()
    run = args.outdir.resolve()
    if run == BUNDLE or run == BUNDLE / 'inputs' or BUNDLE / 'inputs' in run.parents:
        parser.error('outdir must not overwrite the shipped bundle or inputs')
    # Verify all published payloads before copying or running them.
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        data = (BUNDLE / name).read_bytes()
        if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError(f'payload mismatch: {name}')
    run.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copyfile(BUNDLE / 'inputs' / name, run / name)
    for name in OUTPUTS:
        (run / name).unlink(missing_ok=True)
    batch, topk, expected = verify_inputs(run)
    command = [sys.executable, str(BUNDLE / 'run_swimlane.py'),
               '--gfsim', str(model), '--elf', str(BUNDLE / ELF_NAME),
               '--conf', str(model.parent.parent / 'configs/fourpe.conf'),
               '--outdir', str(run), '--name', 'topk', '--timeout', str(args.timeout)]
    (run / 'command.json').write_text(json.dumps({'argv': command, 'cwd': str(run)}, indent=2))
    # gfsim needs a real run directory to resolve CHK_DIR=".". Its config path
    # is supplied explicitly from the collaborator's model, never baked into ELF.
    try:
        with (run / 'model.log').open('w') as log:
            result = subprocess.run(command, cwd=run, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout + 10)
    except subprocess.TimeoutExpired:
        print(f'FAIL simulator timeout; log: {run / "model.log"}')
        return 1
    full = run / 'topk.json'
    if result.returncode:
        print(f'FAIL simulator rc={result.returncode}; log: {run / "model.log"}')
        return 1
    if not full.exists():
        raise ValueError('missing required SwimLane trace')
    summary = run / 'topk_swim.log'
    for line in summary.read_text().splitlines():
        if line.startswith('Total Cycles'):
            print(line)
    verify_outputs(run, batch, topk, expected)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, struct.error) as error:
        raise SystemExit(f'FAIL {error}')
