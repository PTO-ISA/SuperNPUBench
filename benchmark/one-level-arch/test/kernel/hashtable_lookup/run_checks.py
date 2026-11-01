#!/usr/bin/env python3
"""Compile and check real gfrun output, failing on crashes or missing PASS."""
from pathlib import Path
import subprocess
import tempfile

CASE = Path(__file__).resolve().parent
MODEL = Path('/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl')

def main():
    root = Path(tempfile.mkdtemp(prefix='hash-element-check-'))
    print(f'Artifacts: {root}', flush=True)
    for name, options in [('load80', ''), ('full', '-DHASH_FULL_TABLE=1'),
                          ('single', '-DHASH_FULL_TABLE=1 -DHASH_CAPACITY=1')]:
        build = root / name
        build.mkdir()
        with (build / 'compile.log').open('w') as log:
            subprocess.run([str(CASE / 'compile.all'), f'OBJ_ROOT={build}',
                            f'CC_OPTS={options}', 'res_check=on'], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        elf = build / 'kernel/hashtable_lookup/elf/kernel_hashtable_lookup_hashtable_lookup_PE4.elf'
        with (build / 'gfrun.log').open('w') as log:
            result = subprocess.run([str(MODEL / 'bin/gfrun'), '-s',
                                     'softcore.multiThreadNum=4', '-f', str(elf)],
                                    cwd=MODEL, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=120)
        text = (build / 'gfrun.log').read_text()
        if result.returncode or 'hash lookup PASS:' not in text or 'errors=0' not in text:
            raise RuntimeError(f'{name} failed; inspect {build / "gfrun.log"}')
        print(f'{name}: PASS', flush=True)

if __name__ == '__main__':
    main()
