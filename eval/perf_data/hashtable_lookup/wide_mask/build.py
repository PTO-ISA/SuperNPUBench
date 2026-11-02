#!/usr/bin/env python3
"""Build identical-input 32–512 lane bare-metal fixtures with pinned tools."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llvm', type=Path, required=True)
    p.add_argument('--api', type=Path, required=True)
    p.add_argument('--sysroot', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lanes', type=int, nargs='+', default=[128, 256, 512], choices=[32, 64, 128, 256, 512])
    p.add_argument('--query-repeats', type=int, default=1, choices=[1, 4])
    a = p.parse_args()
    a.llvm, a.api, a.sysroot, a.reference, a.output = (
        x.resolve() for x in (a.llvm, a.api, a.sysroot, a.reference, a.output))
    if digest(a.reference) != '960ba161b8c662a74bf3f93ed5ad96a75f3375d4a91eee4607bc870b9fa48bd4':
        p.error('reference ELF must be the archived 6144-query fixture')
    a.output.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    root = src.parents[3]
    cfg = a.output / 'libcxx-config'
    cfg.mkdir(exist_ok=True)
    (cfg / '__config_site').write_text('''#ifndef _LIBCPP_CONFIG_SITE
#define _LIBCPP_CONFIG_SITE
#define _LIBCPP_ABI_VERSION 1
#define _LIBCPP_HAS_MUSL_LIBC
#define _LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS
#endif
''')
    tools = a.llvm / 'build-hash/bin'
    clang, lld, objdump = [tools / x for x in ('clang++', 'ld.lld', 'llvm-objdump')]
    flags = [str(clang), '--target=linx64v5-unknown-linux-musl', '-mlxbc',
             '-fenable-matrix', '-O2', '-std=c++20', '-D__linx',
             '-DENABLE_TENSOR_INSTR', '-ffreestanding', '-fno-exceptions',
             '-fno-rtti', '-fno-vectorize', '-fno-slp-vectorize',
             '-mllvm', '-linxv5-enable-generic-element=true',
             '-mllvm', '-enable-all-vector-as-tilereg=true',
             '--sysroot=' + str(a.sysroot), '-nostdinc++',
             '-isystem', str(cfg), '-isystem', str(a.llvm / 'libcxx/include'),
             '-I' + str(a.api / 'include'), '-I' + str(root),
             '-DHASH_DATA_ELF_PATH="' + str(a.reference) + '"',
             '-DHASH_QUERY_REPEATS=' + str(a.query_repeats)]
    provenance = {'reference_sha256': digest(a.reference),
                  'compiler_sha256': digest(clang), 'linker_sha256': digest(lld),
                  'compiler_version': subprocess.check_output([str(clang), '--version'], text=True),
                  'query_repeats': a.query_repeats, 'query_count': 6144 * a.query_repeats,
                  'commands': []}
    for lanes in a.lanes:
        base = a.output / ('hash' + str(lanes))
        command = flags + ['-DHASH_WIDE_LANES=' + str(lanes)]
        if lanes > 32:
            command += ['-mllvm', '-linxv5-force-element-tile-mask=true']
        command += ['-c', str(src / 'fixture.cpp'), '-o', str(base.with_suffix('.o'))]
        provenance['commands'].append(command)
        (a.output / 'build-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        subprocess.run(command, cwd=root, check=True)
        command = [str(lld), '-T', str(src / 'link.ld'), str(base.with_suffix('.o')),
                   '-o', str(base.with_suffix('.elf'))]
        provenance['commands'].append(command)
        subprocess.run(command, cwd=root, check=True)
        for suffix, options in (('.diss', []), ('.expanded.diss', ['--disassembler-options=no-tile-macros'])):
            with base.with_suffix(suffix).open('w') as stream:
                subprocess.run([str(objdump), '-d', '--no-show-raw-insn'] + options +
                               [str(base.with_suffix('.elf'))], stdout=stream, check=True)
        expanded = base.with_suffix('.expanded.diss').read_text()
        checker = expanded.split('<wide_mask_checker_load_begin>:', 1)[1].split(
            '<wide_mask_checker_load_end>:', 1)[0]
        loads = re.findall(r'^\s*([0-9a-f]+):\s+lw(?:i)?(?:\.u)?\s', checker, re.M)
        finisher = expanded.split('<wide_mask_finisher_store>:', 1)[1].split('<_start>:', 1)[0]
        stores = re.findall(r'^\s*([0-9a-f]+):\s+sw(?:i)?(?:\.u)?\s', finisher, re.M)
        if len(loads) != 4 or len(stores) != 1:
            raise RuntimeError('checker changed: inspect disassembly before selecting witness PCs')
        witness = {'elf_sha256': digest(base.with_suffix('.elf')),
                   'query_count': 6144 * a.query_repeats,
                   'pcs': [int(loads[0],16), int(loads[2],16), int(loads[3],16), int(stores[0],16)]}
        base.with_suffix('.witness.json').write_text(json.dumps(witness, indent=2) + '\n')
        provenance[str(lanes)] = {'elf_sha256': witness['elf_sha256'], 'witness_pcs': witness['pcs']}
        (a.output / 'build-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
