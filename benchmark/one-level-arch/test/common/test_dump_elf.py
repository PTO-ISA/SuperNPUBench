import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dump_elf import dump

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('report_fixp', ROOT / 'microbenchmark/fixp/report_fixp.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class DumpTests(unittest.TestCase):
    def invoke(self, results):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.elf = Path(self.directory.name) / 'case.elf'
        with patch('dump_elf.subprocess.run', side_effect=results) as run:
            rc = dump('objdump', self.elf)
        return rc, run.call_args_list

    def result(self, rc, error=b''):
        return subprocess.CompletedProcess([], rc, stderr=error)

    def test_macro_success(self):
        rc, calls = self.invoke([self.result(0)])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(str(self.elf)+'.diss.mode').read_text(), 'macro\n')

    def test_known_macro_failure_falls_back(self):
        error = b'no complete macro instruction form'
        rc, calls = self.invoke([self.result(1, error), self.result(0)])
        self.assertEqual(rc, 0)
        self.assertIn('--disassembler-options=no-tile-macros', calls[1].args[0])
        self.assertEqual(Path(str(self.elf)+'.diss.macro-errors').read_bytes(), error)
        self.assertEqual(Path(str(self.elf)+'.diss.mode').read_text(), 'physical-fallback\n')

    def test_unrelated_failure_is_fatal(self):
        rc, calls = self.invoke([self.result(2, b'invalid ELF')])
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertFalse(Path(str(self.elf)+'.diss.mode').exists())

    def test_physical_failure_is_fatal(self):
        rc, calls = self.invoke([self.result(1, b'no complete macro instruction form'), self.result(3)])
        self.assertEqual(rc, 3)
        self.assertEqual(len(calls), 2)
        self.assertFalse(Path(str(self.elf)+'.diss.mode').exists())

    def test_rowmax_physical_and_legacy_renderings(self):
        for header, attrs in [('BSTART.TMATMUL FP16', 'PreQuantMode=NoQuant, RowMaxEn=1'),
                              ('BSTART.CUBE TMATMUL FP16', '0, 0, 0, 1, 0, 0, 0, 0, 0, 0')]:
            lines = [f'100: 20031181 {header}', f'104: 00042023 B.FPATR {attrs}',
                     '108: c2002e13 B.IOT m#1, n#1, mask=1111',
                     '10c: 00036e13 B.IOT mask=1111, ->t<4KB>',
                     '110: 0800 C.BSTART.STD',
                     '114: 00010023 B.FPATR unrelated following bundle']
            got = report.parse_bundle(lines)
            self.assertEqual(got, ('TMATMUL', (0,0,0,1,0,0,0,0,0,0), 0x42023, 0, 1, False))


if __name__ == '__main__':
    unittest.main()
