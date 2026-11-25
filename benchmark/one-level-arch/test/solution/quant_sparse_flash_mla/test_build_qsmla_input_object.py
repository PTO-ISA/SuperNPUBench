"""Host tests for compiler argv and embedded input bounds (no Linx required)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BUILDER = Path(__file__).with_name('build_qsmla_input_object.py')


class InputObjectTests(unittest.TestCase):
    def run_builder(self, data_size=8, count=8, exact=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input with spaces.bin'
            source.write_bytes(bytes(data_size))
            capture = root / 'argv.json'
            fake = root / 'fake compiler.py'
            fake.write_text('import json, sys\nfrom pathlib import Path\n'
                            'Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n')
            command = [sys.executable, str(BUILDER), '--compiler', sys.executable,
                       '--compiler-arg=' + str(fake), '--compiler-arg=' + str(capture),
                       '--compiler-arg=-fno-builtin', '--compiler-arg=-fno-exceptions',
                       '--input', str(source), '--output', str(root / 'out.o'),
                       '--symbol', 'q', '--bytes', str(count)]
            if exact:
                command.append('--exact-bytes')
            result = subprocess.run(command, capture_output=True, text=True)
            args = json.loads(capture.read_text()) if capture.exists() else None
            return result, args

    def test_compiler_flags_are_separate_argv(self):
        result, args = self.run_builder(exact=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[:4], ['-fno-builtin', '-fno-exceptions', '-target', 'linx64v5'])
        self.assertIn('-c', args)

    def test_prefix_is_allowed(self):
        result, args = self.run_builder(data_size=16)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(args)

    def test_exact_size_rejects_extra_bytes(self):
        result, args = self.run_builder(data_size=16, exact=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(args)

    def test_short_input_is_rejected_before_compiler(self):
        result, args = self.run_builder(data_size=4)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(args)


if __name__ == '__main__':
    unittest.main()
