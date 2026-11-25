import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import gfrun_matmul as checker


class HostedCheckTests(unittest.TestCase):
    def test_written_zero_is_valid_but_missing_output_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace()
            self.assertEqual(checker.result_compare(directory, args)[0], 'res.bin not exist')
            np.zeros(8, dtype=np.float32).tofile(root/'res.bin')
            np.zeros(8, dtype=np.float32).tofile(root/'golden.bin')
            self.assertEqual(checker.compare_array(root/'res.bin', root/'golden.bin', 'fp32', 0, 0)[0], 'pass')
            np.ones(8, dtype=np.float32).tofile(root/'golden.bin')
            self.assertEqual(checker.compare_array(root/'res.bin', root/'golden.bin', 'fp32', 0, 0)[0], 'fail')

    def test_hosted_root_and_completion_required(self):
        args = SimpleNamespace(gfrun=checker.DEFAULT_GFRUN, gfrun_args=' -f ', timeout=1)
        elf = '/tmp/kernel_matmul_matmul_shared_B1_M64_N64_K64_tM64_tN64_tK64.elf'
        for stdout, expected in [(b'formal_End=1 formal_Abort=0 status=0', 'pass'), (b'', 'fail')]:
            proc = MagicMock(returncode=0)
            proc.communicate.return_value = (stdout, None)
            with patch.object(checker.subprocess, 'Popen', return_value=proc) as popen:
                _, status, _ = checker.run_qemu(elf, args, '/tmp/hosted input')
                self.assertEqual(status, expected)
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index('--hosted-file-root')+1], '/tmp/hosted input')
                self.assertIn('softcore.multiThreadNum=4', command)
                self.assertEqual(popen.call_args.kwargs['cwd'], '/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel')


if __name__ == '__main__':
    unittest.main()
