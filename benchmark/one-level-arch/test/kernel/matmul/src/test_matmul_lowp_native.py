#!/usr/bin/env python3
from pathlib import Path
import tempfile
import unittest

from matmul_lowp_fixture import profiles
from run_matmul_lowp import check_matmul_native, compare_output, dump_windows, read_dump_windows

SYMBOL = '_Z16matmul_hif4x2_mxILi256ELi2048ELi2048ELi32ELi64ELi1024ELb0EEvPargs'
NATIVE = f'''00000100 <main>:
 100: 00000000 C.BSTART.STD DIRECT, 0x200 <{SYMBOL}>
00000200 <{SYMBOL}>:
 200: 00000000 BSTART.CUBE TMATMULMX, HiF4x2
 204: 00000000 BSTART.CUBE TMATMULMX.ACC, HiF4x2
'''


class LowpNativeTest(unittest.TestCase):
    def test_called_original_specialization(self):
        check_matmul_native(NATIVE, profiles()[0])
        check_matmul_native(NATIVE.replace("BSTART.CUBE TMATMULMX, HiF4x2",
                                           "BSTART.TMATMULMX HiF4x2")
                                  .replace("BSTART.CUBE TMATMULMX.ACC, HiF4x2",
                                           "BSTART.TMATMULMX.ACC HiF4x2"),
                            profiles()[0])
        for text in (NATIVE.replace('0x200 <'+SYMBOL+'>', '0x200 <unrelated>'),
                     NATIVE.replace('TMATMULMX', 'TSQRT'),
                     NATIVE.replace('ELi2048ELi2048', 'ELi1280ELi2048')):
            with self.subTest(text=text), self.assertRaises((AssertionError,RuntimeError)):
                check_matmul_native(text, profiles()[0])

    def test_full_range_is_partitioned_without_gaps(self):
        with tempfile.TemporaryDirectory() as work:
            base = 0x10003
            total = 23*1024*1024+17
            windows = dump_windows(base,base+total,Path(work))
            self.assertEqual(sum(size for _,size,_ in windows),total)
            self.assertTrue(all(size<=4*1024*1024 for _,size,_ in windows))
            self.assertEqual(windows[-1][0]+windows[-1][1],base+total)
            for _,size,path in windows:
                path.write_bytes(b'\x6d'*size)
            self.assertEqual(len(read_dump_windows(windows)),total)
            windows[-1][2].write_bytes(b'\x6d'*(windows[-1][1]-1))
            with self.assertRaises(ValueError):
                read_dump_windows(windows)
            with self.assertRaises(ValueError):
                dump_windows(base,base,Path(work))

    def test_exact_fp32_results_and_nonfinite_rejection(self):
        import struct
        expected = struct.pack('<3f',1.25,-2,0)
        compare_output(struct.pack('<3f',1.25,-2,-0.),expected)
        for actual in (struct.pack('<3f',1.25001,-2,0),
                       struct.pack('<3f',float('nan'),-2,0), expected[:-4]):
            with self.assertRaises(ValueError):
                compare_output(actual,expected)


if __name__ == '__main__':
    unittest.main()
