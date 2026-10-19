#!/usr/bin/env python3
import unittest
from run_qli_original import check_score_native, check_topk_native

SCORE = '_Z7qli_ptoIfE'
TOPK = '_Z14qli_topk_radixILi4ELi2048ELi512EE'
POPN = '_Z9RadixPopNILi2048E.pto.local'
NATIVE = f'''00000100 <main>:
 100: 00000000 C.BSTART.STD DIRECT, 0x200 <{SCORE}>
 104: 00000000 C.BSTART.STD DIRECT, 0x300 <{TOPK}>
00000200 <{SCORE}>:
 200: 00000000 BSTART.CUBE TMATMUL, FP8_E4M3
 204: 00000000 BSTART.TEPL TCMP, FP32
 208: 00000000 BSTART.TEPL TSEL, FP32
 20c: 00000000 BSTART.TEPL TMUL, FP32
 210: 00000000 BSTART.TEPL TADD, FP32
 214: 00000000 BSTART.TEPL TCOLSUM, FP32
00000300 <{TOPK}>:
 300: 00000000 BSTART.TLSU MGATHER.ADD, U32
 304: 00000000 BSTART.TEPL TROWSUM, U32
 308: 00000000 C.BSTART.STD DIRECT, 0x400 <{POPN}>
00000400 <{POPN}>:
 400: 00000000 BSTART.TEPL TROWARGMAX, U32
 404: 00000000 BSTART.TEPL TCMP, U32
 408: 00000000 BSTART.TEPL TSEL, U32
'''
IR = f'''define internal inreg <2048 x i32> @{POPN}() {{
  ret <2048 x i32> zeroinitializer
}}
'''


class OriginalQLINativeTest(unittest.TestCase):
    def test_exact_called_score_scope_and_dtype(self):
        check_score_native(NATIVE, 'static')
        for bad in (NATIVE.replace(f'0x200 <{SCORE}>', '0x200 <unrelated>'),
                    NATIVE.replace('TCMP, FP32', 'TCMP, U32'),
                    NATIVE.replace('TCOLSUM, FP32', 'TCOLSUM, U32')):
            with self.assertRaises((AssertionError, RuntimeError)):
                check_score_native(bad, 'static')

    def test_topk_requires_reachable_native_helpers(self):
        profile = ('static', 4, 2048, 512, 1)
        check_topk_native(NATIVE, IR, profile)
        for bad in (NATIVE.replace(f'0x300 <{TOPK}>', '0x300 <unrelated>'),
                    NATIVE.replace(f'0x400 <{POPN}>', f'0x200 <{SCORE}>'),
                    NATIVE.replace('MGATHER.ADD', 'MGATHER'),
                    NATIVE.replace('TROWARGMAX', 'TLOAD'),
                    NATIVE.replace('TSEL, U32', 'TMUL, U32')):
            with self.assertRaises((AssertionError, RuntimeError)):
                check_topk_native(bad, IR, profile)
        with self.assertRaises(AssertionError):
            check_topk_native(NATIVE, IR.replace('  ret', '  %v = load <2048 x i32>, ptr %p\n  ret'), profile)


if __name__ == '__main__':
    unittest.main()
