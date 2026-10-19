#!/usr/bin/env python3
import unittest
from run_qli_original import check_pe0_domain_native, check_score_native, check_topk_native

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
    def test_pe0_domain_covers_all_reachable_helpers(self):
        native = '''00000100 <main>:
 100: 00000000 C.BSTART.STD DIRECT, 0x200 <run_qli_pe0>
00000200 <run_qli_pe0>:
 200: 00000000 B.IOT mask=1000, last, ->t<128B>
 204: 00000000 C.BSTART.STD DIRECT, 0x300 <helper.pto.pe8.0>
00000300 <helper.pto.pe8.0>:
 300: 00000000 BSTART.TLSU MGATHER.ADD, U32
 304: 00000000 B.IOR [a1,a2,zero], ExecMaskPresent
 308: 00000000 B.IOT t#1, mask=1000, last
00000400 <unrelated>:
 400: 00000000 B.IOT mask=1111, last, ->t<128B>
'''
        check_pe0_domain_native(native)
        for bad in (native.replace('200: 00000000 B.IOT mask=1000',
                                   '200: 00000000 B.IOT mask=1111'),
                    native.replace('t#1, mask=1000', 't#1, mask=1111'),
                    native.replace('ExecMaskPresent', ''),
                    native.replace('run_qli_pe0', 'unannotated_stage'),
                    native.replace('MGATHER.ADD', 'MGATHER')):
            with self.assertRaises(AssertionError):
                check_pe0_domain_native(bad)

    def test_pe_domain_clones_preserve_topk_scope(self):
        profile = ('static', 4, 2048, 512, 1)
        clone = POPN + '.pto.pe8.0'
        native, ir = NATIVE.replace(POPN, clone), IR.replace(POPN, clone)
        check_topk_native(native, ir, profile)
        with self.assertRaises(AssertionError):
            check_topk_native(native.replace('TSEL, U32', 'TMUL, U32'), ir, profile)

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
