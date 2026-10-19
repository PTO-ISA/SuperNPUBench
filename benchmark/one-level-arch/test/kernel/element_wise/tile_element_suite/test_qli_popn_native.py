#!/usr/bin/env python3
import unittest
from run_qli_element import check_popn_native

SYMBOL = 'RadixPopN_test.pto.local'
IR = '''define i32 @main() #0 {
  %returned = call inreg <128 x i32> @RadixPopN_test.pto.local(<128 x i32> %input, i32 noundef signext 4096, ptr noundef nonnull @qli_popn_128_32_0_1_indices, i32 noundef signext 1)
  call void asm sideeffect "BSTART.TLSU TSTORE, U32", ""(ptr @qli_popn_128_32_0_1_remaining, <128 x i32> %returned, i32 0)
  call void asm sideeffect "BSTART.TLSU TSTORE, U32", ""(ptr @qli_popn_128_32_0_0_remaining, <128 x i32> %unchanged, i32 0)
  ret i32 0
}
define internal inreg <128 x i32> @RadixPopN_test.pto.local(<128 x i32> %input) #1 {
  ret <128 x i32> %input
}
attributes #1 = { "linx-local-tile-abi"="0:128:32" }
'''
NATIVE = '''00000100 <main>:
 100: 00000000 BSTART.TLSU TMOV, U32
 104: 00000000 B.IOT t#1, mask=1111, last, ->t<512B>
 108: 00000000 C.BSTART.STD DIRECT, 0x200 <RadixPopN_test.pto.local>
00000200 <RadixPopN_test.pto.local>:
 200: 00000000 BSTART.TEPL TROWARGMAX, U32
 204: 00000000 B.DATR NORM.normal, Zero
 208: 00000000 BSTART.TEPL TCMP, U32
 20c: 00000000 BSTART.TEPL TSEL, U32
 210: 00000000 BSTART.TLSU TMOV, U32
 214: 00000000 B.IOT t#1, mask=1111, last, ->t<512B>
 218: 00000000 C.BSTART.STD RET
'''
PROFILES = [(128,32,False,0),(128,32,False,1)]


class PopNNativeTest(unittest.TestCase):
    def test_real_call_and_return_pair(self):
        check_popn_native(NATIVE,IR,PROFILES)
        check_popn_native(NATIVE.replace('B.IOT t#1,', 'B.IOT t#1.reuse,'),
                          IR, PROFILES)
        check_popn_native(NATIVE, IR.replace('BSTART.TLSU TSTORE,',
                                            'BSTART.TSTORE'), PROFILES)

    def test_no_acceptance_from_unrelated_functions(self):
        for text in (NATIVE.replace('<RadixPopN_test.pto.local>:', '<unrelated>:'),
                     NATIVE.replace('TCMP, U32','TXOR, U32')+'\n00000300 <unrelated>:\n BSTART.TEPL TCMP, U32',
                     NATIVE.replace('0x200 <'+SYMBOL+'>','0x200 <unrelated>')):
            with self.subTest(text=text), self.assertRaises((AssertionError,RuntimeError)):
                check_popn_native(text,IR,PROFILES)

    def test_returned_tile_and_scalar_call_attributes(self):
        for text in (IR.replace('BSTART.TLSU TSTORE,', 'BSTART.TLOAD'),
                     IR.replace('<128 x i32> %returned,','<128 x i32> %input,'),
                     IR.replace('i32 noundef signext 1)','i32 noundef signext 2)'),
                     IR.replace('i32 noundef signext 4096,','i32 4096,'),
                     IR.replace('call inreg','call'),
                     IR.replace('0:128:32','0:128:128'),
                     IR.replace('  ret i32 0','  %spill = load <128 x i32>, ptr %slot\n  ret i32 0'),
                     IR.replace('  ret i32 0','  store i32 7, ptr @qli_popn_128_32_0_0_indices\n  ret i32 0')):
            with self.subTest(text=text), self.assertRaises((AssertionError,RuntimeError)):
                check_popn_native(NATIVE,text,PROFILES)

    def test_no_import_allocation_and_mandatory_exports(self):
        for text in (NATIVE.replace(' 200: 00000000 BSTART.TEPL TROWARGMAX, U32',
                        ' 1fc: 00000000 BSTART.TLSU TMOV, U32\n 200: 00000000 BSTART.TEPL TROWARGMAX, U32'),
                     NATIVE.replace(' 100: 00000000 BSTART.TLSU TMOV, U32',' 100: 00000000 BSTART.TLSU TLOAD, U32'),
                     NATIVE.replace(' 210: 00000000 BSTART.TLSU TMOV, U32',' 210: 00000000 BSTART.TLSU TLOAD, U32'),
                     NATIVE.replace(' 214: 00000000 B.IOT t#1, mask=1111, last, ->t<512B>',
                                    ' 214: 00000000 B.IOT m#1, mask=1111, last, ->m<512B>')):
            with self.subTest(text=text), self.assertRaises((AssertionError,RuntimeError)):
                check_popn_native(text,IR,PROFILES)


if __name__ == '__main__':
    unittest.main()
