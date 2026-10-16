#!/usr/bin/env python3
from pathlib import Path
import unittest
from run_qli_element import check_norm_native


class QLINormNativeTest(unittest.TestCase):
    def check(self, text):
        check_norm_native(text, [(128, 32, 0)])

    def test_valid_domain_and_capacity(self):
        source = '''00000100 <qli_norm_stageILj128ELj32EE>:
 100: 00000000 BSTART.TEPL TCMP, U32
 104: 00000000 B.DATR NORM.normal, Zero, GT
 108: 00000000 C.B.DIMI 32, ->lb0
 10c: 00000000 C.B.DIMI 128, ->lb2
 110: 00000000 B.IOT t#1, t#2, last, ->u<128B>
 114: 00000000 BSTART.TEPL TSEL, U32
 118: 00000000 C.B.DIMI 32, ->lb0
 11c: 00000000 C.B.DIMI 128, ->lb2
 120: 00000000 B.IOT u#1, t#1, last, ->n<512B>
 124: 00000000 BSTART.TEPL TEXPANDS, U32
 128: 00000000 C.B.DIMI 32, ->lb0
 12c: 00000000 C.B.DIMI 128, ->lb2
 130: 00000000 B.IOR [a0], []
 134: 00000000 B.IOT last, ->n<512B>
'''
        self.check(source)
        for mutant in (source.replace('128, ->lb2', '32, ->lb2'),
                       source.replace('32, ->lb0', '128, ->lb0'),
                       source.replace('->u<128B>', '->u<512B>'),
                       source.replace('B.IOR [a0], []', 'B.IOR [], []'),
                       source.replace('qli_norm_stage', 'unrelated')):
            with self.subTest(mutant=mutant), self.assertRaises((AssertionError, RuntimeError)):
                self.check(mutant)


if __name__ == '__main__':
    unittest.main()
