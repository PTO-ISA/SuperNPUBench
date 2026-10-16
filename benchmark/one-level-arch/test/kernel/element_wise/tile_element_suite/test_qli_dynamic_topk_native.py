#!/usr/bin/env python3
import unittest

from qli_dynamic_topk_native import check_native


ROOT = "_Z22qli_topk_radix_dynamicPfPiiiiiPj"
POPN = "_ZN13qli_radix_dyn4PopNER...pto.local"


def evidence(main_target=ROOT, popn_target=POPN, include_root_ops=True,
             include_popn_ops=True, unrelated=False):
    root_ops = """
 204: 00000000 BSTART.TLSU MGATHER.ADD, U32
 208: 00000000 BSTART.TEPL TROWSUM, U32
""" if include_root_ops else ""
    popn_ops = """
 300: 00000000 BSTART.TEPL TROWARGMAX, U32
 304: 00000000 BSTART.TEPL TCMP, U32
 308: 00000000 BSTART.TEPL TSEL, U32
""" if include_popn_ops else ""
    other = """
00000400 <unrelated>:
 400: 00000000 BSTART.TLSU MGATHER.ADD, U32
 404: 00000000 BSTART.TEPL TROWSUM, U32
 408: 00000000 BSTART.TEPL TROWARGMAX, U32
 40c: 00000000 BSTART.TEPL TCMP, U32
 410: 00000000 BSTART.TEPL TSEL, U32
""" if unrelated else ""
    dis = f"""
00000100 <main>:
 100: 00000000 C.BSTART.STD DIRECT, 0x200 <{main_target}>
00000200 <{ROOT}>:
 200: 00000000 BSTART.STD DIRECT, 0x300 <{popn_target}>
{root_ops}00000300 <{popn_target}>:
{popn_ops}{other}"""
    ir = f"""define internal void @{popn_target}() {{
entry:
  ret void
}}
"""
    return dis, ir


class DynamicTopKNativeTest(unittest.TestCase):
    def test_called_dynamic_closure_passes(self):
        result = check_native(*evidence())
        self.assertEqual(result["popn_local_helpers"], [POPN])

    def test_unreachable_or_unrelated_operations_rejected(self):
        with self.assertRaisesRegex(ValueError, "called by main"):
            check_native(*evidence(main_target="unrelated", unrelated=True))
        with self.assertRaisesRegex(ValueError, "MGATHER.ADD"):
            check_native(*evidence(include_root_ops=False, unrelated=True))
        with self.assertRaisesRegex(ValueError, "TROWARGMAX"):
            check_native(*evidence(include_popn_ops=False, unrelated=True))

    def test_nonlocal_or_spilled_popn_rejected(self):
        with self.assertRaisesRegex(ValueError, "pto.local"):
            check_native(*evidence(popn_target="ordinary_helper"))
        dis, ir = evidence()
        ir = ir.replace("  ret", "  %v = load <2048 x i32>, ptr %p\n  ret")
        with self.assertRaisesRegex(ValueError, "spilled"):
            check_native(dis, ir)


if __name__ == "__main__":
    unittest.main()
