#!/usr/bin/env python3
"""Negative canaries for the exact MoE element histogram verifier."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_moe_element_histograms as check


SOURCE = """
constexpr uint32_t kPartElements = 32;
void gt_dyn_clear_element_histogram(uint32_t *histogram, uint32_t size) {
#pragma pto element for
  for (uint32_t element = 0; element < 32; ++element)
    if (element < size) histogram[element] = 0;
}
void addElementHistogramPart(const uint32_t *input, uint32_t valid,
                             uint32_t *histogram, uint32_t histogramSize) {
#pragma pto element for
  for (uint32_t element = 0; element < kPartElements; ++element)
    if (element < valid) {
      const uint32_t index = input[element];
      if (index < histogramSize)
      __atomic_fetch_add(histogram + index, 1u, __ATOMIC_RELAXED);
    }
}
"""

CLEAR_IR = """define void @gt_dyn_clear_element_histogram(i32 %size) {
  %n = call i32 @llvm.umin.i32(i32 %size, i32 32)
  %t = call <32 x i32> @llvm.linx.experimental.ew.tci.v32i32()
  %pto.mask.profile = call i64 @llvm.linx.experimental.ew.tcmps.gpr.v32i32()
  %o = call <32 x i64> @llvm.linx.experimental.ew.tlea.v32i64.v32i32()
  call void @llvm.linx.experimental.ew.mscatter.gpr.masked.v32i64.v32i32()
  ret void
}"""

COUNT_IR = """define void @addElementHistogramPart(i32 %valid) {
  %t = call <32 x i32> @llvm.linx.experimental.ew.tci.v32i32()
  %pto.mask.profile = call i64 @llvm.linx.experimental.ew.tcmps.gpr.v32i32()
  %o = call <32 x i64> @llvm.linx.experimental.ew.tlea.v32i64.v32i32()
  %g = call <32 x i32> @llvm.linx.experimental.ew.mgather.gpr.masked.v32i32.v32i64()
  %pto.mask.profile2 = call i64 @llvm.linx.experimental.ew.tcmps.gpr.v32i32()
  %o2 = call <32 x i64> @llvm.linx.experimental.ew.tlea.v32i64.v32i32()
  %a = call <32 x i32> @llvm.linx.experimental.ew.mgather.add.masked.v32i32.v32i64.v32i32()
  ret void
}"""

IR = ("@view = private constant [61 x i8] "
      "c\"pto.element.view:v1;dtype=u32;rows=32;cols=1;layout=cube_m32\\00\"\n" +
      CLEAR_IR + "\n" + COUNT_IR)

CLEAR_NATIVE = """00000100 <gt_dyn_clear_element_histogram>:
  BSTART.TEPL\tTCI
  TCMPS
  BSTART.TEPL\tTLEA
  BSTART.TLSU\tMSCATTER
  B.IOR [], ExecMaskPresent
"""
COUNT_NATIVE = """00000200 <addElementHistogramPart>:
  BSTART.TEPL\tTCI
  TCMPS
  BSTART.TEPL\tTLEA
  BSTART.TLSU\tMGATHER,
  B.IOR input, ExecMaskPresent
  TCMPS
  BSTART.TEPL\tTLEA
  BSTART.TLSU\tMGATHER.ADD
  B.IOR atomic, ExecMaskPresent
"""
DIS = CLEAR_NATIVE + COUNT_NATIVE


class NegativeCanaries(unittest.TestCase):
    def assert_rejected(self, source: str = SOURCE, ir: str = IR,
                        dis: str = DIS) -> None:
        with self.assertRaises(check.CheckError):
            check.validate_artifact(source, ir, dis, check.DYNAMIC)

    def test_baseline_fixture_is_accepted(self) -> None:
        check.validate_artifact(SOURCE, IR, DIS, check.DYNAMIC)

    def test_widened_induction_still_requires_both_tile_masks(self) -> None:
        count = COUNT_IR.replace("ew.tcmps.gpr.v32i32", "ew.tcmps.gpr.v32i64", 1)
        check.validate_artifact(SOURCE, IR.replace(COUNT_IR, count), DIS, check.DYNAMIC)
        count = count.replace(
            "  %pto.mask.profile2 = call i64 @llvm.linx.experimental.ew.tcmps.gpr.v32i32()\n", "")
        self.assert_rejected(ir=IR.replace(COUNT_IR, count))

    def test_scalar_histogram_update_survives_tile_markers(self) -> None:
        poisoned = IR.replace("  ret void\n}",
                              "  %old = atomicrmw add ptr %p, i32 1 monotonic\n"
                              "  ret void\n}", 2)
        self.assert_rejected(ir=poisoned)

    def test_markers_in_wrong_function_are_rejected(self) -> None:
        wrong = IR.replace(COUNT_IR,
                           "define void @addElementHistogramPart(i32 %valid) {\n"
                           "  %x = load i32, ptr %p\n  ret void\n}\n" +
                           COUNT_IR.replace("@addElementHistogramPart",
                                            "@decoyTileFunction"))
        self.assert_rejected(ir=wrong)

    def test_missing_native_mask_is_rejected(self) -> None:
        self.assert_rejected(dis=DIS.replace(
            "B.IOR [], ExecMaskPresent", "B.IOR []", 1))

    def test_missing_input_gather_mask_is_rejected(self) -> None:
        self.assert_rejected(dis=DIS.replace(
            "B.IOR input, ExecMaskPresent", "B.IOR input", 1))

    def test_missing_atomic_mask_is_rejected(self) -> None:
        self.assert_rejected(dis=DIS.replace(
            "B.IOR atomic, ExecMaskPresent", "B.IOR atomic", 1))

    def test_ignored_pragma_and_scalar_loop_are_rejected(self) -> None:
        self.assert_rejected(source=SOURCE.replace("#pragma pto element for\n",
                                                   "", 1))

    def test_scalarized_element_ir_is_rejected(self) -> None:
        scalar = IR.replace("  ret void\n}",
                            "  %v = extractelement <32 x i32> %t, i32 0\n"
                            "  store i32 %v, ptr %p\n  ret void\n}", 2)
        self.assert_rejected(ir=scalar)

    def test_arbitrary_ir_call_is_rejected(self) -> None:
        poisoned_count = COUNT_IR.replace(
            "  ret void\n}", "  call void @scalar_histogram_helper()\n"
            "  ret void\n}")
        self.assert_rejected(ir=IR.replace(COUNT_IR, poisoned_count))

    def test_expanded_scalar_native_memory_is_rejected(self) -> None:
        for mnemonic in ("LD", "LDI", "LWI", "LWI.U", "LWUI.U",
                         "SD", "SDI", "SWI", "C.LW", "C.SW",
                         "C.LWSP", "C.SDSP"):
            with self.subTest(mnemonic=mnemonic):
                poisoned = DIS.replace(
                    "00000200 <addElementHistogramPart>:\n",
                    "00000200 <addElementHistogramPart>:\n"
                    f"  200: 00000013  {mnemonic} a0, [a1]\n")
                self.assert_rejected(dis=poisoned)

    def test_native_call_is_rejected(self) -> None:
        poisoned = DIS.replace(
            "00000200 <addElementHistogramPart>:\n",
            "00000200 <addElementHistogramPart>:\n"
            "  200: 0000  C.BSTART.STD DIRECT, 0x300 <scalar_helper>\n")
        self.assert_rejected(dis=poisoned)

    def test_split_region_regression_is_rejected(self) -> None:
        self.assert_rejected(source=SOURCE.replace(
            "      const uint32_t index = input[element];",
            "#pragma pto element for\n      const uint32_t index = input[element];"))

    def test_tile_staging_regression_is_rejected(self) -> None:
        self.assert_rejected(source=SOURCE.replace(
            "      const uint32_t index = input[element];",
            "      pto::ElementTile<uint32_t, 32> staged;\n"
            "      const uint32_t index = input[element];"))

    def test_wrong_count_capacity_is_rejected(self) -> None:
        self.assert_rejected(ir=IR.replace("<32 x i32>", "<64 x i32>"))

    def test_unbounded_parent_tload_is_rejected(self) -> None:
        poisoned_count = COUNT_IR.replace(
            "%t = call <32 x i32>",
            "%parent = call <128 x i32> asm sideeffect \"BSTART.TLOAD\"\n"
            "  %t = call <32 x i32>", 1)
        self.assert_rejected(ir=IR.replace(COUNT_IR, poisoned_count))

    def test_wrong_source_capacity_is_rejected(self) -> None:
        self.assert_rejected(source=SOURCE.replace(
            "element < kPartElements",
            "element < 64"))

    def test_wrong_element_width_is_rejected(self) -> None:
        self.assert_rejected(source=SOURCE.replace("kPartElements = 32",
                                                   "kPartElements = 16"))


if __name__ == "__main__":
    unittest.main()
