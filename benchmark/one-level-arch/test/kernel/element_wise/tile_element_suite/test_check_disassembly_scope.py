#!/usr/bin/env python3
"""Native checker must not compose evidence across unrelated functions."""

import importlib.util
from pathlib import Path
import subprocess
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "check_disassembly", HERE / "check_disassembly.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CheckerFunctionScopeTest(unittest.TestCase):
    SELECTED_MASK = """
   100: ce619181              BSTART.TEPL TCMPS, U32
   104: 00080813              B.IOR [s5], ->s5
   108: 00f82805              and s5, s4, ->s5
   10c: cad19181              BSTART.TEPL TCMPS, U32
   110: 00020893              B.IOR [a2], ->s6
   114: 0108a785              and s6, s5, ->s4
   118: cae19181              BSTART.TEPL TLEA, U32
   11c: c8c11181              BSTART.TLSU MGATHER.ADD, U32
   120: 01f03ea3              B.DATR CUBE_M32, Zero
   124: 04f28013              B.IOR [a3,s4,zero], ExecMaskPresent
   128: ce619181              BSTART.TEPL TCI, U32
"""

    def test_expression_requires_explicit_reusable_tile_sources(self):
        instructions = ["BSTART.TLSU TLOAD, U32", "B.SUBVIEW"]
        for op in ("TADD", "TSUB", "TSUB", "TMUL", "TDIV", "TREM",
                   "TAND", "TOR", "TXOR", "TXOR", "TXOR", "TSHL", "TSHR"):
            instructions.extend([
                f"BSTART.TEPL {op}, U32", "B.DATR CUBE_M32, Zero",
                "C.B.DIMI 1, ->lb0", "C.B.DIMI 32, ->lb1",
                "B.IOT t#1.reuse, m#2.reuse, mask=1111, last, ->u<128B>",
            ])
        instructions.extend(["BSTART.TLSU TSTORE, U32"] * 3)
        native = "00000100 <element_expression_chain>:\n" + "\n".join(
            f" {0x100 + 4 * index:x}: 00000000 {line}"
            for index, line in enumerate(instructions))
        # Native validation passes, then the separate IR evidence is required.
        with self.assertRaisesRegex(CHECKER.CheckError, "requires LLVM IR evidence"):
            CHECKER.validate_expression(native, None)
        for bad in (native.replace("t#1.reuse", "t#1"),
                    native.replace("m#2.reuse", "m#2"),
                    native.replace("t#1.reuse", "a1"),
                    native.replace("->u<128B>", "->u<256B>")):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                    CHECKER.CheckError, "Tile-to-Tile binding"):
                CHECKER.validate_expression(bad, None)

    def test_concat_f16_store_keeps_full_capacity_with_canonical_header(self):
        native = """00000100 <concat_gather>:
 BSTART.TLSU MGATHER, FP16
 BSTART.TLSU TMOV, FP16
 B.ASSEMBLE 1, 0, zero, 0, 1
 BSTART.TLSU TMOV, FP16
 BSTART.TLSU TMOV, FP16
 B.ASSEMBLE 0, 1, zero, 1, 1
 B.IOT mask=1111, last, ->t<1KB>
 BSTART.TLSU TSTORE, FP16
"""
        lines = native.splitlines()
        native = lines[0] + "\n" + "\n".join(
            f" {0x100 + 4 * index:x}: 00000000 {line}"
            for index, line in enumerate(lines[1:]))
        ir = """define void @concat_gather() {
 %offset = call <64 x i64> @llvm.linx.experimental.ew.tlea(i64 32, i64 2, i64 25, i64 29, <64 x i32> %index, i64 16)
 %value = call <64 x half> @llvm.linx.experimental.ew.mgather.gpr.masked.v64f16.v64i64(i64 32, i64 2, i64 4, i64 0, i64 29, i64 24, ptr %source)
 call void asm sideeffect "BSTART.TSTORE FP16", ""(<512 x half> %parent)
 ret void
}
"""
        for text in (ir, ir.replace("BSTART.TSTORE FP16", "BSTART.TLSU TSTORE, FP16")):
            CHECKER.validate_concat_f16_original(native, text)
            for bad in (text.replace("<512 x half>", "<64 x half>"),
                        text.replace("TSTORE", "TLOAD")):
                with self.assertRaisesRegex(CHECKER.CheckError, "whole 512-element"):
                    CHECKER.validate_concat_f16_original(native, bad)

    def test_group_count_snapshot_order_with_canonical_load(self):
        native = """
00000100 <group_countILi16>:
 100: 00000000 BSTART.TLSU TLOAD, U32
 104: 00000000 BSTART.TEPL TADDS, U32
 108: 00000000 B.SUBVIEW CUBE_M32
 10c: 00000000 BSTART.TEPL TLEA, U32
 110: 00000000 BSTART.TLSU MGATHER.ADD, U32
 114: 00000000 B.IOR [a3,s4,zero], ExecMaskPresent
"""
        ir = """define void @group_countILi16() {
  call void asm sideeffect "BSTART.TLOAD U32", ""()
  store i32 0, ptr %out
  %bound = freeze i32 32
  %wide = zext i32 %bound to i64
  %addr = call <32 x i64> @llvm.linx.experimental.ew.tlea(i64 32, i64 1, i64 25, i64 29, <32 x i32> %index, i64 32)
  %mask = call i64 @llvm.linx.experimental.ew.tcmps.gpr(i64 32, i64 1, i64 25, i64 29, <32 x i32> %index, i64 %wide, i64 2)
  %old = call <32 x i32> @llvm.linx.experimental.ew.mgather.add.masked()
  ret void
}
"""
        for text in (ir, ir.replace("BSTART.TLOAD U32", "BSTART.TLSU TLOAD, U32")):
            CHECKER.validate_group_count(native, text)
            moved = text.replace('  store i32 0, ptr %out\n', '')
            moved = moved.replace('  call void asm', '  store i32 0, ptr %out\n  call void asm')
            with self.assertRaisesRegex(CHECKER.CheckError, 'snapshot before'):
                CHECKER.validate_group_count(native, moved)

    def test_exact_extractors_do_not_select_prefix_clone(self):
        disassembly = """
00000000 <kernel.clone>:
 CLONE_ONLY
00000010 <kernel>:
 EXACT_ONLY
"""
        ir = """
define void @kernel.clone() {
clone.entry:
  ret void
}
define void @kernel() {
exact.entry:
  ret void
}
"""
        self.assertIn(
            "EXACT_ONLY", CHECKER.function_body_exact(disassembly, "kernel"))
        self.assertNotIn(
            "CLONE_ONLY", CHECKER.function_body_exact(disassembly, "kernel"))
        self.assertIn("exact.entry", CHECKER.ir_function_body_exact(ir, "kernel"))
        self.assertNotIn(
            "clone.entry", CHECKER.ir_function_body_exact(ir, "kernel"))

    def test_indexed_snapshot_rejects_corrupt_tail_helper(self):
        disassembly = """
00000000 <transpose_f16_element_tile_full>:
 BSTART.TLSU MGATHER, FP16
 B.IOR ExecMaskPresent
 BSTART.TLSU TMOV, FP16
 B.ASSEMBLE
 B.SUBVIEW
 BSTART.TEPL TLEA, U32
 BSTART.TLSU TSTORE, FP16
 B.IOR ExecMaskPresent
00000010 <transpose_f16_element_tile_tail>:
 BSTART.TLSU MGATHER, FP16
 B.IOR ExecMaskPresent
 BSTART.TLSU TMOV, FP16
 B.ASSEMBLE
 B.SUBVIEW
 BSTART.TEPL TLEA, U32
"""
        ir = """
define void @transpose_f16_element_tile_full() {
  %v = call <64 x half> @llvm.linx.experimental.ew.mgather()
  ret void
}
define void @transpose_f16_element_tile_tail() {
  %v = call <64 x half> @llvm.linx.experimental.ew.mgather()
  ret void
}
"""
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_indexed_snapshot(disassembly, ir)

    def test_inline_broadcast_rejects_split_function_evidence(self):
        disassembly = """
00000000 <broadcast_s32_element_tile_load>:
 BSTART.TLSU MGATHER, U32
 BSTART.TEPL TLEA, U32
 B.ASSEMBLE
 B.IOT ->t<2KB>
00000010 <broadcast_s32_element_tile_store>:
 BSTART.TLSU TSTORE, S32
"""
        ir = """
define void @broadcast_s32_element_tile_load() { ret void }
define void @broadcast_s32_element_tile_store() { ret void }
"""
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_broadcast_inline(disassembly, ir)

    def test_fourpe_broadcast_rejects_split_function_evidence(self):
        disassembly = """
00000000 <unrelated_load>:
 BSTART.TLSU TLOAD, FP32
 C.B.DIMI 128, ->lb0
 C.B.DIMI 128, ->lb2
 B.IOT ->t<512B>
00000010 <unrelated_store>:
 BSTART.TLSU TSTORE, FP32
"""
        ir = "define void @unrelated(<128 x i32> %x) { ret void }\n"
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_broadcast_fourpe(disassembly, ir)

    def test_leaf_checker_rejects_split_function_evidence(self):
        disassembly = """
00000000 <process_gather_tile_load>:
 BSTART.TLSU MGATHER, FP32
 B.IOR ExecMaskPresent
 BSTART.TEPL TLEA, U32
00000010 <process_gather_tile_store>:
 BSTART.TLSU MSCATTER, FP32
 B.IOR ExecMaskPresent
"""
        ir = """
define void @process_gather_tile_load() { ret void }
define void @process_gather_tile_store() { ret void }
"""
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_leaf_gather_scatter(
                disassembly, "gather_v2_f32_0_original", ir)

    def test_selected_radix_mask_requires_both_comparisons(self):
        CHECKER.validate_selected_mask_conjunction(self.SELECTED_MASK)
        broken = self.SELECTED_MASK.replace(
            "and s6, s5, ->s4", "and s5, s5, ->s4")
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_selected_mask_conjunction(broken)

    def test_selected_radix_mask_requires_conjunction(self):
        broken = self.SELECTED_MASK.replace(
            "and s6, s5, ->s4", "or s6, s5, ->s4")
        with self.assertRaises(CHECKER.CheckError):
            CHECKER.validate_selected_mask_conjunction(broken)

    def test_shared_source_profiles_have_distinct_objects(self):
        objects = []
        for case in ("topk_boundaries_0", "topk_boundaries_1",
                     "topk_boundaries_2"):
            result = subprocess.run(
                ["make", "-pn", f"TESTCASE={case}", "SYSROOT=/tmp",
                 "RESOURCE_DIR=/tmp", "API_INCLUDE=/tmp",
                 "COMPILER_DIR=/tmp"],
                cwd=HERE,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assignment = next(
                line for line in result.stdout.splitlines()
                if line.startswith("OBJ := ")
            )
            objects.append(assignment.removeprefix("OBJ := "))
        self.assertEqual(len(set(objects)), 3)
        for case, obj in zip(("topk_boundaries_0", "topk_boundaries_1",
                              "topk_boundaries_2"), objects):
            self.assertIn(f"/src/{case}/topk_boundaries.o", obj)


if __name__ == "__main__":
    unittest.main()
