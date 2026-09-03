#!/usr/bin/env python3
import unittest

import gen_cases


class BenchmarkStabilityTest(unittest.TestCase):
    def test_vector_keeps_main_fixed_inputs(self):
        case = next(case for case in gen_cases.V if case.op == "TADD")
        source = gen_cases.emit_vector(case, "fp32")
        self.assertIn("fill_const(a, 256, (float)2)", source)
        self.assertIn("fill_const(b, 256, (float)1)", source)
        self.assertLess(source.index("BENCHEND"), source.index("publish_cross_model_result"))

    def test_concat_keeps_main_secondary_input(self):
        case = next(case for case in gen_cases.V if case.op == "TCONCAT")
        source = gen_cases.emit_vector(case, "fp32")
        self.assertIn("fill_const(b, 512, (float)3)", source)

    def test_memory_keeps_main_fixed_input_and_byte_offsets(self):
        case = next(case for case in gen_cases.ME if case.op == "MGATHER")
        source = gen_cases.emit_memory(case, "fp32")
        self.assertIn("fill_const(a, M*N, (float)2)", source)
        self.assertIn("idx[i] *= sizeof(float)", source)

    def test_cube_keeps_main_fixed_inputs(self):
        case = next(case for case in gen_cases.C if case.op == "TMATMUL" and case.dtypes == ("fp32",))
        source = gen_cases.emit_cube(case, "fp32")
        self.assertIn("fill_const(a, M*K, (float)0.25)", source)
        self.assertIn("fill_const(b, K*N, (float)0.25)", source)


if __name__ == "__main__":
    unittest.main()
