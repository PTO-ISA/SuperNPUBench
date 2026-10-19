#!/usr/bin/env python3
import unittest
from pathlib import Path

import numpy as np
import matmul_lowp_fixture as fixture


class LowpFixtureTest(unittest.TestCase):
    def test_inventory_matches_every_original_active_command(self):
        self.assertEqual(fixture.active_commands(Path(__file__).parents[1]/'element_original_profiles.json'), fixture.profiles())
        self.assertEqual(len(fixture.profiles()), 7)

    def test_packed_lane_order_and_all_codes(self):
        lanes = np.arange(16, dtype=np.uint8)
        self.assertEqual(fixture.pack(lanes), bytes.fromhex('1032547698badcfe'))
        np.testing.assert_array_equal(fixture.unpack(bytes.fromhex('1032547698badcfe')), lanes)
        with self.assertRaises(ValueError):
            fixture.pack([1])
        with self.assertRaises(ValueError):
            fixture.pack([1, 16])

    def test_all_hif4_and_e2m1_lanes(self):
        lanes = np.arange(16, dtype=np.uint8)
        np.testing.assert_array_equal(fixture.decode_fp4(lanes, True),
            [0,.25,.5,.75,1,1.25,1.5,1.75,-0.,-.25,-.5,-.75,-1,-1.25,-1.5,-1.75])
        np.testing.assert_array_equal(fixture.decode_fp4(lanes, False),
            [0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])

    def test_hif4_scale_both_bit_planes(self):
        word = np.uint32(0xc0 | (1 << (8+2)) | (1 << (16+5)))
        self.assertEqual(float(fixture.hif4_scale(word, 15)), 1)
        self.assertEqual(float(fixture.hif4_scale(word, 16)), 2)
        self.assertEqual(float(fixture.hif4_scale(word, 20)), 4)
        self.assertEqual(float(fixture.hif4_scale(word, 24)), 1)
        self.assertEqual(float(fixture.hif4_scale(np.uint32(0xb8), 0)), .25)
        with self.assertRaises(ValueError):
            fixture.hif4_scale(np.uint32(255), 0)

    def test_scale_expansion_axes_and_group_reset(self):
        left = np.full((2,128), 4, dtype=np.uint8)
        right = np.full((128,3), 4, dtype=np.uint8)
        sa = np.array([[0xc0,0xbc], [0xc4,0xc0]], dtype=np.uint32)
        sb = np.array([[0xc0,0xbc,0xc4],[0xbc,0xc4,0xc0]], dtype=np.uint32)
        a = fixture.expand_hif4(left, sa, True)
        b = fixture.expand_hif4(right, sb, False)
        np.testing.assert_array_equal(a[:,[0,63,64,127]], [[1,1,.5,.5],[2,2,1,1]])
        np.testing.assert_array_equal(b[[0,63,64,127]], [[1,.5,2],[1,.5,2],[.5,2,1],[.5,2,1]])
        np.testing.assert_array_equal(a@b, [[80,96,160],[160,192,320]])

    def test_a16_group_order_per_column_and_defined_tails(self):
        a = np.ones((2,256), dtype=np.float32)
        a[1,128:] = 2
        b = np.full((256,3), .5, dtype=np.float32)
        scale = np.array([[.125,.25,.5],[.5,.25,.125]], dtype=np.float32)
        np.testing.assert_array_equal(fixture.reference_a16w4(a,b,scale), [[40,32,40],[72,48,48]])
        raw, golden = fixture.inputs(('A16W4','',16,16,128,16,16,128))
        self.assertEqual(len(raw['src0']),fixture.original_capacity(16,128)*2)
        self.assertEqual(len(raw['src1']),fixture.original_capacity(128,16))
        self.assertEqual(raw['src1'][128*16//2:], b'\x6d'*(fixture.original_capacity(128,16)-128*16//2))
        self.assertEqual(len(raw['src2']),fixture.original_capacity(128,16)*4)
        self.assertEqual(raw['src2'][16*4:], b'\x6d'*(fixture.original_capacity(128,16)*4-16*4))
        self.assertEqual(len(golden),fixture.original_capacity(16,16)*4)


if __name__ == '__main__':
    unittest.main()
