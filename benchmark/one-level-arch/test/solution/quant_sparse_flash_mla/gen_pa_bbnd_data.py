#!/usr/bin/env python3
"""Generate PA_BBND (page-attention block-table) KV data for the v2.1 / v3.1
QSMLA kernels from an existing BSND data set.

Physical layout produced (BBND, N2 = 1): [BlockNum, BlockSize, 1, D] with a
per-batch block table [B, max_blocks].  A logically ordered row `pos` lives at
pool row `block_table[b * max_blocks + pos / block_size] * block_size
+ pos % block_size` (N2 = 1 keeps the in-pool row stride D), matching the ops
arch35 CSA blockTableGm addressing.

Identity paging (block_size = 128): the paged copies reproduce the same
logical rows as the BSND inputs, so gfrun validates PA against the existing
golden.  ori rows are block-aligned (S2 multiple of the block size in the
csa_small family); the cmp pool is zero-padded to a full block.

Usage: gen_pa_bbnd_data.py --data-dir <case data dir> --s2 128 --cmp-s2 64 \
       --n2 1 --d 512 --b 1 --ext hif8 --block-size 128
"""
import argparse
import os
import struct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--s2", type=int, required=True)
    ap.add_argument("--cmp-s2", type=int, required=True)
    ap.add_argument("--n2", type=int, default=1)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--ext", default="hif8")
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()

    d = args.data_dir
    row = args.d * args.n2
    blk = args.block_size
    ori = open(os.path.join(d, f"ori_kv.{args.ext}.bin"), "rb").read()
    cmp_ = open(os.path.join(d, f"cmp_kv.{args.ext}.bin"), "rb").read()
    assert len(ori) == args.b * args.s2 * row, (len(ori), args.b * args.s2 * row)
    assert len(cmp_) == args.b * args.cmp_s2 * row, len(cmp_)

    # ORI pool: per batch one block when block_size == s2 (block-aligned pool).
    assert args.s2 % blk == 0, "ori s2 must be a multiple of the block size"
    ori_blocks = args.s2 // blk
    with open(os.path.join(d, f"ori_kv_pa.{args.ext}.bin"), "wb") as f:
        f.write(ori)

    # CMP pool: pad each batch's valid rows to a full block.
    cmp_blocks = (args.cmp_s2 + blk - 1) // blk
    with open(os.path.join(d, f"cmp_kv_pa.{args.ext}.bin"), "wb") as f:
        for b in range(args.b):
            chunk = cmp_[b * args.cmp_s2 * row:(b + 1) * args.cmp_s2 * row]
            f.write(chunk + bytes((cmp_blocks * blk - args.cmp_s2) * row))

    # Identity block tables: block b of batch b maps to physical block b.
    with open(os.path.join(d, "ori_block_table.int32.bin"), "wb") as f:
        for b in range(args.b):
            for i in range(ori_blocks):
                f.write(struct.pack("<i", b * ori_blocks + i))
    with open(os.path.join(d, "cmp_block_table.int32.bin"), "wb") as f:
        for b in range(args.b):
            for i in range(cmp_blocks):
                f.write(struct.pack("<i", b * cmp_blocks + i))
    print(f"PA data written to {d}: ori_pa {os.path.getsize(os.path.join(d, f'ori_kv_pa.{args.ext}.bin'))} B,"
          f" cmp_pa {os.path.getsize(os.path.join(d, f'cmp_kv_pa.{args.ext}.bin'))} B,"
          f" tables {args.b * ori_blocks * 4} / {args.b * cmp_blocks * 4} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
