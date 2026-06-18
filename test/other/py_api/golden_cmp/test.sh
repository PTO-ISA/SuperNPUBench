#! /bin/bash

python3 golden_cmp/golden_cmp.py -i tadd
python3 golden_cmp/golden_cmp.py -i tsub
python3 golden_cmp/golden_cmp.py -i texp
python3 golden_cmp/golden_cmp.py -i tmax
python3 golden_cmp/golden_cmp.py -i matmul_frac
python3 golden_cmp/golden_cmp.py -i matmul_mask
python3 golden_cmp/golden_cmp.py -i matmul_rm
python3 golden_cmp/golden_cmp.py -i matmul_tile_rm
python3 golden_cmp/golden_cmp.py -i matmul_tile_frac
python3 golden_cmp/golden_cmp.py -i softmax
python3 golden_cmp/golden_cmp.py -i flash_attention
python3 golden_cmp/golden_cmp.py -i flash_attention_frac
python3 golden_cmp/golden_cmp.py -i flash_attention_rm

