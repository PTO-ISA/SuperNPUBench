#!/usr/bin/python3

import os
import sys
sys.path.append(os.path.dirname(__file__)+"/../../../output/py_api/elf")
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import math
import tileop_py
from ref_func_lib import *

class Process:
    pass

def compare(res, res_ref):
    print("tensor Res:", res)
    print("tensor Res_Ref:", res_ref)
    loss = torch.mean((res - res_ref) ** 2).item()
    print(f"Loss : {loss:.4e}")
    if loss < 0.01:
        print("Compare Passed")
    else:
        print("Compare fail")

def read_config():
    CASE_MAP = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    for case in config["cases"]:
        name = case["name"]
        group = case["group"]
        input_shapes = case["input_shapes"]
        output_shape = case["output_shape"]
        ref_func_str = case["ref_func"]
        test_func_str = case["test_func"]

        input_shapes_tuple = [tuple(shape) for shape in input_shapes]
        res_shape = tuple(output_shape)
        shape_info = (input_shapes_tuple, res_shape)
        func_info = (eval(ref_func_str), eval(test_func_str))
        if name in CASE_MAP.keys():
            exit(f"Error: find same case:\033[31m{name}\033[0m in config.json, check it")
        CASE_MAP[name] = [group, shape_info, func_info]

    return CASE_MAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tileop test")
    parser.add_argument("-i", dest="case", default="tadd", type=str, help="case name")
    args = parser.parse_args()

    CASE_MAP = read_config()
    # print(CASE_MAP)

    try:
        case_info = CASE_MAP[args.case]
    except KeyError:
        exit(f"Error: case \033[31m{args.case}\033[0m not exist in json file")

    input_shapes, res_shape = case_info[1]
    arg_count = len(input_shapes)

    # init tensor
    input_tensors = []
    for shape in input_shapes:
        tensor = torch.randn(shape) * 10 + 10
        input_tensors.append(tensor)
    res = torch.zeros(res_shape)
    res_ref = torch.zeros(res_shape)

    for i, tensor in enumerate(input_tensors):
        print(f"tensor Src{i} (shape {input_shapes[i]}):", tensor)

        group_op = case_info[0]
        ref_func, test_func = case_info[2]
        print(f"{group_op.upper()} Compare")

        res_ref = ref_func(input_tensors)
        # call c/c++ tileop api
        test_func(res, input_tensors)

        compare(res, res_ref)