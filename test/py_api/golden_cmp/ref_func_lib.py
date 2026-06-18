import torch
import torch.nn as nn
import math

def tadd(a, b):
    return a + b

def tsub(a, b):
    return a - b

def texp(a):
    return a.exp()

def tmax(a, b):
    return torch.maximum(a, b)

def softmax(a):
    softmax = nn.Softmax(dim=1)
    return softmax(a)

def attention(q, k, v):
    d = q.shape[1]
    scale = 1.0 / math.sqrt(d)
    out_ref = torch.mm(softmax(torch.mm(q, torch.t(k)) * scale), v)
    return out_ref

def tcvt(col, row, data_type, input_tensor, layout_in, layout_out):
    bytes_per_element = get_bytes_per_element(data_type)
    flat_input = input_tensor.flatten()
    
    if layout_in == "Rowmajor" and layout_out == "Nz":
        return convert_rowmajor_to_nz(col, row, bytes_per_element, flat_input)
    elif layout_in == "Nz" and layout_out == "Rowmajor":
        return convert_nz_to_rowmajor(col, row, bytes_per_element, flat_input)
    elif layout_in == "Nz" and layout_out == "Zn":
        return convert_nz_to_zn(col, row, bytes_per_element, flat_input)
    elif layout_in == "Zn" and layout_out == "Nz":
        return convert_zn_to_nz(col, row, bytes_per_element, flat_input)
    else:
        raise ValueError("不支持的布局转换，仅支持 Nz <-> Zn")


def get_bytes_per_element(data_type):
    if data_type == 'fp32':
        return 4
    elif data_type == 'fp16':
        return 2
    else:
        raise ValueError("不支持的数据类型，应为 'fp32' 或 'fp16'")


def convert_rowmajor_to_nz(col, row, bytes_per_element, flat_input):
    block_row = 16
    block_col = 32 // bytes_per_element
    output = torch.zeros(row * col)
    for i in range(col):
        sub_tile_i = i // block_col
        idx_i = i % block_col
        for j in range(row):
            sub_tile_j = j // block_row
            idx_j = j % block_row
            src_idx = i + j * col
            dst_idx = sub_tile_i * row * block_col + sub_tile_j * block_row * block_col + idx_j * block_col + idx_i
            output[dst_idx] = flat_input[src_idx]
    return output.view(row, col)


def convert_nz_to_rowmajor(col, row, bytes_per_element, flat_input):
    block_col = 32 // bytes_per_element
    block_row = 16
    output = torch.zeros(row * col)
    for i in range(col):
        sub_tile_i = i // block_col
        idx_i = i % block_col
        for j in range(row):
            sub_tile_j = j // block_row
            idx_j = j % block_row
            src_idx = sub_tile_i * row * block_col + sub_tile_j * block_row * block_col + idx_j * block_col + idx_i
            dst_idx = i + j * col
            output[dst_idx] = flat_input[src_idx]
    return output.view(row, col)


def convert_nz_to_zn(col, row, bytes_per_element, flat_input):
    nz_block_col = 64 // bytes_per_element
    nz_block_row = 16
    zn_block_col = 16
    zn_block_row = 32 // bytes_per_element
    output = torch.zeros(row * col)
    for i in range(col):
        zn_sub_tile_i = i // zn_block_col
        zn_idx_i = i % zn_block_col
        nz_sub_tile_i = i // nz_block_col
        nz_idx_i = i % nz_block_col
        for j in range(row):
            zn_sub_tile_j = j // zn_block_row
            nz_sub_tile_j = j // nz_block_row
            zn_idx_j = j % zn_block_row
            nz_idx_j = j % nz_block_row
            src_idx = nz_sub_tile_i * row * nz_block_col + nz_sub_tile_j * nz_block_row * nz_block_col + nz_idx_j * nz_block_col + nz_idx_i
            dst_idx = zn_sub_tile_i * zn_block_row * zn_block_col + zn_sub_tile_j * zn_block_row * col + zn_idx_i * zn_block_row + zn_idx_j
            output[dst_idx] = flat_input[src_idx]
    return output.view(row, col)


def convert_zn_to_nz(col, row, bytes_per_element, flat_input):
    zn_block_col = 16
    zn_block_row = 32 // bytes_per_element
    nz_block_col = 64 // bytes_per_element
    nz_block_row = 16
    output = torch.zeros(row * col)
    for i in range(col):
        zn_sub_tile_i = i // zn_block_col
        zn_idx_i = i % zn_block_col
        nz_sub_tile_i = i // nz_block_col
        nz_idx_i = i % nz_block_col
        for j in range(row):
            zn_sub_tile_j = j // zn_block_row
            nz_sub_tile_j = j // nz_block_row
            zn_idx_j = j % zn_block_row
            nz_idx_j = j % nz_block_row
            src_idx = zn_sub_tile_i * zn_block_row * zn_block_col + zn_sub_tile_j * zn_block_row * col + zn_idx_i * zn_block_row + zn_idx_j
            dst_idx = nz_sub_tile_i * row * nz_block_col + nz_sub_tile_j * nz_block_row * nz_block_col + nz_idx_j * nz_block_col + nz_idx_i
            output[dst_idx] = flat_input[src_idx]
    return output.view(row, col)