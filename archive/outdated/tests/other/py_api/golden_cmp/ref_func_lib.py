import torch
import torch.nn as nn
import math

def tadd(a, b):
    return a + b

def tsub(a, b):
    return a - b

def texp(a):
    return a.exp()

def matmul(a, b):
    return torch.matmul(a, b)

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
