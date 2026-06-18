import torch
import math
import numpy as np
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.Tensor, 
                                 K: torch.Tensor, 
                                 V: torch.Tensor) -> torch.Tensor:
    """
    计算缩放点积注意力
    Args:
        Q: Query 矩阵, 形状为 [batch_size, seq_len_q, d_k]
        K: Key 矩阵, 形状为 [batch_size,  d_k, seq_len_k]
        V: Value 矩阵, 形状为 [batch_size, seq_len_k, d_v]
    
    Returns:
        O: 注意力输出, 形状为 [batch_size, seq_len_q, d_v]
    """
    d_k = Q.size(-1)  # 获取特征维度 d
    
    # 计算 QK
    scores = torch.matmul(Q, K)  # [batch_size, seq_len_q, seq_len_k]
    
    # 缩放操作：除以 sqrt(d)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # 计算 softmax
    attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len_q, seq_len_k]
    # 乘以 V 得到输出
    O = torch.matmul(attn_weights, V)  # [batch_size, seq_len_q, d_v]
    
    return O

# 示例用法
if __name__ == "__main__":
    # 初始化数据 
    BatSz = 1
    globS = 17
    globD = 16
    Q = torch.zeros(BatSz, globS, globD)
    K = torch.zeros(BatSz, globD, globS)
    V = torch.zeros(BatSz, globS, globD)
    for i in range(BatSz):
        for j in range(globS):
            for k in range(globD):
                Q[i][j][k]=math.sin(j * 0.1) + math.cos(k * 0.05)
                V[i][j][k]=math.cos((k*globS+j)*0.2)
                K[i][k][j]=math.sin((j*globD+k)*0.1)
    # 计算注意力输出
    O = scaled_dot_product_attention(Q, K, V)
    print("Q:")  
    for i in range(BatSz):
        for j in range(globS):
            for d in range(globD):
                print(f"{Q[i][j][d].item():.6f}",end=' ')
            print()
        print()

    print("K:")  
    for i in range(BatSz):
        for j in range(globD):
            for d in range(globS):
                print(f"{K[i][j][d].item():.6f}",end=' ')
            print()
        print()

    print("V:")  
    for i in range(BatSz):
        for j in range(globS):
            for d in range(globD):
                print(f"{V[i][j][d].item():.6f}",end=' ')
            print()
        print()

    print("O:")  
    for i in range(BatSz):
        for j in range(globS):
            for d in range(globD):
                print(f"{O[i][j][d].item():.6f}",end=' ')
            print()
        print()