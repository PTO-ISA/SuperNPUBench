#include <common/pto_tileop.hpp>

using namespace pto;

template <int S, int D, int kTm, int kTk>
void flash_attention1(float *out_ptr,
                     float *q_ptr,
                     float *k_ptr,
                     float *v_ptr) {
  // 全局张量形状
  using gmQ = global_tensor<float, RowMajor<S, D>>;  // Q: [S×D]
  using gmK = global_tensor<float, ColMajor<D, S>>;  // K: [D×S]
  using gmV = global_tensor<float, RowMajor<S, D>>;  // V: [S×D]
  using gmO = global_tensor<float, RowMajor<S, D>>;  // O: [S×D]

  // tile 寄存器形状
  using tileQ      = TileLeft<float, kTm, D>;    // [kTm×D]
  using tileK      = TileRight<float, D, kTk>;    // [D×kTk]
  using tileW      = TileLeft<float, kTm, kTk>;  // [kTm×kTk]
  using tileO      = TileAcc<float, kTm, D>;    // [kTm×D]


  using tileV      = TileRight<float, kTk, D>;    // [kTk×D]

  using tileMax    = Tile<Location::Vec, float, kTm, 1, BLayout::ColMajor>;    // [kTm×1]
  using tileSum    = Tile<Location::Vec, float, kTm, 1, BLayout::ColMajor>;    // [kTm×1]
  using tileInvSum = Tile<Location::Vec, float, kTm, 1, BLayout::ColMajor>;    // [kTm×1]

  // 全局迭代器（按行切 Q/O，以 kTm 递增；按列切 K/V，以 kTk 递增）
  using itQ = global_iterator<gmQ, tileQ>;
  using itK = global_iterator<gmK, tileK>;
  using itV = global_iterator<gmV, tileV>;
  using itO = global_iterator<gmO, tileO>;

  itQ gQ(q_ptr);
  itK gK(k_ptr);
  itV gV(v_ptr);
  itO gO(out_ptr);

  const float scale = 1.0f / sqrt((float)D);
  const int Qb = (S + kTm - 1) / kTm;
  const int Kb = (S + kTk - 1) / kTk;

  // 输出 tile 初始化
  tileO tO(0);

  // 对每个 Q‑block (i)
  for (int i = 0; i < Qb; ++i) {
    auto dstO = gO(i, 0);
    // 1. 扫描所有 K‑blocks，计算行最大值
    tileMax tMax;
    // 初始化为极小值
    TEXPANDSCALAR(tMax, -1e30f);

    for (int j = 0; j < Kb; ++j) {
      // load Q_i, K_j
      tileQ tQ; TCOPYIN(tQ, gQ(i, 0));
      tileK tK; TCOPYIN(tK, gK(0, j));
      // 计算分数块
      tileW tW;
      MATMUL(tW, tQ, tK);
      TMULS(tW, tW, scale);
      // 本地行最大
      tileMax tLocalMax;
      TROWMAX(tLocalMax, tW);
      // 全局行最大累积
      tileMax tNextMax;
      TMAX(tNextMax, tMax, tLocalMax);
      tMax = tNextMax;
    }

    // 2. 扫描所有 K‑blocks，计算行 exp 和的累加
    tileSum tSum(0);
    for (int j = 0; j < Kb; ++j) {
      tileQ tQ; TCOPYIN(tQ, gQ(i, 0));
      tileK tK; TCOPYIN(tK, gK(0, j));
      tileW tW;
      MATMUL(tW, tQ, tK);
      TMULS(tW, tW, scale);
      // 减最大值、exp
      TSUB(tW, tW, tMax);
      TEXP(tW, tW);
      // 本地行和
      tileSum tLocalSum;
      TROWSUM(tLocalSum, tW);
      // 累加
      tileSum tNextSum;
      TADD(tNextSum, tSum, tLocalSum);
      tSum = tNextSum;
    }
    // 计算行倒数
    tileInvSum tInv;
    TRECIP(tInv, tSum);

    // 3. 重算加权，乘 V 并累加到输出
    tO = tileO(0);
    for (int j = 0; j < Kb; ++j) {
      tileQ tQ; TCOPYIN(tQ, gQ(i, 0));
      tileK tK; TCOPYIN(tK, gK(0, j));
      tileV tV; TCOPYIN(tV, gV(j, 0));

      tileW tW;
      MATMUL(tW, tQ, tK);
      TMULS(tW, tW, scale);
      // softmax
      TSUB(tW, tW, tMax);
      TEXP(tW, tW);
      // 归一化：展开行倒数 -> 点乘
      tileW tInvExp;
      TEXPANDROW(tInvExp, tInv);
      TMUL(tW, tW, tInvExp);

      // 加权累加
      MATMACC(tO, tW, tV);
    }

    // 写回 global
    TCOPYOUT(dstO, tO);
  }
}
