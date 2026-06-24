#ifndef FLASH_ATTENTION_HPP
#define FLASH_ATTENTION_HPP

// Flash attention v2
#include <common/pto_tileop.hpp>

using namespace pto;

// 以下为带掩码的FA的实现
template <int S, int qD, int vD, int kTm, int kTk>
void flash_attention_frac(float *out_ptr, float *q_ptr, float *k_ptr,
                          float *v_ptr) {
  // block块及掩码信息
  const float scale = 1.0f / sqrt((float)qD);
  const int Qb = S / kTm;
  const int Kb = S / kTk;
  const int rQ = S % kTm;
  const int rK = S % kTk;

  // 全局张量形状
  using gmQ = global_tensor<float, RowMajor<S, qD>>; // Q: [S×qD]
  using gmK = global_tensor<float, ColMajor<qD, S>>; // K: [qD×S]
  using gmV = global_tensor<float, ColMajor<S, vD>>; // V: [S×vD]
  using gmO = global_tensor<float, RowMajor<S, vD>>; // O: [S×vD]

  // tile 寄存器形状
  // ------------无掩码------------
  using tileQ = TileLeft<float, kTm, qD>;     // [kTm×qD]
  using tileK = TileRight<float, qD, kTk>;    // [qD×kTk]
  using tileW_out = TileAcc<float, kTm, kTk>; // [kTm×kTk]
  using tileW = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
  using tileW_left = TileLeft<float, kTm, kTk>;

  using tileO_out = TileAcc<float, kTm, vD>;
  using tileO = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor>; // [kTm×vD]

  using tileV = TileRight<float, kTk, vD>;        // [kTk×vD]
  using tileMax = Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor>;   // [kTm×1]
  using tileSum = Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor>;   // [kTm×1]
  using tileScale = Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor>; // [kTm×1]
  // ------------带掩码------------
  using tileQ_trows = TileLeft<float, kTm, qD, rQ,
                                       qD>; // 大小：[kTm×qD]，有效大小：[rQ,qD]
  using tileK_tcols =
      TileRight<float, qD, kTk, qD,
                        rK>; // 大小：[qD×kTk]，有效大小：[qD,kTk]
  using tileV_trows = TileRight<float, kTk, vD, rK,
                                        vD>; // 大小：[kTk×vD]，有效大小：[rK,
                                             // vD]
  using tileW_out_tcols =
      TileAcc<float, kTm, kTk, kTm,
                      rK>; // 大小：[kTm×kTk]，有效大小：[kTm,rK]
  using tileW_out_trows =
      TileAcc<float, kTm, kTk, rQ,
                      kTk>; // 大小：[kTm×kTk]，有效大小：[rQ,kTk]
  using tileW_out_tcorner =
      TileAcc<float, kTm, kTk, rQ,
                      rK>; // 大小：[kTm×kTk]，有效大小：[rQ,rK]

  using tileW_tcols = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor, kTm, rK>;
  using tileW_trows = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor, rQ, kTk>;
  using tileW_tcorner = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor, rQ, rK>;

  using tileW_left_tcols = TileLeft<float, kTm, kTk, kTm, rK>;
  using tileW_left_trows = TileLeft<float, kTm, kTk, rQ, kTk>;
  using tileW_left_tcorner = TileLeft<float, kTm, kTk, rQ, rK>;

  using tileO_out_trows = TileAcc<float, kTm, vD, rQ, vD>;
  using tileO_trows = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor, rQ, vD>; // 大小：[kTm×vD]，有效大小：[rQ,vD]

  using tileMax_trows = Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor, rQ, 1>; // 大小：[kTm×1]，有效大小：[rQ,1]
  using tileSum_trows = Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor, rQ, 1>; // 大小：[kTm×1]，有效大小：[rQ,1]
  using tileScale_trows = Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor, rQ, 1>; // 大小：[kTm×1], 有效大小：[rQ,1]

  // 全局迭代器
  using itQ = global_iterator<gmQ, tileQ>;
  using itK = global_iterator<gmK, tileK>;
  using itV = global_iterator<gmV, tileV>;
  using itO = global_iterator<gmO, tileO>;

  itQ gQ(q_ptr);
  itK gK(k_ptr);
  itV gV(v_ptr);
  itO gO(out_ptr);

  // =================attention caculation================
  // 计算分为四步，softmax((Q*K)/sqrt(d))*V
  tileQ tQ;
  for (int i = 0; i < Qb; ++i) {
    // 加载当前Q块 (仅一次)
    tileQ tQ;
    TLOAD(tQ, gQ(i, 0));

    // 初始化状态: 最大值/指数和/输出累加
    tileMax tMax;
    TEXPANDSCALAR(tMax, -1e30f); // 初始化为极小值
    tileSum tSum(0);             // 指数和归零
    tileO_out tO_out(0);         // 输出累加归零
    tileO tO(0);

    //-------------------------------------------------------第一步：tile块Q固定(无掩码，即gQ(i,0))，用j遍历tile块k和v(均无掩码)
    // 单次遍历所有K/V块
    for (int j = 0; j < Kb; ++j) {
      // 加载K_j和V_j
      tileK tK;
      TLOAD(tK, gK(0, j));
      tileV tV;
      TLOAD(tV, gV(j, 0));

      // 计算注意力分数块
      tileW_out tW_out;
      MATMUL(tW_out, tQ, tK);

      // Nz -> RowMajor
      tileW tW;
      TCVT(tW, tW_out);
      TMULS(tW, tW, scale);

      // 计算当前块的行最大值
      tileMax tLocalMax;
      TROWMAX(tLocalMax, tW);

      // 更新全局行最大值
      tileMax tNewMax;
      TMAX(tNewMax, tMax, tLocalMax);

      // 计算旧状态的缩放因子: exp(old_max - new_max)
      tileScale tScaleOld;
      TSUB(tScaleOld, tMax, tNewMax);
      TEXP(tScaleOld, tScaleOld);

      // 调整历史指数和: sum = sum * scale_old
      tileSum tScaledSum;
      TMUL(tScaledSum, tSum, tScaleOld);

      // 计算当前块的指数值 (使用新最大值)
      tileW tNewMaxExpanded;
      TEXPANDCOL(tNewMaxExpanded, tNewMax);
      TSUB(tW, tW, tNewMaxExpanded);
      TEXP(tW, tW);

      // 计算当前块的指数和
      tileSum tLocalSum;
      TROWSUM(tLocalSum, tW);

      // 更新全局指数和
      TADD(tSum, tScaledSum, tLocalSum);

      // 调整历史累加值: O = O * scale_old
      tileO tScaleOldExpanded;
      TEXPANDCOL(tScaleOldExpanded, tScaleOld);

      TMUL(tO, tO, tScaleOldExpanded);

      // RowMajor -> Nz
      TCVT(tO_out, tO);
      // RowMajor -> Nz
      tileW_left tW_left;
      TCVT(tW_left, tW);
      // 计算当前块的加权输出: O_j = W * V
      MATMACC(tO_out, tW_left, tV);
      TCVT(tO, tO_out);
      // 更新最大值状态
      tMax = tNewMax;
    }
    ////-------------------------------------------------------第二步：tile块Q固定(无掩码，即gQ(i,0))，计算尾巴块tile块k*v
    ///(有掩码)
    // 掩码块单独处理
    if constexpr (rK) {
      // 加载K_Kb 和V_Kb
      tileK_tcols tK_tcols;
      TLOAD(tK_tcols, gK(0, Kb));
      tileV_trows tV_trows;
      TLOAD(tV_trows, gV(Kb, 0));

      // 计算注意力分数块
      tileW_out_tcols tW_out_tcols;
      MATMUL(tW_out_tcols, tQ, tK_tcols);

      // Nz -> RowMajor
      tileW_tcols tW_tcols;
      TCVT(tW_tcols, tW_out_tcols);
      TMULS(tW_tcols, tW_tcols, scale);

      // 计算当前块的行最大值
      tileMax tLocalMax;
      TROWMAX(tLocalMax, tW_tcols);

      // 更新全局行最大值
      tileMax tNewMax;
      TMAX(tNewMax, tMax, tLocalMax);

      // 计算旧状态的缩放因子: exp(old_max - new_max)
      tileScale tScaleOld;
      TSUB(tScaleOld, tMax, tNewMax);
      TEXP(tScaleOld, tScaleOld);

      // 调整历史指数和: sum = sum * scale_old
      tileSum tScaledSum;
      TMUL(tScaledSum, tSum, tScaleOld);

      // 计算当前块的指数值 (使用新最大值)
      tileW_tcols tNewMaxExpanded_tcols;
      TEXPANDCOL(tNewMaxExpanded_tcols, tNewMax);
      TSUB(tW_tcols, tW_tcols, tNewMaxExpanded_tcols);
      TEXP(tW_tcols, tW_tcols);

      // 计算当前块的指数和
      tileSum tLocalSum;
      TROWSUM(tLocalSum, tW_tcols);

      // 更新全局指数和
      TADD(tSum, tScaledSum, tLocalSum);

      // 调整历史累加值: O = O * scale_old
      tileO tScaleOldExpanded;
      TEXPANDCOL(tScaleOldExpanded, tScaleOld);

      TMUL(tO, tO, tScaleOldExpanded);

      // RowMajor -> Nz
      TCVT(tO_out, tO);
      // RowMajor -> Nz
      tileW_left_tcols tW_left_tcols;
      TCVT(tW_left_tcols, tW_tcols);
      // 计算当前块的加权输出: O_j = W * V
      MATMACC(tO_out, tW_left_tcols, tV_trows);
      TCVT(tO, tO_out);
      // 更新最大值状态
      tMax = tNewMax;
    }

    // 归一化输出: O = O / sum
    tileSum tInvSum;
    TRECIP(tInvSum, tSum);
    tileO tInvSumExpanded;
    TEXPANDCOL(tInvSumExpanded, tInvSum);
    TMUL(tO, tO, tInvSumExpanded);

    // 写回全局内存-------将第一步和第二部合并，即完成输出tile块的计算，并store到
    // gO(i,0)
    auto dstO = gO(i, 0);
    TSTORE(dstO, tO);
  }

  // 最后的Q-block块(Qb)
  if constexpr (rQ) {

    // 加载当前Q块 (仅一次)
    tileQ_trows tQ_trows;
    TLOAD(tQ_trows, gQ(Qb, 0));

    // 初始化状态: 最大值/指数和/输出累加
    tileMax_trows tMax_trows;
    TEXPANDSCALAR(tMax_trows, -1e30f); // 初始化为极小值
    tileSum_trows tSum_trows(0);       // 指数和归零
    tileO_out_trows tO_out_trows(0);   // 输出累加归零
    tileO_trows tO_trows(0);

    ////-------------------------------------------------------第三步：tile块Q固定（有行掩码，即gQ(Qb,0)），用j遍历计算tile块k*v
    ///(无掩码)
    // 单次遍历所有K/V块
    for (int j = 0; j < Kb; ++j) {
      // 加载K_j和V_j
      tileK tK;
      TLOAD(tK, gK(0, j));
      tileV tV;
      TLOAD(tV, gV(j, 0));

      // 计算注意力分数块
      tileW_out_trows tW_out_trows;
      MATMUL(tW_out_trows, tQ_trows, tK);

      // Nz -> RowMajor
      tileW_trows tW_trows;
      TCVT(tW_trows, tW_out_trows);
      TMULS(tW_trows, tW_trows, scale);

      // 计算当前块的行最大值
      tileMax_trows tLocalMax_trows;
      TROWMAX(tLocalMax_trows, tW_trows);

      // 更新全局行最大值
      tileMax_trows tNewMax_trows;
      TMAX(tNewMax_trows, tMax_trows, tLocalMax_trows);

      // 计算旧状态的缩放因子: exp(old_max - new_max)
      tileScale_trows tScaleOld_trows;
      TSUB(tScaleOld_trows, tMax_trows, tNewMax_trows);
      TEXP(tScaleOld_trows, tScaleOld_trows);

      // 调整历史指数和: sum = sum * scale_old
      tileSum_trows tScaledSum_trows;
      TMUL(tScaledSum_trows, tSum_trows, tScaleOld_trows);

      // 计算当前块的指数值 (使用新最大值)
      tileW_trows tNewMaxExpanded_trows;
      TEXPANDCOL(tNewMaxExpanded_trows, tNewMax_trows);
      TSUB(tW_trows, tW_trows, tNewMaxExpanded_trows);
      TEXP(tW_trows, tW_trows);

      // 计算当前块的指数和
      tileSum_trows tLocalSum_trows;
      TROWSUM(tLocalSum_trows, tW_trows);

      // 更新全局指数和
      TADD(tSum_trows, tScaledSum_trows, tLocalSum_trows);

      // 调整历史累加值: O = O * scale_old
      tileO_trows tScaleOldExpanded_trows;
      TEXPANDCOL(tScaleOldExpanded_trows, tScaleOld_trows);

      TMUL(tO_trows, tO_trows, tScaleOldExpanded_trows);

      // RowMajor -> Nz
      TCVT(tO_out_trows, tO_trows);
      // RowMajor -> Nz
      tileW_left_trows tW_left_trows;
      TCVT(tW_left_trows, tW_trows);
      // 计算当前块的加权输出: O_j = W * V
      MATMACC(tO_out_trows, tW_left_trows, tV);
      TCVT(tO_trows, tO_out_trows);
      // 更新最大值状态
      tMax_trows = tNewMax_trows;
    }
    ////-------------------------------------------------------第四步：tile块Q固定
    ///(有行掩码，即gQ(Qb,0))，计算尾巴的tile块k*v (有掩码)
    // 掩码块单独处理
    if constexpr (rK) {
      // 加载K_Kb 和V_Kb
      tileK_tcols tK_tcols;
      TLOAD(tK_tcols, gK(0, Kb));
      tileV_trows tV_trows;
      TLOAD(tV_trows, gV(Kb, 0));

      // 计算注意力分数块
      tileW_out_tcorner tW_out_tcorner;
      MATMUL(tW_out_tcorner, tQ_trows, tK_tcols);

      // Nz -> RowMajor
      tileW_tcorner tW_tcorner;
      TCVT(tW_tcorner, tW_out_tcorner);
      TMULS(tW_tcorner, tW_tcorner, scale);

      // 计算当前块的行最大值
      tileMax_trows tLocalMax_trows;
      TROWMAX(tLocalMax_trows, tW_tcorner);

      // 更新全局行最大值
      tileMax_trows tNewMax_trows;
      TMAX(tNewMax_trows, tMax_trows, tLocalMax_trows);

      // 计算旧状态的缩放因子: exp(old_max - new_max)
      tileScale_trows tScaleOld_trows;
      TSUB(tScaleOld_trows, tMax_trows, tNewMax_trows);
      TEXP(tScaleOld_trows, tScaleOld_trows);

      // 调整历史指数和: sum = sum * scale_old
      tileSum_trows tScaledSum_trows;
      TMUL(tScaledSum_trows, tSum_trows, tScaleOld_trows);

      // 计算当前块的指数值 (使用新最大值)
      tileW_tcorner tNewMaxExpanded_tcorner;
      TEXPANDCOL(tNewMaxExpanded_tcorner, tNewMax_trows);
      TSUB(tW_tcorner, tW_tcorner, tNewMaxExpanded_tcorner);
      TEXP(tW_tcorner, tW_tcorner);

      // 计算当前块的指数和
      tileSum_trows tLocalSum_trows;
      TROWSUM(tLocalSum_trows, tW_tcorner);

      // 更新全局指数和
      TADD(tSum_trows, tScaledSum_trows, tLocalSum_trows);

      // 调整历史累加值: O = O * scale_old
      tileO_trows tScaleOldExpanded_trows;
      TEXPANDCOL(tScaleOldExpanded_trows, tScaleOld_trows);

      TMUL(tO_trows, tO_trows, tScaleOldExpanded_trows);

      // RowMajor -> Nz
      TCVT(tO_out_trows, tO_trows);
      // RowMajor -> Nz
      tileW_left_tcorner tW_left_tcorner;
      TCVT(tW_left_tcorner, tW_tcorner);
      // 计算当前块的加权输出: O_j = W * V
      MATMACC(tO_out_trows, tW_left_tcorner, tV_trows);
      TCVT(tO_trows, tO_out_trows);
      // 更新最大值状态
      tMax_trows = tNewMax_trows;
    }

    // 归一化输出: O = O / sum
    tileSum_trows tInvSum_trows;
    TRECIP(tInvSum_trows, tSum_trows);
    tileO_trows tInvSumExpanded_trows;
    TEXPANDCOL(tInvSumExpanded_trows, tInvSum_trows);
    TMUL(tO_trows, tO_trows, tInvSumExpanded_trows);

    // 写回全局内存-----将第三步和第四步合并，即完成输出tile块的计算，并store到
    // gO(Qb,0)
    auto dstO = gO(Qb, 0);
    TSTORE(dstO, tO_trows);
  }
}

#endif
