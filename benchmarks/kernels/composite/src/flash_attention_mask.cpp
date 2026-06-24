#include "flash_attention_mask.hpp"
#include <common/pto_tileop.hpp>
#include <cstring>
#include <iomanip>

#ifndef BatSz
#define BatSz 1
#endif

#ifndef globS
#define globS 320
#endif

#ifndef globD
#define globD 128
#endif

#ifndef tilM
#define tilM 64
#endif

#ifndef tilK
#define tilK 32
#endif

int main() {

#ifdef LINX_PMC
  PMC_START();
#endif

  float q[BatSz * globS * globD];
  float k[BatSz * globS * globD];
  float v[BatSz * globS * globD];
  float out[BatSz * globS * globD];

  // 计算总元素个数
  const size_t total = BatSz * globS * globD;

  // 循环赋值1.0
  // for (size_t i = 0; i < total; ++i) {
  //   q[i] = 1.0f ; // 用1.0f显式指定float类型，更规范
  //   k[i] = 1.0f;
  //   v[i] = 0.1f;
  // }
  // 按“序列维度(S) + 特征维度(D)”的索引赋值，确保多样性
  for (size_t i = 0; i < total; ++i) {
    // 计算当前元素在序列中的位置 j（0~31）和特征维度 d（0~255）
    size_t j = (i / globD) % globS; // 序列索引（行）
    size_t d = i % globD;           // 特征索引（列）

    // Q：随序列和特征变化（增加随机性）
    q[i] = sin(j * 0.1f) + cos(d * 0.05f); // 不同(j,d)有不同值

    // K：随特征维度变化（打破全1，但保持简单模式）
    // k[i] = 1.0f +
    //        (d % 4) * 0.1f; // 特征每4维一组，值递增0.1（0~3组：1.0,1.1,1.2,1.3）
    k[i]=sin(0.1*i);

    // V：随序列和特征变化（核心：让不同位置的V值不同）
    // v[i] = (j + 1) * 0.1f + (d + 1) * 0.01f; //序列越靠后、特征维度越大，值越大
    v[i]=cos(0.2*i);
  }

  if (!strcmp(MODE, "RM")) {
    flash_attention_rm<BatSz * globS, globD, globD, tilM, tilK>(out, q, k, v);
  } else if (!strcmp(MODE, "FRAC")) {
    flash_attention_frac<BatSz * globS, globD, globD, tilM, tilK>(out, q, k, v);
  } else if (!strcmp(MODE, "ALL")) {
    flash_attention<BatSz * globS, globD, globD, tilM, tilK>(out, q, k, v);
  }

#ifdef LINX_PMC
  PMC_END();
#endif

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Q=";
  for (size_t i = 0; i < BatSz * globS * globD; i++) {
    if (i % globD == 0) {
      std::cout << std::endl;
    }
    std::cout << q[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "K=";
  for (size_t i = 0; i < BatSz * globS * globD; i++) {
    if (i % globS == 0) {
      std::cout << std::endl;
    }
    std::cout << k[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "V=";
  for (size_t i = 0; i < BatSz * globS * globD; i++) {
    if (i % globD == 0) {
      std::cout << std::endl;
    }
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "O=";
  for (size_t i = 0; i < BatSz * globS * globD; i++) {
    if (i % globD == 0) {
      std::cout << std::endl;
    }
    std::cout << out[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}