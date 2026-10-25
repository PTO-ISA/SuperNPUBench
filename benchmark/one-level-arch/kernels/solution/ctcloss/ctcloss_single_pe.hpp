#ifndef PTO_CTC_LOSS_FORWARD_HPP
#define PTO_CTC_LOSS_FORWARD_HPP
#include <common/pto_tileop.hpp>
#include <cstdint>
#include "ctcloss_params.hpp"

namespace ctc_loss_forward {
using pto::ElementTile;
using pto::TPARTELEMENT;
using Word = uint32_t __attribute__((may_alias));
using Words = ElementTile<uint32_t, 32>;
using Floats = ElementTile<float, 32>;
constexpr uint32_t negative_infinity_bits = 0xff800000u;

// Strides and offsets count elements. Caller validates nonnegative dimensions,
// sufficient storage, nonoverlapping outputs, labels and 32-bit offset bounds.
// log_probs contains finite values or -Inf, never raw logits/NaN/+Inf.
// Public zero-instruction reinterpret views keep the same Tile storage.
// Integer element CFG owns memory/predication; every DP arithmetic op is F32.
__attribute__((always_inline)) inline Floats as_float(Words &words) {
  Floats result;
  auto view = pto::reinterpret_tile<float>(words);
  TCVT(result, view);
  return result;
}
__attribute__((always_inline)) inline Words as_words(Floats &values) {
  Words result;
  auto view = pto::reinterpret_tile<uint32_t>(values);
  TCVT(result, view);
  return result;
}
template <unsigned Site>
__attribute__((always_inline)) inline Floats exp_nonpositive(Floats &value) {
  auto bits = as_words(value);
  Words bounded;
  auto &input = TPARTELEMENT(bits);
  auto &output = TPARTELEMENT(bounded);
  // exp(x) rounds to FP32 zero for x <= -104 (below half the least
  // subnormal). Map only that mathematically exact underflow domain to -Inf.
  // The frozen model's finite exp(-1000) overflows; exp(-Inf) is well-defined.
#pragma pto element for
  for (unsigned i=0;i<32;++i) {
    if (input[i] >= 0xc2d00000u) output[i] = negative_infinity_bits;
    else output[i] = input[i];
  }
  auto argument = as_float(bounded);
  Floats result;
  TEXP(result, argument);
  return result;
}
template <unsigned Site>
__attribute__((always_inline)) inline Words logadd3(Words &aa, Words &bb, Words &cc) {
  auto a = as_float(aa), b = as_float(bb), c = as_float(cc);
  Floats ab_max, maximum;
  TMAX(ab_max, a, b); TMAX(maximum, ab_max, c);
  auto maximum_bits = as_words(maximum);
  Words safe_maximum_bits;
  auto &m = TPARTELEMENT(maximum_bits);
  auto &safe = TPARTELEMENT(safe_maximum_bits);
#pragma pto element for
  for (unsigned i = 0; i < 32; ++i) {
    if (m[i] == negative_infinity_bits) safe[i] = 0u;
    else safe[i] = m[i];
  }
  auto safe_maximum = as_float(safe_maximum_bits);
  Floats da, db, dc, eab, total;
  TSUB(da, a, safe_maximum); TSUB(db, b, safe_maximum);
  TSUB(dc, c, safe_maximum);
  auto ea=exp_nonpositive<Site*3>(da), eb=exp_nonpositive<Site*3+1>(db);
  auto ec=exp_nonpositive<Site*3+2>(dc);
  TADD(eab, ea, eb); TADD(total, eab, ec);
  // Explicit all-unreachable branch also avoids invoking TLOG on zero.
  auto total_bits = as_words(total);
  Words safe_total_bits;
  auto &sum = TPARTELEMENT(total_bits);
  auto &safe_sum = TPARTELEMENT(safe_total_bits);
#pragma pto element for
  for (unsigned i = 0; i < 32; ++i) {
    if (m[i] == negative_infinity_bits) safe_sum[i] = 0x3f800000u;
    else safe_sum[i] = sum[i];
  }
  auto safe_total = as_float(safe_total_bits);
  Floats logarithm, value;
  TLOG(logarithm, safe_total); TADD(value, logarithm, safe_maximum);
  auto value_bits = as_words(value);
  Words result;
  auto &v = TPARTELEMENT(value_bits); auto &r = TPARTELEMENT(result);
#pragma pto element for
  for (unsigned i = 0; i < 32; ++i) {
    if (m[i] == negative_infinity_bits) r[i] = negative_infinity_bits;
    else r[i] = v[i];
  }
  return result;
}
__attribute__((always_inline)) inline void compiler_order_boundary() {
  // Compiler ordering only. This backend MUST run on one logical PE, whose
  // scalar/TLSU accesses use the same coherent state (also exercised by the
  // TopK coherence probe). All 32-element regions reconverge on that issuer.
  // There is NO inter-thread/global hardware barrier here. Do not launch this
  // implementation across PEs or use this helper as a portable SIMT barrier.
  asm volatile("" ::: "memory");
}

// Flatten [BatchSlots, StateLanes] into one 32-element group. Time is uniform
// and sequential. State tiles advance in increasing order and retain all
// previous-tile rows in GM for boundary s-1/s-2 predecessors. No scalar DP.
template <unsigned StateLanes = 16, unsigned BatchSlots = 2>
__attribute__((noinline)) void forward(const Arguments &args) {
  static_assert(StateLanes * BatchSlots == 32 &&
                (StateLanes & (StateLanes - 1)) == 0);
  static_assert(StateLanes == 16 && BatchSlots == 2,
                "This frozen compiler profile is validated for 16 states x 2 batch slots only");
  const Word *lp = reinterpret_cast<const Word *>(args.log_probs);
  Word *alpha = reinterpret_cast<Word *>(args.log_alpha);
  Word *loss = reinterpret_cast<Word *>(args.negative_log_likelihood);
  const uint32_t *targets = args.targets, *input_lengths = args.input_lengths;
  const uint32_t *target_lengths = args.target_lengths, *offsets = args.target_offsets;
  const uint32_t batch = args.batch, blank = args.blank, max_states = args.max_states;
  const uint32_t lp_ts = args.lp_time_stride, lp_bs = args.lp_batch_stride;
  const uint32_t lp_cs = args.lp_class_stride, target_stride = args.target_stride;
  const uint32_t alpha_bs = args.alpha_batch_stride;
  const uint32_t alpha_ts = args.alpha_time_stride, alpha_ss = args.alpha_state_stride;
  alignas(128) uint32_t transfer[32];
  for (uint32_t group = 0; group < batch;
       group += (batch-group < BatchSlots ? batch-group : BatchSlots)) {
    for (uint32_t base = 0; base < max_states;
         base += (max_states-base < StateLanes ? max_states-base : StateLanes)) {
      for (uint32_t t = 0; t < args.time; ++t) {
        Words aa, bb, cc, emission_bits;
        auto &a = TPARTELEMENT(aa); auto &b = TPARTELEMENT(bb);
        auto &c = TPARTELEMENT(cc); auto &emission = TPARTELEMENT(emission_bits);
#pragma pto element for
        for (unsigned i = 0; i < 32; ++i) {
          const uint32_t sample = group + i / StateLanes;
          const uint32_t state = base + i % StateLanes;
          uint32_t av = negative_infinity_bits, bv = negative_infinity_bits;
          uint32_t cv = negative_infinity_bits, ev = 0u;
          if (sample < batch && state < max_states) {
            const uint32_t u = target_lengths[sample];
            if (t < input_lengths[sample] && state < 2u * u + 1u) {
              uint32_t label = blank;
              if ((state & 1u) != 0u)
                label = targets[offsets[sample] + (state / 2u) * target_stride];
              ev = lp[t * lp_ts + sample * lp_bs + label * lp_cs];
              if (t == 0u) {
                if (state < 2u) av = 0u;
              } else {
                const uint32_t prev = sample * alpha_bs + (t - 1u) * alpha_ts;
                av = alpha[prev + state * alpha_ss];
                if (state > 0u) bv = alpha[prev + (state - 1u) * alpha_ss];
                if (state > 1u && (state & 1u) != 0u) {
                  const uint32_t earlier = targets[offsets[sample] + (state / 2u - 1u) * target_stride];
                  if (label != earlier) cv = alpha[prev + (state - 2u) * alpha_ss];
                }
              }
            }
          }
          a[i] = av; b[i] = bv; c[i] = cv; emission[i] = ev;
        }
        auto recurrence_bits = logadd3<0>(aa, bb, cc);
        auto recurrence = as_float(recurrence_bits), emission_tile = as_float(emission_bits);
        Floats updated;
        TADD(updated, recurrence, emission_tile);
        auto updated_bits = as_words(updated);
        pto::global_tensor<uint32_t, pto::RowMajor<32, 1>> transfer_memory(transfer);
        TSTORE(transfer_memory, updated_bits);
        compiler_order_boundary();
        // Layout-only transfer: the frozen compiler accepts only unit-stride
        // ordinary element stores. DP math above remains native Tile math.
        for (unsigned i = 0; i < 32; ++i) {
          const uint32_t sample = group + i / StateLanes;
          const uint32_t state = base + i % StateLanes;
          if (sample < batch && state < max_states)
            alpha[sample * alpha_bs + t * alpha_ts + state * alpha_ss] = transfer[i];
        }
        compiler_order_boundary();
      }
    }
    Words last_bits, previous_bits, zero_bits;
    auto &last = TPARTELEMENT(last_bits); auto &previous = TPARTELEMENT(previous_bits);
    auto &zero = TPARTELEMENT(zero_bits);
#pragma pto element for
    for (unsigned i = 0; i < 32; ++i) {
      uint32_t lv = negative_infinity_bits, pv = negative_infinity_bits;
      const uint32_t sample = group + i / StateLanes;
      if (sample < batch) {
        const uint32_t length = input_lengths[sample], u = target_lengths[sample];
        if (length == 0u) {
          if (u == 0u) lv = 0u;
        } else {
          const uint32_t row = sample * alpha_bs + (length - 1u) * alpha_ts;
          lv = alpha[row + (2u * u) * alpha_ss];
          if (u > 0u) pv = alpha[row + (2u * u - 1u) * alpha_ss];
        }
      }
      last[i] = lv; previous[i] = pv; zero[i] = negative_infinity_bits;
    }
    auto likelihood = logadd3<1>(last_bits, previous_bits, zero_bits);
    pto::global_tensor<uint32_t, pto::RowMajor<32, 1>> transfer_memory(transfer);
    TSTORE(transfer_memory, likelihood);
    compiler_order_boundary();
    for (unsigned i = 0; i < BatchSlots; ++i) {
      if (group + i < batch)
        loss[group + i] = transfer[i * StateLanes] ^ 0x80000000u;
    }
    compiler_order_boundary();
  }
}
} // namespace ctc_loss_forward

#include "ctcloss_tiling.hpp"

// Normal solution interface: inputs, host tiling, outputs and offset workspace.
// Inputs/metadata are validated before launch; loss is unreduced.
inline void ctcloss_v2_single_pe(
    const float *log_probs, const int32_t *targets,
    const int32_t *input_lengths, const int32_t *target_lengths,
    const CtcLossTilingData *tiling_info, float *loss, float *log_alpha,
    const uint32_t *target_offsets) {
  const auto args=bind_ctcloss_arguments(log_probs,targets,input_lengths,
      target_lengths,tiling_info,loss,log_alpha,target_offsets);
  ctc_loss_forward::forward(args);
}
#endif
