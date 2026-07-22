#ifndef SUPERNPU_GET_FUSED_MAPPING_PTO_HPP
#define SUPERNPU_GET_FUSED_MAPPING_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// Source: TileKernels/tile_kernels/moe/get_fused_mapping_kernel.py
// Revision: 36d9e45d38e204ebb87e6f6e833821eee0482fe5

namespace supernpu::tile_isa {

// Build an expert-major routing order for one compile-time routing panel.
// Sorting carries the original token-topk position beside each expert ID. The
// remaining maps are then derived from those tile values without scalar access
// to tensor storage.
template <int NumTokens, int NumTopk, int NumExperts, int Alignment = 64>
void get_fused_mapping(std::int32_t *topk_idx,
                       std::int32_t *pos_to_expert,
                       std::int32_t *pos_to_token,
                       std::int32_t *pos_to_token_topk,
                       std::int32_t *token_topk_to_pos,
                       std::int32_t *expert_start,
                       std::int32_t *expert_end,
                       std::int32_t *num_tokens_per_expert) {
  static_assert(NumTokens > 0 && NumTopk > 0 && NumExperts > 0);
  static_assert(Alignment > 0);
  constexpr int kRoutes = NumTokens * NumTopk;
  constexpr int kOutputCapacity = kRoutes + NumExperts * (Alignment - 1);
  constexpr int kExpertStorage = ((NumExperts + 31) / 32) * 32;
  static_assert(kRoutes * static_cast<int>(sizeof(std::int32_t)) >= 128,
                "each routing fragment must contain at least 128 bytes");

  using namespace pto;
  using Routes = Tile<Location::Vec, std::int32_t, 1, kRoutes,
                      BLayout::RowMajor>;
  using OutputRoutes = Tile<Location::Vec, std::int32_t, 1, kOutputCapacity,
                            BLayout::RowMajor>;
  using Reduced = Tile<Location::Vec, std::int32_t, 1, 32,
                       BLayout::RowMajor, 1, 1>;
  using Experts = Tile<Location::Vec, std::int32_t, 1, kExpertStorage,
                       BLayout::RowMajor, 1, NumExperts>;
  using RouteTensor = global_tensor<std::int32_t, RowMajor<1, kRoutes>>;
  using OutputTensor = global_tensor<std::int32_t, RowMajor<1, kOutputCapacity>>;
  using ExpertTensor = global_tensor<std::int32_t, RowMajor<1, NumExperts>>;
  using RouteIterator = global_iterator<RouteTensor, Routes>;
  using OutputIterator = global_iterator<OutputTensor, OutputRoutes>;
  using ExpertIterator = global_iterator<ExpertTensor, Experts>;

  RouteIterator topk(topk_idx);
  OutputIterator expert_out(pos_to_expert);
  OutputIterator token_out(pos_to_token);
  OutputIterator token_topk_out(pos_to_token_topk);
  RouteIterator inverse_out(token_topk_to_pos);
  ExpertIterator start_out(expert_start);
  ExpertIterator end_out(expert_end);
  ExpertIterator count_out(num_tokens_per_expert);

  Routes experts;
  Routes original_position;
  Routes sorted_expert;
  Routes sorted_rank;
  Routes sorted_token;
  Routes sort_tmp;

  TLOAD(experts, topk(0, 0));
  TCI<Routes, std::int32_t, 0>(original_position, 0);
  TSORT(sorted_expert, experts, original_position, sort_tmp);
  TCI<Routes, std::int32_t, 0>(sorted_rank, 0);
  TDIVS(sorted_token, original_position, NumTopk);
  Experts firsts;
  Experts starts;
  Experts ends;
  Experts counts;
  TEXPANDS(firsts, 0);
  TEXPANDS(starts, 0);
  TEXPANDS(ends, 0);
  TEXPANDS(counts, 0);

  Reduced running;
  TEXPANDS(running, 0);
  for (int expert = 0; expert < NumExperts; ++expert) {
    Routes expert_value;
    Routes selected_start;
    Routes selected_end;
    Routes start_fill;
    Routes end_fill;
    Routes mask;
    Routes select_tmp;
    Routes reduce_tmp;
    Reduced first;
    Reduced last;
    Reduced count;
    Reduced aligned_count;
    Reduced aligned_end;

    TEXPANDS(expert_value, expert);
    TEXPANDS(start_fill, kRoutes);
    TEXPANDS(end_fill, -1);
    TCMP(mask, sorted_expert, expert_value);
    TSEL(selected_start, mask, sorted_rank, start_fill, select_tmp);
    TSEL(selected_end, mask, sorted_rank, end_fill, select_tmp);
    TROWMIN(first, selected_start, reduce_tmp);
    TROWMAX(last, selected_end, reduce_tmp);
    TADDS(last, last, 1);
    TSUB(count, last, first);
    TMAXS(count, count, 0);
    TADDS(aligned_count, count, Alignment - 1);
    TDIVS(aligned_count, aligned_count, Alignment);
    TMULS(aligned_count, aligned_count, Alignment);
    TADD(aligned_end, running, aligned_count);
    TINSERT(firsts, first, 0, expert);
    TINSERT(starts, running, 0, expert);
    TINSERT(ends, aligned_end, 0, expert);
    TINSERT(counts, aligned_count, 0, expert);
    TMOV(running, aligned_end);
  }

  Routes route_first;
  Routes route_start;
  Routes route_offset;
  Routes padded_position;
  Routes gather_tmp;
  Routes inverse_position;
  OutputRoutes output_expert;
  OutputRoutes output_token;
  OutputRoutes output_token_topk;
  TEXPANDS(output_expert, -1);
  TEXPANDS(output_token, -1);
  TEXPANDS(output_token_topk, -1);
  TGATHER(route_first, firsts, sorted_expert, gather_tmp);
  TGATHER(route_start, starts, sorted_expert, gather_tmp);
  TSUB(route_offset, sorted_rank, route_first);
  TADD(padded_position, route_start, route_offset);
  TSCATTER(output_expert, sorted_expert, padded_position);
  TSCATTER(output_token, sorted_token, padded_position);
  TSCATTER(output_token_topk, original_position, padded_position);
  TSCATTER(inverse_position, padded_position, original_position);

  TSTORE(expert_out(0, 0), output_expert);
  TSTORE(token_out(0, 0), output_token);
  TSTORE(token_topk_out(0, 0), output_token_topk);
  TSTORE(inverse_out(0, 0), inverse_position);

  TSTORE(start_out(0, 0), starts);
  TSTORE(end_out(0, 0), ends);
  TSTORE(count_out(0, 0), counts);
}

}  // namespace supernpu::tile_isa

#endif
