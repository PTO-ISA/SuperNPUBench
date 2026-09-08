#ifndef QSMLA_MODE_PTO_HPP
#define QSMLA_MODE_PTO_HPP

enum class QsmlaMode { SWA, HCA, CSA, ORI_SPARSE, ORI_CMP_SPARSE };

static constexpr bool qsmla_use_direct_contiguous_tile(
    bool allow_direct, bool indexed, int valid_rows, int tile_rows)
{
    return allow_direct && !indexed &&
           tile_rows > 0 && valid_rows == tile_rows;
}

static constexpr int qsmla_full_o_scratch_elements(
    int pe_count, int pe_rows, int d)
{
    return pe_count * pe_rows * d;
}

static constexpr int qsmla_full_o_scratch_pe_offset(
    int pe_id, int pe_rows, int d)
{
    return pe_id * pe_rows * d;
}

static constexpr int qsmla_sparse_clamp(int value, int lower, int upper)
{
    return value < lower ? lower : (value > upper ? upper : value);
}

static constexpr int qsmla_sparse_floor_div(int numerator, int denominator)
{
    return numerator >= 0
               ? numerator / denominator
               : -((-numerator + denominator - 1) / denominator);
}

static constexpr int qsmla_sparse_ori_valid_end(
    int ori_s2, int q_s1, int q_position)
{
    return qsmla_sparse_clamp(ori_s2 - q_s1 + q_position + 1, 0, ori_s2);
}

static constexpr int qsmla_csa_cmp_valid_end(
    int cmp_s2, int q_s1, int q_position, int cmp_ratio)
{
    return cmp_ratio <= 0
               ? 0
               : qsmla_sparse_clamp(
                     qsmla_sparse_floor_div(
                         cmp_s2 * cmp_ratio - q_s1 + q_position + 1,
                         cmp_ratio),
                     0, cmp_s2);
}

static inline int qsmla_sparse_collect_indices(
    int* selected, int selected_capacity,
    const int* candidates, int candidate_count,
    int source_s2, int causal_end)
{
    if (selected == nullptr || candidates == nullptr ||
        selected_capacity <= 0 || candidate_count <= 0 || source_s2 <= 0) {
        return 0;
    }

    const int valid_end = qsmla_sparse_clamp(causal_end, 0, source_s2);
    int selected_count = 0;
    for (int candidate_id = 0;
         candidate_id < candidate_count && selected_count < selected_capacity;
         ++candidate_id) {
        const int index = candidates[candidate_id];
        if (index == -1) {
            break;
        }
        if (index < 0 || index >= source_s2 || index >= valid_end) {
            continue;
        }
        selected[selected_count++] = index;
    }
    return selected_count;
}

template <typename BaseConfig, QsmlaMode Mode_,
          int CmpS2_, int OriTopK_, int CmpTopK_>
struct QsmlaModeConfig {
    using Base = BaseConfig;

    static constexpr QsmlaMode Mode = Mode_;
    static constexpr int OriS2 = Base::S2;
    static constexpr int CmpS2 = CmpS2_;
    static constexpr int OriTopK = OriTopK_;
    static constexpr int CmpTopK = CmpTopK_;
    static constexpr bool HasIndexedOri =
        Mode == QsmlaMode::ORI_SPARSE ||
        Mode == QsmlaMode::ORI_CMP_SPARSE;
    static constexpr bool HasCmp =
        Mode == QsmlaMode::HCA ||
        Mode == QsmlaMode::CSA ||
        Mode == QsmlaMode::ORI_CMP_SPARSE;
    static constexpr bool HasIndexedCmp =
        Mode == QsmlaMode::CSA ||
        Mode == QsmlaMode::ORI_CMP_SPARSE;

    static_assert(!HasIndexedOri || OriTopK > 0,
                  "ORI sparse modes require a positive OriTopK");
    static_assert(!HasCmp || CmpS2 > 0,
                  "HCA/CSA/CMP sparse modes require positive CmpS2");
    static_assert(!HasIndexedCmp || CmpTopK > 0,
                  "indexed CMP modes require positive CmpTopK");
    static_assert(HasIndexedOri || OriTopK >= 0,
                  "OriTopK must be non-negative");
    static_assert(HasCmp || (CmpS2 >= 0 && CmpTopK >= 0),
                  "unused CMP dimensions must be non-negative");
};

#endif // QSMLA_MODE_PTO_HPP
