#!/usr/bin/env python3

import math


QSMLA_MODES = {"SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"}


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))


def collect_sparse_indices(candidates, candidate_count, source_s2, causal_end):
    selected = []
    valid_end = _clamp(causal_end, 0, source_s2)
    for index in candidates[:max(0, candidate_count)]:
        if index == -1:
            break
        if index < 0 or index >= source_s2 or index >= valid_end:
            continue
        selected.append(index)
    return selected


def _shape4(tensor, name):
    try:
        shape = (
            len(tensor), len(tensor[0]), len(tensor[0][0]),
            len(tensor[0][0][0]),
        )
    except (IndexError, TypeError):
        raise ValueError(f"{name} must be a non-empty rank-4 nested sequence")
    if any(size <= 0 for size in shape):
        raise ValueError(f"{name} dimensions must be positive")
    return shape


def _ori_range(ori_s2, s1, q_position, win_left, win_right):
    diagonal = ori_s2 - s1 + q_position
    begin = 0 if win_left < 0 else diagonal - win_left
    end = ori_s2 if win_right < 0 else diagonal + win_right + 1
    begin = _clamp(begin, 0, ori_s2)
    end = _clamp(end, 0, ori_s2)
    return range(begin, max(begin, end))


def _ori_causal_end(ori_s2, s1, q_position):
    return _clamp(ori_s2 - s1 + q_position + 1, 0, ori_s2)


def _cmp_causal_end(cmp_s2, s1, q_position, cmp_ratio):
    if cmp_ratio <= 0:
        raise ValueError("cmp_ratio must be positive for a CMP source")
    return _clamp(
        (cmp_s2 * cmp_ratio - s1 + q_position + 1) // cmp_ratio,
        0,
        cmp_s2,
    )


def qsmla_reference(
    *, q, ori_kv, cmp_kv,
    ori_sparse_indices, cmp_sparse_indices,
    ori_topk_length, cmp_topk_length,
    mode, softmax_scale, cmp_ratio, win_left, win_right,
):
    if mode not in QSMLA_MODES:
        raise ValueError(f"unsupported QSMLA mode: {mode}")

    b_count, s1, n1, d = _shape4(q, "q")
    ori_b, ori_s2, n2, ori_d = _shape4(ori_kv, "ori_kv")
    if (ori_b, ori_d) != (b_count, d) or n1 % n2 != 0:
        raise ValueError("Q and ORI KV shapes are incompatible")
    group_size = n1 // n2
    has_cmp = mode in {"HCA", "CSA", "ORI_CMP_SPARSE"}
    indexed_ori = mode in {"ORI_SPARSE", "ORI_CMP_SPARSE"}
    indexed_cmp = mode in {"CSA", "ORI_CMP_SPARSE"}

    cmp_s2 = 0
    if has_cmp:
        cmp_b, cmp_s2, cmp_n2, cmp_d = _shape4(cmp_kv, "cmp_kv")
        if (cmp_b, cmp_n2, cmp_d) != (b_count, n2, d):
            raise ValueError("ORI and CMP KV shapes are incompatible")
        if indexed_cmp and cmp_sparse_indices is None:
            raise ValueError("CMP sparse indices are required")
    if indexed_ori and (ori_sparse_indices is None or ori_topk_length is None):
        raise ValueError("ORI sparse indices and ori_topk_length are required")

    output = [[[[0.0 for _ in range(d)] for _ in range(n1)]
               for _ in range(s1)] for _ in range(b_count)]

    for batch in range(b_count):
        for q_position in range(s1):
            for q_head in range(n1):
                kv_head = q_head // group_size
                entries = []

                if indexed_ori:
                    candidates = ori_sparse_indices[batch][q_position][kv_head]
                    count = _clamp(
                        ori_topk_length[batch][q_position][kv_head],
                        0, len(candidates),
                    )
                    indices = collect_sparse_indices(
                        candidates, count, ori_s2,
                        _ori_causal_end(ori_s2, s1, q_position),
                    )
                else:
                    indices = _ori_range(
                        ori_s2, s1, q_position, win_left, win_right)
                entries.extend(ori_kv[batch][index][kv_head] for index in indices)

                if has_cmp:
                    cmp_end = _cmp_causal_end(
                        cmp_s2, s1, q_position, cmp_ratio)
                    if indexed_cmp:
                        candidates = cmp_sparse_indices[batch][q_position][kv_head]
                        count = len(candidates) if cmp_topk_length is None else _clamp(
                            cmp_topk_length[batch][q_position][kv_head],
                            0, len(candidates),
                        )
                        indices = collect_sparse_indices(
                            candidates, count, cmp_s2, cmp_end)
                    else:
                        indices = range(cmp_end)
                    entries.extend(cmp_kv[batch][index][kv_head] for index in indices)

                if not entries:
                    raise ValueError("empty logical attention row is outside v1 scope")

                q_row = q[batch][q_position][q_head]
                scores = [
                    float(softmax_scale) * sum(
                        float(q_value) * float(k_value)
                        for q_value, k_value in zip(q_row, kv_row)
                    )
                    for kv_row in entries
                ]
                row_max = max(scores)
                weights = [math.exp(score - row_max) for score in scores]
                denominator = sum(weights)
                output[batch][q_position][q_head] = [
                    sum(weight * float(kv_row[dim])
                        for weight, kv_row in zip(weights, entries)) / denominator
                    for dim in range(d)
                ]
    return output
