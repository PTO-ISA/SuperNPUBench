#!/usr/bin/env python3

import bisect
import math


QSMLA_MODES = {"SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"}


def decode_hif8(raw):
    """Decode the SuperNPU HIFLOAT8 scalar encoding to Python float."""
    raw = int(raw) & 0xFF
    if raw == 0x80:
        return math.nan
    if raw == 0x6F:
        return math.inf
    if raw == 0xEF:
        return -math.inf
    negative = bool(raw & 0x80)
    payload = raw & 0x7F
    if payload == 0:
        value = 0.0
    elif payload & 0x78 == 0:
        value = math.ldexp(1.0, payload - 23)
    else:
        if payload & 0x78 == 0x08:
            exponent_bits, fraction_bits = 0, 3
        elif payload & 0x70 == 0x10:
            exponent_bits, fraction_bits = 1, 3
        elif payload & 0x60 == 0x20:
            exponent_bits, fraction_bits = 2, 3
        elif payload & 0x60 == 0x40:
            exponent_bits, fraction_bits = 3, 2
        else:
            exponent_bits, fraction_bits = 4, 1
        fraction = payload & ((1 << fraction_bits) - 1)
        exponent = 0
        if exponent_bits:
            encoded = (payload >> fraction_bits) & ((1 << exponent_bits) - 1)
            sign_bit = 1 << (exponent_bits - 1)
            magnitude = sign_bit | (encoded & (sign_bit - 1))
            exponent = -magnitude if encoded & sign_bit else magnitude
        value = math.ldexp(1.0 + fraction / float(1 << fraction_bits), exponent)
    return -value if negative else value


_HIF8_POSITIVE_FINITE = tuple(decode_hif8(code) for code in range(0x6F))
_HIF8_SORTED_CODEBOOK = tuple(sorted(
    (value, code) for code, value in enumerate(_HIF8_POSITIVE_FINITE)
))
_HIF8_SORTED_VALUES = tuple(value for value, _ in _HIF8_SORTED_CODEBOOK)


def encode_hif8(value):
    """Round a scalar to HIFLOAT8 using nearest/even-code tie breaking."""
    value = float(value)
    if math.isnan(value):
        return 0x80
    if math.isinf(value):
        return 0xEF if value < 0 else 0x6F
    if value == 0.0:
        return 0
    negative = value < 0
    magnitude = abs(value)
    if magnitude >= 40960.0:
        return 0xEF if negative else 0x6F
    position = bisect.bisect_left(_HIF8_SORTED_VALUES, magnitude)
    begin = max(0, position - 1)
    end = min(len(_HIF8_SORTED_CODEBOOK), position + 1)
    best = min(
        (code for _, code in _HIF8_SORTED_CODEBOOK[begin:end]),
        key=lambda code: (abs(_HIF8_POSITIVE_FINITE[code] - magnitude), code & 1),
    )
    return best | (0x80 if negative and best else 0)


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
    q_descale=1.0, ori_kv_descale=1.0, cmp_kv_descale=1.0,
    quantize_probability=False,
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
                entries.extend(
                    (ori_kv[batch][index][kv_head], float(ori_kv_descale))
                    for index in indices
                )

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
                    entries.extend(
                        (cmp_kv[batch][index][kv_head], float(cmp_kv_descale))
                        for index in indices
                    )

                if not entries:
                    raise ValueError("empty logical attention row is outside v1 scope")

                q_row = q[batch][q_position][q_head]
                scores = [
                    float(softmax_scale) * float(q_descale) * kv_descale * sum(
                        float(q_value) * float(k_value)
                        for q_value, k_value in zip(q_row, kv_row)
                    )
                    for kv_row, kv_descale in entries
                ]
                row_max = max(scores)
                weights = [math.exp(score - row_max) for score in scores]
                denominator = sum(weights)
                pv_weights = (
                    [decode_hif8(encode_hif8(weight * 16.0)) for weight in weights]
                    if quantize_probability else weights
                )
                output_scale = 1.0 / 16.0 if quantize_probability else 1.0
                output[batch][q_position][q_head] = [
                    output_scale * sum(
                        weight * float(kv_row[dim]) * kv_descale
                        for weight, (kv_row, kv_descale) in zip(pv_weights, entries)
                    ) / denominator
                    for dim in range(d)
                ]
    return output
