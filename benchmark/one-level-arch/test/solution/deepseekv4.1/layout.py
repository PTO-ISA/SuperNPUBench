"""Little-endian Linx ABI for the case and result sections in src/fixture.hpp."""

import numpy as np

D, H, HEAD, RANK, GROUPS, ORANK, FF = 32, 2, 32, 32, 2, 32, 64
IH, ID, HC, EXPERTS, K, WIN, HIST, PAGE, KEEP, POOL = 2, 32, 4, 16, 6, 4, 32, 8, 4, 2
F32, I32, U32, U8, U16 = "<f4", "<i4", "<u4", "u1", "<u2"


def record(**fields):
    return np.dtype(list(fields.items()), align=True)


def mx(out, inn, bits):
    return record(data=(U8, (out, inn * bits // 8)), scale=(U8, (out, inn // 32)))


def expert(bits):
    return record(gate=mx(FF, D, bits), up=mx(FF, D, bits), down=mx(D, FF, bits))


mix = 2 * HC + HC * HC
hc_weights = record(fn=(F32, (mix, HC * D)), scale=(F32, 3), base=(F32, mix), norm=(F32, D))
weights = record(
    attn=hc_weights,
    ffn=hc_weights,
    qa=mx(RANK, D, 8),
    qb=mx(H * HEAD, RANK, 8),
    kv=mx(HEAD, D, 8),
    oa=(mx(ORANK, H * HEAD // GROUPS, 8), GROUPS),
    ob=mx(D, GROUPS * ORANK, 8),
    q_norm=(F32, RANK),
    kv_norm=(F32, HEAD),
    sink=(F32, H),
    compressor=(F32, (HEAD, D)),
    compressor_gate=(F32, (HEAD, D)),
    comp_norm=(F32, HEAD),
    ik=(F32, (ID, HEAD)),
    ik_norm=(F32, ID),
    iw=(F32, (IH, D)),
    iq=mx(IH * ID, RANK, 8),
    engram_kv=mx((HC + 1) * D, D, 8),
    engram_weight=(F32, (HC, D)),
    router=(F32, (EXPERTS, D)),
    bias=(F32, EXPERTS),
    shared=expert(8),
    routed=(expert(4), EXPERTS),
)
hidden = record(x=(F32, (HC, D)), pre=(F32, HC))
rope = record(cosine=(F32, 4), sine=(F32, 4))
token = record(
    hidden=hidden,
    lookup=(F32, D),
    rope=rope,
    compressed_rope=rope,
    position=I32,
    window_slot=I32,
    window_indices=(I32, WIN),
)
window = record(data=(U8, HEAD), scale=(U16, HEAD // 32))
compressed = record(data=(U8, HEAD // 2), scale=(U16, HEAD // 16))
index = record(data=(U8, ID // 2), scale=(U8, ID // 32))
bank = record(kv=(compressed, HIST), index=(index, HIST))
cache = record(
    window=(window, WIN),
    pair_kv=(F32, (2, HEAD)),
    pair_gate=(F32, (2, HEAD)),
    banks=(bank, 4),
    pages=(I32, HIST // PAGE),
)
published = record(indices=(I32, (8, KEEP)), candidates=(I32, POOL))
route = record(expert=I32, slot=I32, weight=F32)
output = record(
    attention=hidden,
    next=hidden,
    selected=(I32, KEEP),
    candidates=(I32, POOL),
    routes=(route, K),
    shared=(F32, D),
    routed=(F32, (K, D)),
)
step = record(token=token, published=published, upstream_slot=I32, upstream_kv=compressed, upstream_key=index)
case = record(layer_idx=I32, shards=I32, weights=weights, initial=cache, steps=(step, 2))
result = record(
    steps=(output, 2),
    published=(published, 2),
    final_cache=cache,
    dynamic_puts=U32,
    dynamic_gets=U32,
    sdma_transfers=U32,
    sdma_bytes=U32,
)
