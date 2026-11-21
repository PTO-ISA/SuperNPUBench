"""NumPy decode oracle. Sparse selection and routed experts use dense array operations."""

import ml_dtypes as ml
import numpy as np

import layout as L

EPS = np.float32(1e-6)
LEVELS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype="f4")
CUTS = (LEVELS[1:] + LEVELS[:-1]) / 2


def bf(x):
    return np.asarray(x).astype(ml.bfloat16).astype("f4")


def fp8(x):
    return np.clip(x, -448, 448).astype(ml.float8_e4m3fn)


def quant8(x):
    grouped = x.reshape(*x.shape[:-1], -1, 32)
    scale = np.exp2(np.ceil(np.log2(np.maximum(np.max(abs(grouped), -1), 1e-4) / 448)))
    return fp8(grouped / scale[..., None]).reshape(x.shape), scale


def pack4(x, group, policy, power):
    g = x.reshape(*x.shape[:-1], -1, group)
    raw = np.max(abs(g), -1) / 6
    scale = np.exp2(np.ceil(np.log2(np.maximum(raw, 2.0**-126)))) if power else bf(np.maximum(raw, 2.0**-9))
    v = (g / scale[..., None]).reshape(x.shape)
    codes = np.searchsorted(CUTS, abs(v), side="right" if policy == "upper" else "left")
    if policy == "even":
        codes += (codes % 2 == 1) & (abs(v) == CUTS[codes.clip(max=6)])
    codes = codes.astype("u1") | (np.signbit(v).astype("u1") << 3)
    return codes[..., ::2] | (codes[..., 1::2] << 4), scale


def unpack4(data):
    codes = np.stack((data & 15, data >> 4), axis=-1).reshape(*data.shape[:-1], data.shape[-1] * 2)
    return np.where(codes & 8, -1.0, 1.0).astype("f4") * LEVELS[codes & 7]


def encode_row(x, dtype, policy="even"):
    row = np.zeros((), dtype)
    if dtype == L.window:
        p, scale = quant8(x)
        row["data"] = p.view("u1")
    else:
        row["data"], scale = pack4(x, 32 if dtype == L.index else 16, policy, dtype == L.index)
    row["scale"] = (
        (np.log2(scale) + 127).astype("u1") if dtype == L.index else scale.astype(ml.bfloat16).view("u2")
    )
    return row


def unpack_rows(rows):
    if rows.dtype == L.window:
        p = rows["data"].view(ml.float8_e4m3fn).astype("f4")
    else:
        p = unpack4(rows["data"])
    if rows.dtype == L.index:
        scale = np.exp2(rows["scale"].astype("f4") - 127)
    else:
        scale = rows["scale"].view(ml.bfloat16).astype("f4")
    return bf(p * np.repeat(scale, p.shape[-1] // scale.shape[-1], axis=-1))


def project(x, w, *, bits=8, rounded=True):
    payload, scale = quant8(x)
    a = (payload.reshape(-1, 32).astype("f4") * scale[:, None]).ravel()
    wdata = w["data"].view(ml.float8_e4m3fn).astype("f4") if bits == 8 else unpack4(w["data"])
    y = (wdata * np.repeat(np.exp2(w["scale"].astype("f4") - 127), 32, axis=1)) @ a
    return bf(y) if rounded else y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def norm(x, weight):
    return bf(x * (1 / np.sqrt(np.mean(x * x) + EPS)) * weight)


def hc_pre(h, w):
    x, pre = h["x"], h["pre"]
    flat = x.ravel()

    def hf(v):
        return ((v.view("u4") + 4096) & np.uint32(0xFFFFE000)).view("f4")

    m = (hf(w["fn"]) @ hf(flat)) / np.sqrt(np.mean(flat * flat) + EPS)
    m = m * np.repeat(w["scale"], [L.HC, L.HC, L.HC**2]) + w["base"]
    pre_next = sigmoid(m[: L.HC]) + EPS
    post = 2 * sigmoid(m[L.HC : 2 * L.HC])
    residual = m[2 * L.HC :].reshape(L.HC, L.HC)
    a = np.exp(residual - residual.max(axis=1, keepdims=True))
    a = a / a.sum(axis=1, keepdims=True) + EPS
    a /= a.sum(axis=0, keepdims=True) + EPS
    for _ in range(19):
        a /= a.sum(axis=1, keepdims=True) + EPS
        a /= a.sum(axis=0, keepdims=True) + EPS
    return norm(bf(pre @ x), w["norm"]), pre_next, post, a


def rotate(x, rope, *, inverse=False, rounded=True):
    x = x.copy()
    cos, sin = rope["cosine"], rope["sine"]
    sine = -sin if inverse else sin
    tail = x[..., -2 * cos.size :].reshape(*x.shape[:-1], -1, 2)
    re, im = tail[..., 0].copy(), tail[..., 1].copy()
    tail[..., 0], tail[..., 1] = re * cos - im * sine, im * cos + re * sine
    return bf(x) if rounded else x


def engram(h, row, w):
    kv = project(row, w["engram_kv"])
    keys, value = kv[: -L.D].reshape(L.HC, L.D), kv[-L.D :]
    x = h["x"]
    z = np.sum(x * w["engram_weight"] * keys, axis=1)
    z /= np.sqrt((np.mean(x * x, axis=1) + EPS) * (np.mean(keys * keys, axis=1) + EPS) * L.D)
    x[:] = bf(x + sigmoid(np.copysign(np.sqrt(np.maximum(abs(z), EPS)), z))[:, None] * value)


# Explicit source lists keep the oracle independent of the C++ bank arithmetic.
KV_SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)


def owner(layer, sources):
    return max(s for s in sources if s <= layer)


def select(scores, keep):
    valid = np.flatnonzero(scores > -np.inf)
    ids = valid[np.argsort(-scores[valid], kind="stable")[:keep]]
    return np.pad(np.sort(ids), (0, keep - ids.size), constant_values=-1).astype("i4")


def sparse(layer, w, t, x, qr, cache, published):
    pos = int(t["position"])
    ratio = 2 if layer < 20 else 1
    n = (pos + 1) // ratio
    bank = cache["banks"][KV_SOURCES.index(owner(layer, KV_SOURCES))]
    ids = np.arange(n)
    physical = cache["pages"][ids // L.PAGE] * L.PAGE + ids % L.PAGE
    ib = INDEX_SOURCES.index(owner(layer, INDEX_SOURCES))
    candidates = np.full(L.POOL, -1, "i4")
    if layer in KV_SOURCES:
        latent = w["compressor"] @ x
        if ratio == 2:
            slot = pos % 2
            cache["pair_kv"][slot] = latent
            cache["pair_gate"][slot] = w["compressor_gate"] @ x
            gates = cache["pair_gate"]
            a = np.exp(gates - gates.max(axis=0))
            latent = (cache["pair_kv"] * a).sum(axis=0) / a.sum(axis=0)
        if (pos + 1) % ratio == 0:
            latent = norm(bf(latent), w["comp_norm"])
            key = norm(bf(w["ik"] @ latent), w["ik_norm"])
            bank["index"][physical[-1]] = encode_row(rotate(key, t["compressed_rope"]), L.index)
            bank["kv"][physical[-1]] = encode_row(
                rotate(latent, t["compressed_rope"]), L.compressed, "even" if ratio == 1 else "lower"
            )
    if layer in INDEX_SOURCES:
        iq = project(qr, w["iq"], rounded=False).reshape(L.IH, L.ID)
        p, s = pack4(rotate(iq, t["rope"], rounded=False), 32, "upper", True)
        iq = unpack4(p) * s
        weights = bf(bf(w["iw"] @ x) / np.float32(np.sqrt(L.IH * L.ID)))
        dots = bf(iq @ unpack_rows(bank["index"][physical]).T)
        scores = np.zeros(n, "f4")
        for h in range(L.IH):
            scores = bf(scores + dots[h].clip(min=0) * weights[h])
        if layer > 20:
            candidates[:] = published["candidates"]
            scores[~np.isin(ids // L.PAGE, candidates)] = -np.inf
        if layer == 20:
            blocks = np.maximum.reduceat(scores, np.arange(0, n, L.PAGE))
            blocks[-1] = np.inf
            candidates[:] = select(blocks, L.POOL)
            published["candidates"] = candidates
        sel = select(scores, L.KEEP)
        valid = sel >= 0
        sel[valid] = physical[sel[valid]]
        published["indices"][ib] = sel
    return bank, published["indices"][ib].copy(), candidates


def attention(layer, w, t, cache, published, out):
    h = t["hidden"].copy()
    if layer in (1, 14):
        engram(h, t["lookup"], w)
    x, pre, post, residual = hc_pre(h, w["attn"])
    qr = norm(project(x, w["qa"]), w["q_norm"])
    q = rotate(project(qr, w["qb"]).reshape(L.H, L.HEAD), t["rope"])
    kv = rotate(norm(project(x, w["kv"]), w["kv_norm"]), t["rope"])
    cache["window"][t["window_slot"]] = encode_row(kv, L.window)
    ids = t["window_indices"]
    chunks = [unpack_rows(cache["window"][ids[ids >= 0]])]
    out["selected"] = out["candidates"] = -1
    if layer >= 2:
        bank, sel, pool = sparse(layer, w, t, x, qr, cache, published)
        out["selected"], out["candidates"] = sel, pool
        chunks.append(unpack_rows(bank["kv"][sel[sel >= 0]]))
    maximum = np.full((L.H, 1), -np.inf, dtype="f4")
    den = np.zeros_like(maximum)
    num = np.zeros_like(q)
    for kvs in chunks:
        if len(kvs) == 0:
            continue
        logits = (q @ kvs.T) / np.float32(np.sqrt(L.HEAD))
        peak = np.maximum(maximum, logits.max(axis=1, keepdims=True))
        corr = np.exp(maximum - peak)
        p = np.exp(logits - peak)
        den, num, maximum = den * corr + p.sum(axis=1, keepdims=True), num * corr + bf(p) @ kvs, peak
    sink = w["sink"][:, None]
    peak = np.maximum(maximum, sink)
    corr = np.exp(maximum - peak)
    ctx = bf(num * (corr / (den * corr + np.exp(sink - peak))))
    groups = rotate(ctx, t["rope"], inverse=True).reshape(L.GROUPS, -1)
    latent = np.concatenate([project(g, wa) for g, wa in zip(groups, w["oa"])])
    out["attention"]["x"] = residual.T @ h["x"] + post[:, None] * project(latent, w["ob"])
    out["attention"]["pre"] = pre


def run_expert(x, w, weight, bits):
    gate = np.minimum(project(x, w["gate"], bits=bits), 10)
    up = np.clip(project(x, w["up"], bits=bits), -10, 10)
    return project(gate * sigmoid(gate) * up * weight, w["down"], bits=bits)


def moe(w, out):
    h = out["attention"]
    x, pre, post, residual = hc_pre(h, w["ffn"])
    raw = np.sqrt(np.logaddexp(np.float32(0), w["router"] @ x))
    ids = np.argsort(-(raw + w["bias"]), kind="stable")[: L.K]
    weights = raw[ids] / (raw[ids].sum() + 1e-20) * 1.5
    out["routes"]["expert"], out["routes"]["slot"], out["routes"]["weight"] = ids, np.arange(L.K), weights
    out["shared"] = run_expert(x, w["shared"], 1.0, 8)
    for k, (e, weight) in enumerate(zip(ids, weights)):
        out["routed"][k] = run_expert(x, w["routed"][e], weight, 4)
    layer_sum = bf(np.vstack([out["shared"], out["routed"]]).sum(axis=0))
    out["next"]["x"] = residual.T @ h["x"] + post[:, None] * layer_sum
    out["next"]["pre"] = pre
