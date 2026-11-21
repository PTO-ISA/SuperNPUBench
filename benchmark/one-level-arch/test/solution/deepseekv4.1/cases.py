"""Synthesize layer inputs and materialize per-query publications from their source layers."""

import math

import numpy as np

import layout as L
import reference as ref


def normal(rng, shape, sigma=1.0):
    return rng.standard_normal(shape, dtype="f4") * np.float32(sigma)


def weights(seed, layer):
    rng = np.random.default_rng([seed, layer, 1])
    w = np.zeros((), L.weights)

    def matrix(m, bits=8):
        n, groups = m["scale"].shape
        raw = normal(rng, (n, groups * 32), 0.8 / np.sqrt(groups * 32))
        if bits == 8:
            data, scale = ref.quant8(raw)
            m["data"] = data.view("u1")
        else:
            m["data"], scale = ref.pack4(raw, 32, "even", True)
        m["scale"] = (np.log2(scale) + 127).astype("u1")

    for a in (w["attn"], w["ffn"]):
        a["fn"] = normal(rng, a["fn"].shape, 0.04)
        a["scale"] = [0.65, 0.8, 1.1]
        a["base"] = normal(rng, a["base"].shape, 0.3)
        a["norm"] = ref.bf(1 + normal(rng, L.D, 0.1))
    for name in ("qa", "qb", "kv", "ob", "iq", "engram_kv"):
        matrix(w[name])
    for group in w["oa"]:
        matrix(group)
    for name in ("q_norm", "kv_norm", "comp_norm", "ik_norm", "engram_weight"):
        w[name] = ref.bf(1 + normal(rng, w[name].shape, 0.1))
    w["sink"] = normal(rng, L.H, 0.6)
    for name in ("compressor", "compressor_gate", "ik", "iw", "router"):
        w[name] = ref.bf(normal(rng, w[name].shape, 0.8 / np.sqrt(w[name].shape[-1])))
    w["bias"] = normal(rng, L.EXPERTS, 0.2)
    for name in ("up", "gate", "down"):
        matrix(w["shared"][name])
        for e in w["routed"]:
            matrix(e[name], 4)
    return w


def tokens(seed, layer, start):
    rng = np.random.default_rng([seed, layer, 2])
    batch = np.zeros(2, L.token)
    dim = 2 * L.rope["cosine"].shape[0]
    lane = np.arange(dim // 2, dtype="f4")
    theta = 10000 if layer < 2 else 160000
    freq = theta ** (-2 * lane / dim)
    if layer >= 2:

        def correction(rotations):
            return dim * math.log(65536 / (2 * math.pi * rotations)) / (2 * math.log(theta))

        low = max(math.floor(correction(32)), 0)
        high = min(math.ceil(correction(1)), dim - 1)
        ramp = np.clip((lane - low) / max(high - low, 1e-3), 0, 1)
        freq *= (1 - ramp) + ramp / 16
    for i, t in enumerate(batch):
        pos = start + i
        t["position"], t["window_slot"] = pos, pos % L.WIN
        t["window_indices"] = [j % L.WIN if j >= 0 else -1 for j in range(pos - L.WIN + 1, pos + 1)]
        # At layer 20 this is the HC boundary emitted by the final encoder layer.
        t["hidden"]["x"] = normal(rng, (L.HC, L.D), 0.65)
        t["hidden"]["pre"] = rng.uniform(0.15, 0.85, L.HC)
        t["lookup"] = ref.bf(normal(rng, L.D))
        pair_position = pos - 1 if pos % 2 else 0
        for key, angle in [("rope", pos), ("compressed_rope", pair_position if layer < 20 else pos)]:
            phase = angle * freq
            t[key]["cosine"], t[key]["sine"] = np.cos(phase), np.sin(phase)
    return batch


def initial_cache(seed):
    rng = np.random.default_rng([seed, 3])
    cache = np.zeros((), L.cache)
    cache["pages"] = [2, 0, 3, 1]
    for i in range(L.WIN):
        cache["window"][i] = ref.encode_row(normal(rng, L.HEAD), L.window)
    for bank in cache["banks"]:
        for i in range(L.HIST):
            bank["kv"][i] = ref.encode_row(normal(rng, L.HEAD), L.compressed)
            bank["index"][i] = ref.encode_row(normal(rng, L.ID), L.index)
    cache["pair_kv"] = normal(rng, (2, L.HEAD))
    cache["pair_gate"] = normal(rng, (2, L.HEAD))
    return cache


def build_case(seed, layer, shards, start=22, empty_shard=False):
    if not 0 <= layer < 40 or shards not in (1, 2) or start < 0:
        raise ValueError("invalid layer, shard count or start position")
    if layer >= 2 and (start + 1) // (2 if layer < 20 else 1) >= L.HIST:
        raise ValueError("two decode positions must fit the compressed history")
    data = np.zeros((), L.case)
    data["layer_idx"], data["shards"] = layer, shards
    data["initial"] = initial_cache(seed)
    data["weights"] = weights(seed, layer)
    if empty_shard:
        data["weights"]["bias"][L.EXPERTS // 2 :] -= 6
    data["steps"]["token"] = tokens(seed, layer, start)
    data["steps"]["published"]["indices"] = -1
    data["steps"]["published"]["candidates"] = -1
    data["steps"]["upstream_slot"] = -1
    expected = np.zeros((), L.result)
    cache = data["initial"].copy()

    if layer >= 2 and layer not in ref.KV_SOURCES:
        source = ref.owner(layer, ref.KV_SOURCES)
        ib = ref.owner(layer, ref.INDEX_SOURCES)
        source_layers = [source] + ([ib] if ib != source and ib != layer else [])
        source_w = {s: weights(seed, s) for s in source_layers}
        source_t = {s: tokens(seed, s, start) for s in source_layers}
        source_cache = {s: data["initial"].copy() for s in source_layers}
        bank = ref.KV_SOURCES.index(source)
        for i, step in enumerate(data["steps"]):
            for s in source_layers:
                sc = source_cache[s]
                if s != source:
                    sc["banks"][bank] = source_cache[source]["banks"][bank]
                ref.attention(s, source_w[s], source_t[s][i], sc, step["published"], np.zeros((), L.output))
            pos = start + i
            if source == 20 or pos % 2:
                logical = pos if source == 20 else pos // 2
                slot = cache["pages"][logical // L.PAGE] * L.PAGE + logical % L.PAGE
                step["upstream_slot"] = slot
                step["upstream_kv"] = source_cache[source]["banks"][bank]["kv"][slot]
                step["upstream_key"] = source_cache[source]["banks"][bank]["index"][slot]

    for i, step in enumerate(data["steps"]):
        out, pub = expected["steps"][i], step["published"].copy()
        slot = int(step["upstream_slot"])
        if slot >= 0:
            bank = cache["banks"][ref.KV_SOURCES.index(ref.owner(layer, ref.KV_SOURCES))]
            bank["kv"][slot], bank["index"][slot] = step["upstream_kv"], step["upstream_key"]
        ref.attention(layer, data["weights"], step["token"], cache, pub, out)
        ref.moe(data["weights"], out)
        expected["published"][i] = pub
    expected["final_cache"] = cache
    return data, expected


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Embed a valid layer 0 SWA case in the default ELF.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data, _ = build_case(41, 0, 1)
    args.output.write_bytes(data.tobytes())
