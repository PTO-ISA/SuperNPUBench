"""Independent guarded fixtures for the runtime-shape QLI radix TopK kernel."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct


CAPACITY = 2048
POISON = 0xDEADBEEF
PAD_SCORE = 0x7FFFFFFF
GUARD_BYTES = 64


@dataclass(frozen=True)
class DynamicTopKCase:
    name: str
    sq: int
    skv: int
    topk: int
    num_pes: int = 1
    pattern: str = "mixed"


CASES = (
    DynamicTopKCase("all_valid_key_zero", 1, 32, 32, pattern="all_zero_key"),
    DynamicTopKCase("mixed_tied_cutoff", 2, 128, 73),
    DynamicTopKCase("topk_zero_noop", 1, 32, 0, pattern="all_zero_key"),
    DynamicTopKCase("sq_zero_noop", 0, 32, 1, pattern="all_zero_key"),
)


def sortable_key(bits):
    bits &= 0xFFFFFFFF
    # Independent arithmetic form of the unsigned raw-bit total order.
    return 0xFFFFFFFF - bits if bits >= 1 << 31 else bits + (1 << 31)


def valid_scores(case, row):
    if case.pattern == "all_zero_key":
        return [0xFFFFFFFF] * case.skv
    values = (0xFFFFFFFF, 0xFFC00001, 0xFF800000, 0xC0000000,
              0x80000000, 0x00000000, 0x3F800000, 0x7F800000,
              0x7FC00001)
    return [values[(row * 3 + column) % len(values)]
            for column in range(case.skv)]


def layout(case):
    stride = ((case.skv + CAPACITY - 1) // CAPACITY) * CAPACITY
    score_words = case.sq * stride
    index_start = score_words
    index_words = case.sq * case.topk
    gap_start = index_start + index_words
    # Preserve the original dynamic driver exactly: 8 KiB, a second
    # Sq*paddedSkv range, then 2 KiB before key scratch.
    gap_words = 2048 + case.sq * stride + 512
    key_start = gap_start + gap_words
    key_words = case.sq * stride
    hist_start = key_start + key_words
    total_words = hist_start + case.num_pes * 288
    return {
        "stride": stride,
        "score_start": 0,
        "score_words": score_words,
        "index_start": index_start,
        "index_words": index_words,
        "gap_start": gap_start,
        "gap_words": gap_words,
        "key_start": key_start,
        "key_words": key_words,
        "hist_start": hist_start,
        "total_words": total_words,
    }


def cumulative_histogram(source, byte, prefix):
    prefix_bytes = prefix.to_bytes(4, "little")
    encoded = [value.to_bytes(4, "little") for value in source]
    return [sum(value[byte] <= bucket and
                value[byte + 1:] == prefix_bytes[byte + 1:]
                for value in encoded)
            for bucket in range(256)]


def oracle(case):
    info = layout(case)
    initial = [POISON] * info["total_words"]
    for row in range(case.sq):
        begin = row * info["stride"]
        initial[begin:begin + case.skv] = valid_scores(case, row)
        initial[begin + case.skv:begin + info["stride"]] = [PAD_SCORE] * (
            info["stride"] - case.skv)
    expected = list(initial)
    rows = []
    if case.sq == 0 or case.topk == 0:
        return initial, expected, rows, info

    for row in range(case.sq):
        score_begin = row * info["stride"]
        raw = initial[score_begin:score_begin + case.skv]
        keys = [sortable_key(value) for value in raw]
        key_begin = info["key_start"] + row * info["stride"]
        expected[key_begin:key_begin + case.skv] = keys
        expected[key_begin + case.skv:key_begin + info["stride"]] = [0] * (
            info["stride"] - case.skv)
        canonical = sorted(range(case.skv), key=lambda index: keys[index],
                           reverse=True)[:case.topk]
        threshold = keys[canonical[-1]]
        strict = {index for index, key in enumerate(keys) if key > threshold}
        equal = {index for index, key in enumerate(keys) if key == threshold}
        quota = case.topk - len(strict)
        out = info["index_start"] + row * case.topk
        expected[out:out + case.topk] = canonical
        rows.append({"keys": keys, "threshold": threshold,
                     "strict": strict, "equal": equal, "quota": quota})

    final = rows[-1]
    padded_keys = final["keys"] + [0] * (info["stride"] - case.skv)
    hist = cumulative_histogram(padded_keys[-CAPACITY:], 0,
                                final["threshold"])
    hist_begin = info["hist_start"]
    expected[hist_begin:hist_begin + 256] = hist
    threshold = final["threshold"]
    prefix = sum(([threshold >> shift & 0xFF] * 8
                  for shift in (24, 16, 8)), []) + [0] * 8
    expected[hist_begin + 256:hist_begin + 288] = prefix
    return initial, expected, rows, info


def generate(directory, case):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    initial, expected, rows, info = oracle(case)
    initial_path = directory / "workspace.initial.bin"
    expected_path = directory / "workspace.expected.bin"
    initial_path.write_bytes(struct.pack(f"<{len(initial)}I", *initial))
    expected_path.write_bytes(struct.pack(f"<{len(expected)}I", *expected))
    assembly = [
        '.section .qli_dynamic_fixture,"aw",@progbits', '.balign 64',
        '.global qli_dynamic_workspace_before',
        '.type qli_dynamic_workspace_before,@object',
        'qli_dynamic_workspace_before:', '.fill 64,1,0xa5',
        '.size qli_dynamic_workspace_before,.-qli_dynamic_workspace_before',
        '.global qli_dynamic_workspace', '.type qli_dynamic_workspace,@object',
        'qli_dynamic_workspace:', f'.incbin "{initial_path.resolve()}"',
        '.size qli_dynamic_workspace,.-qli_dynamic_workspace',
        '.global qli_dynamic_workspace_after',
        '.type qli_dynamic_workspace_after,@object',
        'qli_dynamic_workspace_after:', '.fill 64,1,0x3c',
        '.size qli_dynamic_workspace_after,.-qli_dynamic_workspace_after',
    ]
    (directory / "data.s").write_text("\n".join(assembly) + "\n")
    manifest = {
        "case": asdict(case), "layout": info,
        "initial_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        "expected_is_host_only": True,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return initial, expected, rows, info


def compare(actual_bytes, expected_bytes, case):
    initial, _expected, rows, info = oracle(case)
    actual = list(struct.unpack(f"<{len(actual_bytes) // 4}I", actual_bytes))
    expected = list(struct.unpack(f"<{len(expected_bytes) // 4}I", expected_bytes))
    if len(actual) != info["total_words"] or len(expected) != len(actual):
        raise ValueError("dynamic QLI workspace extent changed")
    if case.sq == 0 or case.topk == 0:
        if actual != initial:
            raise ValueError("dynamic QLI no-op mutated workspace")
        return

    mutable_indices = set(range(info["index_start"],
                                info["index_start"] + info["index_words"]))
    for index, (got, want) in enumerate(zip(actual, expected)):
        if index not in mutable_indices and got != want:
            raise ValueError(f"dynamic QLI exact workspace mismatch at word {index}")
    for row, contract in enumerate(rows):
        begin = info["index_start"] + row * case.topk
        indices = actual[begin:begin + case.topk]
        chosen = set(indices)
        if len(indices) != case.topk or len(chosen) != case.topk:
            raise ValueError(f"row {row}: duplicate or incomplete TopK indices")
        if any(index >= case.skv for index in indices):
            raise ValueError(f"row {row}: TopK index outside runtime Skv")
        if not contract["strict"] <= chosen:
            raise ValueError(f"row {row}: missing strictly-above-threshold index")
        if len(chosen & contract["equal"]) != contract["quota"]:
            raise ValueError(f"row {row}: wrong tied-cutoff quota")
        if not chosen <= contract["strict"] | contract["equal"]:
            raise ValueError(f"row {row}: selected below threshold")
