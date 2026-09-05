---
name: binary-accumulation-cache-id
description: >-
  Explains binary-tree accumulation (二分累加) cache addressing via GetCacheId /
  ctz(idx+1), and how buffer[cacheId] stores partial sums of length 2^k. Use when
  writing or debugging AscendC/CANN reduce kernels, GetCacheId,
  ScalarGetCountOfValue, binary accumulation, or multi-level sum buffers.
---

# Binary accumulation cache ID (二分累加)

## Purpose

Streaming reduce over index `idx`: pick which level of a binary partial-sum tree
to update. Result address is `cacheId`; typical update is:

```cpp
buffer[cacheId] += currentValue;
```

## Core helper

```cpp
__aicore__ inline int64_t GetCacheId(const int64_t idx)
{
    return AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + NUM_ONE)) - NUM_ONE;
}
```

Equivalence:

```text
GetCacheId(idx) == ctz(idx + 1)   // count trailing zeros of (idx+1)
               == popcount(idx ^ (idx + 1)) - 1
```

`ScalarGetCountOfValue<1>(x)` is bit popcount of `x`.

## Level meaning

| cacheId | Partial-sum length |
|---------|--------------------|
| 0       | \(2^0 = 1\)        |
| 1       | \(2^1 = 2\)        |
| 2       | \(2^2 = 4\)        |
| k       | \(2^k\)            |

`idx` sequence → `cacheId`: `0,1,0,2,0,1,0,3,...` (carry every power of two).

## Minimal pattern

```cpp
for (int64_t idx = 0; idx < n; ++idx) {
    int64_t cacheId = GetCacheId(idx);
    buffer[cacheId] += currentValue;  // currentValue = element or already-merged block
}
```

## Carry / merge-up pattern (full binary add)

When lower levels must fold into the landing level (like binary carry):

```cpp
int64_t cacheId = GetCacheId(idx);
auto v = currentValue;
for (int64_t i = 0; i < cacheId; ++i) {
    v += buffer[i];
    // buffer[i] = 0;  // clear if the design reuses lower slots
}
buffer[cacheId] = v;  // use += only if currentValue is not already the merged sum
```

Choose `=` vs `+=` from whether `currentValue` already includes lower-level merges.

## Buffer sizing

Need levels `0 .. floor(log2(n))` inclusive → about `ceil(log2(n)) + 1` slots
(or fixed max depth for the kernel’s max `n`).

## Final fold

After the loop, pending levels still hold unused partial sums. Fold
`buffer[0..maxLevel]` (order depends on which levels are live for that `n`).
For exact power-of-two `n = 2^L`, the total often sits in `buffer[L]` after the
last carry; for general `n`, walk set bits of `n` / remaining levels.

## Checklist

1. `cacheId = GetCacheId(idx)` with `idx` 0-based iteration index.
2. `buffer[k]` means sum of \(2^k\) elements in the tree, not a flat ring buffer.
3. Decide merge-up (`for i < cacheId`) vs direct `+=` and stay consistent.
4. Size `buffer` for max tree depth; clear or define init before the loop.
5. After loop, reduce leftover levels into the scalar/output tile.

## Anti-patterns

- Treating `cacheId` as `idx % numBuffers` (double-buffer ping-pong) — wrong scheme.
- Using `ctz(idx)` instead of `ctz(idx+1)` — off-by-one vs this helper.
- Forgetting the post-loop fold when `n` is not a power of two.
