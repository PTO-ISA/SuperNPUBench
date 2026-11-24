# DeepSeek V4.1 sparse attention — four-PE Cube TileOp

Source contract: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py (`sparse_attn`).

## 输入输出 shape

所有张量连续存储，最后一维连续；输出与 Q 的 shape 相同。

| 张量 | 数据类型 | Shape | 含义 |
|---|---|---|---|
| `q` | `T` | `[batch_size, query_length, num_heads, HeadDim]` | 每个 query、每个 head 的向量 |
| `kv` | `T` | `[batch_size, KVLength, HeadDim]` | 同时作为 K 和 V，所有 heads 共享 |
| `sink` | FP32 | `[num_heads]` | 每个 head 的 sink logit，仅参与分母 |
| `indices` | int32 | `[batch_size, query_length, TopK]` | 每个 query 的 KV 行号列表，所有 heads 共享 |
| `out` | `T` | `[batch_size, query_length, num_heads, HeadDim]` | attention 输出 |

`T` 为 FP16/BF16 存储类型（当前已验证 FP16 编译），内部统计和累加为 FP32。
`HeadDim` 就是 `head_dim`；`KVLength` 是 KV 总行数；`TopK` 是索引槽位数。
有效索引为 `[0, KVLength)`，`-1` 是 padding，重复索引保留重复贡献。
`HeadsPerGroup` 只影响内部计算分组，**不改变输入输出 shape**。

例如 `batch_size=1, query_length=4, num_heads=96, HeadDim=128, KVLength=4096, TopK=64`：

```text
q/out:   [1, 4, 96, 128]
kv:      [1, 4096, 128]
sink:    [96]
indices: [1, 4, 64]
```

## 计算语义与实现

Contiguous Q/O [B,S,H,D], KV [B,N,D], FP32 sink [H], int32 indices [B,S,K].
O = sum(exp(scale*dot(Q,KV_i))*KV_i) /
(exp(sink_h)+sum(exp(scale*dot(Q,KV_i)))). KV is shared by all heads.
Duplicates retain multiplicity. -1 anywhere is skipped, not a terminator.
Other out-of-range indices are also skipped defensively. No implicit causal mask.

The default `deepseek_v41::sparse_attn<T,D,N,K>` uses SharedTile TMATMUL
for both QK and PV. All logic is local to the header: no lambdas, QSMLA
configuration types, source visitors or external gather helpers.
HeadsPerGroup selects 64 heads (16 per PE, CubeM16) or 128 heads (32 per PE,
CubeM32). The default is 64. Runtime num_heads may be any positive count;
incomplete groups are padded and stripped.
KV is gathered in 32-row tiles. D is reduced in 64-element chunks with
TMATMUL_ACC. A single loop gathers KV, computes QK, updates online softmax,
then computes PV. Initialize m=sink, l=1 and O=0. For each block:

```text
m_new = max(m, rowmax(scores))
alpha = exp(m - m_new)
P = exp(scores - m_new)
l = alpha*l + rowsum(P)
O = alpha*O + cast_T(P) @ KV
m = m_new
```

Finally output O/l. Empty selections return zero. Row max/sum and output accumulators remain in tiles across KV blocks.
There is no explicit GM state buffer for them. QK directly produces the CUBE-layout score tile consumed
by scaling, masking and softmax. PV directly feeds the output TADD; neither
result uses a store/reload bridge. Reduction buffers preserve CUBE layout
and expose their reduced column with TREDUCEPREFIXVIEW.

API: sparse_attn<T,D,N,K,HeadsPerGroup=64>(out,q,kv,sink,indices,B,S,H,scale,workspace).
All four PEs invoke the same call; workspace is caller-owned and shared by the
PE group. Different concurrent groups must use separate workspaces. Buffers
must not alias. T is FP16/BF16 storage (FP16 compiled); D is a positive multiple
of 64, N>0, K>=0; B/S/H are positive runtime dimensions. Inputs/sinks are finite
and scale positive. Intermediate statistics and output accumulation are FP32.
Probability casting now occurs before PV within each online block, as in the
source algorithm. Block size, sink initialization, and reduction order can
still change rounding; this is not a bit-exact port. Performance is unmeasured.

Conservative Q/K/P/V SharedTile payload bound: 20 KiB for group 64,
32 KiB for group 128 at 16-bit storage.
Workspace: sizeof(T)*(2*HeadsPerGroup*D+HeadsPerGroup*32) + sizeof(float)*HeadsPerGroup bytes.
Additionally the implementation has per-PE FP32 output tiles (HeadsPerGroup/4*D elements),
bounded per-PE KV staging (32*D elements),
and mask storage; no K-sized index copy is needed. No N-by-D materialization is allocated.

Files:
- /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/kernels/basic_op/sparse_attn/sparse_attn.hpp
- /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/kernels/basic_op/sparse_attn/sparse_attn_vector_reference.hpp (previous vector baseline, renamed entry point)
- /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/sparse_attn/src/sparse_attn.cpp

Validation: main compiler checkout builds the FP16 ELF; disassembly confirms
QK and PV TMATMUL instructions. Smoke source covers a middle -1, duplicate
indices, two KV blocks, head tail, D accumulation and empty selection.
Numerical execution remains unverified: the mandated functional simulator
worktree /Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl is absent.

Why more code than dense attention:

- Sparse selection requires indexed KV staging and padding masks.
- KV is both K and V and is shared by heads: each matrix row is a query head.
- Sink adds a denominator-only term; treating it as a zero-valued virtual token
  keeps the normal online-softmax equations and handles empty selections.
- Four-PE TileOp requires explicit SharedTile, CUBE-layout vector operations, probability
  publication and output shards. These are backend mechanics, not extra math.
- Blocking avoids storing the full H-by-K score/probability matrix. Online
  max/sum/output rescaling is standard FlashAttention, not specific to sparsity.

The previous two-pass/source-visitor structure was inherited from multi-mode
QSMLA and was unnecessary for this standalone operator. It has been removed.

Intermediate-storage audit:

- Removed explicit row_max/row_sum and FP32 pv workspaces and their round trips.
- Final reciprocal overwrites the denominator tile; max/sum reductions reuse
  one physical reduction buffer after its previous prefix view is consumed.
- Complete head groups read Q/sink and write O directly. Only head tails use
  the padded q/sink/out buffers in the workspace.
- P publication stays: four local per-PE shards become the HeadsPerGroup-row SharedTile
  left operand. This is inter-PE operand assembly, not a redundant local copy.
- KV/mask staging stays for scalar sparse indices; TCVT of P and final O stays
  for the FP32-to-storage-type conversion.

Keeping output tiles live increases local tile pressure as D grows. Compilation
of the D=128/H=65 smoke succeeds, but compiler spilling and numerical execution
have not been validated across larger head dimensions. No claim of zero
compiler-generated memory traffic is made.

Readable identifier mapping (implementation parameter names):

| Previous | Current | Meaning |
|---|---|---|
| D | HeadDim | Per-head feature dimension (head_dim) |
| N | KVLength | KV row count |
| K | TopK | Index slots per query, not QK reduction width |
| B / S / H | batch_size / query_length / num_heads | Runtime tensor dimensions |
| kGroupM / kPeRows | kHeadsPerGroup / kHeadsPerPE | HeadsPerGroup heads per group, HeadsPerGroup/4 per PE |
| kTk / kTd | kKeysPerBlock / kDimsPerBlock | 32 selected keys, 64 feature dimensions |
| tMax / tSum | running_max / running_sum | Per-head online softmax state |
| output | output_numerator | Unnormalized weighted KV sum, normalized at the end |

Template argument order and function behavior are unchanged.

Configurable heads example:

```cpp
constexpr int HeadDim = 128, KVLength = 4096, TopK = 64;
constexpr int HeadsPerGroup = 128; // 64 or 128; independent of model head count
int num_heads = 96;
deepseek_v41::SparseAttnWorkspace<_Float16, HeadDim, HeadsPerGroup> workspace;
deepseek_v41::sparse_attn<_Float16, HeadDim, KVLength, TopK, HeadsPerGroup>(
    out, q, kv, sink, indices, batch_size, query_length, num_heads,
    softmax_scale, workspace);
```

The workspace object must be shared by the four PEs, not separately constructed
on each PE's stack. The example shows types/call syntax, not allocation policy.
Head count 64 was an implementation tiling default, not a DeepSeek requirement.
Compile checks: num_heads=16/group=64 and num_heads=129/group=128.
These verify template/code generation only, not numerical execution.
