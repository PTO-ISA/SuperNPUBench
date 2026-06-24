# Flash Attention Sample

`flash_attention_block_template.diss` is a checked-in block-template TileOP
disassembly of a compiler-produced Linx object for a larger flash attention
case. It is intended to show representative `BSTART.*`, `B.ARG`, `B.IOR`, and
`B.IOTI` sequences instead of the older scalar smoke output.

Related SuperNPUBench sources:

| Path | Role |
| --- | --- |
| [`../../benchmarks/npu/fusion`](../../benchmarks/npu/fusion) | Active NPU flash-attention-style fusion benchmark suite. |
| [`../../benchmarks/kernels/composite/src/flash_attention.cpp`](../../benchmarks/kernels/composite/src/flash_attention.cpp) | Composite flash attention benchmark entrypoint. |

Regenerate from a compatible Linx compiler object with:

```sh
llvm-objdump -dl flash_attention.o > flash_attention_block_template.diss
```
