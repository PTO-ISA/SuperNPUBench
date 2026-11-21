# DeepSeek-V4.1-Flash on WSE

This single-core decode example uses the same layer weights, schedule and arithmetic as the [traditional kernel](../deepseekv4.1/README.md). Its memory backend models separate PGAS receive slots for the attention and expert owners, plus SoC L1.SRAM windows for bulk transfers.

| Transfer | Producer and consumer | WSE path |
| --- | --- | --- |
| Compressed KV and index keys | A Full layer publishes newly completed rows. Indexers and sparse attention read selected physical rows. | PGAS TLOAD/TSTORE, 20 or 17 bytes per row |
| Row selection and candidate blocks | Full or Reindex publishes per-query IDs for downstream layers. | PGAS TLOAD/TSTORE, 16 or 8 bytes |
| Routed token, shard count, expert return | Router writes the selected owner's inbox, each shard writes the original Top-K return slot. | PGAS TLOAD/TSTORE, 48, 4 or 128 bytes |
| Resident MXFP4 expert weights and scales | The loader stages each known expert range in its assigned SRAM window once per sequence. | SDMA, 52,224 bytes total, split evenly between logical shards |
| Attention to MoE | Attention publishes the four outgoing HC streams and pre-mix. MoE consumes the destination window after completion. | SDMA, 528 bytes per token |

The window and C2A pair ring belong to the current attention layer. Its own matrix weights, shared expert and intermediate vectors are local. The caller supplies a separately addressable PGAS inbox for each expert owner, the published cache and index banks, and the output containing the return slots. The backend uses a byte-typed GM view and sets the tile's valid columns to the packet size. This preserves adjacent rows and return slots. See the [PTO-SPEC GM addressing decision](https://github.com/PTO-ISA/pto-spec/blob/main/docs/status/decisions/ADR-MEM-0008-tload-tstore-gm-byte-row-stride.md).

`Memory<Sdma>` submits a window ID, byte offset, byte count and ticket. The consumer checks the ticket, successful status and completed byte count before reading the destination. In the hosted example, `SingleCoreSdma` checks window bounds, copies into distinct guest-memory ranges representing L1.SRAM, and returns completion. One PE evaluates the layer on one simulated Core.

`MessageSdma` and `serve_one_copy` implement the client and service for a provisioned endpoint pair. They send a 24-byte request and a 12-byte completion using `PTOSendMessage` and `PTORecvMessage`. They order local accesses around the exchange. The SoC service owns the payload movement and must wait for destination visibility before replying. The typed MSGB helpers come from [Linx-TileOP-API #315](https://github.com/LinxISA/Linx-TileOP-API/pull/315).

## Build and verify

From `benchmark/one-level-arch`, use the same toolchain, `COMPILER_DIR`, headers and Python dependencies as the [traditional example](../deepseekv4.1/README.md).

```sh
test/solution/deepseekv4.1/compile.all
out=output/solution/deepseekv4.1/elf
wse="$out/solution_deepseekv4.1_wse.elf"
baseline="$out/solution_deepseekv4.1_decode.elf"
python3 test/solution/deepseekv4.1/run.py --gfrun "$GFRUN" \
  --elf "$wse" --wse --shards 2 --baseline-elf "$baseline" --all-layers --jobs 4
```

The runner checks the numerical oracle, exact PGAS packet counts and completed SDMA byte totals. With two shards and two tokens the expected total is four transfers and 53,280 bytes. With `--baseline-elf`, each fixture also runs on the one-shard traditional kernel and the runner compares every byte of the layer outputs, publications and final cache. Add `--start 0 --seed 42 --empty-shard` to exercise partially filled caches and a shard receiving zero routed tokens.

## Run the MSGB service on a second PE

Use the same 0.59 TileOP headers for the layer math. Provide the additional typed message headers from [Linx-TileOP-API #315](https://github.com/LinxISA/Linx-TileOP-API/pull/315) and the resident runtime and matching `gfrun` from [SuperScalarModel #1243](https://github.com/LinxISA/SuperScalarModel/pull/1243).

```sh
make -C test/solution/deepseekv4.1 msgb \
  MSGB_RUNTIME="$MSGB_MODEL/runtime/linx-runtime" MSGB_INCLUDE="$MSGB_API/include"
python3 test/solution/deepseekv4.1/run.py --gfrun "$MSGB_MODEL/bin/gfrun" \
  --elf "$out/solution_deepseekv4.1_wse_msgb.elf" --shards 2 --all-layers --jobs 4 \
  --baseline-gfrun "$GFRUN" --baseline-elf "$baseline"
```

The resident runtime provisions the endpoints on one simulated Core. PE0 computes the full layer. PE1 receives the actual typed copy requests, executes the bounds-checked SRAM copy stub, and replies with the ticket and completed byte count. The runner requires successful scheduler completions from both PEs and a return to idle in addition to the same guest-memory and transport checks.
