---
hide:
  - toc
---

<div class="snpu-masthead" markdown>
<span class="snpu-masthead__label">LinxISA PTO 0.57 · one-level · four PEs</span>

# SuperNPUBench golden manual

Program one core as a four-PE group with tile-ID dataflow, PE-local tile
versions, and versioned shared tile registers.

<div class="snpu-statusbar">
  <span><strong>ACTIVE</strong> 0.57 contract</span>
  <span>target: linx64-linx-none-elf</span>
  <span>ISA: 111 PTO + 2 identity intrinsics</span>
</div>
</div>

<div class="snpu-grid">
  <div class="snpu-card">
    <h3>Program four PEs</h3>
    <p>Start from one entry, branch by PE ID, partition tensors by core ID, and reconverge for group operations.</p>
    <a href="model/group-execution/">Open group execution →</a>
  </div>
  <div class="snpu-card" data-accent="compute">
    <h3>Browse PTO 0.57</h3>
    <p>Read only the workbook-selected operations, plus the two group identity intrinsics.</p>
    <a href="intrinsics/">Open the 113-entry manual →</a>
  </div>
  <div class="snpu-card" data-accent="vector">
    <h3>Inspect one-level benchmarks</h3>
    <p>See complete source, reached kernel code, exact build variants, and the 0.57 PTO surface.</p>
    <a href="benchmarks/">Open the one-level catalog →</a>
  </div>
</div>

## One model, five contracts

<div class="snpu-model" data-model-switcher>
  <div class="snpu-model__tabs" role="tablist" aria-label="NPU programming contracts">
    <button id="tab-group" role="tab" aria-controls="model-group" aria-selected="true" data-target="model-group">Group</button>
    <button id="tab-values" role="tab" aria-controls="model-values" aria-selected="false" data-target="model-values" tabindex="-1">Values</button>
    <button id="tab-shared" role="tab" aria-controls="model-shared" aria-selected="false" data-target="model-shared" tabindex="-1">Shared tiles</button>
    <button id="tab-execution" role="tab" aria-controls="model-execution" aria-selected="false" data-target="model-execution" tabindex="-1">Execution</button>
    <button id="tab-memory" role="tab" aria-controls="model-memory" aria-selected="false" data-target="model-memory" tabindex="-1">Memory</button>
  </div>
  <section class="snpu-model__panel" id="model-group" role="tabpanel" aria-labelledby="tab-group">
    <h3>Four PEs start together, then execute independently</h3>
    <p><code>get_thread_idx()</code> selects a PE path. <code>get_block_idx()</code> selects the core's tensor slice. Group operations require four compatible dynamic participants.</p>
    <p><a href="model/group-execution/">Group execution →</a></p>
  </section>
  <section class="snpu-model__panel" id="model-values" role="tabpanel" aria-labelledby="tab-values" hidden>
    <h3>Logical tiles carry shape and distribution</h3>
    <p>A tile may be split into four PE-local quarters or materialized as one core-local shared tile version.</p>
    <p><a href="model/tiles-and-layouts/">Tiles and distribution →</a></p>
  </section>
  <section class="snpu-model__panel" id="model-shared" role="tabpanel" aria-labelledby="tab-shared" hidden>
    <h3>Shared tiles are versioned registers</h3>
    <p><code>TLOAD</code>, <code>TSTORE</code>, and <code>TMOV</code> connect global, local, and shared tile values. A group matrix operation reads its right tile from shared storage.</p>
    <p><a href="model/shared-tile-registers/">Shared tile registers →</a></p>
  </section>
  <section class="snpu-model__panel" id="model-execution" role="tabpanel" aria-labelledby="tab-execution" hidden>
    <h3>Tile IDs expose superscalar work</h3>
    <p>Definitions create versions and uses bind exact versions. Independent operations may issue out of order when source versions are ready.</p>
    <p><a href="model/execution/">Tile-ID execution →</a></p>
  </section>
  <section class="snpu-model__panel" id="model-memory" role="tabpanel" aria-labelledby="tab-memory" hidden>
    <h3>Core and PE IDs select tensor slices</h3>
    <p>Global-memory movement uses explicit typed views. Shared tile registers handle register-sized inter-PE exchange inside a core.</p>
    <p><a href="model/memory/">Memory and movement →</a></p>
  </section>
</div>

## Grouped matrix path

<ol class="snpu-flow" aria-label="Grouped matrix execution path">
  <li><strong>Core slice</strong><span>get_block_idx()</span></li>
  <li><strong>PE quarter</strong><span>get_thread_idx()</span></li>
  <li><strong>Shared RHS</strong><span>TLOAD to shared tile</span></li>
  <li><strong>Group MMA</strong><span>four PE-local result quarters</span></li>
</ol>

```cpp title="Architecture-level grouped matmul" linenums="1"
SharedTile<half, K, N> shared_b;
if (get_thread_idx() == 0) {
  TLOAD(shared_b, global_b);
}
TMATMUL(local_c, local_a, shared_b);
```

Continue with [group programming examples](tutorials/group-programming.md) for
grouped matmul, reduce-to-one, all-reduce, and neighbor exchange.

!!! info "Compiler status"
    Shared tile-register lowering is an architecture contract and is not yet a
    required compiler feature. The one-level benchmark catalog remains the
    compile-checked implementation surface.
