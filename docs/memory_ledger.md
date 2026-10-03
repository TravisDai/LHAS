# Per-operation reservation ledger (final, D8)

This ledger implements the manuscript's conservative **accumulated-reservation
model** (Eq. `memory_feasibility`): every modeled reservation of every layer and
every selected operation is summed on each physical node, and the sum must not
exceed the modeled budget of 12 GiB minus the runtime reserve h = 1 GiB (D7), that
is 11 GiB. It is a *documented reservation bound*, not a measured or simulated peak
and not a lifetime-based memory timeline. Buffer reuse inside an operation only
sizes that operation's own reservation; nothing is reused across operations or
layers. "Feasible" means "satisfies this modeled reservation budget".

Implementation: `lhas.model.layercost` (layer reservations, shared by every
planner and baseline), `lhas.transport.schedule.evaluate` (per-operation
quantities), `lhas.collectives.service.Alt` (ledger categories),
`lhas.model.chain` (`_assembled_mask`, `_u_storage`), `lhas.model.branch`
(`ag_mask`, `consumer_full_T_buffer`, `shape_preserving_add`) and
`lhas.planner.graph_eval` (interfaces).

## Layer reservations m_i (Eqs. memory_dp, memory_mp)

w counts weights, biases and trainable BatchNorm γ/β; C is the number of BatchNorm
channels; ψ = 4 bytes; x, y include the global batch.

| Term | DP (nodes 1..p) | MP (nodes 1..p) |
|---|---|---|
| tensors and gradients | ψ[2(x+y)/p + 2w] | ψ[2(x+y) + 2w/p] |
| SGD momentum (D6, k = 1) | ψ·k·w | ψ·k·w/p |
| BatchNorm running mean and variance (D4, D8.3) | ψ·2C | ψ·2C/p |
| BatchNorm saved batch statistics (mean, inverse std) for backward | ψ·2C | ψ·2C/p |
| BatchNorm `num_batches_tracked` (int64 scalar) | 8 bytes | 8 bytes on every owner node (never divided by p) |
| per-layer workspace (D7) | 0 (declared) | 0 (declared) |

Any initially replicated input buffer is inside x. The runtime reserve h is charged
once per node through the budget, not per layer.

## Operation ledger categories

Every collective alternative carries four per-node byte vectors.

| Category | Contents | Size | Becomes reusable | Rule |
|---|---|---|---|---|
| `relay` | double buffer at every physical relay of a relayed route, including non-execution nodes | 2·min(κ, V) per concurrently reserved relayed route (κ = 0: 2V), summed per round, maximum over rounds | end of each round | manuscript sec:transport |
| `retained` | All-reduce contributions received in a phase and kept until that phase's local reduction completes | a_v·V at a reducer, maximum over phases | after the phase's local reduction | R1 |
| `accum` | reduction output accumulator (not in place, conservative) | V at a reducer, maximum over phases | after the phase's local reduction | R1 |
| `assembled` | All-gather shards received and kept | received bytes per node | not reused (they are the output) | A1 decides aliasing |

WRHT distribution phases write into the existing contribution tensor (inside m_i).
All-to-all receptions are written directly into the destination tensors listed
below, so an All-to-all reserves only `relay`. An operation's reservation is the
**sum** of its category peaks (conservative: the peaks need not coincide). When the
fastest alternative of an operation does not fit, slower nondominated alternatives
are retained (chain label DP, GPipe All-reduce selection); the typed-interface
engine checks the minimum-charge combination and reports the status of D10.

## Aliasing rules (corrected)

An exemption from a charge is allowed only where a **named** reservation holds the
same elements with sufficient capacity and a compatible use. Hosting an MP consumer
is not sufficient by itself.

* **A1 (All-gather output).** Assembled bytes at node v are not charged when
  (i) v hosts an MP consumer whose full-input reservation holds exactly the
  delivered tensor T: `x` if the consumer has no consumer-side operators and
  numel(x) = numel(T); the J1 operand buffer at addition joins; the C1 buffer when
  the consumer applies its own operators; or (ii) v is an MP producer node and every
  operator of Φ is storage-preserving (in-place ReLU), so the delivered tensor is
  the producer's own full output y. Dropout, normalization, pooling and flattening
  are not storage-preserving. Otherwise, for example at DP consumer nodes that
  receive the full tensor before selecting their batch columns, the bytes are
  charged.
* **U1 (separately stored boundary tensor).** When Φ is not storage-preserving, or
  the join is an addition, the producer's local shard of the boundary tensor and its
  gradient (2 × shard) are charged at producer nodes, except at a node where an MP
  consumer's named full-T buffer (above) holds the same elements, or for DP → DP
  with equal configurations (the consumer's batch-shard input holds the shard).
* **A2A destinations.** DP → MP FP writes into the MP consumer's full input (or
  its J1/C1 buffer). DP → MP BP and DD BP write into the DP producer's gradient
  shard (or the U1 buffer). MP → DP BP and MM write into the MP producer's full
  gradient reservation. DD FP writes into the DP consumer's input shard.
* **B1 (backward contributions at a fork or join).** With K ≥ 2 consumer
  contributions, a producer reserves (K+1)·V_loc: K contributions retained until
  the local sum, plus one accumulator. An identity holder reserves (K+1)·V_loc:
  K received contributions and an accumulator; its own input-gradient contribution
  is already inside its layer reservation (the earlier (K+2) double-counted it).
* **J1 (addition joins).** When the consumer's operators preserve shape (ReLU only)
  and numel(x) = numel(T): the first operand is summed in place into the consumer
  input, and one operand-sized buffer V holds the second operand, unless the
  identity holder has the consumer's configuration and the operand is read in place.
  When the consumer's operators change shape (for example ReLU, average pooling and
  flattening before ResNet-50's final FC layer): 3V, holding both pre-pooling
  operands (the sum and the ReLU in place) and the pre-pooling gradient.
* **C1 (consumer-side operators).** A consumer that applies its own operators to
  the interface tensor (for example the max pooling of a GoogLeNet branch 4) holds
  T and its gradient in a separate 2V buffer before its own input x.

## Regression evidence

* ResNet-50 final interface (delivered 2048×7×7, FC input 2048): with an MP4
  producer and an MP4 FC consumer, node 1 is charged at least 3·ψ·100352·B bytes
  (`test_resnet_final_interface_charges_prepooling_operands`); the earlier code
  zeroed the 308,281,344 received bytes on the basis of an 8,388,608-byte FC
  input reservation (`scripts/diagnostics/review_repro.py` at commit 757bd4c).
* GoogLeNet branch-4 max pooling: C1 charged at the MP consumer
  (`test_googlenet_consumer_side_pooling_is_charged_C1`).
* BatchNorm state and the int64 counter (`test_batchnorm_state_bytes_and_int64_counter`).

## What is not modeled

Element-wise intermediate tensors other than those covered by U1, J1 and C1 are
not modeled (the manuscript's weighted-layer approximation). Framework allocator
overheads, CUDA contexts and cuDNN workspaces are covered only by the declared
runtime reserve h. Collective scratch space beyond the categories above is not
modeled.
