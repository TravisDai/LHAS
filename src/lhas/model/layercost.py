"""Layer-level quantities shared by the chain engine, the typed-interface engine
and the GPipe adaptation, so that every planner uses identical rules.

Reservation per participating node (Eqs. memory_dp, memory_mp), float32:
  tensors and gradients    DP: psi[2(x+y)/p + 2w]      MP: psi[2(x+y) + 2w/p]
  momentum (D6)            DP: psi*k*w                 MP: psi*k*w/p
  BatchNorm (D4, D8.3), per normalization module attached to the layer:
    running mean/variance  DP: psi*2C                  MP: psi*2C/p   (channel owner)
    saved batch statistics DP: psi*2C (mean, invstd)   MP: psi*2C/p
    num_batches_tracked    8 bytes (int64 scalar) on every owner node; never divided by p
  per-layer workspace      declared constant (nominally 0; the runtime reserve h is
                           charged once per node through the budget, D7)
Here w counts weights, biases and trainable BatchNorm gamma/beta.
"""
from __future__ import annotations

import numpy as np

DP, MP = 0, 1
INT64_BYTES = 8

# Operators whose output may occupy the producer's own output reservation
# (in-place ReLU, as in the pinned torchvision graphs). Any other operator
# (dropout, normalization, pooling, flattening) produces a separately stored
# tensor for the purposes of the aliasing rules (docs/memory_ledger.md).
STORAGE_PRESERVING_OPS = {"relu"}


def aliasable_ops(ops) -> bool:
    return all(op in STORAGE_PRESERVING_OPS for op in ops)


def param_numel(L, norm_policy: str) -> int:
    n = L.weight_numel + L.bias_numel
    if L.norm is not None and norm_policy in ("local", "sync", "fixed"):
        n += L.norm["params"]
    return n


def norm_state_bytes(L, t: int, p: int, psi: int, norm_policy: str) -> float:
    """BatchNorm non-trainable state and saved statistics on one owner node."""
    if L.norm is None or norm_policy == "none":
        return 0.0
    C = L.norm["num_features"]
    share = C if t == DP else C / p
    running = 2 * share * psi                      # running_mean, running_var (float32)
    saved = 2 * share * psi                        # saved mean and inverse std for backward
    counter = INT64_BYTES if "num_batches_tracked" in L.norm.get("buffers", {}) else 0
    return running + saved + counter


def layer_reservation(L, c, B: int, psi: int, norm_policy: str, opt_factor: float,
                      workspace: float, N: int) -> np.ndarray:
    t, p = c
    x, y, w = L.x_numel * B, L.y_numel * B, param_numel(L, norm_policy)
    if t == DP:
        v = psi * (2 * (x + y) / p + 2 * w) + psi * opt_factor * w
    else:
        v = psi * (2 * (x + y) + 2 * w / p) + psi * opt_factor * w / p
    v += norm_state_bytes(L, t, p, psi, norm_policy) + workspace
    out = np.zeros(N)
    out[:p] = v
    return out


__all__ = ["DP", "MP", "param_numel", "norm_state_bytes", "layer_reservation", "aliasable_ops",
           "STORAGE_PRESERVING_OPS"]
