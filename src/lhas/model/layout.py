"""Boundary ownership and All-to-all payload matrices (manuscript sec:inter-layer).

Conventions (1-based physical node identifiers, prefixes P_p = {1..p}):

* Batch interval of node u under DP with p nodes: I_u(p) = [floor((u-1)B/p), floor(uB/p)).
  With p | B (Eq. bg_feasible_nodes) these are the manuscript's contiguous B/p intervals.
* Feature rows of the boundary tensor U_i (nu_i rows in the feature-by-batch view).
  An output-feature MP producer with p | n_i owns the rows derived from its channel
  block [(u-1)n_i/p, u n_i/p); with a local, ownership-preserving Phi_i each channel
  maps to r_i = nu_i / n_i consecutive rows (channel-major flattening).
* DP-to-MP backward forwarding rows F_{i,b} = [floor((b-1)nu/q), floor(b nu/q)) for
  sources b in P_q (Eq. boundary_dp_mp_bp_payload); sets may be empty.

Every matrix returns (src, dst, bytes) with self-transfers and zero entries removed.
"""
from __future__ import annotations

import numpy as np

from ..collectives.families import circ


def batch_interval(u: int, p: int, B: int) -> tuple[int, int]:
    return ((u - 1) * B) // p, (u * B) // p


def row_block_mp(u: int, p: int, n_channels: int, rows_per_channel: int) -> tuple[int, int]:
    if n_channels % p:
        raise ValueError("MP node count must divide the output-feature count")
    c = n_channels // p
    return (u - 1) * c * rows_per_channel, u * c * rows_per_channel


def forwarding_rows(b: int, q: int, nu: int) -> tuple[int, int]:
    return ((b - 1) * nu) // q, (b * nu) // q


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _pack(entries):
    src, dst, nb = [], [], []
    for u, v, b in entries:
        if u != v and b > 0:
            src.append(u); dst.append(v); nb.append(int(b))
    return np.asarray(src, np.int64), np.asarray(dst, np.int64), np.asarray(nb, np.int64)


# ----------------------------------------------------------------------------
# strategy-switch exchanges (Table tab:inter)
# ----------------------------------------------------------------------------
def dp_to_mp_fp(p_i: int, p_j: int, nu: int, B: int, psi: int):
    """Each DP source sends its complete batch shard to every MP destination
    (Eq. boundary_alltoall_payload): psi * nu * |I_u(p_i)|."""
    return _pack((u, v, psi * nu * (batch_interval(u, p_i, B)[1] - batch_interval(u, p_i, B)[0]))
                 for u in range(1, p_i + 1) for v in range(1, p_j + 1))


def dp_to_mp_bp(p_i: int, p_j: int, nu: int, B: int, psi: int):
    """After the consumer's MP All-reduce, source b in P_j forwards rows F_{i,b}
    for each DP destination's batch columns (Eq. boundary_dp_mp_bp_payload)."""
    out = []
    for b in range(1, p_j + 1):
        f = forwarding_rows(b, p_j, nu)
        for v in range(1, p_i + 1):
            I = batch_interval(v, p_i, B)
            out.append((b, v, psi * (f[1] - f[0]) * (I[1] - I[0])))
    return _pack(out)


def mp_to_dp_bp(p_i: int, p_j: int, n_channels: int, rows_per_channel: int, B: int, psi: int):
    """DP source u in P_j holds nu x |I_u(p_j)|; MP destination v in P_i needs its
    feature rows across the full batch (Eq. boundary_mp_dp_bp_payload)."""
    out = []
    for u in range(1, p_j + 1):
        I = batch_interval(u, p_j, B)
        for v in range(1, p_i + 1):
            r = row_block_mp(v, p_i, n_channels, rows_per_channel)
            out.append((u, v, psi * (r[1] - r[0]) * (I[1] - I[0])))
    return _pack(out)


# ----------------------------------------------------------------------------
# same-strategy ownership changes (Eqs. dd_matrix, mm_matrix)
# ----------------------------------------------------------------------------
def dd_fp(p_i: int, p_j: int, nu: int, B: int, psi: int):
    return _pack((u, v, psi * nu * _overlap(batch_interval(u, p_i, B), batch_interval(v, p_j, B)))
                 for u in range(1, p_i + 1) for v in range(1, p_j + 1))


def dd_bp(p_i: int, p_j: int, nu: int, B: int, psi: int):
    s, d, b = dd_fp(p_i, p_j, nu, B, psi)
    return d, s, b


def mm_bp(p_i: int, p_j: int, n_channels: int, rows_per_channel: int, B: int, psi: int, N: int):
    """Missing producer nodes v in P_i minus P_j receive their rows over the full
    batch from the nearest consumer node pi(v) (circular distance, smaller id)."""
    out = []
    for v in range(p_j + 1, p_i + 1):
        src = min(range(1, p_j + 1), key=lambda u: (circ(u, v, N), u))
        r = row_block_mp(v, p_i, n_channels, rows_per_channel)
        out.append((src, v, psi * (r[1] - r[0]) * B))
    return _pack(out)


__all__ = ["batch_interval", "row_block_mp", "forwarding_rows", "dp_to_mp_fp", "dp_to_mp_bp",
           "mp_to_dp_bp", "dd_fp", "dd_bp", "mm_bp"]
