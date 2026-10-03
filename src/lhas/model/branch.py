"""Typed fork/join interface charges and exact search for branch networks.

Protocol (manuscript sec:branch, with the author's join-layout guidance):

FP delivery of an interface tensor T to its consumers (each consumer's input
layout is derived from its configuration: a DP consumer needs its batch shard
with all features; an MP consumer needs the complete tensor on every node):
  * MP producer (a channel slice or operand): one All-gather to the largest
    required prefix over all consumers, charged once;
  * DP producer: the per-consumer boundary matrix (DP->DP or DP->MP);
  * identity operand of an addition: delivered from the input layout of the
    consumer that holds it (the first consumer of the earlier interface) to each
    consumer's input layout (input-layout redistribution, nearest holder for
    replicated sources);
  * addition: each consumer sums the aligned operands locally, 3V/beta_red.
BP: each consumer's gradient contribution is returned to every producer's
layout by the per-pair boundary matrix (DP->DP, MP->DP, DP->MP with F rows,
MP->MP nearest-node) and the producer adds its K contributions locally,
(K+1)V/beta_red for K >= 2. For an identity operand, contributions are returned
to the holder's input layout and summed there with the holder's own input gradient.

Chain interfaces (one producer, one consumer) reduce exactly to Eqs. intra_cost,
inter_cost, dd_matrix and mm_matrix; tests compare this engine with ChainModel.

Search: exact minimization of the additive objective over all layer
configurations with a tuple state at every multi-consumer or join interface
(min-plus contractions in numba). Every operation uses its minimum-latency
alternative (minimum charge under alpha); the resulting plan's reservations are
then checked. When that plan is memory-feasible it is optimal for the
memory-constrained problem as well (its cost is a lower bound). Otherwise the
result is reported as not established.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from ..collectives.service import CollectiveService, Alt
from ..collectives.families import circ
from . import layout as Lt
from .chain import Settings, DP, MP
from .graph import GraphStructure, Interface
from . import layercost as _lc

SHAPE_OPS = {"maxpool", "avgpool", "adaptive_avgpool", "flatten", "batchnorm", "dropout"}  # non-aliasable (see layercost)


def _pack(entries):
    src, dst, nb = [], [], []
    for u, v, b in entries:
        if u != v and b > 0:
            src.append(u); dst.append(v); nb.append(int(b))
    return np.asarray(src, np.int64), np.asarray(dst, np.int64), np.asarray(nb, np.int64)


def input_to_input(ts, ps, td, pd, nu, B, psi, N):
    """Redistribute a tensor held in a consumer *input* layout (DP: batch shard with
    all rows; MP: complete tensor on every node of the prefix) to another input
    layout. Unique holders send their own part; for a replicated source each
    missing block comes from the nearest holder (circular distance, smaller id)."""
    if ts == DP and td == DP:
        return Lt.dd_fp(ps, pd, nu, B, psi)
    if ts == DP and td == MP:
        return Lt.dp_to_mp_fp(ps, pd, nu, B, psi)
    out = []
    for v in range(ps + 1, pd + 1):
        src = min(range(1, ps + 1), key=lambda u: (circ(u, v, N), u))
        if td == MP:
            out.append((src, v, psi * nu * B))
        else:
            I = Lt.batch_interval(v, pd, B)
            out.append((src, v, psi * nu * (I[1] - I[0])))
    return _pack(out)


@dataclass
class OpChoice:
    name: str
    alt: Alt
    mask: object = None
    alts: object = None          # all retained alternatives of this operation (memory lower bound)
    spec: object = None          # rebuild specification for independent verification

    def mem(self):
        return self.alt.mem(self.mask)

    def mem_lower_bound(self):
        if not self.alts:
            return self.mem()
        return np.min(np.stack([a.mem(self.mask) for a in self.alts]), axis=0)


class GraphModel:
    def __init__(self, gs: GraphStructure, svc: CollectiveService, st: Settings, compute_table: dict | None = None):
        st.validate()
        self.compute_table = compute_table
        self.gs, self.svc, self.st = gs, svc, st
        self.tp, self.cp = svc.tp, svc.cp
        self.N, self.psi, self.B = self.tp.N, self.cp.psi, st.B
        self.L = {L.name: L for L in gs.workload.layers}
        self.order = [L.name for L in gs.workload.layers]
        self.budget = float(self.cp.budget_bytes - self.cp.runtime_reserve_bytes)
        self._charge = {}
        self._pair = {}

    # ------------------------------------------------------------------ layers
    def configs(self, name):
        L = self.L[name]
        out = [(DP, p) for p in range(1, self.N + 1) if self.B % p == 0]
        out += [(MP, p) for p in range(1, self.N + 1) if L.n_out % p == 0]
        out = [c for c in out if self.m(name, c).max() <= self.budget]
        if self.compute_table is not None:
            from .profile import has_entry      # profiled mode: unmeasured shapes excluded
            out = [c for c in out if has_entry(self.compute_table, self.gs.workload, L, c)]
        return out

    def missing_profile(self, cfg: dict) -> list:
        """Profiled mode: keys of the (layer, configuration) pairs of `cfg` without a measured
        compute time (T6); empty in analytical mode."""
        if self.compute_table is None:
            return []
        from .profile import has_entry, profile_key
        wl = self.gs.workload
        return [profile_key(wl, self.L[n], cfg[n]) for n in self.order
                if not has_entry(self.compute_table, wl, self.L[n], cfg[n])]

    def memory_allowed(self, name, c) -> bool:
        """Divisibility and the layer's own reservation within the budget (profile ignored)."""
        L = self.L[name]
        ok = (self.B % c[1] == 0) if c[0] == DP else (L.n_out % c[1] == 0)
        return ok and c[1] <= self.N and self.m(name, c).max() <= self.budget

    def param_numel(self, L):
        return _lc.param_numel(L, self.st.norm_policy)

    def needs_input_grad(self, L):
        return not (self.st.raw_input_grad == "omit" and L.input_is_graph_input)

    def tcp(self, name, c):
        L = self.L[name]
        if self.compute_table is not None:
            from .chain import _profiled_time
            return _profiled_time(self.compute_table, self.gs.workload, L, c, self.B, self.needs_input_grad(L))
        mults = 3 if self.needs_input_grad(L) else 2
        return mults * L.mult_flops_per_example() * self.B / (c[1] * self.cp.utilization * self.cp.peak_flops)

    def m(self, name, c):
        return _lc.layer_reservation(self.L[name], c, self.B, self.psi, self.st.norm_policy,
                                     self.st.optimizer_state_factor, self.st.workspace_bytes, self.N)

    def _own_ops(self, name, c):
        """Intra-layer reductions (and normalization statistics) of a layer."""
        L, (t, p) = self.L[name], c
        ops = []
        if t == DP:
            V = self.psi * self.param_numel(L)
            ops.append(("AR_w", self.svc.allreduce(p, V), ("AR", p, V)))
            if L.norm is not None and self.st.norm_policy == "sync" and p > 1:
                C = L.norm["num_features"]
                ops.append(("AR_bnstat_fp", self.svc.allreduce(p, self.psi * 2 * C), ("AR", p, self.psi * 2 * C)))
                ops.append(("AR_bnstat_bp", self.svc.allreduce(p, self.psi * 2 * C), ("AR", p, self.psi * 2 * C)))
        elif self.needs_input_grad(L):
            V = self.psi * L.x_numel * self.B
            ops.append(("AR_x", self.svc.allreduce(p, V), ("AR", p, V)))
        return ops

    def charge(self, name, c, ag=None):
        """Layer charge with optional output All-gather `ag` = (p, q, shard bytes,
        assembled mask): T_cp + intra - alpha min(T_cp, intra transport).
        Returns (charge, [OpChoice])."""
        key = (name, c, None if ag is None else (ag[0], ag[1], ag[2], None if ag[3] is None else ag[3].tobytes()))
        if key in self._charge:
            return self._charge[key]
        t = self.tcp(name, c)
        own = self._own_ops(name, c)
        groups = [alts for _, alts, _ in own]
        names = [n for n, _, _ in own]
        specs = [sp for _, _, sp in own]
        if ag is not None:
            groups.append(self.svc.allgather(ag[0], ag[1], ag[2]))
            names.append("AG_y")
            specs.append(("AG", ag[0], ag[1], ag[2]))
        best = None
        # AR-type operations are independent of alpha except through the overlap
        # credit on total intra transport; enumerate combinations (small sets).
        for combo in itertools.product(*groups):
            tr = sum(a.transport for a in combo)
            red = sum(a.reduction for a in combo)
            g = t + tr + red - self.st.alpha * min(t, tr)
            if best is None or g < best[0] - 1e-15:
                best = (g, combo)
        g, combo = best if best is not None else (t, ())
        choices = [OpChoice(n, a, (ag[3] if n == "AG_y" else None), alts=g_, spec=sp)
                   for n, a, g_, sp in zip(names, combo, groups, specs)]
        self._charge[key] = (g, choices)
        return self._charge[key]

    # ------------------------------------------------------------------ interface pieces
    def rows(self, I: Interface, prod) -> int:
        return (prod.ch1 - prod.ch0) * I.rows_per_channel

    def _a2a(self, key, mat):
        s, d, b = mat
        alts = self.svc.alltoall(key, s, d, b)
        a = min(alts, key=lambda a: a.latency)
        a._spec = ("A2A", s, d, b)          # rebuild specification for verification
        return a

    # ------------------------------------------------------------------ aliasing (docs/memory_ledger.md)
    def consumer_full_T_buffer(self, I: Interface, cname: str) -> str | None:
        """Name of the reservation at an MP consumer's nodes that holds the complete
        interface tensor T with matching elements, or None.
          'x'  : the consumer's own full-input reservation, when T is exactly its input
                 (no consumer-side operators and equal element counts);
          'J1' : the explicitly charged addition-operand buffer (addition joins);
          'C1' : the explicitly charged consumer-side input buffer (consumer operators)."""
        if I.join == "add":
            return "J1"
        if I.consumer_ops.get(cname):
            return "C1"
        return "x" if self.L[cname].x_numel == I.numel else None

    def ag_mask(self, I: Interface, prod, ccfg: dict, pb: int) -> np.ndarray:
        """1 where assembled All-gather bytes must be reserved in e_i; 0 only where a
        named reservation with matching elements, capacity and compatible use holds
        them: a full-T buffer of an MP consumer on that node, or (for storage-
        preserving producer maps) the producer's own full-output reservation."""
        mask = np.ones(self.N)
        for cn, (t, p) in ccfg.items():
            if t == MP and self.consumer_full_T_buffer(I, cn) is not None:
                mask[:p] = 0.0
        if _lc.aliasable_ops(prod.ops) and I.join != "add":
            mask[:pb] = 0.0
        return mask

    def shape_preserving_add(self, I: Interface, cname: str) -> bool:
        return all(op == "relu" for op in I.consumer_ops.get(cname, [])) and self.L[cname].x_numel == I.numel

    def fp_pair_dp(self, I, prod, cb, cc):
        """DP producer slice -> one consumer (per-consumer boundary matrix)."""
        (tb, pb), (tc, pc) = cb, cc
        nu = self.rows(I, prod)
        key = ("fp", I.idx, prod.ch0, cb, cc)
        if key not in self._pair:
            mat = Lt.dd_fp(pb, pc, nu, self.B, self.psi) if tc == DP else Lt.dp_to_mp_fp(pb, pc, nu, self.B, self.psi)
            a = self._a2a(("fpdp", pb, pc, tc, nu, self.B), mat)
            self._pair[key] = (a.latency, [OpChoice("A2A_fp", a, alts=[a], spec=a._spec)])
        return self._pair[key]

    def bp_pair(self, I, prod, cb, cc):
        """Consumer gradient contribution -> layer producer layout."""
        (tb, pb), (tc, pc) = cb, cc
        nu = self.rows(I, prod)
        key = ("bp", I.idx, prod.ch0, cb, cc)
        if key not in self._pair:
            ch = prod.ch1 - prod.ch0
            r = I.rows_per_channel
            if tb == MP and tc == MP:
                mat, k = Lt.mm_bp(pb, pc, ch, r, self.B, self.psi, self.N), ("mm", pb, pc, ch, r)
            elif tb == MP and tc == DP:
                mat, k = Lt.mp_to_dp_bp(pb, pc, ch, r, self.B, self.psi), ("mpdp", pb, pc, ch, r)
            elif tb == DP and tc == MP:
                mat, k = Lt.dp_to_mp_bp(pb, pc, nu, self.B, self.psi), ("dpmp_bp", pb, pc, nu)
            else:
                mat, k = Lt.dd_bp(pb, pc, nu, self.B, self.psi), ("dd_bp", pb, pc, nu)
            a = self._a2a(("bp",) + k + (self.B,), mat)
            self._pair[key] = (a.latency, [OpChoice("A2A_bp", a, alts=[a], spec=a._spec)])
        return self._pair[key]

    def id_fp_pair(self, I, ch, cc):
        """Identity operand: holder input layout -> consumer input layout."""
        key = ("idfp", I.idx, ch, cc)
        if key not in self._pair:
            nu = I.numel
            mat = input_to_input(ch[0], ch[1], cc[0], cc[1], nu, self.B, self.psi, self.N)
            a = self._a2a(("id", ch, cc, nu, self.B), mat)
            self._pair[key] = (a.latency, [OpChoice("A2A_id_fp", a, alts=[a], spec=a._spec)])
        return self._pair[key]

    def id_bp_pair(self, I, ch, cc):
        """Consumer gradient contribution -> identity holder's input layout."""
        key = ("idbp", I.idx, ch, cc)
        if key not in self._pair:
            nu = I.numel
            mat = input_to_input(cc[0], cc[1], ch[0], ch[1], nu, self.B, self.psi, self.N)
            a = self._a2a(("id", cc, ch, nu, self.B), mat)
            self._pair[key] = (a.latency, [OpChoice("A2A_id_bp", a, alts=[a], spec=a._spec)])
        return self._pair[key]

    def local_sum_time(self, V_loc, K):
        return 0.0 if K < 2 else (K + 1) * V_loc / self.cp.beta_red

    def local_bytes_layer(self, I, prod, cb):
        tb, pb = cb
        nu = self.rows(I, prod)
        return self.psi * (nu / pb) * self.B if tb == MP else self.psi * nu * self.B / pb

    def local_bytes_input(self, I, cc):
        tc, pc = cc
        return self.psi * I.numel * self.B / (pc if tc == DP else 1)


__all__ = ["GraphModel", "input_to_input", "OpChoice"]
