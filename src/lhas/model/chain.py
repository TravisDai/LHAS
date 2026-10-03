"""Layer and transition charges for chain networks (manuscript Section 4).

Implements, for consecutive weighted layers i -> j=i+1:
  * feasible configurations A_i (Eq. bg_feasible_nodes), minus individually
    memory-infeasible ones;
  * T_cp (Eq. comp) with analytical FLOPs and assumed utilization;
  * m_i reservation vectors (Eqs. memory_dp, memory_mp) plus declared optimizer
    state, parameter-attached normalization buffers and workspace;
  * intra-layer charges (Eq. intra_cost) and boundary charges (Eqs. inter_cost,
    dd_matrix, mm_matrix) as combinations of collective alternatives;
  * the additional reservation e_i of each schedule combination (docs/memory_ledger.md);
  * the layer-and-boundary charge G_i (Eq. layer_charge) and the terminal charge
    (Eq. terminal_charge).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..collectives.service import CollectiveService, Alt
from . import layout as Lt
from .workload import Workload, WLayer
from . import layercost as _lc

DP, MP = 0, 1


@dataclass(frozen=True)
class Settings:
    B: int
    alpha: float = 0.0
    raw_input_grad: str = "retain"          # 'retain' | 'omit' (explicit flag; see decision table)
    optimizer_state_factor: float = 0.0     # extra parameter-sized state (e.g. 1 for SGD momentum)
    workspace_bytes: int = 0                # modeled per-node layer workspace
    norm_policy: str = "none"               # 'none' (no normalization in graph) | 'local' | 'sync' | 'fixed'
    max_mp_p: Optional[int] = None          # optional declared restriction (None = all divisors)

    def validate(self):
        assert self.raw_input_grad in ("retain", "omit")
        assert 0.0 <= self.alpha <= 1.0


@dataclass
class TransAlt:
    """One schedule combination for a transition (or the terminal charge)."""
    tr_intra: float
    red_intra: float
    inter: float
    e: np.ndarray
    choices: tuple

    def G(self, tcp: float, alpha: float) -> float:
        a_intra = self.tr_intra + self.red_intra
        return tcp + a_intra + self.inter - alpha * min(tcp, a_intra - self.red_intra)


def _profiled_time(table: dict, wl, L, c, B: int, input_grad: bool) -> float:
    """Measured local time for (layer, strategy, p) with the D17 safeguards
    (lhas.model.profile); refuses missing or mismatched entries."""
    from .profile import profiled_time
    return profiled_time(table, wl, L, c, B, input_grad)


def _divisors_upto(n: int, N: int) -> list[int]:
    return [p for p in range(1, N + 1) if n % p == 0]


class ChainModel:
    def __init__(self, wl: Workload, svc: CollectiveService, st: Settings, compute_table: dict | None = None):
        self.compute_table = compute_table   # profiled_<gpu> input mode (scripts/ingest_colab.py); None = analytic
        if not wl.is_chain:
            raise ValueError("ChainModel requires a chain workload")
        st.validate()
        self.wl, self.svc, self.st = wl, svc, st
        self.tp, self.cp = svc.tp, svc.cp
        self.N = self.tp.N
        self.psi = self.cp.psi
        self.L = wl.layers
        self.budget = float(self.cp.budget_bytes - self.cp.runtime_reserve_bytes)
        self._trans_cache = {}

    # ------------------------------------------------------------------ basics
    def configs(self, i: int, memory_filter: bool = True) -> list[tuple]:
        L = self.L[i]
        out = [(DP, p) for p in _divisors_upto(self.st.B, self.N)]
        mp = _divisors_upto(L.n_out, self.N)
        if self.st.max_mp_p:
            mp = [p for p in mp if p <= self.st.max_mp_p]
        out += [(MP, p) for p in mp]
        if memory_filter:
            out = [c for c in out if self.m(i, c).max() <= self.budget]
        if self.compute_table is not None:
            # profiled input mode: configurations whose local shape was not measured are
            # excluded (declared restriction; never extrapolated)
            from .profile import has_entry
            out = [c for c in out if has_entry(self.compute_table, self.wl, L, c)]
        return out

    def missing_profile(self, seq) -> list:
        """Profiled mode: keys of the (layer, configuration) pairs of `seq` without a measured
        compute time (T6); empty in analytical mode."""
        if self.compute_table is None:
            return []
        from .profile import has_entry, profile_key
        return [profile_key(self.wl, self.L[i], c) for i, c in enumerate(seq)
                if not has_entry(self.compute_table, self.wl, self.L[i], c)]

    def param_numel(self, L: WLayer) -> int:
        return _lc.param_numel(L, self.st.norm_policy)

    def needs_input_grad(self, L: WLayer) -> bool:
        return not (self.st.raw_input_grad == "omit" and L.input_is_graph_input)

    def flops(self, i: int) -> float:
        L = self.L[i]
        mults = 3 if self.needs_input_grad(L) else 2
        return float(mults) * L.mult_flops_per_example() * self.st.B

    def tcp(self, i: int, c) -> float:
        if self.compute_table is not None:
            return _profiled_time(self.compute_table, self.wl, self.L[i], c, self.st.B,
                                  self.needs_input_grad(self.L[i]))
        return self.flops(i) / (c[1] * self.cp.utilization * self.cp.peak_flops)

    def m(self, i: int, c) -> np.ndarray:
        return _lc.layer_reservation(self.L[i], c, self.st.B, self.psi, self.st.norm_policy,
                                     self.st.optimizer_state_factor, self.st.workspace_bytes, self.N)

    # ------------------------------------------------------------------ operations
    def _intra_ops(self, i: int, c, q: int):
        """(name, alternatives, is_intra_transport_candidate) for layer i's intra charge."""
        L, (t, p) = self.L[i], c
        psi, B = self.psi, self.st.B
        ops = []
        if t == DP:
            ops.append(("AR_w", self.svc.allreduce(p, psi * self.param_numel(L)), None))
        else:
            S = psi * L.u_numel * B
            ops.append(("AG_y", self.svc.allgather(p, q, S // p), "AG"))
            if self.needs_input_grad(L):
                ops.append(("AR_x", self.svc.allreduce(p, psi * L.x_numel * B), None))
        return ops

    def _inter_ops(self, i: int, ci, cj):
        return [(name, self.svc.alltoall(spec[0], *spec[1:])) for name, spec in self._inter_specs(i, ci, cj)]

    def _assembled_mask(self, i: int, ci, cj) -> np.ndarray:
        """1 where All-gather output must be reserved in e_i, 0 where it aliases
        an existing reservation (docs/memory_ledger.md, rule A1)."""
        L = self.L[i]
        (ti, pi), (tj, pj) = ci, cj
        mask = np.ones(self.N)
        if tj == MP:
            mask[:pj] = 0.0                 # consumer's full-input reservation x_j
        if L.phi_identity and ti == MP:
            mask[:pi] = 0.0                 # producer's full-output reservation y_i
        return mask

    def _u_storage(self, i: int, ci, cj) -> np.ndarray:
        """Separately stored U_i shard and its gradient at producer nodes when
        Phi_i is not an identity map and no reservation covers it (rule U1)."""
        L = self.L[i]
        out = np.zeros(self.N)
        if L.phi_identity:
            return out
        (ti, pi), (tj, pj) = ci, cj
        shard = self.psi * L.u_numel * self.st.B / pi
        for u in range(1, pi + 1):
            covered = (tj == MP and u <= pj) or (ti == DP and tj == DP and pi == pj)
            if not covered:
                out[u - 1] = 2 * shard
        return out

    def demands(self) -> list:
        """Every collective evaluation the planner can request (for grouped prefetch)."""
        out = []
        psi, B = self.psi, self.st.B
        A = [self.configs(i) for i in range(len(self.L))]
        for i, L in enumerate(self.L):
            for ci in A[i]:
                t, p = ci
                if t == DP:
                    out.append(("AR", p, psi * self.param_numel(L)))
                else:
                    if self.needs_input_grad(L):
                        out.append(("AR", p, psi * L.x_numel * B))
                    if i + 1 < len(self.L):
                        for cj in A[i + 1]:
                            out.append(("AG", p, cj[1], (psi * L.u_numel * B) // p))
                    else:
                        out.append(("AG", p, p, (psi * L.y_numel * B) // p))
                if i + 1 < len(self.L):
                    for cj in A[i + 1]:
                        for name, spec in self._inter_specs(i, ci, cj):
                            out.append(("A2A",) + spec)
        return out

    def _inter_specs(self, i: int, ci, cj):
        L = self.L[i]
        (ti, pi), (tj, pj) = ci, cj
        psi, B, N = self.psi, self.st.B, self.N
        nu, n, r = L.u_numel, L.n_out, L.rows_per_channel
        if (ti, tj) == (DP, MP):
            return [("A2A_fp", (("dpmp_fp", pi, pj, nu, B),) + Lt.dp_to_mp_fp(pi, pj, nu, B, psi)),
                    ("A2A_bp", (("dpmp_bp", pi, pj, nu, B),) + Lt.dp_to_mp_bp(pi, pj, nu, B, psi))]
        if (ti, tj) == (MP, DP):
            return [("A2A_bp", (("mpdp_bp", pi, pj, n, r, B),) + Lt.mp_to_dp_bp(pi, pj, n, r, B, psi))]
        if (ti, tj) == (DP, DP):
            return [("A2A_fp", (("dd_fp", pi, pj, nu, B),) + Lt.dd_fp(pi, pj, nu, B, psi)),
                    ("A2A_bp", (("dd_bp", pi, pj, nu, B),) + Lt.dd_bp(pi, pj, nu, B, psi))]
        return [("A2A_bp", (("mm_bp", pi, pj, n, r, B),) + Lt.mm_bp(pi, pj, n, r, B, psi, N))]

    def _combine(self, parts, base_e: np.ndarray, pareto: bool = True) -> list[TransAlt]:
        """parts: list of (name, alts, kind, mask) with kind in {'intra', 'inter'}."""
        combos = [TransAlt(0.0, 0.0, 0.0, base_e.copy(), ())]
        for name, alts, kind, mask in parts:
            new = []
            for cb in combos:
                for a in alts:
                    e = cb.e + a.mem(mask)
                    if kind == "intra":
                        new.append(TransAlt(cb.tr_intra + a.transport, cb.red_intra + a.reduction, cb.inter, e,
                                            cb.choices + ((name, a.cand),)))
                    else:
                        new.append(TransAlt(cb.tr_intra, cb.red_intra, cb.inter + a.latency, e,
                                            cb.choices + ((name, a.cand),)))
            combos = _pareto_trans(new) if pareto else new
        return combos

    def transition(self, i: int, ci, cj, pareto: bool = True) -> list[TransAlt]:
        key = (i, ci, cj, pareto)
        if key not in self._trans_cache:
            mask = self._assembled_mask(i, ci, cj)
            parts = [(n, a, "intra", mask if k == "AG" else None) for n, a, k in self._intra_ops(i, ci, cj[1])]
            parts += [(n, a, "inter", None) for n, a in self._inter_ops(i, ci, cj)]
            self._trans_cache[key] = self._combine(parts, self._u_storage(i, ci, cj), pareto)
        return self._trans_cache[key]

    def local_charge(self, i: int, c) -> list[TransAlt]:
        """Layer i's own charge: computation plus its intra-layer operations, with the
        MP output All-gather delivered to its own prefix (q = p_i). Independent of
        the next layer's configuration set (used by the greedy ablation)."""
        key = ("local", i, c)
        if key not in self._trans_cache:
            parts = [(n, a, "intra", None) for n, a, k in self._intra_ops(i, c, c[1])]
            self._trans_cache[key] = self._combine(parts, np.zeros(self.N), True)
        return self._trans_cache[key]

    def terminal(self, ci, pareto: bool = True) -> list[TransAlt]:
        i = len(self.L) - 1
        key = (i, ci, None, pareto)
        if key not in self._trans_cache:
            L = self.L[i]
            (t, p) = ci
            mask = np.ones(self.N)
            mask[:p] = 0.0                  # logits assembled into the producer's full y reservation
            parts = []
            if t == DP:
                parts.append(("AR_w", self.svc.allreduce(p, self.psi * self.param_numel(L)), "intra", None))
            else:
                S = self.psi * L.y_numel * self.st.B
                parts.append(("AG_y", self.svc.allgather(p, p, S // p), "intra", mask))
                if self.needs_input_grad(L):
                    parts.append(("AR_x", self.svc.allreduce(p, self.psi * L.x_numel * self.st.B), "intra", None))
            self._trans_cache[key] = self._combine(parts, np.zeros(self.N), pareto)
        return self._trans_cache[key]


def _pareto_trans(alts: list[TransAlt]) -> list[TransAlt]:
    """Nondominated combinations over (intra transport, remaining time, e vector).
    G is nondecreasing in both times for every alpha in [0,1], so this is exact."""
    if len(alts) <= 1:
        return alts
    k1 = np.array([a.tr_intra for a in alts])
    k2 = np.array([a.red_intra + a.inter for a in alts])
    E = np.stack([a.e for a in alts])
    order = np.lexsort((E.sum(axis=1), k2, k1))
    keep = []
    for i in order:
        dominated = False
        for j in keep:
            if k1[j] <= k1[i] + 1e-15 and k2[j] <= k2[i] + 1e-15 and np.all(E[j] <= E[i] + 1e-9):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return [alts[i] for i in keep]


__all__ = ["Settings", "ChainModel", "TransAlt", "DP", "MP"]
