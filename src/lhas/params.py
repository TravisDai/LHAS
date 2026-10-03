"""Model parameters.

Every numerical input is loaded from `configs/approved_reference_parameters.json`
(author-approved analytical reference, 2026-09-28) unless an experiment
configuration overrides it explicitly. Nothing here is a measured hardware value.
Manuscript: Section 2.1 (sec:optical_model), 2.2 (sec:reference_compute),
2.4.1 (sec:transport).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APPROVED = REPO / "configs" / "approved_reference_parameters.json"


@dataclass(frozen=True)
class TransportParams:
    """Physical transport model (Section 2.1 and Eq. bg_generic_collective).

    reach = 0 means unrestricted direct reach (idealized reference only).
    d_peer = 0 means no peer-union cap (separately labeled relaxation).
    kappa_bytes = 0 means whole-item store-and-forward (sensitivity only).
    """
    N: int
    lam: int = 64                 # labels per directed ring
    w_tx: int = 64                # transmit channels per node (both directions)
    w_rx: int = 64                # receive channels per node (both directions)
    reach: int = 8                # directed fibre segments per optical hop
    d_peer: int = 16              # union of in/out direct peers per node and round
    seg_len_m: float = 1.0
    v_g: float = 2.0e8
    rate_bps: float = 25.0e9
    eps: float = 1.0              # effective bandwidth fraction
    t_setup: float = 25.0e-6
    t_tx: float = 10.0e-9
    t_rx: float = 10.0e-9
    t_fwd: float = 0.0            # additional relay handling (abstraction)
    kappa_bytes: int = 1 << 20    # max chunk size in pipelined rounds

    @property
    def b_eff(self) -> float:
        return self.eps * self.rate_bps

    @property
    def seg_time(self) -> float:
        return self.seg_len_m / self.v_g

    @property
    def eff_reach(self) -> int:
        return self.reach if self.reach > 0 else max(1, self.N)

    def structural_key(self) -> tuple:
        """Fields that change routes, packing or circuit reuse (not payload timing)."""
        return (self.N, self.lam, self.w_tx, self.w_rx, self.eff_reach, self.d_peer)

    def validate(self) -> None:
        assert self.N >= 1 and self.lam >= 1 and self.w_tx >= 1 and self.w_rx >= 1
        assert self.reach >= 0 and self.d_peer >= 0
        assert 0 < self.eps <= 1 and self.rate_bps > 0
        assert self.kappa_bytes >= 0
        if self.d_peer:
            assert self.d_peer >= 2, "a relay needs one incoming and one outgoing peer"


@dataclass(frozen=True)
class ComputeParams:
    """Hypothetical P100 PCIe 16 GB analytical reference (Section 2.2)."""
    peak_flops: float = 9.3e12
    utilization: float = 0.5
    peak_mem_bw: float = 732e9
    reduction_efficiency: float = 0.5
    psi: int = 4
    budget_bytes: int = 12 * 2 ** 30
    runtime_reserve_bytes: int = 0     # must fit inside budget_bytes
    input_mode: str = "analytic_reference"

    @property
    def beta_red(self) -> float:
        return self.reduction_efficiency * self.peak_mem_bw

    def validate(self) -> None:
        assert 0 < self.utilization <= 1, "rho=0 has no finite execution time"
        assert 0 < self.reduction_efficiency <= 1
        assert 0 <= self.runtime_reserve_bytes <= self.budget_bytes


def load_approved(N: int, path: Path = APPROVED) -> tuple[TransportParams, ComputeParams]:
    cfg = json.loads(Path(path).read_text())
    o, c = cfg["optical"], cfg["compute"]
    assert o["per_label_resources_shared_across_directions"] is True
    assert o["concurrent_tx_rx"] is True and o["optical_multicast"] is False
    assert o["direct_only_rounds_send_whole_items"] is True
    tp = TransportParams(
        N=N, lam=o["lambda_per_direction"], w_tx=o["tx_channels_per_node_total"],
        w_rx=o["rx_channels_per_node_total"], reach=o["reach_segments"],
        d_peer=o["peer_union_limit_per_round"], seg_len_m=o["segment_length_m"],
        v_g=o["propagation_speed_m_s"], rate_bps=o["wavelength_rate_bit_s"],
        eps=o["effective_bandwidth_fraction"], t_setup=o["setup_s"],
        t_tx=o["endpoint_tx_s"], t_rx=o["endpoint_rx_s"], t_fwd=o["extra_relay_handling_s"],
        kappa_bytes=o["pipelined_round_max_chunk_bytes"])
    cp = ComputeParams(
        peak_flops=c["peak_fp32_flop_s"], utilization=c["utilization_assumed"],
        peak_mem_bw=c["peak_memory_bandwidth_byte_s"],
        reduction_efficiency=c["reduction_bandwidth_efficiency_assumed"],
        psi=c["bytes_per_element"], budget_bytes=c["modeled_total_per_node_budget_bytes"],
        input_mode=c["input_mode"])
    assert abs(cp.beta_red - c["effective_reduction_bandwidth_byte_s"]) < 1.0
    tp.validate(); cp.validate()
    return tp, cp


__all__ = ["TransportParams", "ComputeParams", "load_approved", "replace", "asdict"]
