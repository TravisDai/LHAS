"""Shared builders for tests."""
import copy
from dataclasses import replace

from lhas.params import load_approved
from lhas.transport.export import export_record, params_dict, spec_allgather, spec_allreduce, spec_alltoall

TP4, CP = load_approved(4)
BETA = CP.beta_red


def stressed(N=24, lam=2, w=2, reach=3, d_peer=3, kappa=1000):
    """Small resources that force several rounds, relays, setups and chunk tails."""
    return replace(load_approved(N)[0], lam=lam, w_tx=w, w_rx=w, reach=reach, d_peer=d_peer,
                   kappa_bytes=kappa)


def rec_and_params(op, payload, tp):
    return export_record(op, payload, tp, BETA), params_dict(tp, BETA)


def deep(x):
    return copy.deepcopy(x)
