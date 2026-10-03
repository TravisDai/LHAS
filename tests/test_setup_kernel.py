"""The numba setup-flag kernel equals the vectorized reference on many operations."""
import numpy as np
import pytest

from lhas.collectives.families import allgather_ops, wrht_ops, osm
from lhas.transport.schedule import assign_setup_flags_reference
from helpers import stressed


@pytest.mark.parametrize("pr", [dict(N=24, lam=2, w=2, reach=3, d_peer=3, kappa=1000),
                                dict(N=16, lam=1, w=1, reach=2, d_peer=2, kappa=700),
                                dict(N=20, lam=4, w=3, reach=8, d_peer=16, kappa=1 << 20)])
def test_setup_kernel_matches_reference(pr):
    tp = stressed(**pr)
    ops = []
    for p, q in [(12, 4), (7, 4), (4, 9), (6, 6), (3, 17), (10, 10)]:
        if max(p, q) <= tp.N:
            ops += allgather_ops(p, q, tp)
    for p in (5, 9, 12):
        ops += wrht_ops(list(range(1, p + 1)), tp)
    n_skip = 0
    for op in ops:
        ref = assign_setup_flags_reference(op.phases, tp)
        got = [ph.setup for ph in op.phases]
        for a, b in zip(ref, got):
            assert np.array_equal(a, b)
            n_skip += int((b == 0).sum())
    assert n_skip > 0           # reuse actually occurs in these instances
