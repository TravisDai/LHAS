"""Constructor output must pass the independent verifier on stressed instances.

Coverage: unequal counts (C-A both regimes, C-B, C-C, E-A, E-B, E-C),
nonconsecutive representatives, multi-round packing, relays, multi-chunk items,
short tails (V = kappa + 1), sub-chunk items, whole-item sensitivity (kappa=0),
unrestricted reach and no peer cap relaxations, and All-to-all with arbitrary
nonzero matrices. Claimed relay reservations must equal the manuscript bound.
"""
import numpy as np
import pytest

from lhas.collectives.families import allgather_ops, wrht_ops, osm, optree_ops
from lhas.transport.export import spec_allgather, spec_allreduce, spec_alltoall
from lhas_verify import verify
from helpers import stressed, rec_and_params

PARAMS = [
    dict(N=24, lam=2, w=2, reach=3, d_peer=3, kappa=1000),
    dict(N=16, lam=1, w=1, reach=2, d_peer=2, kappa=700),
    dict(N=20, lam=4, w=3, reach=8, d_peer=16, kappa=1 << 20),
    dict(N=18, lam=3, w=2, reach=0, d_peer=0, kappa=500),     # idealized relaxations
    dict(N=24, lam=2, w=2, reach=3, d_peer=3, kappa=0),       # whole-item sensitivity
]
AG_CASES = [(12, 4), (10, 4), (8, 4), (7, 4), (5, 4), (6, 5), (4, 9), (3, 17), (5, 12), (2, 16), (6, 6), (12, 12)]
SIZES = [1, 699, 700, 701, 1000, 1001, 2500]


def _check(op, payload, tp, spec):
    rec, prm = rec_and_params(op, payload, tp)
    out = verify(rec, spec, prm)
    # implementation consistency: claimed relay reservation equals the manuscript bound
    claimed = {int(k): v for k, v in rec["claimed"]["relay_peak"].items()}
    assert claimed == pytest.approx(out["relay_reservation"])
    return rec, out


@pytest.mark.parametrize("pr", PARAMS, ids=lambda d: "N{N}l{lam}w{w}L{reach}d{d_peer}k{kappa}".format(**d))
@pytest.mark.parametrize("pq", AG_CASES, ids=lambda t: f"{t[0]}to{t[1]}")
def test_allgather_candidates(pr, pq):
    tp = stressed(**pr)
    p, q = pq
    if max(p, q) > tp.N:
        pytest.skip("participants exceed N")
    for op in allgather_ops(p, q, tp):
        for V in SIZES:
            _check(op, V, tp, spec_allgather(range(1, p + 1), range(1, q + 1), V))


@pytest.mark.parametrize("pr", PARAMS, ids=lambda d: "N{N}l{lam}w{w}L{reach}d{d_peer}k{kappa}".format(**d))
@pytest.mark.parametrize("p", [2, 3, 5, 7, 9, 12, 16])
def test_wrht_candidates(pr, p):
    tp = stressed(**pr)
    if p > tp.N:
        pytest.skip()
    for op in wrht_ops(list(range(1, p + 1)), tp):
        for V in (1, 700, 1001, 2500):
            _check(op, V, tp, spec_allreduce(range(1, p + 1), V))


@pytest.mark.parametrize("pr", PARAMS, ids=lambda d: "N{N}l{lam}w{w}L{reach}d{d_peer}k{kappa}".format(**d))
def test_alltoall_random(pr):
    tp = stressed(**pr)
    rng = np.random.default_rng(7)
    for _ in range(15):
        k = int(rng.integers(3, 70))
        pairs = {}
        for s, d in zip(rng.integers(1, tp.N + 1, k), rng.integers(1, tp.N + 1, k)):
            pairs[(int(s), int(d))] = int(rng.choice([1, 699, 700, 701, 1500, 3000]))
        S = [a for a, _ in pairs]; D = [b for _, b in pairs]; B = [pairs[x] for x in pairs]
        op = osm(S, D, tp)
        _check(op, np.array(B), tp, spec_alltoall(S, D, B))


def test_nonconsecutive_representatives():
    """OpTree over physical representatives [1,4,7,10] with actual holdings."""
    tp = stressed(N=24, lam=2, w=2, reach=3, d_peer=3, kappa=1000)
    reps = [1, 4, 7, 10]
    hold = {1: {1, 2, 3}, 4: {4, 5, 6}, 7: {7, 8, 9}, 10: {10, 11, 12}}
    for op, final in optree_ops(reps, tp, hold=hold, tag="rep"):
        for V in (1, 1001, 2500):
            spec = dict(P=list(range(1, 13)), Q=reps, initial={u: sorted(h) for u, h in hold.items()},
                        required={r: list(range(1, 13)) for r in reps},
                        shard_bytes={u: V for u in range(1, 13)})
            _check(op, V, tp, spec)
        assert all(final[r] == set(range(1, 13)) for r in reps)
