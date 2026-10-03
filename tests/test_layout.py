"""Element-level ownership checks of the boundary matrices (sec:inter-layer).

Each matrix is checked against an explicit feature-by-batch element grid: every
destination must end up with exactly its required elements (local holdings plus
received blocks), with no element received twice, and sources may only send
elements they hold. Closed forms S/(p_i p_j) and S/p_i are checked when exact.
"""
import itertools

import numpy as np
import pytest

from lhas.model import layout as Lt

PSI = 4


def grid_dp(p, nu, B):
    """node -> set of (row, col) under DP with p nodes: all rows, batch interval."""
    return {u: {(r, c) for r in range(nu) for c in range(*Lt.batch_interval(u, p, B))} for u in range(1, p + 1)}


def grid_mp(p, n, rpc, B):
    return {u: {(r, c) for r in range(*Lt.row_block_mp(u, p, n, rpc)) for c in range(B)} for u in range(1, p + 1)}


def grid_full(p, nu, B):
    return {u: {(r, c) for r in range(nu) for c in range(B)} for u in range(1, p + 1)}


def check(matrix, have, need, sender_blocks):
    """sender_blocks(u, v) -> set of elements u sends to v under the stated rule."""
    s, d, b = matrix
    recv = {v: set() for v in need}
    for u, v, nb in zip(s, d, b):
        blk = sender_blocks(int(u), int(v))
        assert blk <= have[int(u)], "source sends elements it does not hold"
        assert nb == PSI * len(blk)
        assert not (recv[int(v)] & blk), "element received twice"
        recv[int(v)] |= blk
    for v, req in need.items():
        got = recv[v] | (have.get(v, set()) & req)
        assert req <= got, f"node {v} misses elements"
        assert recv[v] <= req, f"node {v} receives unrequired elements"


CASES = [(p, q) for p, q in itertools.product([1, 2, 3, 4, 6, 8], repeat=2)]


@pytest.mark.parametrize("p,q", CASES)
def test_dp_to_mp_fp(p, q):
    nu, B = 5, 24
    have, need = grid_dp(p, nu, B), grid_full(q, nu, B)
    check(Lt.dp_to_mp_fp(p, q, nu, B, PSI), have, need, lambda u, v: have[u])
    if q > 1 or p > 1:
        s, d, b = Lt.dp_to_mp_fp(p, q, nu, B, PSI)
        assert np.all(b == PSI * nu * B // p)          # S_i / p_i per pair


@pytest.mark.parametrize("p,q", CASES)
def test_dp_to_mp_bp(p, q):
    nu, B = 12, 24
    have = grid_full(q, nu, B)                          # after the consumer's All-reduce
    need = grid_dp(p, nu, B)
    def blocks(b, v):
        f = Lt.forwarding_rows(b, q, nu); I = Lt.batch_interval(v, p, B)
        return {(r, c) for r in range(*f) for c in range(*I)}
    s, d, bb = Lt.dp_to_mp_bp(p, q, nu, B, PSI)
    # rows F_{i,b} partition the rows, so each destination receives every row exactly once
    check((s, d, bb), have, {v: need[v] - {(r, c) for r in range(*Lt.forwarding_rows(v, q, nu)) for c in range(B)}
                              if v <= q else need[v] for v in need}, blocks)
    if nu % q == 0:
        assert np.all(bb == PSI * nu * B // (p * q))


@pytest.mark.parametrize("p,q", CASES)
def test_mp_to_dp_bp(p, q):
    n, rpc, B = 24, 2, 24
    nu = n * rpc
    have = grid_dp(q, nu, B)                             # DP consumer gradients
    need = grid_mp(p, n, rpc, B)
    def blocks(u, v):
        r = Lt.row_block_mp(v, p, n, rpc); I = Lt.batch_interval(u, q, B)
        return {(x, c) for x in range(*r) for c in range(*I)}
    s, d, b = Lt.mp_to_dp_bp(p, q, n, rpc, B, PSI)
    check((s, d, b), have, need, blocks)
    assert np.all(b == PSI * nu * B // (p * q))


@pytest.mark.parametrize("p,q", CASES)
def test_dd(p, q):
    nu, B = 3, 24
    have, need = grid_dp(p, nu, B), grid_dp(q, nu, B)
    check(Lt.dd_fp(p, q, nu, B, PSI), have, need, lambda u, v: have[u] & need[v])
    have2, need2 = grid_dp(q, nu, B), grid_dp(p, nu, B)
    check(Lt.dd_bp(p, q, nu, B, PSI), have2, need2, lambda u, v: have2[u] & need2[v])
    if p == q:
        assert Lt.dd_fp(p, q, nu, B, PSI)[0].size == 0   # identical counts and mapping


@pytest.mark.parametrize("p,q", CASES)
def test_mm(p, q):
    n, rpc, B, N = 24, 3, 8, 16
    have = grid_full(q, n * rpc, B)
    need = {v: grid_mp(p, n, rpc, B)[v] for v in range(1, p + 1)}
    def blocks(u, v):
        return grid_mp(p, n, rpc, B)[v]
    s, d, b = Lt.mm_bp(p, q, n, rpc, B, PSI, N)
    check((s, d, b), have, need, blocks)
    assert set(d.tolist()) == set(range(q + 1, p + 1))


def test_mm_nearest_rule_wraps():
    # N=16, q=2: node 16 is nearest to node 1 (distance 1), node 3 to node 2
    s, d, b = Lt.mm_bp(16, 2, 16, 1, 4, PSI, 16)
    src = dict(zip(d.tolist(), s.tolist()))
    assert src[3] == 2 and src[16] == 1 and src[9] == 2   # distance 8 to node 1, 7 to node 2
