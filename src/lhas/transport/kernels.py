"""Numba kernels for the approved transport model (Section 2.4.1, sec:transport).

Nodes are 0-based inside kernels; public APIs use the manuscript's 1-based IDs.
A route is (src, direction, H hops, last-hop segment count). With fixed-stride
relays every hop except the last spans exactly `L` segments.
Direction index: 0 = clockwise (+1), 1 = counter-clockwise (-1).
"""
from __future__ import annotations

import numpy as np
from numba import njit

WORD = 64


@njit(cache=True, inline="always")
def _lowest_free(used_or, words, lam):
    # used_or: array[words] of uint64; return lowest free label index (0-based) or -1
    for w in range(words):
        free = ~used_or[w]
        if w == words - 1:
            rem = lam - 64 * w
            if rem < 64:
                free &= (np.uint64(1) << np.uint64(rem)) - np.uint64(1)
        if free != np.uint64(0):
            # index of lowest set bit
            b = 0
            x = free
            while (x & np.uint64(1)) == np.uint64(0):
                x >>= np.uint64(1)
                b += 1
            return w * 64 + b
    return -1


@njit(cache=True, inline="always")
def _fw_add(fw, di, i, v, N):
    j = i + 1
    while j <= N:
        fw[di, j] += v
        j += j & (-j)


@njit(cache=True, inline="always")
def _fw_prefix(fw, di, i):
    # sum of indicator over [0, i)
    s = 0
    j = i
    while j > 0:
        s += fw[di, j]
        j -= j & (-j)
    return s


@njit(cache=True, inline="always")
def _fw_range(fw, di, lo, length, N):
    # circular range [lo, lo+length) with 0 < length <= N
    hi = lo + length
    if hi <= N:
        return _fw_prefix(fw, di, hi) - _fw_prefix(fw, di, lo)
    return (_fw_prefix(fw, di, N) - _fw_prefix(fw, di, lo)) + _fw_prefix(fw, di, hi - N)


@njit(cache=True)
def pack_first_fit(order, src0, dirn, H, last, L, N, lam, wtx, wrx, dpeer,
                   hop_start, out_round, out_label):
    """Deterministic first-fit rounds (manuscript sec:transport).

    `order` is the priority order (already sorted by decreasing hop count,
    decreasing total segments, increasing src, dst, item). Every hop of an
    admitted route gets the lowest label that is free on all its directed
    segments, free for Tx at its sender and Rx at its receiver (labels are shared
    across directions), within aggregate budgets and within the peer-union cap.
    A route is admitted only if all hops succeed; otherwise partial
    reservations are undone and it is deferred to the next round.

    Acceleration (exactness-preserving): before trying a route, reject it if any
    directed segment on its arc already carries all `lam` labels (Fenwick tree
    over saturated segments), or if any hop endpoint has exhausted its aggregate
    Tx/Rx budget. Either condition makes the route fail under the rule above, so
    the resulting rounds and labels are identical to `pack_first_fit_plain`.
    Returns the number of rounds, or -1 if a round admitted nothing.
    """
    words = (lam + 63) // 64
    n = order.shape[0]
    pending = order.copy()
    newp = np.empty_like(pending)
    npend = n
    seg = np.zeros((2, N, words), np.uint64)
    segc = np.zeros((2, N), np.int32)
    fw = np.zeros((2, N + 1), np.int32)
    txm = np.zeros((N, words), np.uint64)
    rxm = np.zeros((N, words), np.uint64)
    txc = np.zeros(N, np.int32)
    rxc = np.zeros(N, np.int32)
    pcap = dpeer if dpeer > 0 else 1
    peers = np.full((N, pcap), -1, np.int32)
    pc = np.zeros(N, np.int32)
    maxh = 1
    for r in range(n):
        if H[r] > maxh:
            maxh = H[r]
    t_lab = np.empty(maxh, np.int32)
    t_pa = np.zeros(maxh, np.uint8)
    t_pb = np.zeros(maxh, np.uint8)
    used = np.empty(words, np.uint64)
    rnd = 0
    while npend > 0:
        seg[:] = 0
        segc[:] = 0
        fw[:] = 0
        txm[:] = 0
        rxm[:] = 0
        txc[:] = 0
        rxc[:] = 0
        pc[:] = 0
        nfull = 0
        nnew = 0
        placed = 0
        for idx in range(npend):
            r = pending[idx]
            s = src0[r]
            d = dirn[r]
            nh = H[r]
            di = 0 if d > 0 else 1
            Dtot = (nh - 1) * L + last[r]
            dst_last = (s + d * Dtot) % N
            if txc[s] >= wtx or rxc[dst_last] >= wrx:
                newp[nnew] = r
                nnew += 1
                continue
            if nfull > 0:
                lo = s if d > 0 else (s - Dtot + 1) % N
                if _fw_range(fw, di, lo, Dtot, N) > 0:
                    newp[nnew] = r
                    nnew += 1
                    continue
            sat = False
            for k in range(1, nh):
                a = (s + d * k * L) % N
                if txc[a] >= wtx or rxc[a] >= wrx:
                    sat = True
                    break
            if sat:
                newp[nnew] = r
                nnew += 1
                continue
            ok = True
            k = 0
            while k < nh:
                a = (s + d * k * L) % N
                ns = L if k < nh - 1 else last[r]
                b = (a + d * ns) % N
                if txc[a] >= wtx or rxc[b] >= wrx:
                    ok = False
                    break
                ha = False
                hb = False
                if dpeer > 0:
                    for q in range(pc[a]):
                        if peers[a, q] == b:
                            ha = True
                            break
                    for q in range(pc[b]):
                        if peers[b, q] == a:
                            hb = True
                            break
                    if (not ha and pc[a] >= dpeer) or (not hb and pc[b] >= dpeer):
                        ok = False
                        break
                for w in range(words):
                    used[w] = txm[a, w] | rxm[b, w]
                for t in range(ns):
                    sg = (a + d * t) % N
                    for w in range(words):
                        used[w] |= seg[di, sg, w]
                lab = _lowest_free(used, words, lam)
                if lab < 0:
                    ok = False
                    break
                w0 = lab // 64
                bit = np.uint64(1) << np.uint64(lab - 64 * w0)
                for t in range(ns):
                    sg = (a + d * t) % N
                    seg[di, sg, w0] |= bit
                    segc[di, sg] += 1
                    if segc[di, sg] == lam:
                        _fw_add(fw, di, sg, 1, N)
                        nfull += 1
                txm[a, w0] |= bit
                rxm[b, w0] |= bit
                txc[a] += 1
                rxc[b] += 1
                t_pa[k] = 0
                t_pb[k] = 0
                if dpeer > 0:
                    if not ha:
                        peers[a, pc[a]] = b
                        pc[a] += 1
                        t_pa[k] = 1
                    if not hb:
                        peers[b, pc[b]] = a
                        pc[b] += 1
                        t_pb[k] = 1
                t_lab[k] = lab
                k += 1
            if ok:
                out_round[r] = rnd
                for kk in range(nh):
                    out_label[hop_start[r] + kk] = t_lab[kk] + 1
                placed += 1
            else:
                for kk in range(k - 1, -1, -1):
                    a = (s + d * kk * L) % N
                    ns = L if kk < nh - 1 else last[r]
                    b = (a + d * ns) % N
                    lab = t_lab[kk]
                    w0 = lab // 64
                    bit = np.uint64(1) << np.uint64(lab - 64 * w0)
                    for t in range(ns):
                        sg = (a + d * t) % N
                        seg[di, sg, w0] &= ~bit
                        if segc[di, sg] == lam:
                            _fw_add(fw, di, sg, -1, N)
                            nfull -= 1
                        segc[di, sg] -= 1
                    txm[a, w0] &= ~bit
                    rxm[b, w0] &= ~bit
                    txc[a] -= 1
                    rxc[b] -= 1
                    if dpeer > 0:
                        if t_pb[kk] == 1:
                            pc[b] -= 1
                            peers[b, pc[b]] = -1
                        if t_pa[kk] == 1:
                            pc[a] -= 1
                            peers[a, pc[a]] = -1
                newp[nnew] = r
                nnew += 1
        if placed == 0:
            return -1
        tmp = pending
        pending = newp
        newp = tmp
        npend = nnew
        rnd += 1
    return rnd


@njit(cache=True)
def pack_first_fit_plain(order, src0, dirn, H, last, L, N, lam, wtx, wrx, dpeer,
                   hop_start, out_round, out_label):
    """Unaccelerated reference version of `pack_first_fit` (kept for equivalence tests).

    Deterministic first-fit rounds (manuscript sec:transport).

    `order` is the priority order (already sorted by decreasing hop count,
    decreasing total segments, increasing src, dst, item). Every hop of an
    admitted route gets the lowest label that is free on all its directed
    segments, free for Tx at its sender and Rx at its receiver (labels are shared
    across directions), within aggregate budgets and within the peer-union cap.
    A route is admitted only if all hops succeed; otherwise partial
    reservations are undone and it is deferred to the next round.
    Returns the number of rounds, or -1 if a round admitted nothing (should not
    happen with positive capacities).
    """
    words = (lam + 63) // 64
    n = order.shape[0]
    pending = order.copy()
    newp = np.empty_like(pending)
    npend = n
    seg = np.zeros((2, N, words), np.uint64)
    txm = np.zeros((N, words), np.uint64)
    rxm = np.zeros((N, words), np.uint64)
    txc = np.zeros(N, np.int32)
    rxc = np.zeros(N, np.int32)
    pcap = dpeer if dpeer > 0 else 1
    peers = np.full((N, pcap), -1, np.int32)
    pc = np.zeros(N, np.int32)
    maxh = 1
    for r in range(n):
        if H[r] > maxh:
            maxh = H[r]
    t_lab = np.empty(maxh, np.int32)
    t_pa = np.zeros(maxh, np.uint8)
    t_pb = np.zeros(maxh, np.uint8)
    used = np.empty(words, np.uint64)
    rnd = 0
    while npend > 0:
        seg[:] = 0
        txm[:] = 0
        rxm[:] = 0
        txc[:] = 0
        rxc[:] = 0
        pc[:] = 0
        nnew = 0
        placed = 0
        for idx in range(npend):
            r = pending[idx]
            s = src0[r]
            d = dirn[r]
            nh = H[r]
            di = 0 if d > 0 else 1
            # quick exact rejections: first sender / last receiver saturated
            dst_last = (s + d * ((nh - 1) * L + last[r])) % N
            if txc[s] >= wtx or rxc[dst_last] >= wrx:
                newp[nnew] = r
                nnew += 1
                continue
            ok = True
            k = 0
            while k < nh:
                a = (s + d * k * L) % N
                ns = L if k < nh - 1 else last[r]
                b = (a + d * ns) % N
                if txc[a] >= wtx or rxc[b] >= wrx:
                    ok = False
                    break
                # peer-union cap
                ha = False
                hb = False
                if dpeer > 0:
                    for q in range(pc[a]):
                        if peers[a, q] == b:
                            ha = True
                            break
                    for q in range(pc[b]):
                        if peers[b, q] == a:
                            hb = True
                            break
                    if (not ha and pc[a] >= dpeer) or (not hb and pc[b] >= dpeer):
                        ok = False
                        break
                for w in range(words):
                    used[w] = txm[a, w] | rxm[b, w]
                for t in range(ns):
                    sg = (a + d * t) % N
                    for w in range(words):
                        used[w] |= seg[di, sg, w]
                lab = _lowest_free(used, words, lam)
                if lab < 0:
                    ok = False
                    break
                w0 = lab // 64
                bit = np.uint64(1) << np.uint64(lab - 64 * w0)
                for t in range(ns):
                    sg = (a + d * t) % N
                    seg[di, sg, w0] |= bit
                txm[a, w0] |= bit
                rxm[b, w0] |= bit
                txc[a] += 1
                rxc[b] += 1
                t_pa[k] = 0
                t_pb[k] = 0
                if dpeer > 0:
                    if not ha:
                        peers[a, pc[a]] = b
                        pc[a] += 1
                        t_pa[k] = 1
                    if not hb:
                        peers[b, pc[b]] = a
                        pc[b] += 1
                        t_pb[k] = 1
                t_lab[k] = lab
                k += 1
            if ok:
                out_round[r] = rnd
                for kk in range(nh):
                    out_label[hop_start[r] + kk] = t_lab[kk] + 1
                placed += 1
            else:
                # undo hops 0..k-1 in reverse order (stack discipline for peers)
                for kk in range(k - 1, -1, -1):
                    a = (s + d * kk * L) % N
                    ns = L if kk < nh - 1 else last[r]
                    b = (a + d * ns) % N
                    lab = t_lab[kk]
                    w0 = lab // 64
                    bit = np.uint64(1) << np.uint64(lab - 64 * w0)
                    for t in range(ns):
                        sg = (a + d * t) % N
                        seg[di, sg, w0] &= ~bit
                    txm[a, w0] &= ~bit
                    rxm[b, w0] &= ~bit
                    txc[a] -= 1
                    rxc[b] -= 1
                    if dpeer > 0:
                        if t_pb[kk] == 1:
                            pc[b] -= 1
                            peers[b, pc[b]] = -1
                        if t_pa[kk] == 1:
                            pc[a] -= 1
                            peers[a, pc[a]] = -1
                newp[nnew] = r
                nnew += 1
        if placed == 0:
            return -1
        tmp = pending
        pending = newp
        newp = tmp
        npend = nnew
        rnd += 1
    return rnd


@njit(cache=True)
def pack_dissemination_round(targets, need, hold, N, L, lam, wtx, wrx, dpeer,
                             out_src, out_dst, out_item, out_label):
    """One E-A dissemination round (manuscript sec:group_comm, Expansion).

    targets: ordered target node indices (0-based). need[t, k] = 1 if target
    targets[t] still misses shard k. hold[v, k] = 1 if node v holds shard k at the
    start of the round. For each target and missing shard (ascending), direct
    senders holding the shard are tried by increasing circular distance then
    identifier; the first feasible lowest-label assignment is accepted.
    Returns the number of transfers placed (all single-hop, direct-only round).
    """
    words = (lam + 63) // 64
    seg = np.zeros((2, N, words), np.uint64)
    txm = np.zeros((N, words), np.uint64)
    rxm = np.zeros((N, words), np.uint64)
    txc = np.zeros(N, np.int32)
    rxc = np.zeros(N, np.int32)
    pcap = dpeer if dpeer > 0 else 1
    peers = np.full((N, pcap), -1, np.int32)
    pc = np.zeros(N, np.int32)
    used = np.empty(words, np.uint64)
    nshard = need.shape[1]
    # candidate sender offsets ordered by circular distance, then identifier
    # (identifier order is resolved per target below)
    nplaced = 0
    cand = np.empty(2 * L, np.int32)
    cdist = np.empty(2 * L, np.int32)
    for ti in range(targets.shape[0]):
        v = targets[ti]
        if rxc[v] >= wrx:
            continue
        # build candidate list: nodes within reach in either direction
        nc = 0
        for dd in range(1, L + 1):
            for sgn in (1, -1):
                u = (v + sgn * dd) % N
                dup = False
                for q in range(nc):
                    if cand[q] == u:
                        dup = True
                        break
                if not dup and u != v:
                    cand[nc] = u
                    cdist[nc] = min((v - u) % N, (u - v) % N)
                    nc += 1
        # sort by (distance, id)
        for i in range(1, nc):
            j = i
            while j > 0 and (cdist[j - 1] > cdist[j] or (cdist[j - 1] == cdist[j] and cand[j - 1] > cand[j])):
                tc = cand[j]; cand[j] = cand[j - 1]; cand[j - 1] = tc
                td = cdist[j]; cdist[j] = cdist[j - 1]; cdist[j - 1] = td
                j -= 1
        for k in range(nshard):
            if need[ti, k] == 0:
                continue
            if rxc[v] >= wrx:
                break
            for ci in range(nc):
                u = cand[ci]
                if hold[u, k] == 0:
                    continue
                if txc[u] >= wtx:
                    continue
                # shortest-arc direction from u to v, clockwise on ties
                cw = (v - u) % N
                ccw = (u - v) % N
                if cw <= ccw:
                    d = 1
                    ns = cw
                else:
                    d = -1
                    ns = ccw
                if ns > L:
                    continue
                di = 0 if d > 0 else 1
                ha = False
                hb = False
                if dpeer > 0:
                    for q in range(pc[u]):
                        if peers[u, q] == v:
                            ha = True
                            break
                    for q in range(pc[v]):
                        if peers[v, q] == u:
                            hb = True
                            break
                    if (not ha and pc[u] >= dpeer) or (not hb and pc[v] >= dpeer):
                        continue
                for w in range(words):
                    used[w] = txm[u, w] | rxm[v, w]
                for t in range(ns):
                    sg = (u + d * t) % N
                    for w in range(words):
                        used[w] |= seg[di, sg, w]
                lab = _lowest_free(used, words, lam)
                if lab < 0:
                    continue
                w0 = lab // 64
                bit = np.uint64(1) << np.uint64(lab - 64 * w0)
                for t in range(ns):
                    sg = (u + d * t) % N
                    seg[di, sg, w0] |= bit
                txm[u, w0] |= bit
                rxm[v, w0] |= bit
                txc[u] += 1
                rxc[v] += 1
                if dpeer > 0:
                    if not ha:
                        peers[u, pc[u]] = v
                        pc[u] += 1
                    if not hb:
                        peers[v, pc[v]] = u
                        pc[v] += 1
                out_src[nplaced] = u
                out_dst[nplaced] = v
                out_item[nplaced] = k
                out_label[nplaced] = lab + 1
                nplaced += 1
                break
    return nplaced


@njit(cache=True)
def round_body_times(order_by_round, round_start, nrounds, H, last, nbytes, L,
                     kappa, pipelined, rate, t_tx, t_rx, t_fwd, seg_time,
                     out_steps, out_body):
    """Per-round step counts and step-duration sums, excluding setup.

    Direct-only round: one step, each item whole.
    Pipelined round: every item is split into c = ceil(V/kappa) chunks of
    min(kappa, V-(j-1)kappa) bytes (tail not rounded up); hop a carries chunk j at
    step a+j-1. Step duration is the max over active transmissions of
    8*bytes/rate + t_tx + nseg*seg_time + t_rx + [a>1]*t_fwd.
    kappa == 0 means whole-item store-and-forward (c = 1).
    """
    maxkey = 2 * (L + 1)
    for rr in range(nrounds):
        a0 = round_start[rr]
        a1 = round_start[rr + 1]
        if a1 == a0:
            out_steps[rr] = 0
            out_body[rr] = 0.0
            continue
        if pipelined[rr] == 0:
            best = 0.0
            for idx in range(a0, a1):
                r = order_by_round[idx]
                v = 8.0 * nbytes[r] / rate + t_tx + t_rx + last[r] * seg_time
                if v > best:
                    best = v
            out_steps[rr] = 1
            out_body[rr] = best
            continue
        S = 0
        for idx in range(a0, a1):
            r = order_by_round[idx]
            c = 1
            if kappa > 0:
                c = (nbytes[r] + kappa - 1) // kappa
            s_r = H[r] + c - 1
            if s_r > S:
                S = s_r
        cover = np.zeros((maxkey, S + 1), np.int64)
        point = np.zeros(S, np.float64)
        fullval = np.zeros(maxkey, np.float64)
        keyused = np.zeros(maxkey, np.uint8)
        for idx in range(a0, a1):
            r = order_by_round[idx]
            V = nbytes[r]
            # Full chunks (exactly kappa bytes) share one duration per key and are
            # tracked with cover arrays; any other chunk (a short tail, or a whole
            # item when kappa == 0) is a point maximum at its own step.
            if kappa > 0:
                c = (V + kappa - 1) // kappa
                chunk = kappa
                tail = V - (c - 1) * kappa
                has_tail = tail != kappa
                nfull = c - 1 if has_tail else c
            else:
                c = 1
                chunk = 0
                tail = V
                has_tail = True
                nfull = 0
            for a in range(1, H[r] + 1):
                ns = L if a < H[r] else last[r]
                f = 1 if a > 1 else 0
                if nfull > 0:
                    key = ns * 2 + f
                    keyused[key] = 1
                    fullval[key] = 8.0 * chunk / rate + t_tx + t_rx + ns * seg_time + f * t_fwd
                    cover[key, a - 1] += 1
                    cover[key, a - 1 + nfull] -= 1
                if has_tail:
                    t0 = a - 1 + c - 1
                    val = 8.0 * tail / rate + t_tx + t_rx + ns * seg_time + f * t_fwd
                    if val > point[t0]:
                        point[t0] = val
        run = np.zeros(maxkey, np.int64)
        tot = 0.0
        for t in range(S):
            best = point[t]
            for key in range(maxkey):
                if keyused[key] == 1:
                    run[key] += cover[key, t]
                    if run[key] > 0 and fullval[key] > best:
                        best = fullval[key]
            tot += best
        out_steps[rr] = S
        out_body[rr] = tot


@njit(cache=True)
def relay_peaks(order_by_round, round_start, nrounds, src0, dirn, H, nbytes, L, N,
                kappa, pipelined, out_peak):
    """Peak relay reservation per physical node: 2*min(kappa, V) per concurrently
    reserved relayed route, summed per round and maximized over rounds."""
    cur = np.zeros(N, np.float64)
    touched = np.empty(N, np.int32)
    for rr in range(nrounds):
        if pipelined[rr] == 0:
            continue
        nt = 0
        for idx in range(round_start[rr], round_start[rr + 1]):
            r = order_by_round[idx]
            if H[r] < 2:
                continue
            V = nbytes[r]
            buf = 2.0 * (min(kappa, V) if kappa > 0 else V)
            for k in range(1, H[r]):
                x = (src0[r] + dirn[r] * k * L) % N
                if cur[x] == 0.0:
                    touched[nt] = x
                    nt += 1
                cur[x] += buf
        for q in range(nt):
            x = touched[q]
            if cur[x] > out_peak[x]:
                out_peak[x] = cur[x]
            cur[x] = 0.0


@njit(cache=True)
def _circuit_key(a, d, ns, lab, L, lam):
    dirbit = 0 if d > 0 else 1
    return ((a * 2 + dirbit) * (L + 1) + ns) * (lam + 1) + lab


@njit(cache=True)
def setup_flags_phase(order_by_round, round_start, nrounds, src0, dirn, H, last, labels, hop_start,
                      L, N, lam, retained):
    """Setup flags of one phase given the retained configuration (sorted unique
    circuit keys) left by earlier phases of the same operation. A round skips
    setup only if its circuit set is a subset of the retained set; otherwise it
    is configured and becomes the retained set. Memory is O(hops per round).
    Returns (flags, retained)."""
    flags = np.ones(nrounds, np.uint8)
    for rr in range(nrounds):
        a0 = round_start[rr]
        a1 = round_start[rr + 1]
        nh = 0
        for idx in range(a0, a1):
            nh += H[order_by_round[idx]]
        keys = np.empty(nh, np.int64)
        t = 0
        for idx in range(a0, a1):
            r = order_by_round[idx]
            d = dirn[r]
            for k in range(H[r]):
                a = (src0[r] + d * k * L) % N
                ns = L if k < H[r] - 1 else last[r]
                keys[t] = _circuit_key(a, d, ns, labels[hop_start[r] + k], L, lam)
                t += 1
        keys = np.unique(keys)
        sub = retained.shape[0] > 0 and keys.shape[0] <= retained.shape[0]
        if sub:
            j = 0
            for i in range(keys.shape[0]):
                while j < retained.shape[0] and retained[j] < keys[i]:
                    j += 1
                if j >= retained.shape[0] or retained[j] != keys[i]:
                    sub = False
                    break
        if sub:
            flags[rr] = 0
        else:
            retained = keys
    return flags, retained
