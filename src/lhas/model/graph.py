"""Typed interface structure of a pinned graph (manuscript sec:branch).

Every tensor that crosses between weighted layers is an *interface* T:

* its producers are weighted layers (each contributing a channel slice for a
  concatenation, the full tensor for an addition operand or a single producer),
  or, for an identity operand of an addition, an earlier interface;
* its consumers are weighted layers whose input is T (possibly after exclusive,
  consumer-side local operators such as the stride-1 max pooling of an
  inception pooling branch), plus identity uses by later additions.

Producer-side local operators (normalization, activation, pooling, flattening,
dropout) between a producer (or a join) and the fork point are applied per
channel slice by the producer, so the delivered tensor is T itself. Channel
offsets of concatenated slices follow the concatenation input order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .workload import Workload, LOCAL_OPS, NORM_OPS

PASS_OPS = LOCAL_OPS | NORM_OPS


@dataclass
class Producer:
    kind: str                 # 'layer' | 'iface'
    ref: object               # layer name or interface index
    ch0: int                  # channel offset in T
    ch1: int
    ops: list = field(default_factory=list)   # producer-side local operators


@dataclass
class Interface:
    idx: int
    node: str
    shape: tuple              # per example (C, H, W) or (F,)
    join: str                 # 'input' | 'single' | 'cat' | 'add'
    producers: list
    consumers: list           # weighted layer names
    consumer_ops: dict        # layer name -> exclusive consumer-side ops
    identity_users: list = field(default_factory=list)   # interface indices using T as identity operand
    post_join_ops: list = field(default_factory=list)

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def rows_per_channel(self) -> int:
        return self.numel // self.channels


@dataclass
class GraphStructure:
    workload: Workload
    interfaces: list
    iface_of_input: dict       # layer name -> interface index of its input
    iface_of_output: dict      # layer name -> interface index its output feeds (None for the last layer)
    output_layer: str


def build_structure(wl: Workload) -> GraphStructure:
    m = wl.manifest
    nodes = {n["id"]: n for n in m["nodes"]}
    order = [n["id"] for n in m["nodes"]]
    consumers = {k: [] for k in nodes}
    for n in m["nodes"]:
        for a in n["inputs"]:
            consumers[a].append(n["id"])
    wnames = {L.name for L in wl.layers}

    def is_pass(nid):
        return nodes[nid].get("op") in PASS_OPS

    def walk_exclusive(start):
        """Walk back from node `start` (inclusive) through single-consumer pass
        operators; return (stop node, ops walked in forward order)."""
        ops, cur = [], start
        while is_pass(cur) and len(consumers[cur]) == 1:
            ops.append(nodes[cur]["op"])
            cur = nodes[cur]["inputs"][0]
        return cur, list(reversed(ops))

    # fork point of each weighted layer's input
    fork_of, cons_ops = {}, {}
    for L in wl.layers:
        inp = nodes[L.name]["inputs"][0]
        stop, ops = walk_exclusive(inp)
        if len(consumers[stop]) > 1:
            # fork point shared with other consumers; walked ops are consumer-side
            fork_of[L.name] = stop
            cons_ops[L.name] = ops
        else:
            # exclusive path to a producer, join or the graph input: T is the
            # layer's own input and the walked ops are producer-side (Phi)
            fork_of[L.name] = inp
            cons_ops[L.name] = []
    # add operands that are identity uses of a fork point
    add_nodes = [k for k in order if nodes[k].get("op") == "add"]
    fork_points = []
    for k in order:
        if k in set(fork_of.values()) and k not in fork_points:
            fork_points.append(k)
    ifaces, idx_of_node = [], {}

    def source_of(t_node):
        """Walk back from interface node through pass ops to its source."""
        ops, cur = [], t_node
        while is_pass(cur):
            ops.append(nodes[cur]["op"])
            cur = nodes[cur]["inputs"][0]
        return cur, list(reversed(ops))

    for t in fork_points:
        src, post = source_of(t)
        shape = tuple(nodes[t]["shape"])
        if nodes[src].get("op") == "add":
            # post-join operators do not commute with addition in general (relu):
            # operands are delivered with the addition's input shape and the sum and
            # post-join operators are applied at each consumer
            shape = tuple(nodes[nodes[src]["inputs"][0]]["shape"])
        cons = [L.name for L in wl.layers if fork_of[L.name] == t]
        if nodes[src].get("op") == "input":
            prods, join = [], "input"
        elif src in wnames:
            prods, join = [Producer("layer", src, 0, shape[0], post)], "single"
        elif nodes[src]["op"] == "cat":
            prods, off, join = [], 0, "cat"
            for a in nodes[src]["inputs"]:
                ps, pops = walk_exclusive(a)
                if ps not in wnames:
                    raise ValueError(f"concatenation input {a} is not produced by a weighted layer")
                c = nodes[a]["shape"][0]
                prods.append(Producer("layer", ps, off, off + c, pops + post))
                off += c
        elif nodes[src]["op"] == "add":
            prods, join = [], "add"
            for a in nodes[src]["inputs"]:
                ps, pops = walk_exclusive(a)
                if ps in wnames:
                    prods.append(Producer("layer", ps, 0, shape[0], pops))
                else:
                    prods.append(Producer("iface", ps, 0, shape[0], pops))   # resolved below
        else:
            raise ValueError(f"unsupported interface source {src} ({nodes[src].get('op')})")
        cops = {c: (post if join == "add" else []) + cons_ops[c] for c in cons}
        I = Interface(len(ifaces), t, shape, join, prods, cons, cops, post_join_ops=post)
        idx_of_node[t] = I.idx
        ifaces.append(I)
    for I in ifaces:
        for p in I.producers:
            if p.kind == "iface":
                if p.ref not in idx_of_node:
                    raise ValueError(f"identity operand {p.ref} is not an interface")
                p.ref = idx_of_node[p.ref]
                ifaces[p.ref].identity_users.append(I.idx)
    iface_in = {L.name: idx_of_node[fork_of[L.name]] for L in wl.layers}
    iface_out = {}
    for I in ifaces:
        for p in I.producers:
            if p.kind == "layer":
                iface_out[p.ref] = I.idx
    last = wl.layers[-1].name
    return GraphStructure(wl, ifaces, iface_in, iface_out, last)


def describe(gs: GraphStructure) -> str:
    lines = []
    for I in gs.interfaces:
        pr = ", ".join(f"{p.kind}:{p.ref}[{p.ch0}:{p.ch1}]{('+' + '/'.join(p.ops)) if p.ops else ''}" for p in I.producers)
        co = ", ".join(c + (f"({'/'.join(I.consumer_ops[c])})" if I.consumer_ops[c] else "") for c in I.consumers)
        lines.append(f"T{I.idx} {I.node} {I.join} {I.shape}: producers [{pr}] -> consumers [{co}]"
                     + (f" identity->{I.identity_users}" if I.identity_users else ""))
    return "\n".join(lines)


__all__ = ["Producer", "Interface", "GraphStructure", "build_structure", "describe"]
