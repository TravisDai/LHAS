"""Weighted-layer view of an exported graph manifest (configs/workloads/*.json).

A weighted layer is a CONV or FC (Linear) node. Non-weighted nodes between
weighted layers form the boundary transformation Phi_i (sec:inter-layer). For a
chain network each weighted layer has one weighted successor, and U_i = Phi_i(Y_i)
is the successor's input tensor X_{i+1}. Shapes are per example; the global batch
B is applied by the cost model.

Branch networks (concatenation and addition joins) are represented by the same
WLayer objects plus typed interface nodes; see `lhas.model.regions`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
WORKLOAD_DIR = REPO / "configs" / "workloads"

# Non-weighted operators that are local and ownership-preserving under both the
# batch (DP) and output-channel (MP) layouts when channels are the slowest
# varying feature index. BatchNorm is local only under the chosen statistics
# policy (see docs/decision_table.md); it is listed separately.
LOCAL_OPS = {"relu", "maxpool", "avgpool", "adaptive_avgpool", "dropout", "flatten"}
NORM_OPS = {"batchnorm"}
JOIN_OPS = {"cat", "add"}


@dataclass
class WLayer:
    name: str
    kind: str                         # 'conv' | 'fc'
    index: int
    n_in: int
    n_out: int
    kernel: tuple
    x_shape: tuple                    # per example
    y_shape: tuple
    weight_numel: int
    bias_numel: int
    norm: Optional[dict] = None       # BatchNorm immediately following: params, buffers
    producers: list = field(default_factory=list)   # names of upstream weighted layers
    input_is_graph_input: bool = False
    phi_ops: list = field(default_factory=list)     # ops from Y to successor input (chains)
    u_shape: Optional[tuple] = None                 # boundary tensor per example (chains)

    @property
    def x_numel(self) -> int:
        return math.prod(self.x_shape)

    @property
    def y_numel(self) -> int:
        return math.prod(self.y_shape)

    @property
    def u_numel(self) -> int:
        return math.prod(self.u_shape) if self.u_shape is not None else self.y_numel

    @property
    def rows_per_channel(self) -> int:
        r, rem = divmod(self.u_numel, self.n_out)
        if rem:
            raise ValueError(f"{self.name}: boundary tensor is not channel-major divisible")
        return r

    @property
    def phi_identity(self) -> bool:
        """True when U_i can occupy the producer's own output reservation: same
        shape and only storage-preserving (in-place ReLU) operators. Dropout,
        normalization, pooling and flattening create separately stored tensors."""
        from .layercost import aliasable_ops
        return self.u_shape is None or (tuple(self.u_shape) == tuple(self.y_shape) and aliasable_ops(self.phi_ops))

    def mult_flops_per_example(self) -> int:
        """FLOPs of one modeled multiplication per example (Eq. comp): 2 per MAC."""
        if self.kind == "fc":
            return 2 * self.n_in * self.n_out
        spatial = math.prod(self.y_shape[1:])
        return 2 * spatial * self.n_out * self.n_in * math.prod(self.kernel)


@dataclass
class Workload:
    name: str
    manifest: dict
    layers: list
    is_chain: bool
    manifest_sha256: str = ""        # sha256 of the manifest file (configs/workloads/SHA256SUMS)

    def layer(self, name: str) -> WLayer:
        return next(l for l in self.layers if l.name == name)


def load_workload(name: str, path: Path = WORKLOAD_DIR) -> Workload:
    raw = (Path(path) / f"{name}.json").read_bytes()
    m = json.loads(raw)
    nodes = {n["id"]: n for n in m["nodes"]}
    order = [n["id"] for n in m["nodes"]]
    consumers = {k: [] for k in nodes}
    for n in m["nodes"]:
        for a in n["inputs"]:
            consumers[a].append(n["id"])
    weighted = [k for k in order if nodes[k].get("op") in ("conv", "linear")]
    layers = []
    for i, k in enumerate(weighted):
        n = nodes[k]
        src = nodes[n["inputs"][0]]
        if n["op"] == "conv":
            if n.get("groups", 1) != 1 or n.get("dilation", [1, 1]) != [1, 1]:
                raise ValueError(f"{k}: grouped or dilated convolution is outside the model")
            kind, n_in, n_out, kernel = "conv", n["in_channels"], n["out_channels"], tuple(n["kernel"])
        else:
            kind, n_in, n_out, kernel = "fc", n["in_features"], n["out_features"], (1,)
        norm = None
        cons = consumers[k]
        if len(cons) == 1 and nodes[cons[0]].get("op") == "batchnorm":
            b = nodes[cons[0]]
            norm = dict(name=cons[0], params=sum(b["param_numel"].values()),
                        buffers={kk: v for kk, v in b["buffer_numel"].items()}, num_features=b["num_features"])
        layers.append(WLayer(name=k, kind=kind, index=i, n_in=n_in, n_out=n_out, kernel=kernel,
                             x_shape=tuple(src["shape"]), y_shape=tuple(n["shape"]),
                             weight_numel=n["param_numel"].get("weight", 0),
                             bias_numel=n["param_numel"].get("bias", 0), norm=norm))
    # upstream weighted producers and graph-input reachability
    wset = set(weighted)
    anc_cache = {}

    def weighted_ancestors(node_id):
        if node_id in anc_cache:
            return anc_cache[node_id]
        res = set()
        for a in nodes[node_id]["inputs"]:
            if a in wset:
                res.add(a)
            else:
                res |= weighted_ancestors(a)
        anc_cache[node_id] = res
        return res

    def direct_weighted_producers(node_id):
        res = []
        for a in nodes[node_id]["inputs"]:
            if a in wset:
                res.append(a)
            else:
                res += direct_weighted_producers(a)
        return res

    for L in layers:
        L.producers = sorted(set(direct_weighted_producers(L.name)), key=weighted.index)
        L.input_is_graph_input = len(weighted_ancestors(L.name)) == 0
    # chain detection and Phi paths
    is_chain = True
    for i, L in enumerate(layers):
        succ = [c for c in layers if L.name in c.producers]
        if i < len(layers) - 1 and (len(succ) != 1 or succ[0].index != i + 1 or len(succ[0].producers) != 1):
            is_chain = False
    if is_chain:
        for i, L in enumerate(layers[:-1]):
            nxt = layers[i + 1]
            path, cur = [], nodes[nxt.name]["inputs"][0]
            while cur != L.name:
                op = nodes[cur].get("op")
                if op not in LOCAL_OPS | NORM_OPS:
                    raise ValueError(f"nonlocal boundary operator {op} between {L.name} and {nxt.name}")
                path.append(op)
                cur = nodes[cur]["inputs"][0]
            L.phi_ops = list(reversed(path))
            L.u_shape = tuple(nxt.x_shape)
    import hashlib
    return Workload(name=name, manifest=m, layers=layers, is_chain=is_chain,
                    manifest_sha256=hashlib.sha256(raw).hexdigest())


__all__ = ["WLayer", "Workload", "load_workload", "LOCAL_OPS", "JOIN_OPS"]


def synthetic_chain(name: str, specs: list) -> Workload:
    """Small synthetic chain for tests. specs: list of dicts with keys
    kind ('conv'|'fc'), n_in, n_out, x_shape, y_shape, kernel, u_shape (optional),
    phi_ops (optional)."""
    layers = []
    for i, s in enumerate(specs):
        L = WLayer(name=f"{name}_{i}", kind=s["kind"], index=i, n_in=s["n_in"], n_out=s["n_out"],
                   kernel=tuple(s.get("kernel", (1,))), x_shape=tuple(s["x_shape"]), y_shape=tuple(s["y_shape"]),
                   weight_numel=s["n_in"] * s["n_out"] * math.prod(s.get("kernel", (1,))),
                   bias_numel=s.get("bias", s["n_out"]))
        L.input_is_graph_input = i == 0
        L.producers = [] if i == 0 else [f"{name}_{i-1}"]
        layers.append(L)
    for i, L in enumerate(layers[:-1]):
        L.u_shape = tuple(specs[i].get("u_shape", layers[i + 1].x_shape))
        L.phi_ops = list(specs[i].get("phi_ops", ["relu"]))
    return Workload(name=name, manifest={}, layers=layers, is_chain=True)
