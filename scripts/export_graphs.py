#!/usr/bin/env python
"""Export pinned torchvision graphs to machine-readable manifests.

For each workload this records the constructor and options, package versions,
every traced node (op type, inputs, per-example output shape, attributes,
trainable parameter and buffer element counts). No weights are downloaded and
no timing is performed; shapes are obtained with torch.fx shape propagation on
a random per-example input (batch size 2, CPU, float32).

Usage: python scripts/export_graphs.py [--out configs/workloads]
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import torch.fx
import torchvision
from torch.fx.passes.shape_prop import ShapeProp

WORKLOADS = {
    "alexnet": ("torchvision.models.alexnet", dict(weights=None, num_classes=1000)),
    "vgg16": ("torchvision.models.vgg16", dict(weights=None, num_classes=1000)),
    "googlenet": ("torchvision.models.googlenet",
                  dict(weights=None, num_classes=1000, aux_logits=False, transform_input=False, init_weights=True)),
    "googlenet_aux": ("torchvision.models.googlenet",
                      dict(weights=None, num_classes=1000, aux_logits=True, transform_input=False, init_weights=True)),
    "resnet50": ("torchvision.models.resnet50", dict(weights=None, num_classes=1000)),
}
INPUT = (3, 224, 224)


def _mod_record(m: torch.nn.Module) -> dict:
    rec = {"module": type(m).__name__}
    if isinstance(m, torch.nn.Conv2d):
        rec.update(op="conv", in_channels=m.in_channels, out_channels=m.out_channels,
                   kernel=list(m.kernel_size), stride=list(m.stride), padding=list(m.padding),
                   dilation=list(m.dilation), groups=m.groups, bias=m.bias is not None)
    elif isinstance(m, torch.nn.Linear):
        rec.update(op="linear", in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
    elif isinstance(m, torch.nn.BatchNorm2d):
        rec.update(op="batchnorm", num_features=m.num_features, eps=m.eps, momentum=m.momentum,
                   affine=m.affine, track_running_stats=m.track_running_stats)
    elif isinstance(m, (torch.nn.ReLU,)):
        rec.update(op="relu", inplace=m.inplace)
    elif isinstance(m, torch.nn.MaxPool2d):
        rec.update(op="maxpool", kernel=m.kernel_size, stride=m.stride, padding=m.padding, ceil_mode=m.ceil_mode)
    elif isinstance(m, torch.nn.AvgPool2d):
        rec.update(op="avgpool", kernel=m.kernel_size, stride=m.stride, padding=m.padding)
    elif isinstance(m, torch.nn.AdaptiveAvgPool2d):
        rec.update(op="adaptive_avgpool", output_size=m.output_size)
    elif isinstance(m, torch.nn.Dropout):
        rec.update(op="dropout", p=m.p)
    elif isinstance(m, torch.nn.Flatten):
        rec.update(op="flatten", start_dim=m.start_dim)
    else:
        rec.update(op="module")
    rec["param_numel"] = {n: int(p.numel()) for n, p in m.named_parameters(recurse=False)}
    rec["buffer_numel"] = {n: int(b.numel()) for n, b in m.named_buffers(recurse=False)}
    return rec


def _fn_op(target) -> str:
    name = getattr(target, "__name__", str(target))
    return {"flatten": "flatten", "cat": "cat", "add": "add", "iadd": "add", "relu": "relu",
            "dropout": "dropout", "adaptive_avg_pool2d": "adaptive_avgpool", "max_pool2d": "maxpool",
            "getitem": "getitem", "mul": "mul", "unsqueeze": "unsqueeze"}.get(name, name)


def export(name: str, out: Path) -> dict:
    ctor_path, kwargs = WORKLOADS[name]
    ctor = getattr(torchvision.models, ctor_path.rsplit(".", 1)[1])
    torch.manual_seed(0)
    model = ctor(**kwargs)
    model.train()
    gm = torch.fx.symbolic_trace(model)
    x = torch.randn(2, *INPUT)
    ShapeProp(gm).propagate(x)
    mods = dict(gm.named_modules())
    nodes = []
    for n in gm.graph.nodes:
        meta = n.meta.get("tensor_meta")
        shape = None
        if meta is not None and hasattr(meta, "shape"):
            shape = list(meta.shape)[1:]
        rec = {"id": n.name, "fx_op": n.op, "inputs": [a.name for a in n.all_input_nodes], "shape": shape}
        if n.op == "call_module":
            rec["target"] = n.target
            rec.update(_mod_record(mods[n.target]))
        elif n.op in ("call_function", "call_method"):
            rec["target"] = str(getattr(n.target, "__name__", n.target))
            rec["op"] = _fn_op(n.target)
            if rec["op"] == "cat":
                dim = n.kwargs.get("dim", n.args[1] if len(n.args) > 1 else 0)
                rec["dim"] = int(dim)
            if rec["op"] == "flatten":
                rec["start_dim"] = int(n.args[1] if len(n.args) > 1 else n.kwargs.get("start_dim", 0))
        elif n.op == "placeholder":
            rec["op"] = "input"
        elif n.op == "output":
            rec["op"] = "output"
        nodes.append(rec)
    manifest = {
        "workload": name,
        "constructor": ctor_path,
        "constructor_kwargs": kwargs,
        "mode": "train",
        "input_per_example": list(INPUT),
        "dtype": "float32",
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "python": platform.python_version()},
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "buffers": int(sum(b.numel() for b in model.buffers())),
        "nodes": nodes,
        "note": "Shapes are per example (batch dimension removed). Traced in train mode on CPU; "
                "no pretrained weights, no timing.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "configs" / "workloads"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for name in (a.only or WORKLOADS):
        m = export(name, Path(a.out))
        conv = sum(1 for n in m["nodes"] if n.get("op") == "conv")
        lin = sum(1 for n in m["nodes"] if n.get("op") == "linear")
        bn = sum(1 for n in m["nodes"] if n.get("op") == "batchnorm")
        print(f"{name}: {len(m['nodes'])} nodes, {conv} conv, {lin} linear, {bn} batchnorm, "
              f"{m['trainable_params']} trainable parameters")


if __name__ == "__main__":
    main()
