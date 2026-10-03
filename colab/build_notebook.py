#!/usr/bin/env python
"""Generate colab/LHAS_compute_profiling.ipynb (run locally; the notebook itself
runs on a Colab GPU runtime). Keeping the notebook as generated code makes its
content reviewable in plain text.

The repository's workload manifests (configs/workloads/*.json) and their sha256
hashes are embedded in the notebook; the notebook stops before profiling if the
Colab torchvision graphs do not reproduce the manifest shapes (D2, D17)."""
import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
WLS = ["alexnet", "vgg16", "googlenet", "resnet50"]
NB_VERSION = "3"


def expected_layers():
    out, hashes = {}, {}
    for w in WLS:
        f = ROOT / "configs" / "workloads" / f"{w}.json"
        hashes[w] = hashlib.sha256(f.read_bytes()).hexdigest()
        m = json.loads(f.read_text())
        nodes = {n["id"]: n for n in m["nodes"]}
        rows = []
        for n in m["nodes"]:
            if n.get("op") not in ("conv", "linear"):
                continue
            src = nodes[n["inputs"][0]]
            r = dict(layer=n["id"], kind="conv" if n["op"] == "conv" else "fc", x=src["shape"], y=n["shape"],
                     bias=n["bias"])
            if n["op"] == "conv":
                r.update(n_in=n["in_channels"], n_out=n["out_channels"], kernel=n["kernel"], stride=n["stride"],
                         padding=n["padding"], dilation=n["dilation"], groups=n["groups"])
            else:
                r.update(n_in=n["in_features"], n_out=n["out_features"], kernel=[], stride=[], padding=[],
                         dilation=[], groups=1)
            rows.append(r)
        out[w] = dict(constructor=m["constructor"], constructor_kwargs=m["constructor_kwargs"],
                      versions=m["versions"], layers=rows)
    return out, hashes


EXPECTED, HASHES = expected_layers()

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md(r"""
# LHAS: local compute profiling on the *actual* Colab GPU (notebook version """ + NB_VERSION + r""")

**What this notebook measures.** Per-layer local computation times of the pinned
torchvision graphs (AlexNet, VGG16, GoogLeNet without auxiliary heads, ResNet-50)
at the **local partition shapes** evaluated by the planner for the global batch
`B = 1024`:

* DP with `p` nodes: local batch `B/p`, complete weights;
* output-feature MP with `p` nodes: `n_out/p` output channels, complete batch `B`.

For each shape it times, separately and with CUDA-event synchronization, the
forward product, the input-gradient product and the weight-gradient product
(`aten.convolution_backward` with a single output mask, or the separate GEMMs of a
linear layer). It also times a local elementwise reduction and reports a
**model-equivalent rate** for the planner's reduction term.

**Safeguards.** The repository's workload manifests and their sha256 hashes are
embedded below. The notebook re-exports the layer shapes from the torchvision
constructors on Colab and **stops before profiling** if any layer differs from
the manifests. Results are appended to CSV files after every shape, so a
disconnected runtime keeps its completed measurements; rerunning the notebook in
the same runtime resumes. Every allocation is released in `finally` blocks.

**What it does not do.** It does not measure the optical ring or any collective,
does not train on ImageNet (inputs are **synthetic random tensors**), and does not
produce P100 timings. Results belong to the GPU recorded in the environment cell.
Never relabel them as P100, and never rescale them to P100 by peak FLOP/s.
Dividing a full-layer time by `p` is **not** a measurement; unmeasured shapes are
recorded as such. Profiled times validate the local compute model on this GPU;
they are not a validation of the optical transport model.

**How to run.** Runtime → Change runtime type → a GPU. Then Runtime → Run all.
The final cell writes `lhas_profile_<gpu>.zip` and offers a download (optionally
also copies it to Google Drive). Send that ZIP back unchanged; the repository's
`scripts/ingest_colab.py` rejects results whose batch, manifests, precision or
local shapes do not match.
""")

code(r"""
# ---- environment record (run first) ----
import json, os, platform, subprocess, sys, time, datetime, math, random, csv, zipfile, hashlib
import torch, torchvision
assert torch.cuda.is_available(), "Select a GPU runtime (Runtime > Change runtime type)."
dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(dev)
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"
# precision policy: float32 everywhere, TF32 and autocast disabled
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = True        # algorithm selection happens during warm-up; recorded below
torch.backends.cudnn.deterministic = False
ENV = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gpu_name": props.name,
    "gpu_total_memory_bytes": int(props.total_memory),
    "gpu_multiprocessors": props.multi_processor_count,
    "compute_capability": f"{props.major}.{props.minor}",
    "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv"),
    "driver_version": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    "cuda_runtime_torch": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "torch": torch.__version__, "torchvision": torchvision.__version__, "python": platform.python_version(),
    "cpu": platform.processor() or sh("lscpu | grep 'Model name'"),
    "precision": "float32",
    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "autocast": "not used",
    "inputs": "synthetic torch.randn tensors (not ImageNet)",
    "input_mode_label": "profiled_" + props.name.replace(" ", "_"),
}
print(json.dumps(ENV, indent=1))
""")

code(r"""
# ---- configuration and embedded repository manifests (do not edit the manifests) ----
NOTEBOOK_VERSION = "__NBV__"
B_GLOBAL = 1024                   # fixed global batch of the main comparison (D1)
N_MAX = 1024                      # largest system size considered (N = 64..512 primary, 1024 supplementary)
WARMUP = 10                       # warm-up launches per kernel (also triggers cuDNN algorithm selection)
REPS = 30                         # timed repetitions per kernel (raw samples are all saved)
MIN_REPS = 10                     # a kernel with fewer completed samples is recorded as not measured
MIN_CAL_SHAPES, MIN_VAL_SHAPES = 10, 20   # below these counts no error statistics are reported
MEM_FRACTION = 0.80               # skip shapes whose tensors would exceed this fraction of GPU memory
TIME_BUDGET_S = 3 * 3600          # stop profiling new shapes after this wall-clock budget (recorded)
SEED = 0
assert REPS >= MIN_REPS
EXPECTED = json.loads(r'''__EXPECTED__''')
EXPECTED_MANIFEST_SHA256 = json.loads(r'''__HASHES__''')
WORKLOADS = {w: (v["constructor"].split(".")[-1], v["constructor_kwargs"]) for w, v in EXPECTED.items()}
assert WORKLOADS["googlenet"][1].get("aux_logits") is False          # D3: no auxiliary heads
torch.manual_seed(SEED); random.seed(SEED)
OUT = f"/content/lhas_profile_{ENV['gpu_name'].replace(' ', '_')}"
os.makedirs(OUT, exist_ok=True)
CONFIG = dict(notebook_version=NOTEBOOK_VERSION, B_GLOBAL=B_GLOBAL, N_MAX=N_MAX, WARMUP=WARMUP, REPS=REPS,
              MIN_REPS=MIN_REPS, MEM_FRACTION=MEM_FRACTION, TIME_BUDGET_S=TIME_BUDGET_S, SEED=SEED,
              expected_manifest_sha256=EXPECTED_MANIFEST_SHA256)
# metadata files are written only after the resume-compatibility check (profiling cell)
""")

code(r"""
# ---- export weighted-layer shapes on this runtime and compare with the manifests ----
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
FIELDS = ("layer", "kind", "x", "y", "n_in", "n_out", "kernel", "stride", "padding", "dilation", "groups", "bias")
def export_layers(name):
    ctor, kw = WORKLOADS[name]
    m = getattr(torchvision.models, ctor)(**kw).train()
    gm = torch.fx.symbolic_trace(m)
    ShapeProp(gm).propagate(torch.randn(2, 3, 224, 224))
    mods = dict(gm.named_modules())
    layers = []
    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        mod = mods[n.target]
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            rec = dict(layer=n.name, x=list(n.all_input_nodes[0].meta["tensor_meta"].shape)[1:],
                       y=list(n.meta["tensor_meta"].shape)[1:], bias=mod.bias is not None)
            if isinstance(mod, torch.nn.Conv2d):
                rec.update(kind="conv", n_in=mod.in_channels, n_out=mod.out_channels, kernel=list(mod.kernel_size),
                           stride=list(mod.stride), padding=list(mod.padding), dilation=list(mod.dilation),
                           groups=mod.groups)
            else:
                rec.update(kind="fc", n_in=mod.in_features, n_out=mod.out_features, kernel=[], stride=[],
                           padding=[], dilation=[], groups=1)
            layers.append(rec)
    return layers

def compare_with_manifests(exported):
    problems = []
    for w, exp in EXPECTED.items():
        got = exported[w]
        if len(got) != len(exp["layers"]):
            problems.append(f"{w}: {len(got)} weighted layers, manifest has {len(exp['layers'])}")
            continue
        for a, b in zip(got, exp["layers"]):
            for f in FIELDS:
                if a[f] != b[f]:
                    problems.append(f"{w}/{b['layer']}: {f} = {a[f]} on this runtime, manifest {b[f]}")
    return problems

EXPORTED = {w: export_layers(w) for w in WORKLOADS}
PROBLEMS = compare_with_manifests(EXPORTED)
CHECK = dict(status="match" if not PROBLEMS else "mismatch", problems=PROBLEMS,
             runtime_versions=dict(torch=torch.__version__, torchvision=torchvision.__version__),
             manifest_versions={w: v["versions"] for w, v in EXPECTED.items()},
             expected_manifest_sha256=EXPECTED_MANIFEST_SHA256)
if PROBLEMS:
    print("\n".join(PROBLEMS[:50]))
    raise RuntimeError("Layer shapes differ from the repository manifests; profiling stopped (see the list above; "
                       "no file was written or modified).")
LAYERS = [dict(L, workload=w) for w in WORKLOADS for L in EXPECTED[w]["layers"]]
print(sum(len(v) for v in EXPORTED.values()), "weighted layers match the manifests")
""")

code(r"""
# ---- local partition shapes, deduplicated by TIMED KERNEL SIGNATURE (notebook v3) ----
# The key holds exactly what the timed kernels depend on. The parent layer's full output
# width does not enter the kernels, so it is recorded per use, not in the key (v2 keyed it,
# which timed some kernels twice under different IDs).
def divisors_upto(n, m):
    return [p for p in range(1, m + 1) if n % p == 0]
for L in LAYERS:          # the timing code implements dilation 1 and groups 1 only
    assert L["dilation"] in ([], [1, 1]) and L["groups"] == 1, (L["workload"], L["layer"])
def shape_key(L, batch, n_loc):
    return (L["kind"], tuple(L["x"]), tuple(L["kernel"]), tuple(L["stride"]), tuple(L["padding"]),
            L["bias"], batch, n_loc)
SHAPES = {}
for L in LAYERS:
    for p in divisors_upto(B_GLOBAL, N_MAX):                 # DP: local batch B/p, full weights
        SHAPES.setdefault(shape_key(L, B_GLOBAL // p, L["n_out"]), []).append(
            dict(workload=L["workload"], layer=L["layer"], strategy="DP", p=p, n_out_full=L["n_out"]))
    for p in divisors_upto(L["n_out"], N_MAX):                # MP: n_out/p output features, full batch
        SHAPES.setdefault(shape_key(L, B_GLOBAL, L["n_out"] // p), []).append(
            dict(workload=L["workload"], layer=L["layer"], strategy="MP", p=p, n_out_full=L["n_out"]))
print(len(SHAPES), "timed kernel signatures;", sum(len(v) for v in SHAPES.values()), "layer/configuration uses")
""")

code(r"""
# ---- timing helpers: CUDA events, warm-up, raw samples; incremental CSV ----
import torch.nn.functional as F
def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()
def timed(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) / 1e3)       # seconds
    return samples

def quantile(sorted_vals, q):
    # linear interpolation between order statistics (defined for any n >= 1)
    n = len(sorted_vals)
    if n == 0:
        return None
    k = (n - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

def release():
    sync()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

def conv_kernels(x_shape, n_out_local, kernel, stride, padding, bias, batch):
    C, H, W = x_shape
    x = torch.randn(batch, C, H, W, device=dev)
    w = torch.randn(n_out_local, C, *kernel, device=dev)
    b = torch.randn(n_out_local, device=dev) if bias else None
    y = F.conv2d(x, w, b, stride, padding)
    gy = torch.randn_like(y)
    bsz = [n_out_local] if bias else None
    cb = torch.ops.aten.convolution_backward
    fwd = lambda: F.conv2d(x, w, b, stride, padding)
    igrad = lambda: cb(gy, x, w, bsz, stride, padding, [1, 1], False, [0, 0], 1, [True, False, False])
    wgrad = lambda: cb(gy, x, w, bsz, stride, padding, [1, 1], False, [0, 0], 1, [False, True, bias])
    flops = 2 * batch * y.shape[2] * y.shape[3] * n_out_local * C * kernel[0] * kernel[1]
    return dict(forward=fwd, input_grad=igrad, weight_grad=wgrad), flops, [x, w, b, y, gy]

def fc_kernels(n_in, n_out_local, bias, batch):
    x = torch.randn(batch, n_in, device=dev)
    w = torch.randn(n_out_local, n_in, device=dev)
    b = torch.randn(n_out_local, device=dev) if bias else None
    gy = torch.randn(batch, n_out_local, device=dev)
    fwd = lambda: F.linear(x, w, b)
    igrad = lambda: gy @ w
    wgrad = (lambda: (gy.t() @ x, gy.sum(0))) if bias else (lambda: gy.t() @ x)
    return dict(forward=fwd, input_grad=igrad, weight_grad=wgrad), 2 * batch * n_in * n_out_local, [x, w, b, gy]

def est_bytes(key):
    kind, x, kernel, stride, padding, bias, batch, n_loc = key
    xin = math.prod(x) * batch
    if kind == "conv":
        H = (x[1] + 2 * padding[0] - kernel[0]) // stride[0] + 1
        W = (x[2] + 2 * padding[1] - kernel[1]) // stride[1] + 1
        y = n_loc * H * W * batch
        w = n_loc * x[0] * kernel[0] * kernel[1]
    else:
        y, w = n_loc * batch, n_loc * x[0]
    return 4 * (2 * xin + 2 * y + 2 * w)        # tensors plus gradients, float32

PARTS = ("forward", "input_grad", "weight_grad")
SUM_FIELDS = (["shape_id", "kind", "x", "n_out_full", "n_out_local", "batch_local", "kernel", "stride", "padding",
               "bias", "uses", "status", "flops_per_product", "reps_requested"]
              + [f"{k}_{s}" for k in PARTS for s in ("median_s", "p25_s", "p75_s", "n")])
RAW_FIELDS = ["shape_id", "kernel_part", "rep", "seconds"]

class AppendCSV:
    # appends rows and flushes them to disk immediately (survives a disconnected runtime)
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="raise")
        if new:
            self.w.writeheader(); self.flush()
    def write(self, row):
        self.w.writerow({k: row.get(k, "") for k in self.fields}); self.flush()
    def flush(self):
        self.f.flush(); os.fsync(self.f.fileno())
    def close(self):
        self.f.close()

def done_shape_ids(path):
    # shapes already finished in an earlier run of this runtime (resume); time-budget skips are retried
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {int(r["shape_id"]) for r in csv.DictReader(f) if not r["status"].startswith("not measured: time")}
""")

code(r"""
# ---- profile every local shape (skips are recorded, never extrapolated) ----
t_start = time.time()
cap = MEM_FRACTION * props.total_memory
ORDERED = sorted(SHAPES.items(), key=lambda kv: (est_bytes(kv[0]), kv[0]))
# Match the recorded hardware class, runtime and precision policy; this does not
# authenticate a physical GPU identity. Session timestamps are intentionally excluded.
RESUME_ENV_KEYS = ("gpu_name", "gpu_total_memory_bytes", "gpu_multiprocessors", "compute_capability",
                   "driver_version", "cuda_runtime_torch", "torch", "torchvision", "python", "cudnn_version",
                   "precision", "allow_tf32_matmul", "allow_tf32_cudnn", "float32_matmul_precision",
                   "cudnn_benchmark", "cudnn_deterministic", "autocast")
FP_SRC = dict(CONFIG, resume_environment={k: ENV.get(k) for k in RESUME_ENV_KEYS},
              shapes=[[str(v) for v in k] for k, _ in ORDERED])
FINGERPRINT = hashlib.sha256(json.dumps(FP_SRC, sort_keys=True).encode()).hexdigest()
fp_path = f"{OUT}/resume_fingerprint.json"
DATA_FILES = ("environment.json", "config.json", "manifest_check.json", "shape_manifest.json",
              "layer_timings_summary.csv", "layer_timings_raw.csv", "reduction_timings.csv")
if os.path.exists(fp_path):
    FP = json.load(open(fp_path))
    if FP["fingerprint"] != FINGERPRINT:
        raise RuntimeError(f"{OUT} holds partial results of a different configuration, device, software or shape "
                           "list; move that directory away before profiling (no file was modified).")
    FP["resumed_utc"].append(datetime.datetime.now(datetime.timezone.utc).isoformat())
    FP.setdefault("resumed_environments", []).append(ENV)
else:
    present = [f for f in DATA_FILES if os.path.exists(f"{OUT}/{f}")]
    if present:
        raise RuntimeError(f"{OUT} holds {present} without a resume fingerprint; move that directory away before "
                           "profiling (no file was modified).")
    FP = dict(fingerprint=FINGERPRINT, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              resumed_utc=[])
def _write_meta(name, obj):
    # the first session's metadata is kept on a compatible resume; later sessions are recorded
    # in resume_fingerprint.json (resumed_utc, resumed_environments)
    if not os.path.exists(f"{OUT}/{name}"):
        json.dump(obj, open(f"{OUT}/{name}", "w"), indent=1)
_write_meta("environment.json", ENV)
_write_meta("config.json", CONFIG)
_write_meta("manifest_check.json", CHECK)
_write_meta("shape_manifest.json", EXPORTED)
json.dump(FP, open(fp_path, "w"), indent=1)
done = done_shape_ids(f"{OUT}/layer_timings_summary.csv")
summ = AppendCSV(f"{OUT}/layer_timings_summary.csv", SUM_FIELDS)
raw = AppendCSV(f"{OUT}/layer_timings_raw.csv", RAW_FIELDS)
try:
    for i, (key, uses) in enumerate(ORDERED):
        if i in done:
            continue
        kind, x, kernel, stride, padding, bias, batch, n_loc = key
        rec = dict(shape_id=i, kind=kind, x=json.dumps(list(x)),
                   n_out_full=json.dumps(sorted({u["n_out_full"] for u in uses})), n_out_local=n_loc,
                   batch_local=batch, kernel=json.dumps(list(kernel)), stride=json.dumps(list(stride)),
                   padding=json.dumps(list(padding)), bias=bias, uses=json.dumps(uses), reps_requested=REPS)
        if time.time() - t_start > TIME_BUDGET_S:
            summ.write(dict(rec, status="not measured: time budget exhausted")); continue
        if est_bytes(key) > cap:
            summ.write(dict(rec, status="not measured: exceeds memory cap")); continue
        ks = keep = None
        try:
            ks, flops, keep = (conv_kernels(x, n_loc, kernel, stride, padding, bias, batch) if kind == "conv"
                               else fc_kernels(x[0], n_loc, bias, batch))
            out = dict(rec, status="measured", flops_per_product=flops)
            for kname in PARTS:
                s = timed(ks[kname])
                for j, v in enumerate(s):
                    raw.write(dict(shape_id=i, kernel_part=kname, rep=j, seconds=v))
                ss = sorted(s)
                out.update({f"{kname}_median_s": quantile(ss, 0.5), f"{kname}_p25_s": quantile(ss, 0.25),
                            f"{kname}_p75_s": quantile(ss, 0.75), f"{kname}_n": len(ss)})
                if len(ss) < MIN_REPS:
                    out["status"] = f"not measured: {len(ss)} samples < MIN_REPS for {kname}"
            summ.write(out)
        except torch.cuda.OutOfMemoryError:
            summ.write(dict(rec, status="not measured: CUDA out of memory"))
        except RuntimeError as e:
            summ.write(dict(rec, status=f"not measured: runtime error: {str(e)[:200]}"))
        finally:
            del ks, keep
            release()
        if i % 25 == 0:
            print(f"{i}/{len(SHAPES)} shapes, {time.time() - t_start:.0f}s elapsed", flush=True)
finally:
    summ.close(); raw.close()
with open(f"{OUT}/layer_timings_summary.csv") as f:
    ROWS = list(csv.DictReader(f))
print("measured:", sum(r["status"] == "measured" for r in ROWS), "of", len(SHAPES), "shapes")
""")

code(r"""
# ---- local reduction microbenchmark: model-equivalent rate for the reduction term ----
# The planner charges a local reduction of `a` received contributions of V bytes as
# (a + 2) V / beta_red. The value reported below, (a + 2) V / t_measured, is the rate
# at which that formula reproduces the measured time on this GPU (a model-equivalent
# rate). It is not a memory-bandwidth measurement, and it applies to this GPU only.
# A compatible Run-all resumes reductions as well as layer kernels. Keep the
# first complete measurement of each case; retry only absent/incomplete cases.
red_path = f"{OUT}/reduction_timings.csv"
red_done = set()
if os.path.exists(red_path):
    with open(red_path) as f:
        for r in csv.DictReader(f):
            if (r["method"] in ("fused_sum", "sequential_inplace")
                    and int(r.get("n") or 0) >= MIN_REPS
                    and math.isfinite(float(r.get("median_s") or "nan"))
                    and float(r["median_s"]) > 0):
                red_done.add((int(r["V_bytes"]), int(r["a"]), r["method"]))
red = AppendCSV(red_path,
                ["V_bytes", "a", "method", "median_s", "p25_s", "p75_s", "n", "model_traffic_bytes",
                 "model_equivalent_rate_byte_s", "note", "samples"])
try:
    for V_mib in [1, 4, 16, 64, 256]:
        n = V_mib * 2**20 // 4
        for a in [1, 2, 4, 8]:
            if all((V_mib * 2**20, a, name) in red_done
                   for name in ("fused_sum", "sequential_inplace")):
                continue
            if (a + 3) * V_mib * 2**20 > MEM_FRACTION * props.total_memory:
                continue
            stacked = acc = None
            try:
                stacked = torch.randn(a + 1, n, device=dev)       # own contribution + a received ones
                acc = torch.empty(n, device=dev)
                def fused():
                    torch.sum(stacked, dim=0, out=acc)          # reads (a + 1) V, writes V
                def sequential():
                    torch.add(stacked[0], stacked[1], out=acc)
                    for t in stacked[2:]:
                        acc.add_(t)
                for name, fn, note in (("fused_sum", fused, "traffic equals the model's (a+2)V"),
                                       ("sequential_inplace", sequential, "traffic 3aV; rate still (a+2)V/t")):
                    if (V_mib * 2**20, a, name) in red_done:
                        continue
                    s = sorted(timed(fn))
                    med = quantile(s, 0.5)
                    red.write(dict(V_bytes=V_mib * 2**20, a=a, method=name, median_s=med, p25_s=quantile(s, 0.25),
                                   p75_s=quantile(s, 0.75), n=len(s), model_traffic_bytes=(a + 2) * V_mib * 2**20,
                                   model_equivalent_rate_byte_s=(a + 2) * V_mib * 2**20 / med, note=note,
                                   samples=json.dumps(s)))
            except torch.cuda.OutOfMemoryError:
                red.write(dict(V_bytes=V_mib * 2**20, a=a, method="-", note="not measured: CUDA out of memory"))
            finally:
                del stacked, acc
                release()
finally:
    red.close()
print("reduction measurements written")
""")

code(r"""
# ---- constant-throughput model of THIS device: held-out validation ----
# Model: t_pred = 3 * flops_per_product / R, with R fitted on a random half of the
# measured kernel signatures (calibration) and evaluated on the other half. From v3 each
# record is one timed kernel signature, so the split shares no signature (the repository's
# scripts/analyse_profile.py applies the same grouped protocol and is the reported analysis).
# This mirrors the planner's analytic form (rho * peak replaced by the fitted rate)
# and says nothing about P100 or about the optical transport model.
PEAK_FP32_FLOPS = None            # optional vendor FP32 peak of this GPU for a utilization figure

def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra)); vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else None

def error_stats(pred, meas):
    rel = [(p - m) / m for p, m in zip(pred, meas)]
    ab = sorted(abs(e) for e in rel)
    return dict(n=len(rel), signed_relative_error_median=quantile(sorted(rel), 0.5),
                absolute_relative_error_median=quantile(ab, 0.5), absolute_relative_error_p90=quantile(ab, 0.9),
                absolute_relative_error_p95=quantile(ab, 0.95), absolute_relative_error_max=ab[-1],
                weighted_absolute_error=sum(abs(p - m) for p, m in zip(pred, meas)) / sum(meas),
                summed_time_error=(sum(pred) - sum(meas)) / sum(meas),
                spearman_rank_correlation=spearman(pred, meas))

def validate(rows, seed=SEED):
    meas = [r for r in rows if r["status"] == "measured"]
    tot = lambda r: sum(float(r[f"{k}_median_s"]) for k in PARTS)
    ids = sorted({int(r["shape_id"]) for r in meas}); random.Random(seed).shuffle(ids)
    calib = set(ids[: len(ids) // 2])
    cal = [r for r in meas if int(r["shape_id"]) in calib]
    val = [r for r in meas if int(r["shape_id"]) not in calib]
    out = dict(n_measured=len(meas), n_calibration=len(cal), n_validation=len(val),
               unit="timed kernel signature (one record each from notebook v3)",
               model="t = 3 * flops_per_product / R (R fitted on calibration signatures of this GPU)")
    if len(cal) < MIN_CAL_SHAPES or len(val) < MIN_VAL_SHAPES:
        out.update(status="insufficient samples",
                   note=f"need >= {MIN_CAL_SHAPES} calibration and >= {MIN_VAL_SHAPES} validation shapes")
        return out
    R = sum(3 * float(r["flops_per_product"]) for r in cal) / sum(tot(r) for r in cal)
    pred = [3 * float(r["flops_per_product"]) / R for r in val]
    out.update(status="ok", achieved_flops_calibration=R,
               utilization_vs_entered_peak=(R / PEAK_FP32_FLOPS if PEAK_FP32_FLOPS else None),
               validation=error_stats(pred, [tot(r) for r in val]))
    for kind in ("conv", "fc"):
        sub = [(p, tot(r)) for p, r in zip(pred, val) if r["kind"] == kind]
        out[f"validation_{kind}"] = (error_stats([a for a, _ in sub], [b for _, b in sub])
                                     if len(sub) >= MIN_VAL_SHAPES else dict(n=len(sub), status="insufficient samples"))
    return out

VALIDATION = validate(ROWS)
json.dump(VALIDATION, open(f"{OUT}/device_model_validation.json", "w"), indent=1)
print(json.dumps(VALIDATION, indent=1))
""")

code(r"""
# ---- package results (send this ZIP back unchanged) ----
import shutil
zip_path = shutil.make_archive(OUT, "zip", OUT)
print("wrote", zip_path, hashlib.sha256(open(zip_path, "rb").read()).hexdigest())
try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("download helper unavailable:", e)
# Optional: copy to Google Drive
# from google.colab import drive; drive.mount('/content/drive'); shutil.copy(zip_path, '/content/drive/MyDrive/')
""")

for c in cells:
    if c.cell_type == "code":
        c.source = (c.source.replace("__EXPECTED__", json.dumps(EXPECTED))
                    .replace("__HASHES__", json.dumps(HASHES)).replace("__NBV__", NB_VERSION))
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = Path(__file__).resolve().parent / "LHAS_compute_profiling.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
