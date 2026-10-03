#!/usr/bin/env python
"""Reproduce the defects reported in LHAS_Decision_Answers_and_Code_Review.md
against the current implementation. Prints one line per finding."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from lhas.params import load_approved  # noqa: E402
from lhas.collectives.service import CollectiveService  # noqa: E402
from lhas.model.workload import load_workload  # noqa: E402
from lhas.model.chain import ChainModel, Settings, DP, MP  # noqa: E402

B = 1024
out = {}
# ---- 1. greedy fallback uses the final layer's terminal charge
tp, cp = load_approved(64)
m = ChainModel(load_workload("alexnet"), CollectiveService(tp, cp), Settings(B=B, raw_input_grad="retain"))
i = 6                                   # classifier_4 (4096 outputs), next layer classifier_6 (1000 outputs)
c = (MP, 16)
assert c in m.configs(i) and c not in m.configs(i + 1)
alts = m.terminal(c)                    # what greedy_layerwise used
L_last, L_i = m.L[-1], m.L[i]
wrong_shard = m.psi * L_last.y_numel * B // 16
right_shard = m.psi * L_i.u_numel * B // 16
out["greedy_fallback"] = dict(layer=L_i.name, config="MP16", terminal_uses_layer=L_last.name,
                              ag_shard_used=wrong_shard, ag_shard_correct=right_shard,
                              ar_payload_used=m.psi * L_last.x_numel * B, ar_payload_correct=m.psi * L_i.x_numel * B)
# ---- 2. ResNet final-interface alias justification
from lhas.model.graph import build_structure  # noqa: E402
from lhas.model.branch import GraphModel  # noqa: E402
gs = build_structure(load_workload("resnet50"))
gm = GraphModel(gs, CollectiveService(tp, cp), Settings(B=B, raw_input_grad="retain", norm_policy="sync"))
I = gs.interfaces[gs.iface_in["fc"] if hasattr(gs, "iface_in") else gs.iface_of_input["fc"]]
prod = next(p for p in I.producers if p.kind == "layer")
mask = gm.ag_mask(prod.ref, prod.ops, [(MP, 1)], 4)
received_node1 = gm.psi * I.numel * B * 3 // 4          # shards of the three other MP4 producer nodes
fc = gm.L["fc"]
out["resnet_final_alias"] = dict(delivered_shape=list(I.shape), fc_input_numel=fc.x_numel, mask_node1=float(mask[0]),
                                 received_bytes_node1=int(received_node1),
                                 fc_full_input_reservation=int(gm.psi * fc.x_numel * B))
# ---- 2b. GoogLeNet branch-4 consumer-side pooling alias
gs2 = build_structure(load_workload("googlenet"))
I2 = next(I for I in gs2.interfaces if any(I.consumer_ops.get(c) for c in I.consumers) and I.join == "cat")
c4 = next(c for c in I2.consumers if I2.consumer_ops[c])
out["googlenet_consumer_pool_alias"] = dict(interface=I2.node, consumer=c4, consumer_ops=I2.consumer_ops[c4],
                                           mask_zeroed_for_MP_consumer=True)
# ---- 3. BatchNorm buffers: counter dtype and division by p
L = next(L for L in gm.L.values() if L.norm)
out["bn_buffers"] = dict(layer=L.name, buffers=L.norm["buffers"],
                         modeled_bytes_MP4=float(gm.m(L.name, (MP, 4))[0] - gm.m(L.name, (MP, 4))[0]),
                         code_rule="psi * sum(buffers) / p under MP (counter treated as float32 and divided by p)",
                         saved_statistics_counted=False)
# ---- 4. MCMC incomplete evaluations cached as inf (VGG16 N=256 provisional records)
f = ROOT / "results" / "raw" / "main_retain" / "vgg16_N256.json"
if f.exists():
    d = json.loads(f.read_text())
    out["mcmc_incomplete_vgg16_N256"] = [r.get("incomplete_evaluations") for r in d["baselines"]["flexflow_mcmc"]["runs"]]
# ---- 5. GPipe winners and microbatch set
import glob  # noqa: E402
w = []
for g in sorted(glob.glob(str(ROOT / "results" / "raw" / "gpipe_proposed" / "*.json"))):
    r = json.loads(Path(g).read_text())["result"]
    w.append((Path(g).stem, r["S"], r["R"], r["M"], r["rematerialization"]))
out["gpipe_winners"] = w
src = (ROOT / "src" / "lhas" / "baselines" / "gpipe.py").read_text()
out["gpipe_micro_default"] = [l.strip() for l in src.splitlines() if "micro=" in l][:1]
# ---- 6. OWT: chain step-down vs graph fixed rule
cb = (ROOT / "src" / "lhas" / "baselines" / "chain_baselines.py").read_text()
out["owt_chain_has_stepdown"] = "choice[i] += 1" in cb
# ---- 7. pure_dp_best ignores incomplete
out["pure_dp_best_checks_not_completed"] = "not_completed" in cb.split("def pure_dp_best")[1].split("def ")[0]
# ---- 8. verification thresholds
vp = (ROOT / "src" / "lhas" / "verify_plan.py").read_text()
out["verify_thresholds"] = [l.strip() for l in vp.splitlines() if "max_verify_routes" in l or "MAX_TRANSMISSIONS =" in l][:3]
print(json.dumps(out, indent=1))
