#!/usr/bin/env python
"""Write the command lists of the final experiment set (configs/nominal_experiment.json)
to results/queues/*.txt; run each with scripts/run_list.sh <file> <logname>.

  main       N = 64, 128, 256, 512; all baselines; verification of every selected operation
  gpipe      AlexNet and VGG16, N = 64..512 (and AlexNet N = 1024, supplementary)
  ablations  planner ablations (chain workloads), N = 64 and 256
  family     collective-family comparison (all / one-stage / deepest), N = 64, candidate IDs exported
  batch      B = 256 at N = 64: LHAS, DP(N), best-common DP, OWT (D1)
  sens       one-parameter-at-a-time sensitivity at N = 64 (DP(N) and OWT with the same model)
  cold       planner runtime with an empty packing cache (D16: cold- vs reused-cache construction time)
  supp       AlexNet N = 1024 (supplementary, D16)
  profiled   supplementary planning sensitivity with measured Tesla T4 computation (D17 option b):
             N = 64 and 256, LHAS, DP(N), best-common DP, OWT; beta_red = measured T4 model-equivalent
             reduction rate (median fused-sum rate for V >= 16 MiB, scripts/analyse_profile.py)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOM = json.loads((ROOT / "configs" / "nominal_experiment.json").read_text())
CHAIN, BRANCH = ("alexnet", "vgg16"), ("googlenet", "resnet50")


def runner(w):
    return "experiments/run_chain.py" if w in CHAIN else "experiments/run_graph.py"


def cmd(w, N, tag, *extra):
    return " ".join(["python", runner(w), "--workload", w, "--N", str(N), "--tag", tag, *extra])


SENS = [("alpha_0.25", "--alpha 0.25"), ("alpha_0.5", "--alpha 0.5"), ("alpha_0.75", "--alpha 0.75"),
        ("alpha_1.0", "--alpha 1.0"), ("rho_0.25", "--rho 0.25"), ("rho_0.75", "--rho 0.75"), ("rho_1.0", "--rho 1.0"),
        ("redeff_0.25", "--red-eff 0.25"), ("redeff_1.0", "--red-eff 1.0"), ("tsetup_5e-6", "--t-setup 5e-6"),
        ("tsetup_100e-6", "--t-setup 100e-6"), ("lamW_32", "--lam 32 --w 32"), ("lamW_128", "--lam 128 --w 128"),
        ("lamW_256", "--lam 256 --w 256"), ("reach_15", "--reach 15"), ("reach_16", "--reach 16"), ("reach_unrestricted_nopeer", "--reach 0 --dpeer 0"), ("eps_0.5", "--eps 0.5"),
        ("whole_item", "--kappa 0"), ("raw_retain", "--raw retain"), ("reserve_0", "--reserve 0"),
        ("no_momentum", "--opt-state 0")]


def main():
    q = {}
    Ns = NOM["N_primary"]
    q["chain_main"] = [cmd(w, N, "final") for w in CHAIN for N in Ns]
    q["graph_main"] = [cmd(w, N, "final") for w in BRANCH for N in Ns]
    q["gpipe"] = [f"python experiments/run_gpipe.py --workload {w} --N {N} --tag final_gpipe"
                  for w in NOM["gpipe"]["workloads"] for N in Ns]
    q["ablations"] = [cmd(w, N, "final_ablations", "--baselines ablations --no-verify") for w in CHAIN for N in (64, 256)]
    q["family"] = [cmd(w, 64, f"final_family/{f}", f"--family {f} --baselines ''")
                   for w in CHAIN + BRANCH for f in ("all", "one-stage", "deepest")]
    bs = NOM["batch_sensitivity"]
    q["batch"] = [cmd(w, bs["N"], f"final_batch/B{bs['B']}", f"--B {bs['B']} --baselines dp,dpbest,owt")
                  for w in CHAIN + BRANCH]
    q["sens_chain"] = [cmd(w, 64, f"final_sens/{name}", args, "--baselines dp,owt") for w in CHAIN
                       for name, args in [("nominal", "")] + SENS]
    q["sens_graph"] = [cmd(w, 64, f"final_sens/{name}", args, "--baselines dp,owt") for w in BRANCH
                       for name, args in [("nominal", "")] + SENS]
    cold = "rm -rf results/cache/cold && mkdir -p results/cache/cold && LHAS_PACK_CACHE_DIR=results/cache/cold "
    q["cold"] = [cold + cmd(w, N, "final_cold", "--baselines '' --no-verify")
                 for w, N in (("alexnet", 256), ("vgg16", 256), ("googlenet", 128), ("resnet50", 128), ("alexnet", 512))]
    q["supp"] = [cmd("alexnet", 1024, "final_supp"),
                 "python experiments/run_gpipe.py --workload alexnet --N 1024 --tag final_supp_gpipe"]
    prof = "--profile configs/profiles/profiled_Tesla_T4_B1024.json --red-rate 218.6e9 --baselines dp,dpbest,owt"
    q["profiled"] = [cmd(w, N, "final_profiled_T4", prof) for N in (64, 256) for w in CHAIN + BRANCH]
    d = ROOT / "results" / "queues"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in q.items():
        (d / f"{k}.txt").write_text("\n".join(v) + "\n")
        print(f"{k}: {len(v)} commands")


if __name__ == "__main__":
    main()
