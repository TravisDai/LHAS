#!/usr/bin/env python
"""Chain workloads (AlexNet, VGG16): exact memory-aware LHAS planner, baselines,
optional planner ablations and verification coverage. Defaults come from
configs/nominal_experiment.json.

Example:
  python experiments/run_chain.py --workload alexnet --N 64 --tag final
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict

from common import add_common_args, build_params, provenance, cfg_str, write, rss_peak, load_profile, profile_provenance, NOMINAL  # noqa: E402

from lhas.model.workload import load_workload  # noqa: E402
from lhas.model.chain import ChainModel, Settings  # noqa: E402
from lhas.collectives.service import CollectiveService  # noqa: E402
from lhas.planner.chain_dp import plan_chain  # noqa: E402
from lhas.baselines import common as CC  # noqa: E402
from lhas.baselines import chain_baselines as BL  # noqa: E402
from lhas.transport.schedule import PACK_STATS  # noqa: E402


def plan_record(model, r):
    rec = dict(status=r.status, feasible=r.feasible, stats=r.stats)
    if not r.feasible:
        rec["cost"] = math.inf
        return rec
    rec.update(cost=r.cost, configs=[cfg_str(c) for c in r.configs], layers=[L.name for L in model.L],
               choices=[list(map(list, a.choices)) for a in r.choices], breakdown=r.breakdown,
               memory_max=float(r.memory.max()), memory_node1=float(r.memory[0]),
               memory_budget_after_reserve=float(model.budget))
    return rec


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser())
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    tp, cp = build_params(a)
    wl = load_workload(a.workload)
    st = Settings(B=a.B, alpha=a.alpha, raw_input_grad=a.raw, optimizer_state_factor=a.opt_state,
                  workspace_bytes=a.workspace, norm_policy="none")
    svc = CollectiveService(tp, cp, family=a.family)
    model = ChainModel(wl, svc, st, compute_table=load_profile(a.profile) if a.profile else None)
    out = dict(schema="lhas-run-v2", engine="chain (exact memory-aware label DP)", workload=a.workload, N=a.N,
               B=a.B, family=a.family, settings=asdict(st), transport=asdict(tp), compute=asdict(cp),
               input_mode=cp.input_mode, provenance=provenance(a.workload),
               profile=profile_provenance(a.profile) if a.profile else None)
    t0 = time.perf_counter()
    pre = svc.prefetch(model.demands(), log=lambda m: print(m, flush=True))
    t_constr = time.perf_counter() - t0
    if a.family != "all":
        out["family_candidates"] = svc.family_log() if hasattr(svc, "family_log") else None
    t0 = time.perf_counter()
    r = plan_chain(model)
    t_plan = time.perf_counter() - t0
    out["lhas"] = plan_record(model, r)
    names = [x for x in a.baselines.split(",") if x]
    bl = {}
    t0 = time.perf_counter()
    starts = []
    if "single" in names:
        s = BL.single_node(model)
        s["configs"] = [cfg_str(c) for c in s["configs"]]
        bl["single_node"] = s
    if "dp" in names:
        bl["dp_N"] = plan_record(model, BL.pure_dp(model, a.N)) if a.B % a.N == 0 else \
            dict(status="infeasible", reason="N does not divide B", cost=math.inf)
        if a.B % a.N == 0:
            starts.append(("DP(N)", [(0, a.N)] * len(model.L)))
    if "dpbest" in names:
        summ, best = BL.pure_dp_best(model)
        rec = dict(summ)
        if best is not None:
            rec.update(configs=[cfg_str(c) for c in best.configs])
            starts.insert(0, ("best-common DP", list(best.configs)))
        bl["dp_best"] = rec
    if "owt" in names:
        seq, rr = BL.owt(model)
        if seq is None:
            bl["owt"] = dict(status="infeasible", reason="N does not divide B", cost=math.inf)
        else:
            bl["owt"] = dict(plan_record(model, rr), definition_configs=[cfg_str(c) for c in seq])
            starts.insert(1, ("adapted OWT", seq))
    if "mcmc" in names:
        runs = []
        for seed in range(a.mcmc_seeds):
            m = BL.flexflow_mcmc(model, iterations=a.mcmc_iters, seed=seed, starts=starts)
            m["best"] = [cfg_str(c) for c in m["best"]] if m.get("best") else None
            runs.append(m)
        costs = [x["best_cost"] for x in runs]
        bl["flexflow_mcmc"] = dict(runs=runs, best_cost=min(costs),
                                   status="feasible" if math.isfinite(min(costs)) else "not_established",
                                   **CC.aggregate_mcmc(runs))
    if "ablations" in names:
        ab = {}
        for key, fn in (("strategy_only", BL.strategy_only), ("dp_only_counts", BL.dp_only_counts),
                        ("mp_only_counts", BL.mp_only_counts)):
            tt = time.perf_counter()
            ab[key] = plan_record(model, fn(model))
            ab[key]["seconds"] = time.perf_counter() - tt
        seq, rr = BL.greedy_layerwise(model)
        ab["greedy_layerwise"] = dict(plan_record(model, rr), greedy_configs=[cfg_str(c) for c in seq])
        bl["ablations"] = ab
    t_base = time.perf_counter() - t0
    out["baselines"] = bl
    t0 = time.perf_counter()
    if not a.no_verify and r.feasible:
        from lhas.verify_plan import verify_chain_plan
        out["verification"] = verify_chain_plan(model, r)
    else:
        out["verification"] = dict(summary=dict(not_run=True, reason="--no-verify" if a.no_verify else "no plan"))
    t_ver = time.perf_counter() - t0
    out["runtime"] = dict(construction_seconds=t_constr, construction=pre, plan_seconds=t_plan,
                          baselines_seconds=t_base, verification_seconds=t_ver,
                          total_seconds=time.perf_counter() - t_start, pack=dict(PACK_STATS), peak_rss_bytes=rss_peak())
    f = write(a, out)
    lh = out["lhas"]
    print(f"{a.workload} N={a.N} [{a.tag}]: LHAS {lh['status']} {lh.get('cost')} | "
          + " ".join(f"{k}={v.get('status', '')}:{v.get('cost', v.get('best_cost'))}" for k, v in bl.items()
                     if isinstance(v, dict) and k != "ablations")
          + f" | verify {out['verification']['summary']} | {out['runtime']['total_seconds']:.0f}s -> {f}", flush=True)


if __name__ == "__main__":
    main()
