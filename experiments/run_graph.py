#!/usr/bin/env python
"""Typed-interface engine (all four workloads; required for GoogLeNet and ResNet-50):
exact search of the additive objective with every operation at its minimum-charge
schedule alternative, then the corrected reservation ledger (D8). Statuses (D10):
the LHAS plan is `feasible` (and then the memory-constrained optimum within the
candidate family) when the unconstrained minimizer satisfies the budget; otherwise
`not_established`. Baselines use the same evaluator and definitions as the chain
runner (lhas.baselines.common). Defaults come from configs/nominal_experiment.json.

Example:
  python experiments/run_graph.py --workload googlenet --N 64 --tag final
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict

from common import add_common_args, build_params, provenance, cfg_str, write, rss_peak, load_profile, profile_provenance  # noqa: E402

from lhas.model.workload import load_workload  # noqa: E402
from lhas.model.graph import build_structure  # noqa: E402
from lhas.model.chain import Settings, DP  # noqa: E402
from lhas.model.branch import GraphModel  # noqa: E402
from lhas.collectives.service import CollectiveService  # noqa: E402
from lhas.planner.graph_dp import plan_graph  # noqa: E402
from lhas.baselines import common as CC  # noqa: E402
from lhas.baselines import graph_baselines as GB  # noqa: E402
from lhas.baselines import common as C  # noqa: E402
from lhas.transport.schedule import PACK_STATS  # noqa: E402


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser())
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    tp, cp = build_params(a)
    wl = load_workload(a.workload)
    norm = a.norm if any(L.norm for L in wl.layers) else "none"
    st = Settings(B=a.B, alpha=a.alpha, raw_input_grad=a.raw, optimizer_state_factor=a.opt_state,
                  workspace_bytes=a.workspace, norm_policy=norm)
    svc = CollectiveService(tp, cp, keep_structures=False, family=a.family)
    t0 = time.perf_counter()
    model = GraphModel(build_structure(wl), svc, st, compute_table=load_profile(a.profile) if a.profile else None)
    t_constr = time.perf_counter() - t0
    out = dict(schema="lhas-run-v2", engine="typed-interface (exact additive minimizer, min-charge alternatives, "
                                            "reservation check)",
               workload=a.workload, N=a.N, B=a.B, family=a.family, settings=asdict(st), transport=asdict(tp),
               compute=asdict(cp), input_mode=cp.input_mode, provenance=provenance(a.workload),
               profile=profile_provenance(a.profile) if a.profile else None)
    t0 = time.perf_counter()
    r = plan_graph(model)
    t_plan = time.perf_counter() - t0
    lh = dict(status=r.status, cost=r.cost if r.memory_feasible else math.inf, cost_unconstrained_minimizer=r.cost,
              memory_feasible=r.memory_feasible, memory_max=float(r.memory.max()), memory_node1=float(r.memory[0]),
              memory_budget_after_reserve=float(model.budget), configs=[cfg_str(r.configs[n]) for n in model.order],
              layers=list(model.order), stats=r.stats,
              certificate=("unconstrained minimizer satisfies the budget: memory-constrained optimum within the "
                           "candidate family" if r.memory_feasible else
                           "memory-constrained optimum not established"))
    out["lhas"] = lh
    names = [x for x in a.baselines.split(",") if x]
    bl = {}
    starts = []
    t0 = time.perf_counter()
    if "single" in names:
        bl["single_node"] = GB.single_node(model)
    if "dp" in names:
        if a.B % a.N == 0:
            bl["dp_N"] = GB.pure_dp(model, a.N)
            starts.append(("DP(N)", [(DP, a.N)] * len(model.order)))
        else:
            bl["dp_N"] = dict(status=C.INFEASIBLE, reason="N does not divide B", cost=math.inf)
    if "dpbest" in names:
        rec = GB.pure_dp_best(model)
        bl["dp_best"] = rec
        if rec["p"] is not None:
            starts.insert(0, ("best-common DP", [(DP, rec["p"])] * len(model.order)))
    if "owt" in names:
        seq = C.owt_configs([model.L[n] for n in model.order], a.N, a.B)
        if seq is None:
            bl["owt"] = dict(status=C.INFEASIBLE, reason="N does not divide B", cost=math.inf)
        else:
            bl["owt"] = GB.fixed(model, dict(zip(model.order, seq)))
            starts.insert(min(1, len(starts)), ("adapted OWT", seq))
    if "mcmc" in names:
        runs = []
        for seed in range(a.mcmc_seeds):
            m = GB.flexflow_mcmc(model, iterations=a.mcmc_iters, seed=seed, starts=starts)
            m["best"] = [cfg_str(c) for c in m["best"]] if m.get("best") else None
            runs.append(m)
        costs = [x["best_cost"] for x in runs]
        bl["flexflow_mcmc"] = dict(runs=runs, best_cost=min(costs),
                                   status=C.FEASIBLE if math.isfinite(min(costs)) else C.NOT_ESTABLISHED,
                                   **CC.aggregate_mcmc(runs))
    t_base = time.perf_counter() - t0
    out["baselines"] = bl
    t0 = time.perf_counter()
    if not a.no_verify:
        from lhas.verify_plan import verify_graph_plan
        out["verification"] = verify_graph_plan(model, r.ops)
    else:
        out["verification"] = dict(summary=dict(not_run=True, reason="--no-verify"))
    t_ver = time.perf_counter() - t0
    out["runtime"] = dict(construction_seconds=t_constr, plan_seconds=t_plan, baselines_seconds=t_base,
                          verification_seconds=t_ver, total_seconds=time.perf_counter() - t_start,
                          pack=dict(PACK_STATS), peak_rss_bytes=rss_peak())
    if a.family != "all":
        out["family_candidates"] = svc.family_log()
    f = write(a, out)
    print(f"{a.workload} N={a.N} [{a.tag}]: LHAS {lh['status']} {lh['cost_unconstrained_minimizer']:.6f} | "
          + " ".join(f"{k}={v.get('status', '')}:{v.get('cost', v.get('best_cost'))}" for k, v in bl.items()
                     if isinstance(v, dict))
          + f" | verify {out['verification']['summary']} | {out['runtime']['total_seconds']:.0f}s -> {f}", flush=True)


if __name__ == "__main__":
    main()
