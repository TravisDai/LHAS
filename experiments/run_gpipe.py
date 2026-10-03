#!/usr/bin/env python
"""Optical-adapted GPipe baseline (final contract D13; lhas.baselines.gpipe) for the
chain workloads, under the nominal specification (configs/nominal_experiment.json).

Example:
  python experiments/run_gpipe.py --workload alexnet --N 64 --tag final_gpipe
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict

from common import add_common_args, build_params, provenance, write, rss_peak, NOMINAL  # noqa: E402

from lhas.model.workload import load_workload  # noqa: E402
from lhas.model.chain import ChainModel, Settings  # noqa: E402
from lhas.collectives.service import CollectiveService  # noqa: E402
from lhas.baselines.gpipe import gpipe, MICRO  # noqa: E402
from lhas.transport.schedule import PACK_STATS  # noqa: E402


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser())
    ap.set_defaults(tag="final_gpipe", baselines="")
    ap.add_argument("--micro", default=",".join(str(m) for m in NOMINAL["gpipe"]["micro"]))
    ap.add_argument("--remat", default="both", choices=["both", "off", "on"])
    a = ap.parse_args(argv)
    if a.profile is not None:
        # T5: the GPipe evaluator computes stage-microbatch times analytically; a profiled
        # table would need stage-local microbatch shapes, which are not supported yet
        ap.error("GPipe does not yet support profiled microbatch computation; omit --profile.")
    t_start = time.perf_counter()
    tp, cp = build_params(a)
    st = Settings(B=a.B, alpha=a.alpha, raw_input_grad=a.raw, optimizer_state_factor=a.opt_state,
                  workspace_bytes=a.workspace, norm_policy="none")
    svc = CollectiveService(tp, cp, family=a.family)
    model = ChainModel(load_workload(a.workload), svc, st)
    micro = tuple(int(x) for x in a.micro.split(","))
    remats = dict(both=(False, True), off=(False,), on=(True,))[a.remat]
    t0 = time.perf_counter()
    best, stats = gpipe(model, micro=micro, remats=remats)
    secs = time.perf_counter() - t0
    if best is None:
        # F3: an exhausted search is a proof of infeasibility only if no candidate evaluation
        # was stopped by the All-reduce selection node limit
        unresolved = stats.get("allreduce_selection_unresolved", 0) > 0
        res = dict(status="not_established" if unresolved else "infeasible", cost=math.inf,
                   reason=("no feasible incumbent; one or more candidate evaluations remain unresolved" if unresolved
                           else "no configuration in the searched space satisfies the modeled budget"))
    else:
        res = dict(status=best.status, cost=best.cost, S=best.S, R=best.R, active_nodes=best.P, M=best.M,
                   rematerialization=best.remat, degenerate_dp=best.degenerate_dp, stages=best.stages,
                   breakdown=best.breakdown, memory_max=best.memory_max,
                   memory_budget_after_reserve=float(model.budget))
    out = dict(schema="lhas-gpipe-v2", contract="D13 (final)", workload=a.workload, N=a.N, B=a.B,
               settings=asdict(st), transport=asdict(tp), compute=asdict(cp), provenance=provenance(a.workload),
               search_space=dict(stages="1..number of weighted layers (S=1 labeled degenerate DP)",
                                 replicas="R | B, R*S <= N", micro=list(micro), rematerialization=list(remats),
                                 partition="min-max linear partition of per-example stage compute"),
               result=res, stats=stats,
               runtime=dict(search_seconds=secs, total_seconds=time.perf_counter() - t_start,
                            pack=dict(PACK_STATS), peak_rss_bytes=rss_peak()))
    f = write(a, out)
    print(f"{a.workload} N={a.N} GPipe: {res['status']} {res['cost']}"
          + ("" if best is None else f" S={best.S} R={best.R} P={best.P} M={best.M} remat={best.remat}")
          + f" | {stats} | {secs:.0f}s -> {f}", flush=True)


if __name__ == "__main__":
    main()
