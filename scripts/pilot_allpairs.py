#!/usr/bin/env python
"""Pilot: construction time of the deterministic first-fit packing for one
all-pairs phase (every ordered pair among nodes 1..p), nominal parameters."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from lhas.params import load_approved
from lhas.transport.schedule import build_phase

out = []
for N, p in [(int(a.split(':')[0]), int(a.split(':')[1])) for a in sys.argv[1:]]:
    tp, _ = load_approved(N)
    P = np.arange(1, p + 1); S = np.repeat(P, p); D = np.tile(P, p); m = S != D
    t0 = time.perf_counter(); ph = build_phase("allpairs", S[m], D[m], S[m], tp); dt = time.perf_counter() - t0
    rec = dict(N=N, p=p, routes=int(m.sum()), hops=int(ph.H.sum()), rounds=int(ph.nrounds), seconds=dt)
    print(json.dumps(rec), flush=True)
    out.append(rec)
