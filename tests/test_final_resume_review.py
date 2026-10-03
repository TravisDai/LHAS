"""CPU-only control-flow regressions; synthetic stubs perform no GPU measurements."""
import ast
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]

def cells():
    nb = json.loads((ROOT / 'colab/LHAS_compute_profiling.ipynb').read_text())
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

@pytest.mark.parametrize('field,new', [('driver_version','driver-B'), ('cuda_runtime_torch','cuda-B'),
                                      ('allow_tf32_matmul', True), ('gpu_total_memory_bytes', 24*2**30)])
def test_resume_rejects_changed_recorded_environment_without_writes(tmp_path, field, new):
    s = next(x for x in cells() if '# ---- profile every local shape' in x)
    # Include fingerprint policy declarations but omit the timing loop.
    source = s[s.index('# resume fingerprint:') if '# resume fingerprint:' in s else s.index('# Match the recorded'):s.index('done = done_shape_ids')]
    env = dict(gpu_name='Tesla T4', driver_version='driver-A', cuda_runtime_torch='cuda-A',
               torch='X', cudnn_version=1, gpu_total_memory_bytes=16*2**30, allow_tf32_matmul=False)
    g = dict(CONFIG={'REPS':30}, ENV=env, ORDERED=[], OUT=str(tmp_path), json=json,
             hashlib=hashlib, os=os, datetime=datetime, CHECK={'status':'match'}, EXPORTED={})
    exec(source, g)
    snap = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    g['ENV'] = dict(env, **{field:new})
    with pytest.raises(RuntimeError):
        exec(source, g)
    assert snap == {p.name:p.read_bytes() for p in tmp_path.iterdir()}

def test_reduction_resume_keeps_completed_rows_and_finishes_missing_case(tmp_path):
    code = cells()
    tree = ast.parse(next(s for s in code if 'class AppendCSV' in s))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AppendCSV')
    g = dict(os=os, csv=csv)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<AppendCSV>', 'exec'), g)
    class TensorStub:
        def __getitem__(self, i): return self
        def add_(self, x): return self
    fake = types.SimpleNamespace(randn=lambda *a,**k:TensorStub(), empty=lambda *a,**k:TensorStub(),
             cuda=types.SimpleNamespace(OutOfMemoryError=type('OOM',(Exception,),{})))
    calls = []
    def timed(fn):
        calls.append(1)
        return [.01]*10
    g.update(OUT=str(tmp_path), torch=fake, dev=None, MEM_FRACTION=.8,
             props=types.SimpleNamespace(total_memory=16*2**30), timed=timed,
             quantile=lambda s,q:s[0], release=lambda:None, json=json, math=math, MIN_REPS=10)
    reduction = next(s for s in code if '# ---- local reduction microbenchmark' in s)
    exec(reduction, g)
    path = tmp_path/'reduction_timings.csv'; original=path.read_bytes()
    assert len(calls)==40
    exec(reduction, g)
    assert path.read_bytes()==original and len(calls)==40
    rows=list(csv.DictReader(path.open()))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows[:-1])
    exec(reduction, g)
    out=list(csv.DictReader(path.open()))
    assert len(calls)==41 and len(out)==40
    assert len({(r['V_bytes'],r['a'],r['method']) for r in out})==40
