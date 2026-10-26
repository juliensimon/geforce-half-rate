"""Join gemm.csv rows to telemetry segments and print per-clock FP32/FP16-accumulate ratios.

    python3 analyze.py results/<run>/ [results/<run>/ ...]

A segment is a run of samples at >= 90% utilization; runs since 2026-10-26 09:20 UTC have a 3 s
idle gap between configurations, so segments map one to one onto gemm.csv rows. If the counts
differ, clocks are not printed for that run.
"""

import csv
import sys
from pathlib import Path


def num(s):
    try:
        return float(s.strip().split()[0])
    except (ValueError, IndexError):
        return None


def segments(path):
    segs, cur = [], []
    for r in list(csv.reader(open(path)))[1:]:
        if (num(r[5]) or 0) >= 90:
            cur.append(r)
        else:
            if len(cur) >= 4:
                segs.append(cur)
            cur = []
    if len(cur) >= 4:
        segs.append(cur)
    return segs


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


for run in map(Path, sys.argv[1:]):
    rows = list(csv.DictReader(open(run / "gemm.csv")))
    segs = segments(run / "telemetry.csv")
    limit = next(csv.reader(open(run / "gpu.csv")))  # header
    gpu = list(csv.reader(open(run / "gpu.csv")))[1]
    print(f"\n== {rows[0]['gpu']} | driver {gpu[1].strip()} | power limit {gpu[2].strip()}")
    aligned = len(segs) == len(rows)
    if not aligned:
        print(f"   {len(segs)} segments for {len(rows)} rows: clocks not joined")
    by = {}
    for i, r in enumerate(rows):
        tf = float(r["tflops_median"])
        clk = pw = None
        if aligned:
            clk = median(num(x[1]) for x in segs[i])
            pw = median(num(x[3]) for x in segs[i])
        by[(r["mode"], r["shape"])] = (tf, clk, pw)
    print(f"   {'shape':8s} {'acc16 TF@MHz W':>20s} {'acc32 TF@MHz W':>20s} {'raw':>5s} {'per clk':>7s}")
    for shape in dict.fromkeys(r["shape"] for r in rows):
        a, b = by.get(("fp16_acc16", shape)), by.get(("fp16_acc32", shape))
        if not a or not b:
            continue
        raw = b[0] / a[0]
        fmt = lambda t: f"{t[0]:6.1f}@{t[1]:4.0f} {t[2]:3.0f}" if t[1] else f"{t[0]:6.1f}"
        per = f"{(b[0] / b[1]) / (a[0] / a[1]):.2f}" if a[1] else "-"
        print(f"   {shape:8s} {fmt(a):>20s} {fmt(b):>20s} {raw:5.2f} {per:>7s}")
    for mode in ("bf16_acc32", "tf32", "int8_acc32"):
        vals = [v[0] for (m, _), v in by.items() if m == mode]
        if vals:
            print(f"   {mode}: {min(vals):.1f} to {max(vals):.1f}")
