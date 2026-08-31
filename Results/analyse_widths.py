#!/usr/bin/env python3
"""Summarise the loss-transition width for every scan in ``results/``.

Prints, for each scan, the 90%-to-10% transition range in plasma current, that
range as a percentage of the 50% crossing, and — importantly — how many grid
points actually lie inside the transition. A width described by fewer than ~3
interior points is set by the grid, not measured; see docs/DOC_SELF_REVIEW.md
finding 17.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

SCANS = [
    ("scan_pitch_+0.80", "xi = +0.80"),
    ("scan_pitch_+0.45", "xi = +0.45"),
    ("scan_pitch_+0.00", "xi =  0.00"),
    ("scan_pitch_-0.45", "xi = -0.45"),
    ("scan_ensemble", "ensemble"),
]


def summarise(name):
    with open(os.path.join(RES, name + ".json")) as fh:
        d = json.load(fh)
    ip = np.array([p["plasma_current"] for p in d]) / 1e3
    lost = np.array([p["lost_fraction"] for p in d])
    order = np.argsort(ip)
    ip, lost = ip[order], lost[order]
    if lost.max() == 0:
        return None

    def crossing(y):
        for k in range(len(lost) - 1):
            if (lost[k] - y) * (lost[k + 1] - y) <= 0 and lost[k] != lost[k + 1]:
                return ip[k] + (y - lost[k]) * (ip[k + 1] - ip[k]) / (
                    lost[k + 1] - lost[k])
        return float("nan")

    hi = crossing(0.9 * lost.max())
    mid = crossing(0.5 * lost.max())
    lo = crossing(0.1 * lost.max())
    return dict(
        n_points=int(len(ip)),
        n_particles=int(d[0]["n_particles"]),
        max_loss=float(lost.max()),
        upper_kA=float(hi), lower_kA=float(lo), midpoint_kA=float(mid),
        width_kA=float(lo - hi),
        width_pct_of_midpoint=float(100 * (lo - hi) / mid),
        points_inside_transition=int(np.sum((ip > hi) & (ip < lo))),
        highest_current_with_loss_kA=float(ip[lost > 0].max()),
    )


def main():
    out = {}
    header = ("%-12s  N  npart  maxloss   90->10%% (kA)     width  %% of mid"
              "  pts in  threshold" % "scan")
    print(header)
    print("-" * len(header))
    for key, label in SCANS:
        s = summarise(key)
        out[key] = s
        if s is None:
            print("%-12s  never lost at any current in range" % label)
            continue
        print("%-12s %3d  %4d  %6.1f%%  %6.1f -> %6.1f  %6.1f  %6.1f%%  %4d  %7.1f"
              % (label, s["n_points"], s["n_particles"], 100 * s["max_loss"],
                 s["upper_kA"], s["lower_kA"], s["width_kA"],
                 s["width_pct_of_midpoint"], s["points_inside_transition"],
                 s["highest_current_with_loss_kA"]))

    fixed = [out[k] for k, _ in SCANS[:-1] if out.get(k)]
    thr = [s["highest_current_with_loss_kA"] for s in fixed]
    print()
    print("threshold spread across pitch: %.1f -> %.1f kA = %.1fx"
          % (min(thr), max(thr), max(thr) / min(thr)))
    # Only rows with grid points INSIDE the transition are measurements; the
    # others are upper bounds set by the grid, so a ratio taken against them
    # would be a lower bound quoted as a result. Report both, labelled.
    ens = out["scan_ensemble"]["width_pct_of_midpoint"]
    resolved = [s for s in fixed if s["points_inside_transition"] >= 3]
    if resolved:
        sharp_res = min(s["width_pct_of_midpoint"] for s in resolved)
        print("ensemble / sharpest RESOLVED single pitch = %.0fx  (%.1f%% vs %.1f%%)"
              % (ens / sharp_res, ens, sharp_res))
    sharp_all = min(s["width_pct_of_midpoint"] for s in fixed)
    print("ensemble / narrowest bound (unresolved)  >= %.0fx  (%.1f%% vs <=%.1f%%)"
          % (ens / sharp_all, ens, sharp_all))
    with open(os.path.join(RES, "transition_widths.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("-> results/transition_widths.json")


if __name__ == "__main__":
    main()
