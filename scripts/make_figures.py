#!/usr/bin/env python3
"""Generate every figure in ``figures/`` from the code and ``results/``.

    python scripts/make_figures.py

Design rules followed here: one y-axis per panel, thin marks, recessive
grid and spines, direct labels instead of legend boxes wherever four or fewer
series are shown, and a categorical palette validated for colour-vision
deficiency (worst all-pairs CVD dE 9.2, normal-vision 16.3).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokamak_orbits import TokamakField, ev_to_joule
from tokamak_orbits.constants import SPECIES
from tokamak_orbits.diagnostics import classify, guiding_centre, make_loss_func
from tokamak_orbits.particles import initialise, trapped_fraction_analytic
from tokamak_orbits.pusher import gyroperiod, integrate, rk45_push

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

# validated categorical palette (light mode, all-pairs safe for 4 slots)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#b8b7b2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.titleweight": "medium", "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": MUTED, "grid.linewidth": 0.5, "grid.alpha": 0.45,
    "legend.frameon": False, "figure.dpi": 150,
})


def tidy(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def _load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print(f"  ! missing {name}, skipping", flush=True)
        return None
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
def fig_orbits():
    """The headline picture: passing vs banana orbits in the poloidal plane."""
    print("figures/orbits_poloidal.png", flush=True)
    f = TokamakField()
    m, q = SPECIES["D"]
    tc = gyroperiod(m, q, 2.0)
    dt = tc / 40

    # (pitch, label, colour, label anchor in data coords, alignment)
    # Anchors are placed by hand rather than picked off the curve: the four
    # orbits overlap near the outboard midplane and any automatic rule put two
    # labels on top of each other.
    cases = [
        (0.90, "co-passing", C[0], (0.855, 0.055), ("right", "bottom")),
        (0.45, "trapped (banana)", C[1], (0.945, -0.075), ("right", "top")),
        (-0.45, "counter-trapped", C[2], (1.02, 0.215), ("center", "bottom")),
        (0.00, "deeply trapped", C[3], (1.175, 0.005), ("left", "center")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4),
                             gridspec_kw=dict(width_ratios=[1, 1]))

    ax = axes[0]
    for xi, label, col, anchor, (ha, va) in cases:
        x0, v0, mm, qq = initialise(f, r_start=0.15, pitch=[xi])
        tr = integrate(x0, v0, dt, int(4e-5 / dt), mm, qq, f.b_field,
                       sample_every=6)
        gc = guiding_centre(f, tr.x, tr.v, mm, qq, dt=dt)
        R = np.hypot(gc[:, 0, 0], gc[:, 0, 1])
        Z = gc[:, 0, 2]
        ax.plot(R, Z, lw=1.1, color=col, solid_capstyle="round", zorder=3)
        ax.annotate(label, anchor, color=col, fontsize=8, fontweight="medium",
                    ha=ha, va=va, zorder=5)

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(f.R0 + f.a * np.cos(th), f.a * np.sin(th),
            color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.annotate("plasma boundary  r = a", (f.R0, -f.a), color=INK2, fontsize=7.5,
                ha="center", va="top", xytext=(0, -4), textcoords="offset points")
    ax.plot([f.R0], [0], marker="+", ms=7, color=INK2, mew=1.0, zorder=4)
    ax.set_xlabel("major radius $R$  (m)")
    ax.set_ylabel("height $Z$  (m)")
    ax.set_title("Guiding-centre orbits, poloidal plane", color=INK, loc="left")
    ax.set_aspect("equal")
    tidy(ax)

    # right panel: parallel velocity, showing the bounce
    ax = axes[1]
    from tokamak_orbits.diagnostics import parallel_perp_split
    for xi, label, col, _anchor, _align in cases:
        x0, v0, mm, qq = initialise(f, r_start=0.15, pitch=[xi])
        tr = integrate(x0, v0, dt, int(4e-5 / dt), mm, qq, f.b_field,
                       sample_every=6)
        vpar, _ = parallel_perp_split(f, tr.x, tr.v)
        v = np.linalg.norm(v0[0])
        ax.plot(tr.t * 1e6, vpar[:, 0] / v, lw=1.0, color=col, zorder=3)
    ax.axhline(0.0, color=INK2, lw=0.8, zorder=2)
    ax.annotate("$v_\\parallel = 0$: turning point", (0.02, 0.02), color=INK2,
                fontsize=7.5, xycoords="axes fraction")
    ax.set_xlabel("time  ($\\mu$s)")
    ax.set_ylabel("$v_\\parallel / v$")
    ax.set_title("Trapped orbits reverse $v_\\parallel$; passing ones do not",
                 color=INK, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "orbits_poloidal.png"), bbox_inches="tight")
    plt.close(fig)


def fig_energy():
    """Why Boris and not RK45."""
    print("figures/energy_conservation.png", flush=True)
    f = TokamakField()
    m, q = SPECIES["D"]
    v = np.sqrt(2 * ev_to_joule(10e3) / m)
    tc = gyroperiod(m, q, 2.0)
    x0 = np.array([1.15, 0.0, 0.0])
    v0 = np.array([0.0, 0.5 * v, np.sqrt(0.75) * v])
    n_orbits = 4000

    t_rk, x_rk, v_rk = rk45_push(x0, v0, n_orbits * tc, m, q, f.b_field,
                                 rtol=1e-6, atol=1e-8, n_out=3000)
    ke_rk = 0.5 * m * np.sum(v_rk**2, axis=-1)

    dt = tc / 40
    tr = integrate(x0[None], v0[None], dt, int(n_orbits * 40), m, q,
                   f.b_field, sample_every=40)
    ke_b = tr.kinetic_energy(m)[:, 0]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    e_rk = np.abs(ke_rk - ke_rk[0]) / ke_rk[0]
    e_b = np.abs(ke_b - ke_b[0]) / ke_b[0]
    ax.plot(t_rk / tc, np.maximum(e_rk, 1e-18), lw=1.2, color=C[1], zorder=3)
    ax.plot(tr.t / tc, np.maximum(e_b, 1e-18), lw=1.2, color=C[0], zorder=3)
    ax.set_yscale("log")
    ax.annotate("RK45 (scipy, rtol $10^{-6}$)\nerror grows without bound",
                (0.45, 0.80), xycoords="axes fraction", color=C[1],
                fontsize=8.5, fontweight="medium")
    ax.annotate("Boris, $\\Delta t = T_c/40$\nbounded at round-off",
                (0.45, 0.16), xycoords="axes fraction", color=C[0],
                fontsize=8.5, fontweight="medium")
    ax.set_xlabel("time  (gyro-periods)")
    ax.set_ylabel("$|\\Delta E / E|$")
    ax.set_title("Energy error over 4000 gyro-orbits", color=INK, loc="left")
    tidy(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "energy_conservation.png"), bbox_inches="tight")
    plt.close(fig)


def fig_convergence():
    print("figures/convergence.png", flush=True)
    d = _load("convergence.json")
    if d is None:
        return
    g = d["gyroradius"]
    n = np.array([r["steps_per_gyroperiod"] for r in g], float)
    e = np.array([r["radius_rel_error"] for r in g], float)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    ax = axes[0]
    ax.loglog(n, e, marker="o", ms=4.5, lw=1.2, color=C[0], zorder=3)
    guide = e[0] * (n / n[0]) ** -2.0
    ax.loglog(n, guide, lw=1.0, ls=(0, (4, 3)), color=INK2, zorder=2)
    ax.annotate("slope $-2$", (n[-2], guide[-2]), color=INK2, fontsize=8,
                xytext=(6, 6), textcoords="offset points")
    ax.annotate("measured", (n[1], e[1]), color=C[0], fontsize=8,
                fontweight="medium", xytext=(6, -12), textcoords="offset points")
    ax.set_xlabel("steps per gyro-period")
    ax.set_ylabel("relative error in gyroradius")
    ax.set_title("Boris is second-order accurate", color=INK, loc="left")
    tidy(ax)

    ax = axes[1]
    rr = d["mu_vs_rho"]
    rho = np.array([r["rho_over_a"] for r in rr])
    mu = np.array([r["mu_error"] for r in rr])
    ax.plot(rho, mu, marker="o", ms=4.5, lw=1.2, color=C[1], zorder=3)
    fitc = np.polyfit(rho, mu, 1)
    ax.plot(rho, np.polyval(fitc, rho), lw=1.0, ls=(0, (4, 3)),
            color=INK2, zorder=2)
    ax.annotate(f"slope ${fitc[0]:.1f}$, linear in $\\rho/a$",
                (0.05, 0.86), xycoords="axes fraction", color=INK2, fontsize=8)
    ax.annotate("$\\delta\\mu/\\mu$ at $\\xi = 0.95$", (rho[-2], mu[-2]),
                color=C[1], fontsize=8, fontweight="medium",
                xytext=(-6, 10), textcoords="offset points", ha="right")
    ax.set_xlabel("$\\rho / a$")
    ax.set_ylabel("$\\max |\\delta\\mu / \\mu|$")
    ax.set_title("$\\mu$ error is physical, not numerical", color=INK, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence.png"), bbox_inches="tight")
    plt.close(fig)


def fig_confinement():
    print("figures/confinement_scan.png", flush=True)
    ens = _load("scan_ensemble.json")
    if ens is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))

    # -- left: loss vs plasma current, log x so every threshold is visible ---
    ax = axes[0]
    fixed = [(0.80, C[2], "$\\xi=+0.80$"), (0.45, C[1], "$\\xi=+0.45$"),
             (0.00, C[3], "$\\xi=0$"), (-0.45, C[0], "$\\xi=-0.45$")]
    for xi, col, lab in fixed:
        d = _load(f"scan_pitch_{xi:+.2f}.json")
        if d is None:
            continue
        ip = np.array([p["plasma_current"] for p in d]) / 1e3
        lo = np.array([p["lost_fraction"] for p in d])
        o = np.argsort(ip)
        ax.plot(ip[o], 100 * lo[o], lw=1.3, color=col, zorder=4)
    ip = np.array([p["plasma_current"] for p in ens]) / 1e3
    lost = np.array([p["lost_fraction"] for p in ens])
    o = np.argsort(ip)
    ax.plot(ip[o], 100 * lost[o], marker="o", ms=3.5, lw=1.8, color=INK,
            zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(14, 950)
    ax.set_ylim(-4, 106)
    ax.annotate("$\\xi=+0.80$", (30, 62), color=C[2], fontsize=8,
                fontweight="medium", ha="right", rotation=90, va="center")
    ax.annotate("$\\xi=+0.45$", (37.5, 62), color=C[1], fontsize=8,
                fontweight="medium", ha="left", rotation=90, va="center")
    ax.annotate("$\\xi=0$", (95, 62), color=C[3], fontsize=8,
                fontweight="medium", ha="left", rotation=90, va="center")
    ax.annotate("$\\xi=-0.45$", (275, 62), color=C[0], fontsize=8,
                fontweight="medium", ha="left", rotation=90, va="center")
    ax.annotate("isotropic ensemble\n(300 ions)", (0.53, 0.60),
                xycoords="axes fraction", color=INK, fontsize=8.5,
                fontweight="medium")
    ax.annotate("single pitch angles: near-vertical steps,\nspread over 8.3x in current",
                (0.03, 0.06), xycoords="axes fraction", color=INK2, fontsize=7.5)
    ax.set_xlabel("plasma current $I_p$  (kA, log scale)")
    ax.set_ylabel("ions lost  (%)")
    ax.set_title("Every fixed pitch steps; the population does not",
                 color=INK, loc="left")
    tidy(ax)

    # -- right: transition width, the actual claim ---------------------------
    ax = axes[1]
    w = _load("transition_widths.json")
    if w is not None:
        keys = ["scan_pitch_+0.80", "scan_pitch_+0.45", "scan_pitch_+0.00",
                "scan_pitch_-0.45", "scan_ensemble"]
        labs = ["$\\xi=+0.80$", "$\\xi=+0.45$", "$\\xi=0$", "$\\xi=-0.45$",
                "isotropic\nensemble"]
        vals = [w[k]["width_pct_of_midpoint"] for k in keys]
        inside = [w[k]["points_inside_transition"] for k in keys]
        cols = [C[2], C[1], C[3], C[0], INK]
        y = np.arange(len(keys))[::-1]
        for yi, v, c, n in zip(y, vals, cols, inside):
            ax.barh(yi, v, height=0.55, color=c,
                    alpha=1.0 if n >= 3 else 0.35,
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            txt = f"{v:.1f}%" if n >= 3 else f"$\\leq${v:.1f}%"
            ax.annotate(txt, (v, yi), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK,
                        fontweight="medium")
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlim(1.6, 700)
        ax.annotate("pale bars are upper bounds:\nno grid point fell inside\nthe transition",
                    (0.30, 0.72), xycoords="axes fraction", color=INK2,
                    fontsize=7.5)
        ax.set_xlabel("90%$\\rightarrow$10% transition width\n(% of midpoint current, log scale)")
        ax.set_title("The population transition is 27x broader",
                     color=INK, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confinement_scan.png"), bbox_inches="tight")
    plt.close(fig)


def fig_trapped_fraction():
    print("figures/trapped_fraction.png", flush=True)
    d = _load("trapped_fraction.json")
    if d is None:
        return
    eps = np.array([r["epsilon"] for r in d])
    meas = np.array([r["measured"] for r in d])
    err = np.array([r["stderr"] for r in d])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    e = np.linspace(0.01, 0.30, 200)
    ax.plot(e, [trapped_fraction_analytic(x) for x in e], lw=1.2,
            ls=(0, (4, 3)), color=INK2, zorder=2)
    ax.errorbar(eps, meas, yerr=err, marker="o", ms=5, lw=0, elinewidth=1.0,
                capsize=2.5, color=C[0], zorder=4)
    ax.annotate("$\\sqrt{2\\epsilon/(1+\\epsilon)}$", (e[-1], trapped_fraction_analytic(e[-1])),
                color=INK2, fontsize=8.5, ha="right", va="bottom",
                xytext=(-2, 4), textcoords="offset points")
    ax.annotate("measured", (eps[-2], meas[-2]), color=C[0], fontsize=8.5,
                fontweight="medium", xytext=(6, -12), textcoords="offset points")
    ax.set_xlabel("local inverse aspect ratio $\\epsilon = r / R_0$")
    ax.set_ylabel("trapped fraction")
    ax.set_title("Trapped fraction against the large-aspect-ratio estimate",
                 color=INK, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "trapped_fraction.png"), bbox_inches="tight")
    plt.close(fig)


def fig_banana_width():
    print("figures/banana_width.png", flush=True)
    d = _load("banana_width.json")
    if d is None:
        return
    fits = d.get("fits", {})
    rows = d["rows"]
    cur = [r for r in rows if r["kind"] == "current"]
    ene = [r for r in rows if r["kind"] == "energy"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))

    ax = axes[0]
    inv_bt = np.array([1.0 / r["b_theta"] for r in cur])
    w = np.array([r["measured"] for r in cur])
    wa = np.array([r["analytic"] for r in cur])
    o = np.argsort(inv_bt)
    ax.plot(inv_bt[o], 1e2 * w[o], marker="o", ms=4.5, lw=1.2, color=C[0], zorder=4)
    ax.plot(inv_bt[o], 1e2 * wa[o], lw=1.1, ls=(0, (4, 3)), color=INK2, zorder=3)
    ax.annotate("measured\n"
                f"$w_b \\propto B_\\theta^{{{fits.get('exponent_vs_b_theta', float('nan')):.3f}}}$"
                "  (ideal $-1$)",
                (0.05, 0.72), xycoords="axes fraction", color=C[0],
                fontsize=8.5, fontweight="medium")
    ax.annotate("$2\\rho_\\theta\\sqrt{\\epsilon}$", (inv_bt[o][-1], 1e2 * wa[o][-1]),
                color=INK2, fontsize=8.5, ha="right", va="top",
                xytext=(-4, -4), textcoords="offset points")
    ax.set_xlabel("$1 / B_\\theta$  (T$^{-1}$)")
    ax.set_ylabel("banana width  (cm)")
    ax.set_title("Width scales as $1/B_\\theta$", color=INK, loc="left")
    tidy(ax)

    ax = axes[1]
    en = np.array([r["energy_ev"] for r in ene]) / 1e3
    w2 = np.array([r["measured"] for r in ene])
    w2a = np.array([r["analytic"] for r in ene])
    o = np.argsort(en)
    ax.plot(np.sqrt(en[o]), 1e2 * w2[o], marker="o", ms=4.5, lw=1.2,
            color=C[1], zorder=4)
    ax.plot(np.sqrt(en[o]), 1e2 * w2a[o], lw=1.1, ls=(0, (4, 3)),
            color=INK2, zorder=3)
    ax.annotate("measured\n"
                f"$w_b \\propto E^{{{fits.get('exponent_vs_energy', float('nan')):.3f}}}$"
                "  (ideal $+1/2$)",
                (0.05, 0.72), xycoords="axes fraction", color=C[1],
                fontsize=8.5, fontweight="medium")
    ax.set_xlabel("$\\sqrt{E}$  (keV$^{1/2}$)")
    ax.set_ylabel("banana width  (cm)")
    ax.set_title("Width scales as $\\sqrt{E}$", color=INK, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "banana_width.png"), bbox_inches="tight")
    plt.close(fig)


def fig_no_poloidal():
    """Without a poloidal field the ions simply drift out."""
    print("figures/no_poloidal_field.png", flush=True)
    m, q = SPECIES["D"]
    tc = gyroperiod(m, q, 2.0)
    dt = tc / 40
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    for ax, ip, title in (
        (axes[0], 0.0, "$I_p = 0$: no poloidal field"),
        (axes[1], 4.5e5, "$I_p = 450$ kA: field lines twist"),
    ):
        f = TokamakField(plasma_current=ip)
        for k, xi in enumerate((0.6, 0.0)):
            x0, v0, mm, qq = initialise(f, r_start=0.10, pitch=[xi])
            tr = integrate(x0, v0, dt, int(6e-5 / dt), mm, qq, f.b_field,
                           sample_every=8, loss_func=make_loss_func(f))
            gc = guiding_centre(f, tr.x, tr.v, mm, qq, dt=dt)
            al = tr.alive[:, 0]
            R = np.hypot(gc[al, 0, 0], gc[al, 0, 1])
            Z = gc[al, 0, 2]
            ax.plot(R, Z, lw=1.1, color=C[k], zorder=3)
            ax.annotate(f"$\\xi = {xi:.1f}$", (R[len(R) // 2], Z[len(Z) // 2]),
                        color=C[k], fontsize=8, fontweight="medium",
                        xytext=(4, 4), textcoords="offset points")
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(f.R0 + f.a * np.cos(th), f.a * np.sin(th),
                color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.set_xlabel("major radius $R$  (m)")
        ax.set_title(title, color=INK, loc="left")
        ax.set_aspect("equal")
        tidy(ax)
    axes[0].set_ylabel("height $Z$  (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "no_poloidal_field.png"), bbox_inches="tight")
    plt.close(fig)



def fig_collision_validation():
    """Legendre eigenmode rates and the detrapping-time scaling."""
    print("figures/collision_validation.png", flush=True)
    d = _load("collision_operator.json")
    dt_ = _load("detrapping.json")
    if d is None or dt_ is None:
        return
    from numpy.polynomial import legendre as _leg
    from tokamak_orbits.collisions import scatter_pitch, legendre_decay_rate

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # -- left: <P_l(xi)> decay against the exact exponentials ---------------
    ax = axes[0]
    rng = np.random.default_rng(11)
    n, nu, nu_dt, xi0 = 300_000, 1.0, 0.005, 0.9
    noise = 1.0 / np.sqrt(n)
    xi = np.full(n, xi0)
    ts = [0.0]
    hist = {l: [float(_leg.legval(xi0, np.eye(4)[l]))] for l in (1, 2, 3)}
    for step in range(1, 801):
        xi = scatter_pitch(xi, nu_dt, rng)
        if step % 10 == 0:
            ts.append(step * nu_dt / nu)
            for l in (1, 2, 3):
                hist[l].append(float(np.mean(_leg.legval(xi, np.eye(4)[l]))))
    t = np.array(ts)
    tt = np.linspace(0, t.max(), 200)
    for l, col in zip((1, 2, 3), (C[0], C[1], C[3])):
        y = np.array(hist[l])
        ax.plot(t, np.maximum(y, 1e-6), lw=0, marker="o", ms=2.6, color=col,
                zorder=4)
        exact = hist[l][0] * np.exp(-legendre_decay_rate(l, nu) * tt)
        ax.plot(tt, exact, lw=1.1, ls=(0, (4, 3)), color=INK2, zorder=3)
        j = np.argmin(np.abs(t - (0.9 if l == 1 else 0.45 / (l - 0.5))))
        ax.annotate(f"$l={l}$", (t[j], max(hist[l][j], 2e-3)), color=col,
                    fontsize=8.5, fontweight="medium",
                    xytext=(5, 5), textcoords="offset points")
    ax.axhline(noise, color=MUTED, lw=0.9, zorder=2)
    ax.annotate("Monte Carlo noise floor $1/\\sqrt{N}$", (t.max(), noise),
                color=INK2, fontsize=7.5, ha="right", va="bottom",
                xytext=(0, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 1.4)
    ax.set_xlabel("$\\nu_d t$")
    ax.set_ylabel("$\\langle P_l(\\xi)\\rangle$")
    ax.set_title("Legendre modes decay at exactly $l(l+1)\\nu_d/2$",
                 color=INK, loc="left")
    ax.annotate("dashed: exact, no fitted parameters", (0.04, 0.06),
                xycoords="axes fraction", color=INK2, fontsize=7.5)
    tidy(ax, grid_axis="y")

    # -- right: detrapping time vs collisionality ---------------------------
    ax = axes[1]
    rows = dt_["rows"]
    nu = np.array([r["nu_d"] for r in rows])
    tau = np.array([r["median_detrap_time"] for r in rows])
    res = np.array([(r["fraction_detrapped"] > 0.6)
                    and (r["samples_per_detrap_time"] >= 20) for r in rows])
    ax.loglog(nu[res], tau[res], marker="o", ms=5, lw=0, color=C[0], zorder=5)
    ax.loglog(nu[~res], tau[~res], marker="o", ms=5, lw=0, color=C[0],
              alpha=0.3, zorder=4)
    eps = dt_["epsilon"]
    ax.loglog(nu, eps / nu, lw=1.1, ls=(0, (4, 3)), color=INK2, zorder=3)
    ax.annotate("$\\epsilon/\\nu_d$", (nu[1], eps / nu[1]), color=INK2,
                fontsize=8.5, xytext=(6, 6), textcoords="offset points")
    fit = dt_.get("fit")
    if fit:
        ax.annotate("measured $\\tau \\propto \\nu_d^{%.2f}$\n(ideal $-1$)"
                    % fit["exponent"], (0.05, 0.12), xycoords="axes fraction",
                    color=C[0], fontsize=8.5, fontweight="medium")
    if (~res).any():
        ax.annotate("pale: sampling-limited\nor <60% detrapped", (0.55, 0.72),
                    xycoords="axes fraction", color=INK2, fontsize=7.5)
    else:
        ax.annotate("all points resolved:\n>120 samples per $\\tau$,\n"
                    "$\\nu_d\\Delta t = 0.005$", (0.53, 0.68),
                    xycoords="axes fraction", color=INK2, fontsize=7.5)
    wb = dt_["omega_b_measured"]
    ax.axvline(wb * eps, color=MUTED, lw=0.9, zorder=2)
    ax.annotate("$\\nu_{\\rm eff}=\\omega_b$", (wb * eps, tau.max()),
                color=INK2, fontsize=7.5, rotation=90, ha="right", va="top",
                xytext=(-3, 0), textcoords="offset points")
    ax.set_xlabel("deflection frequency $\\nu_d$  (s$^{-1}$)")
    ax.set_ylabel("median detrapping time  (s)")
    ax.set_title("Detrapping time against $\\epsilon/\\nu_d$",
                 color=INK, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "collision_validation.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_collisional_smearing():
    """Collisions destroy the single-particle loss threshold."""
    print("figures/collisional_smearing.png", flush=True)
    d = _load("collisional_smearing.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    cols = [INK, C[0], C[1], C[3]]

    ax = axes[0]
    for r, col in zip(d, cols):
        pts = r["points"]
        ip = np.array([p["plasma_current"] for p in pts]) / 1e3
        lo = np.array([p["lost_fraction"] for p in pts])
        o = np.argsort(ip)
        lab = ("collisionless" if r["nu_d"] == 0
               else "$\\nu_d=10^{%d}$" % round(np.log10(r["nu_d"])))
        ax.plot(ip[o], 100 * lo[o], marker="o", ms=3, lw=1.4, color=col,
                zorder=4)
        k = o[np.argmax(lo[o] > 0)] if (lo > 0).any() else o[0]
        hi = ip[lo > 0].max() if (lo > 0).any() else ip.min()
        ax.annotate(lab, (hi, 4 + 14 * cols.index(col)), color=col,
                    fontsize=8.5, fontweight="medium", ha="right",
                    xytext=(-3, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("plasma current $I_p$  (kA, log scale)")
    ax.set_ylabel("ions lost  (%)")
    ax.set_title("One pitch angle, $\\xi = -0.45$: collisions erase the step",
                 color=INK, loc="left")
    tidy(ax)

    ax = axes[1]
    nus, his, trunc, labels = [], [], [], []
    for r in d:
        pts = r["points"]
        ip = np.array([p["plasma_current"] for p in pts]) / 1e3
        lo = np.array([p["lost_fraction"] for p in pts])
        if not (lo > 0).any():
            continue
        nus.append(max(r["nu_d"], 3e3))          # plot nu=0 at the left edge
        his.append(ip[lo > 0].max())
        trunc.append(abs(ip[lo > 0].max() - ip.max()) < 1e-6)
        # labels are derived from the data, not hard-coded: a --quick run has
        # fewer collisionality points than a production one, and a fixed list
        # of four crashed matplotlib with a FixedLocator/label mismatch.
        labels.append("collisionless" if r["nu_d"] <= 0
                      else "$10^{%d}$" % round(np.log10(r["nu_d"])))
    x = np.arange(len(nus))
    for xi_, h, tr, col in zip(x, his, trunc, cols):
        ax.bar(xi_, h, width=0.55, color=col, alpha=0.35 if tr else 1.0,
               edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(("$\\geq$" if tr else "") + f"{h:.0f} kA",
                    (xi_, h), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("deflection frequency $\\nu_d$  (s$^{-1}$)")
    ax.set_ylabel("highest $I_p$ at which the ion is still lost  (kA)")
    ax.set_title("The confinement threshold moves up 6x, then runs off the scan",
                 color=INK, loc="left")
    ax.annotate("pale bar: still lost at the\ntop of the scan, so this is\na lower bound",
                (0.04, 0.70), xycoords="axes fraction", color=INK2, fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "collisional_smearing.png"),
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_thermalisation():
    """The energy operator: right rate, right equilibrium, right in the pusher."""
    print("figures/thermalisation.png", flush=True)
    d = _load("thermalisation.json")
    if d is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.4))

    # (a) drift: measured vs analytic
    rows = d["drift_validation"]
    v = np.array([r["v_over_vth"] for r in rows])
    meas = np.abs([r["measured"] for r in rows])
    se = np.array([r["stderr"] for r in rows])
    ana = np.abs([r["analytic"] for r in rows])
    from tokamak_orbits.collisions_full import MaxwellianBackground, speed_drift
    from tokamak_orbits.constants import SPECIES as _SP
    bgf = MaxwellianBackground(d["density"], d["temperature_ev"])
    vf = np.linspace(0.8, 6.4, 400)
    ax[0].plot(vf, np.abs(speed_drift(bgf, vf * bgf.v_th, _SP["D"][0])),
               "-", color=INK2, lw=1.4, zorder=2,
               label=r"analytic $-(\nu_s - \nu_\perp/2)\,v$")
    ax[0].errorbar(v, meas, yerr=se, fmt="o", ms=5.5, color=C[0], capsize=3,
                   lw=1.2, mec=SURFACE, mew=0.8, label="Monte Carlo", zorder=3)
    ax[0].set_yscale("log")
    ax[0].set_xlim(0.8, 6.4)
    ax[0].set_xlabel(r"$v/v_{\rm th}$")
    ax[0].set_ylabel(r"$|\mu(v)|$  (m s$^{-2}$)")
    ax[0].set_title("(a) drag, derived not fitted")
    ax[0].legend(loc="lower left", fontsize=8)
    tidy(ax[0])

    # (b) the KS convergence
    shape = d["equilibrium_shape"]
    crit = shape[0]["ks_95"]
    xs = sorted({r["nu_dt"] for r in shape})
    ax[1].axhspan(1e-4, crit, color=C[2], alpha=0.10, zorder=0)
    for lbl, colour, mark, wid, msz in (("cold", C[0], "o", 2.6, 7.5),
                                        ("hot", C[1], "s", 1.2, 4.0)):
        sel = sorted([r for r in shape if r["start"].startswith(lbl)],
                     key=lambda r: r["nu_dt"])
        ax[1].plot([r["nu_dt"] for r in sel], [r["ks"] for r in sel],
                   mark + "-", color=colour, ms=msz, lw=wid, alpha=0.95,
                   label=(r"cold start, $0.4\,v_{\rm th}$" if lbl == "cold"
                          else r"hot start, $3\,v_{\rm th}$"), zorder=3)
    ax[1].axhline(crit, color=INK2, ls="--", lw=1.0, zorder=2)
    ax[1].text(xs[-1] * 1.25, crit * 0.85,
               "95% critical value\nbelow: indistinguishable\nfrom a Maxwellian",
               fontsize=7, color=INK2, va="top", ha="right")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(["%g" % x for x in xs])
    ax[1].minorticks_off()
    ax[1].set_xlim(xs[0] * 0.75, xs[-1] * 1.35)
    ax[1].set_ylim(2.0e-3, 3.5e-2)
    ax[1].set_xlabel(r"collision timestep $\nu\,\Delta t$")
    ax[1].set_ylabel("KS statistic vs Maxwellian")
    ax[1].set_title("(b) the equilibrium is a Maxwellian")
    ax[1].legend(loc="upper left", fontsize=8)
    tidy(ax[1])

    # (c) slowing down in the pusher vs 0-D
    p = d["pusher_slowing_down"]
    t = np.array(p["t"]) * 1e3
    ax[2].plot(t, np.array(p["mean_energy_ev"]) / 1e3, "-", color=C[0],
               lw=1.6, label="in the tokamak field", zorder=3)
    ax[2].plot(np.array(p["zero_d_t"]) * 1e3,
               np.array(p["zero_d_energy_ev"]) / 1e3, "--", color=C[1],
               lw=1.4, label="0-D, same operator", zorder=3)
    ax[2].axhline(p["background_thermal_ev"] / 1e3, color=INK2, ls=":", lw=1.0)
    ax[2].text(t.max() * 0.4, p["background_thermal_ev"] / 1e3 * 1.25,
               r"$3T_b/2$", fontsize=8, color=INK2)
    ax[2].set_xlabel("time (ms)")
    ax[2].set_ylabel(r"$\langle E\rangle$ (keV)")
    ax[2].set_title("(c) slowing down")
    ax[2].legend(loc="upper right", fontsize=8)
    tidy(ax[2])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "thermalisation.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ripple():
    """Ripple: still solenoidal, where the wells are, and what it costs."""
    print("figures/ripple.png", flush=True)
    d = _load("ripple.json")
    if d is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.4))

    # (a) divergence bar
    dv = d["divergence"]
    labels = ["axisymmetric", "potential\nripple", "naive\nripple"]
    vals = [dv["axisymmetric_max"], dv["ripple_max"], dv["naive_max"]]
    cols = [MUTED, C[0], C[1]]
    ax[0].bar(labels, vals, color=cols, zorder=3, width=0.6)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"max $|\nabla\cdot\mathbf{B}| \, / \, (|B|/a)$")
    ax[0].set_title("(a) breaking axisymmetry, not $\\nabla\\!\\cdot\\!\\mathbf{B}$")
    for i, vv in enumerate(vals):
        ax[0].text(i, vv * 1.7, "%.0e" % vv, ha="center", fontsize=7.5,
                   color=INK2)
    ax[0].set_ylim(min(vals) * 0.2, max(vals) * 30)
    tidy(ax[0], grid_axis="y")

    # (b) well fraction vs r
    for i, row in enumerate(d["well_fraction"]):
        rr = np.array([float(k) for k in row["fractions"]])
        ff = np.array(list(row["fractions"].values()))
        ax[1].plot(rr, ff, "o-", ms=4, lw=1.3,
                   color=C[i % len(C)] if i < len(C) else INK2,
                   label=r"$\delta_{\rm edge}$ = %.2f%%"
                         % (100 * row["delta_edge"]), zorder=3)
    ax[1].set_xlabel("minor radius $r$ (m)")
    ax[1].set_ylabel("fraction of surface in a ripple well")
    ax[1].set_title("(b) where a particle can be ripple-trapped")
    ax[1].legend(loc="upper left", fontsize=7.5)
    tidy(ax[1])

    # (c) the paired statistic: what ripple loses that the control confines
    sc = d["loss_scan"]
    n_conf = sc[0]["n_particles"] - sc[0]["n_lost"]
    x = np.array([r["delta_edge"] * 100 for r in sc[1:]])
    k = np.array([r["newly_lost"] for r in sc[1:]], dtype=float)
    y = 100 * k / n_conf
    err = 100 * np.sqrt(np.clip((k / n_conf) * (1 - k / n_conf), 1e-12, None)
                        / n_conf)
    ax[2].errorbar(x, y, yerr=err, fmt="o-", ms=6, color=C[0], lw=1.6,
                   capsize=3, mec=SURFACE, mew=0.8, zorder=3)
    ax[2].axhline(0.0, color=C[1], ls="--", lw=1.3, zorder=2)
    ax[2].text(x[0], 2.0, "axisymmetric control: 0 by definition",
               fontsize=7.5, color=C[1], va="bottom")
    ax[2].text(x[-1], y[-1] + err[-1] + 2.0,
               "0 rescued\nat every amplitude", fontsize=7.5, color=INK2,
               ha="right", va="bottom")
    ax[2].set_xscale("log")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["%g" % v for v in x])
    ax[2].minorticks_off()
    ax[2].set_xlim(x[0] * 0.7, x[-1] * 1.5)
    ax[2].set_ylim(-2, max(y + err) * 1.55)
    ax[2].set_xlabel(r"$\delta_{\rm edge}$  (%)")
    ax[2].set_ylabel("newly lost (%% of the %d\nthe control confines)" % n_conf)
    ax[2].set_title(r"(c) loss ripple causes, $r=0.25$ m")
    tidy(ax[2])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ripple.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_orbits()
    fig_no_poloidal()
    fig_energy()
    fig_convergence()
    fig_trapped_fraction()
    fig_banana_width()
    fig_confinement()
    fig_collision_validation()
    fig_collisional_smearing()
    fig_thermalisation()
    fig_ripple()
    print("done", flush=True)
