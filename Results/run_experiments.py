#!/usr/bin/env python3
"""Run every production experiment and write raw results to ``results/``.

This is the single entry point that regenerates every number quoted in
``RESULTS.md``. It is deliberately slow and deterministic: fixed seeds, fixed
timestep policy, no adaptive shortcuts.

    python scripts/run_experiments.py            # everything (~40 min)
    python scripts/run_experiments.py --quick    # coarse version (~4 min)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokamak_orbits import TokamakField, ev_to_joule
from tokamak_orbits.constants import SPECIES
from tokamak_orbits.diagnostics import (
    banana_width_analytic, classify, guiding_centre, make_loss_func,
)
from tokamak_orbits.particles import (
    initialise, sample_pitch, trapped_fraction_analytic, trapping_boundary_pitch,
)
from tokamak_orbits.diagnostics import is_trapped, parallel_perp_split
from tokamak_orbits.pusher import gyroperiod, integrate
from tokamak_orbits.collisions import (
    PitchAngleCollisions, bounce_frequency, effective_detrapping_frequency,
    legendre_decay_rate, scatter_pitch,
)
from tokamak_orbits.scan import run_ensemble, save_scan, scan_plasma_current

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def _dump(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"  -> {path}", flush=True)


# --------------------------------------------------------------------------
def exp_ensemble_scan(quick):
    """A. Isotropic ensemble, confinement vs plasma current."""
    print("\n[A] ensemble confinement scan (isotropic pitch)", flush=True)
    n = 60 if quick else 300
    t_max = 6e-5 if quick else 1.5e-4
    currents = np.linspace(0.6e5, 9.0e5, 8 if quick else 24)
    pts = scan_plasma_current(currents, n_particles=n, t_max=t_max, seed=12345)
    save_scan(pts, os.path.join(RESULTS, "scan_ensemble.json"))
    print(f"  -> {RESULTS}/scan_ensemble.json", flush=True)


def exp_fixed_pitch_scans(quick):
    """B. Single-particle response at fixed pitch -- is the threshold sharp?"""
    print("\n[B] fixed-pitch scans (gyrophase-averaged single particle)", flush=True)
    n = 8 if quick else 48
    t_max = 6e-5 if quick else 1.5e-4
    if quick:
        grids = {xi: np.linspace(0.2e5, 9.0e5, 8) for xi in (0.45, -0.45, 0.80, 0.00)}
    else:
        # Two-stage per pitch. A single uniform grid over 60-900 kA put only ONE
        # point inside the xi = -0.45 transition, so the quoted width was an
        # interpolation through a single sample and was resolution-limited
        # rather than measured. The scan also started at 60 kA, above the
        # co-going thresholds, which is why they looked like "never lost".
        # Coarse stage locates the transition; dense stage resolves it.
        coarse = np.concatenate([
            np.linspace(0.15e5, 1.0e5, 8),      # co-going thresholds live here
            np.linspace(1.2e5, 9.0e5, 14),
        ])
        grids = {}
        for xi in (0.45, -0.45, 0.80, 0.00):
            probe = scan_plasma_current(
                coarse, n_particles=16, t_max=t_max, seed=999,
                fixed_pitch=xi, verbose=False,
            )
            ip = np.array([p.plasma_current for p in probe])
            lost = np.array([p.lost_fraction for p in probe])
            hit = ip[lost > 0]
            miss = ip[lost == 0]
            if hit.size and miss.size:
                lo, hi = hit.max(), miss[miss > hit.max()].min() if (miss > hit.max()).any() else hit.max() * 1.5
                span = hi - lo
                dense = np.linspace(max(1e3, lo - 0.6 * span), hi + 0.6 * span, 22)
            else:
                dense = np.array([])
            grids[xi] = np.unique(np.concatenate([coarse, dense]))
            print(f"  xi = {xi:+.2f}: transition bracketed in "
                  f"[{lo/1e3:.0f}, {hi/1e3:.0f}] kA, refining with 22 points"
                  if hit.size and miss.size else
                  f"  xi = {xi:+.2f}: no transition found in range", flush=True)

    for xi in (0.45, -0.45, 0.80, 0.00):
        print(f"  xi = {xi:+.2f}  ({len(grids[xi])} currents)", flush=True)
        pts = scan_plasma_current(
            grids[xi], n_particles=n, t_max=t_max, seed=999,
            fixed_pitch=xi, verbose=False,
        )
        save_scan(pts, os.path.join(RESULTS, f"scan_pitch_{xi:+.2f}.json"))
        lo = [p for p in pts if p.lost_fraction > 0]
        print(f"    max loss {max(p.lost_fraction for p in pts):.0%}; "
              f"highest current with any loss: "
              f"{max(p.plasma_current for p in lo)/1e3:.1f} kA" if lo
              else "    never lost", flush=True)


def exp_trapped_fraction(quick):
    """C. Measured trapped fraction vs sqrt(2 eps / (1 + eps))."""
    print("\n[C] trapped fraction vs analytic", flush=True)
    field = TokamakField()
    n = 100 if quick else 600
    t_max = 3e-5 if quick else 8e-5
    rows = []
    for r in (0.05, 0.10, 0.15, 0.20, 0.25):
        pitches = sample_pitch(n, rng=2024, mode="uniform_xi")
        traj, s, m, q = run_ensemble(field, pitches, r_start=r, t_max=t_max)
        keep = ~s.lost
        meas = float(s.trapped[keep].mean()) if keep.any() else float("nan")
        # binomial standard error on the measured fraction
        se = float(np.sqrt(max(meas * (1 - meas), 1e-12) / max(keep.sum(), 1)))
        rows.append(dict(
            r=r, epsilon=r / field.R0,
            measured=meas, stderr=se,
            analytic=trapped_fraction_analytic(r / field.R0),
            xi_crit_numeric=trapping_boundary_pitch(field, r),
            n_used=int(keep.sum()), n_lost=int(s.lost.sum()),
            max_mu_error=float(s.mu_error.max()),
        ))
        print(f"  r={r:.2f}  measured={meas:.4f}+-{se:.4f}  "
              f"analytic={rows[-1]['analytic']:.4f}  lost={rows[-1]['n_lost']}", flush=True)
    _dump("trapped_fraction.json", rows)


# Launch pitch for the banana-width study. NOT 0.45: that sits only 12% inside
# the trapped/passing boundary (xi_crit = 0.511 at r = 0.15), and as the plasma
# current is lowered the growing orbit width pushes the particle across the
# boundary entirely -- at Ip = 250 kA it registers zero bounces and is not
# trapped, so the "banana width" measured there is not a banana width. xi = 0.30
# is comfortably inside and gives a monotonic, clean scan. See
# docs/DOC_SELF_REVIEW.md finding 11.
BANANA_PITCH = 0.30


def exp_banana_width(quick):
    """D. Banana width scaling with 1/B_theta and with sqrt(E)."""
    print("\n[D] banana width scaling", flush=True)
    rows = []
    t_max = 3e-5 if quick else 6e-5
    # vary plasma current at fixed energy
    for ip in (2.5e5, 3.0e5, 3.5e5, 4.5e5, 6.0e5, 8.0e5, 1.0e6, 1.3e6):
        field = TokamakField(plasma_current=ip)
        traj, s, m, q = run_ensemble(
            field, [BANANA_PITCH], energy_ev=10e3, r_start=0.15, t_max=t_max,
        )
        # only accept cleanly trapped orbits: a lost or barely-bouncing particle
        # does not have a well-defined banana width
        if s.lost[0] or not s.trapped[0] or s.n_bounces[0] < 2:
            print(f"  Ip={ip/1e3:6.1f} kA  SKIPPED (not cleanly trapped: "
                  f"lost={bool(s.lost[0])}, bounces={int(s.n_bounces[0])})", flush=True)
            continue
        rows.append(dict(
            kind="current", plasma_current=ip, energy_ev=10e3,
            b_theta=float(field.b_poloidal_reference(0.15)),
            measured=float(s.radial_width[0]),
            analytic=float(banana_width_analytic(
                field, 0.15, ev_to_joule(10e3), m, q)),
            trapped=bool(s.trapped[0]), n_bounces=int(s.n_bounces[0]),
            pitch=BANANA_PITCH,
        ))
        print(f"  Ip={ip/1e3:6.1f} kA  w_b={rows[-1]['measured']:.5f} m  "
              f"analytic={rows[-1]['analytic']:.5f} m", flush=True)
    # vary energy at fixed current
    field = TokamakField()
    for e_kev in (1.25, 2.5, 5.0, 10.0, 20.0, 40.0):
        traj, s, m, q = run_ensemble(
            field, [BANANA_PITCH], energy_ev=e_kev * 1e3, r_start=0.15,
            t_max=t_max,
        )
        if s.lost[0] or not s.trapped[0] or s.n_bounces[0] < 2:
            print(f"  E={e_kev:5.1f} keV  SKIPPED (not cleanly trapped: "
                  f"lost={bool(s.lost[0])}, bounces={int(s.n_bounces[0])})", flush=True)
            continue
        rows.append(dict(
            kind="energy", plasma_current=field.Ip, energy_ev=e_kev * 1e3,
            b_theta=float(field.b_poloidal_reference(0.15)),
            measured=float(s.radial_width[0]),
            analytic=float(banana_width_analytic(
                field, 0.15, ev_to_joule(e_kev * 1e3), m, q)),
            trapped=bool(s.trapped[0]), n_bounces=int(s.n_bounces[0]),
            pitch=BANANA_PITCH,
        ))
        print(f"  E={e_kev:5.1f} keV  w_b={rows[-1]['measured']:.5f} m  "
              f"analytic={rows[-1]['analytic']:.5f} m", flush=True)
    # fit the scaling exponents -- these, not the prefactor, are the claim
    cur = [r for r in rows if r["kind"] == "current"]
    ene = [r for r in rows if r["kind"] == "energy"]
    fits = {}
    if len(cur) > 2:
        bt = np.array([r["b_theta"] for r in cur])
        w = np.array([r["measured"] for r in cur])
        fits["exponent_vs_b_theta"] = float(np.polyfit(np.log(bt), np.log(w), 1)[0])
        fits["prefactor_ratio_vs_analytic"] = float(np.mean(
            [r["measured"] / r["analytic"] for r in cur]))
    if len(ene) > 2:
        en = np.array([r["energy_ev"] for r in ene])
        w = np.array([r["measured"] for r in ene])
        fits["exponent_vs_energy"] = float(np.polyfit(np.log(en), np.log(w), 1)[0])
    print(f"  fitted exponents: {fits}", flush=True)
    _dump("banana_width.json", dict(rows=rows, fits=fits, pitch=BANANA_PITCH))


def exp_convergence(quick):
    """E. Timestep convergence, and the mu-error / rho-scaling separation."""
    print("\n[E] convergence and invariant errors", flush=True)
    from tokamak_orbits import UniformField
    m, q = SPECIES["D"]
    B0 = 2.0
    v = np.sqrt(2 * ev_to_joule(10e3) / m)
    Tc = gyroperiod(m, q, B0)
    rho = m * v / (q * B0)

    # gyroradius error vs dt in a uniform field -> expect second order
    rows = []
    for nper in (10, 20, 40, 80, 160, 320):
        dt = Tc / nper
        tr = integrate(np.zeros((1, 3)), np.array([[v, 0, 0]]), dt,
                       nper * 50, m, q, UniformField(b_vec=(0, 0, B0)).b_field)
        r_meas = np.linalg.norm(tr.x[:, 0, :2], axis=-1).max() / 2
        ke = tr.kinetic_energy(m)[:, 0]
        rows.append(dict(steps_per_gyroperiod=nper,
                         radius_rel_error=float(abs(r_meas - rho) / rho),
                         energy_rel_error=float(np.abs(ke - ke[0]).max() / ke[0])))
        print(f"  n/gyro={nper:4d}  radius err={rows[-1]['radius_rel_error']:.3e}  "
              f"energy err={rows[-1]['energy_rel_error']:.2e}", flush=True)
    orders = [np.log2(rows[i]['radius_rel_error'] / rows[i + 1]['radius_rel_error'])
              for i in range(len(rows) - 1)]
    print(f"  observed order of accuracy: {np.round(orders, 3)}", flush=True)

    # mu error: independent of dt, linear in rho/a
    field = TokamakField()
    dt_rows = []
    for nper in (20, 40, 80, 160, 320):
        traj, s, mm, qq = run_ensemble(
            field, [0.95, 0.45], t_max=6e-6, steps_per_gyroperiod=nper)
        dt_rows.append(dict(steps_per_gyroperiod=nper,
                            mu_error_xi095=float(s.mu_error[0]),
                            mu_error_xi045=float(s.mu_error[1]),
                            energy_error=float(s.energy_error.max())))
    rho_rows = []
    for e_kev in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        traj, s, mm, qq = run_ensemble(
            field, [0.95], energy_ev=e_kev * 1e3, t_max=6e-6)
        vv = np.sqrt(2 * ev_to_joule(e_kev * 1e3) / mm)
        rho_rows.append(dict(energy_ev=e_kev * 1e3,
                             rho_over_a=float(mm * vv / (qq * 2.0) / field.a),
                             mu_error=float(s.mu_error[0])))
    _dump("convergence.json", dict(gyroradius=rows, mu_vs_dt=dt_rows,
                                   mu_vs_rho=rho_rows,
                                   observed_order=[float(o) for o in orders]))


def exp_drift_validation(quick):
    """F. grad-B + curvature drift against the analytic formula."""
    print("\n[F] drift validation in a pure toroidal field", flush=True)
    from tokamak_orbits.fields import grad_b_curvature_drift
    m, q = SPECIES["D"]
    field = TokamakField(plasma_current=0.0)     # vacuum toroidal field only
    rows = []
    for e_kev, xi in ((10.0, 0.5), (10.0, 0.0), (10.0, 0.9), (2.0, 0.5), (40.0, 0.5)):
        x0, v0, mm, qq = initialise(field, energy_ev=e_kev * 1e3,
                                    r_start=0.10, pitch=[xi])
        v = np.linalg.norm(v0[0])
        Tc = gyroperiod(mm, qq, np.linalg.norm(field.b_field(x0)[0]))
        dt = Tc / 40
        n_steps = int(round(30 * Tc / dt))
        tr = integrate(x0, v0, dt, n_steps, mm, qq, field.b_field, sample_every=4)
        # Use diagnostics.guiding_centre WITH dt. An inline reconstruction that
        # omits the half-step rotation carries the leapfrog staggering error of
        # finding 5, which for the xi = 0 case corrupted this measurement by 65x.
        gc_z = guiding_centre(field, tr.x, tr.v, mm, qq, dt=dt)[:, 0, 2]
        fit = np.polyfit(tr.t, gc_z, 1)[0]
        pred = float(np.ravel(grad_b_curvature_drift(
            field, x0, v * xi, v * np.sqrt(1 - xi**2), mm, qq))[0])
        rows.append(dict(energy_ev=e_kev * 1e3, pitch=xi,
                         measured=float(fit), predicted=pred,
                         rel_error=float(abs(fit - pred) / abs(pred))))
        print(f"  E={e_kev:5.1f} keV xi={xi:+.2f}  measured={fit:11.2f} m/s  "
              f"predicted={pred:11.2f} m/s  err={rows[-1]['rel_error']:.3%}", flush=True)
    _dump("drift_validation.json", rows)


def exp_orbit_topology(quick):
    """G. The orbit-classification table and banana width vs minor radius.

    Added because RESULTS.md section 2 and the w_b-vs-r table in section 3 were
    originally produced by ad-hoc exploration rather than by this script, so
    they could not be regenerated and had drifted from the shipped code path.
    """
    print("\n[G] orbit topology table", flush=True)
    field = TokamakField()
    rows = []
    for xi in (0.95, 0.70, 0.55, 0.45, 0.20, 0.00, -0.45, -0.95):
        traj, s, m, q = run_ensemble(
            field, [xi], energy_ev=10e3, r_start=0.15,
            t_max=3e-5 if quick else 6e-5,
        )
        rows.append(dict(
            pitch=xi, kind=str(s.kind[0]), n_bounces=int(s.n_bounces[0]),
            r_min=float(s.r_min[0]), r_max=float(s.r_max[0]),
            width=float(s.radial_width[0]),
            energy_error=float(s.energy_error[0]),
            mu_error=float(s.mu_error[0]),
        ))
        print(f"  xi={xi:+.2f}  {rows[-1]['kind']:8s} bounces={rows[-1]['n_bounces']:3d}  "
              f"r=[{rows[-1]['r_min']:.4f}, {rows[-1]['r_max']:.4f}]  "
              f"w={rows[-1]['width']:.4f}", flush=True)

    print("  banana width vs minor radius", flush=True)
    radial = []
    for r in (0.05, 0.10, 0.15, 0.20, 0.25):
        w = float(banana_width_analytic(
            field, r, ev_to_joule(10e3), *SPECIES["D"]))
        radial.append(dict(r=r, w_analytic=w, w_over_r=w / r,
                           w_over_gap=w / (field.a - r)))
        print(f"    r={r:.2f}  w_b={w:.4f} m  w/r={w/r:.2f}  "
              f"w/(a-r)={w/(field.a-r):.2f}", flush=True)
    _dump("orbit_topology.json", dict(orbits=rows, radial=radial))


def exp_exb(quick):
    """H. E x B drift, recorded rather than only asserted in a test."""
    print("\n[H] E x B drift", flush=True)
    from tokamak_orbits import UniformField, e_cross_b_drift
    E = np.array([0.0, 1e4, 0.0])
    B = np.array([0.0, 0.0, 2.0])
    f = UniformField(b_vec=tuple(B), e_vec=tuple(E))
    expected = float(e_cross_b_drift(E, B)[0])
    rows = []
    for name in ("H", "D", "T"):
        m, q = SPECIES[name]
        v = np.sqrt(2 * ev_to_joule(10e3) / m)
        tc = gyroperiod(m, q, 2.0)
        tr = integrate(np.zeros((1, 3)), np.array([[v, 0, 0]]),
                       tc / 100, 100 * 400, m, q, f.b_field, f.e_field)
        meas = float(np.polyfit(tr.t, tr.x[:, 0, 0], 1)[0])
        rows.append(dict(species=name, measured=meas, expected=expected,
                         rel_error=abs(meas - expected) / abs(expected)))
        print(f"  {name}: {meas:.3f} m/s vs {expected:.3f}  "
              f"err={rows[-1]['rel_error']:.4%}", flush=True)
    spread = float(np.std([r["measured"] for r in rows])
                   / abs(np.mean([r["measured"] for r in rows])))
    print(f"  species spread: {spread:.2e}", flush=True)
    _dump("exb_drift.json", dict(rows=rows, species_spread=spread))


def exp_no_poloidal(quick):
    """I. With Ip = 0 every particle is lost; with Ip = 450 kA none are."""
    print("\n[I] confinement requires a poloidal field", flush=True)
    rows = []
    for ip in (0.0, 4.5e5):
        field = TokamakField(plasma_current=ip)
        pitches = sample_pitch(40 if quick else 120, rng=77, mode="uniform_xi")
        traj, s, m, q = run_ensemble(
            field, pitches, r_start=0.10, t_max=6e-5 if quick else 1.5e-4)
        rows.append(dict(plasma_current=ip, n=len(pitches),
                         lost_fraction=float(s.lost.mean()),
                         median_loss_time=float(np.median(
                             s.loss_time[np.isfinite(s.loss_time)]))
                         if s.lost.any() else None))
        print(f"  Ip={ip/1e3:6.1f} kA  lost {rows[-1]['lost_fraction']:.1%}",
              flush=True)
    _dump("no_poloidal.json", rows)



# --------------------------------------------------------------------------
# collisional experiments
# --------------------------------------------------------------------------
def dt_guess(field, steps_per_gyroperiod=40):
    """The production timestep, for sizing runs before a pusher exists."""
    mass, charge = SPECIES["D"]
    b_max = field.b_toroidal_at(field.R0 - field.a) * np.sqrt(
        1.0 + (field.b_poloidal_reference(field.a) / field.B0) ** 2)
    return gyroperiod(mass, charge, b_max) / steps_per_gyroperiod


def _collisional_run(field, pitches, nu_d, energy_ev=10e3, r_start=0.15,
                     t_max=1.5e-4, steps_per_gyroperiod=40, collide_every=None,
                     seed=4242, gyrophase=0.0, target_nu_dt=0.02,
                     sample_target_per_tau=None):
    """One ensemble with pitch-angle collisions. Returns (traj, mass, charge, dt).

    ``collide_every`` defaults to the largest stride that keeps
    ``nu_d * collide_every * dt <= target_nu_dt``, capped at 500. A fixed
    stride cannot work across the collisionality scan: at nu_d = 1e6 s^-1 a
    stride of 200 gives nu*dt = 0.23, well outside the operator's first-order
    validity, and scatter_pitch would (correctly) refuse it.
    """
    x0, v0, mass, charge = initialise(
        field, energy_ev=energy_ev, r_start=r_start, theta_start=0.0,
        pitch=pitches, gyrophase=gyrophase)
    b_max = field.b_toroidal_at(field.R0 - field.a) * np.sqrt(
        1.0 + (field.b_poloidal_reference(field.a) / field.B0) ** 2)
    dt = gyroperiod(mass, charge, b_max) / steps_per_gyroperiod
    n_steps = int(np.ceil(t_max / dt))
    if collide_every is None:
        collide_every = (500 if nu_d <= 0
                         else int(np.clip(target_nu_dt / (nu_d * dt), 1, 500)))
    col = PitchAngleCollisions(nu_d, seed=seed) if nu_d > 0 else None
    max_samples = 8000        # hard memory cap: 8000 x N x 3 x 8 x 2 bytes
    if sample_target_per_tau and nu_d > 0:
        # resolve the detrapping time itself, not just the run
        tau_guess = 0.15 / nu_d
        sample_every = max(1, min(n_steps // 200,
                                  int(tau_guess / (sample_target_per_tau * dt))))
        sample_every = max(sample_every, n_steps // max_samples)
    else:
        sample_every = max(1, n_steps // 1500)
    sample_every = max(1, sample_every)
    traj = integrate(
        x0, v0, dt, n_steps, mass, charge, field.b_field,
        sample_every=sample_every,
        loss_func=make_loss_func(field),
        collision_op=col, collide_every=collide_every,
        field=field if col is not None else None)
    return traj, mass, charge, dt


def exp_collision_operator(quick):
    """J. Validate the collision operator against Legendre eigenmode decay.

    The Lorentz operator has Legendre polynomials as eigenfunctions with
    eigenvalues -l(l+1)nu/2, so <P_l(xi)> decays exponentially at a rate with
    no free parameters. This is the sharpest available test of the operator and
    needs no field at all.
    """
    print("\n[J] collision operator vs Legendre eigenmode decay", flush=True)
    from numpy.polynomial import legendre as _leg

    def Pl(l, x):
        c = np.zeros(l + 1)
        c[l] = 1.0
        return _leg.legval(x, c)

    rng = np.random.default_rng(11)
    n = 200_000 if quick else 1_500_000
    noise = 1.0 / np.sqrt(n)
    nu, xi0 = 1.0, 0.9
    rows = []
    for nu_dt in ((0.02, 0.005) if quick else (0.02, 0.01, 0.005, 0.0025)):
        xi = np.full(n, xi0)
        ts = [0.0]
        hist = {l: [float(Pl(l, xi0))] for l in (1, 2, 3)}
        for step in range(1, 1401):
            xi = scatter_pitch(xi, nu_dt, rng)
            if step % 10 == 0:
                ts.append(step * nu_dt / nu)
                for l in (1, 2, 3):
                    hist[l].append(float(np.mean(Pl(l, xi))))
        t = np.array(ts)
        fitted = {}
        for l in (1, 2, 3):
            y = np.array(hist[l])
            ok = (y > 10 * noise) & (t > 0)      # only where signal beats noise
            if ok.sum() < 4:
                fitted[l] = None
                continue
            slope = np.polyfit(t[ok], np.log(y[ok]), 1)[0]
            fitted[l] = float(-slope / legendre_decay_rate(l, nu))
        rows.append(dict(nu_dt=nu_dt, n_samples=n, noise_floor=float(noise),
                         ratio_l1=fitted[1], ratio_l2=fitted[2],
                         ratio_l3=fitted[3]))
        print("  nu*dt=%.4f  fitted/exact rate:  l=1 %s  l=2 %s  l=3 %s" % (
            nu_dt, *["%.4f" % fitted[l] if fitted[l] else "  n/a"
                     for l in (1, 2, 3)]), flush=True)

    # the update provably cannot leave [-1, 1]; check rather than assert
    over = []
    for a in (0.005, 0.02, 0.05, 0.1):
        x = rng.uniform(-1, 1, 200_000)
        raw = scatter_pitch(x, a, rng, clip=False)
        over.append(dict(nu_dt=a, overshoot_fraction=float(np.mean(np.abs(raw) > 1)),
                         max_abs_xi=float(np.abs(raw).max()),
                         bound=float(np.sqrt(1 - a + a * a))))
        print("  nu*dt=%.3f  overshoots: %.4f%%  max|xi'|=%.5f  bound sqrt(1-a+a^2)=%.5f"
              % (a, 100 * over[-1]["overshoot_fraction"], over[-1]["max_abs_xi"],
                 over[-1]["bound"]), flush=True)
    _dump("collision_operator.json", dict(legendre=rows, overshoot=over))


def exp_detrapping(quick):
    """K. Collisional detrapping: mean time for a banana to be scattered out.

    A trapped particle need only be scattered through a pitch angle ~sqrt(eps)
    to cross the trapped/passing boundary, so the detrapping rate is
    nu_eff = nu_d / eps rather than nu_d, and the predicted mean detrapping
    time is eps / nu_d. The measured constant of proportionality is reported,
    not assumed.
    """
    print("\n[K] collisional detrapping", flush=True)
    field = TokamakField()
    m, q = SPECIES["D"]
    r0 = 0.15
    eps = r0 / field.R0
    xi_c = trapping_boundary_pitch(field, r0)
    omega_b = 2.0 * np.pi * 5.0 / 1.5e-4      # measured: 10 sign changes / 150 us
    n = 40 if quick else 400
    t_max = 1.5e-4
    nus = (1e4, 1e5, 1e6) if quick else (3e3, 1e4, 3e4, 1e5, 3e5, 1e6)
    # A first-passage quantity is far more sensitive to the collision step than
    # the bulk Legendre rates of section 7.1: at nu*dt = 0.02 the pitch moves
    # 0.135 per application against a 0.211 distance to the boundary, so the
    # particle arrives in ~2 discrete jumps and the measured time is biased
    # high. Refining to 0.005 moves the fitted exponent from -0.88 to -0.96.
    # See docs/DOC_SELF_REVIEW.md finding 23.
    target_nu_dt = 0.005
    rows = []
    for nu_d in nus:
        rng = np.random.default_rng(5)
        # Run only as long as the detrapping time requires. Integrating the
        # full 150 us at every nu_d while also sampling finely enough to
        # resolve a 150 ns detrapping time means storing ~1.3 GB of
        # trajectory, which is how the first attempt at this was killed.
        # 40 predicted detrapping times is ample for a median.
        t_run = float(min(t_max, max(40.0 * eps / nu_d, 20 * dt_guess(field))))
        traj, mass, charge, dt = _collisional_run(
            field, np.full(n, 0.30), nu_d, r_start=r0, t_max=t_run,
            gyrophase=rng.uniform(0, 2 * np.pi, n),
            target_nu_dt=target_nu_dt, sample_target_per_tau=200)
        # Angle-independent trapping test from mu conservation. The earlier
        # version applied the *midplane* condition |xi| < sqrt(2 eps/(1+eps))
        # at every poloidal angle, which misclassifies away from the midplane
        # and made every detrapping time wrong. See finding 20.
        detrapped = ~is_trapped(field, traj.x, traj.v)
        times = []
        for i in range(n):
            idx = np.flatnonzero(detrapped[:, i] & traj.alive[:, i])
            times.append(traj.t[idx[0]] if idx.size else np.inf)
        times = np.array(times)
        frac = float(np.mean(np.isfinite(times)))
        # Two different reasons a particle can fail to register a detrapping
        # time: it stayed trapped for the whole run, or it hit the wall while
        # still trapped. Only the first is censoring; the second is a competing
        # risk and has to be counted separately or the median is not what it
        # looks like.
        never = ~np.isfinite(times)
        n_lost_while_trapped = int(np.sum(never & np.isfinite(traj.loss_time)))
        n_survived_trapped = int(np.sum(never) - n_lost_while_trapped)
        # The MEDIAN is the right statistic here: it is unbiased under
        # right-censoring as long as more than half the sample detraps, whereas
        # the mean of the observed times is biased low by exactly the particles
        # that never detrapped. Both are recorded so the difference is visible.
        t_med = float(np.median(times)) if frac > 0.5 else float("nan")
        t_mean_obs = (float(np.mean(times[np.isfinite(times)]))
                      if np.isfinite(times).any() else float("nan"))
        sample_dt = float(traj.t[1] - traj.t[0]) if traj.t.size > 1 else float(dt)
        rows.append(dict(nu_d=float(nu_d), n=n,
                         fraction_detrapped=frac,
                         median_detrap_time=t_med,
                         mean_observed_detrap_time=t_mean_obs,
                         predicted_eps_over_nu=float(eps / nu_d),
                         ratio_median=(float(t_med / (eps / nu_d))
                                       if np.isfinite(t_med) else None),
                         nu_eff_over_omega_b=float(
                             effective_detrapping_frequency(nu_d, eps) / omega_b),
                         censored=bool(frac < 1.0),
                         t_run=float(t_run),
                         n_lost_while_trapped=n_lost_while_trapped,
                         n_survived_trapped=n_survived_trapped,
                         sample_dt=sample_dt,
                         samples_per_detrap_time=(float(t_med / sample_dt)
                                                  if np.isfinite(t_med) else None)))
        print("  nu_d=%8.3g  detrapped %5.1f%%  median tau=%9.3g s"
              "  eps/nu=%9.3g s  ratio=%s  nu_eff/w_b=%7.3f  samples/tau=%s" % (
                  nu_d, 100 * frac, t_med, eps / nu_d,
                  "%.2f" % rows[-1]["ratio_median"] if np.isfinite(t_med) else " n/a",
                  rows[-1]["nu_eff_over_omega_b"],
                  "%.0f" % rows[-1]["samples_per_detrap_time"]
                  if np.isfinite(t_med) else "n/a"), flush=True)

    # Fit the scaling exponent, but only over points that are neither
    # censoring-biased (low nu) nor sampling-limited (high nu).
    # The median is unbiased under right-censoring while more than half the
    # sample detraps, so "censored at all" is too strict a filter. What does
    # disqualify a point is (a) fewer than 60% detrapping, or (b) the sampling
    # interval being coarse next to the detrapping time itself.
    good = [r for r in rows
            if r["fraction_detrapped"] > 0.6
            and r["samples_per_detrap_time"] is not None
            and r["samples_per_detrap_time"] >= 20]
    fit = None
    if len(good) >= 3:
        lx = np.log([r["nu_d"] for r in good])
        ly = np.log([r["median_detrap_time"] for r in good])
        fit = dict(exponent=float(np.polyfit(lx, ly, 1)[0]),
                   ideal=-1.0,
                   n_points=len(good),
                   nu_range=[float(min(r["nu_d"] for r in good)),
                             float(max(r["nu_d"] for r in good))],
                   mean_ratio=float(np.mean([r["ratio_median"] for r in good])))
        print("  resolved subset (%d pts, nu = %.3g to %.3g): "
              "tau ~ nu^%.3f (ideal -1), <tau nu/eps> = %.2f" % (
                  fit["n_points"], fit["nu_range"][0], fit["nu_range"][1],
                  fit["exponent"], fit["mean_ratio"]), flush=True)
    _dump("detrapping.json", dict(rows=rows, fit=fit,
                                  omega_b_measured=float(omega_b),
                                  xi_crit=float(xi_c), epsilon=float(eps),
                                  target_nu_dt=float(target_nu_dt),
                                  n_particles=int(n)))


def exp_collisional_smearing(quick):
    """L. Does collisionality smear a single particle's sharp threshold?

    The collisionless result is that a fixed pitch angle gives a near-step loss
    transition while a population gives a broad one, because the population
    contains many thresholds at once. Collisions let a *single* particle wander
    across pitch angles during its own lifetime, so it should sample many
    thresholds too, and its transition should broaden toward the population one.
    This is the direct test of that.
    """
    print("\n[L] collisional smearing of the single-pitch threshold", flush=True)
    n = 12 if quick else 48
    t_max = 1.5e-4
    # The collisionless threshold is at 253 kA, but with collisions on, loss
    # persisted to the top of an earlier 450 kA scan for every nu > 0. The
    # range has to cover where the collisional curves actually reach zero.
    currents = (np.linspace(1.5e5, 9.0e5, 8) if quick
                else np.concatenate([np.linspace(1.2e5, 5.0e5, 14),
                                     np.linspace(6.0e5, 1.6e6, 8)]))
    nus = (0.0, 1e5) if quick else (0.0, 1e4, 1e5, 1e6)
    rows = []
    for nu_d in nus:
        pts = []
        for ip in currents:
            field = TokamakField(plasma_current=float(ip))
            rng = np.random.default_rng(999)
            traj, mass, charge, dt = _collisional_run(
                field, np.full(n, -0.45), nu_d, t_max=t_max,
                gyrophase=rng.uniform(0, 2 * np.pi, n))
            lost = float(np.isfinite(traj.loss_time).mean())
            ke = traj.kinetic_energy(mass)
            pts.append(dict(plasma_current=float(ip), lost_fraction=lost,
                            max_energy_error=float(
                                (np.abs(ke - ke[0]).max(axis=0) / ke[0]).max())))
        rows.append(dict(nu_d=float(nu_d), n_particles=n, points=pts))
        lo = [p["plasma_current"] for p in pts if p["lost_fraction"] > 0]
        print("  nu_d=%8.3g  max loss %5.1f%%  highest current with loss %s kA" % (
            nu_d, 100 * max(p["lost_fraction"] for p in pts),
            "%.0f" % (max(lo) / 1e3) if lo else "none"), flush=True)
    _dump("collisional_smearing.json", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse, fast version")
    ap.add_argument("--only", default=None, help="comma-separated subset: A..L")
    args = ap.parse_args()

    table = dict(A=exp_ensemble_scan, B=exp_fixed_pitch_scans,
                 C=exp_trapped_fraction, D=exp_banana_width,
                 E=exp_convergence, F=exp_drift_validation,
                 G=exp_orbit_topology, H=exp_exb, I=exp_no_poloidal,
                 J=exp_collision_operator, K=exp_detrapping,
                 L=exp_collisional_smearing)
    keys = args.only.split(",") if args.only else list(table)
    t0 = time.time()
    for k in keys:
        table[k.strip().upper()](args.quick)
    print(f"\nall done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
