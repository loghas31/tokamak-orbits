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

from tokamak_orbits import TokamakField, SolovevField, ev_to_joule
from tokamak_orbits.equilibrium import grad_shafranov_residual
from tokamak_orbits.constants import SPECIES
from tokamak_orbits.diagnostics import (
    banana_width_analytic, classify, guiding_centre, make_loss_func,
)
from tokamak_orbits.particles import (
    initialise, sample_pitch, trapped_fraction_analytic, trapping_boundary_pitch,
)
from tokamak_orbits.diagnostics import (
    is_trapped, mirror_ratio, parallel_perp_split,
)
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
    return gyroperiod(mass, charge, field.b_max_in_domain) / steps_per_gyroperiod


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
    dt = gyroperiod(mass, charge, field.b_max_in_domain) / steps_per_gyroperiod
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



def exp_cogoing_refine(quick):
    """M. Resolve the two co-going transition widths, still upper bounds.

    In experiment B the xi = +0.45 and xi = +0.80 scans flip from 100% lost to
    0% lost between adjacent grid points 1.27 kA apart, with no intermediate
    sample, so their quoted widths (<= 2.7% and <= 3.5% of the midpoint) are
    limits set by the grid rather than measurements. This bisects into those
    two brackets with a 25x finer grid and 200 particles per point.

    Two outcomes are possible and both are reportable. Either intermediate loss
    fractions appear, in which case the width is finally measured; or every
    particle still flips together, in which case the width is below the new
    spacing and the bound tightens by a factor of ~25. The second outcome would
    mean the transition is narrower than the spread introduced by gyrophase
    alone -- i.e. that at fixed pitch the loss threshold is, to within anything
    this code can resolve, a step function.
    """
    print("\n[M] resolving the co-going transition widths", flush=True)
    n = 40 if quick else 200
    t_max = 6e-5 if quick else 1.5e-4
    brackets = {0.45: (35.5e3, 39.0e3), 0.80: (29.0e3, 32.5e3)}
    out = {}
    for xi, (lo, hi) in brackets.items():
        npts = 8 if quick else 25
        currents = np.linspace(lo, hi, npts)
        pts = scan_plasma_current(currents, n_particles=n, t_max=t_max,
                                  seed=999, fixed_pitch=xi, verbose=False)
        ip = np.array([p.plasma_current for p in pts])
        frac = np.array([p.lost_fraction for p in pts])
        partial = [(float(a), float(b)) for a, b in zip(ip, frac)
                   if 0.0 < b < 1.0]
        spacing = float(currents[1] - currents[0])
        # 90% -> 10% crossings by linear interpolation, as elsewhere
        def cross(y):
            for k in range(len(frac) - 1):
                if (frac[k] - y) * (frac[k + 1] - y) <= 0 and frac[k] != frac[k + 1]:
                    return float(ip[k] + (y - frac[k]) * (ip[k + 1] - ip[k])
                                 / (frac[k + 1] - frac[k]))
            return float("nan")
        upper, mid, lower = cross(0.9), cross(0.5), cross(0.1)
        width = lower - upper
        out[f"{xi:+.2f}"] = dict(
            pitch=xi, n_particles=n, n_points=npts, spacing_A=spacing,
            currents=[float(v) for v in ip], lost=[float(v) for v in frac],
            n_partial=len(partial), partial=partial,
            upper_A=upper, midpoint_A=mid, lower_A=lower, width_A=width,
            width_pct=float(100 * width / mid) if mid == mid else float("nan"),
            resolved=bool(len(partial) >= 3),
            bound_pct=float(100 * spacing / mid) if mid == mid else float("nan"))
        r = out[f"{xi:+.2f}"]
        print("  xi=%+.2f  grid spacing %.0f A  intermediate points: %d" % (
            xi, spacing, len(partial)), flush=True)
        if partial:
            for a, b in partial:
                print("      %8.1f A -> %5.1f%% lost" % (a, 100 * b), flush=True)
        print("      midpoint %.0f A   width %s" % (
            mid, ("%.0f A = %.2f%% of midpoint" % (width, r["width_pct"]))
            if r["resolved"] else
            ("< %.0f A = < %.2f%% of midpoint (still a step)"
             % (spacing, r["bound_pct"]))), flush=True)
    _dump("cogoing_refine.json", out)



def exp_equilibrium(quick):
    """N. Trapped fraction in a real equilibrium versus the circular model.

    In the circular model |B| is exactly proportional to 1/R on a flux
    surface, so sqrt(2 eps/(1+eps)) is exact BY CONSTRUCTION and measuring
    against it tests the integrator, not the field (finding 4). A Solov'ev
    equilibrium breaks that identity, so the same comparison becomes a real
    test of a large-aspect-ratio expansion.

    Also records how badly the circular model violates Grad-Shafranov, which
    is the quantitative version of the caveat the repository has been carrying
    in prose.
    """
    print("\n[N] trapped fraction: circular model vs Solov'ev equilibrium",
          flush=True)
    n = 100 if quick else 600
    t_max = 3e-5 if quick else 8e-5
    radii = (0.10, 0.15, 0.20, 0.25)

    # how well does each field satisfy Grad-Shafranov?
    eq = SolovevField()
    rng = np.random.default_rng(0)
    R = rng.uniform(eq.r_inboard + 0.03, eq.r_outboard - 0.03, 1500)
    Z = rng.uniform(-0.26, 0.26, 1500)
    keep = eq.psi_of(R, Z) < 0.95 * eq.psi_b
    gs = grad_shafranov_residual(eq, R[keep], Z[keep])
    print("  Solov'ev Grad-Shafranov residual: max %.2e  median %.2e"
          % (np.abs(gs).max(), np.median(np.abs(gs))), flush=True)

    rows = []
    for label, field in (("circular", TokamakField()), ("solovev", eq)):
        for r in radii:
            pitches = sample_pitch(n, rng=2024, mode="uniform_xi")
            traj, s, m, q = run_ensemble(field, pitches, r_start=r,
                                         t_max=t_max)
            keep_i = ~s.lost
            meas = float(s.trapped[keep_i].mean()) if keep_i.any() else float("nan")
            se = float(np.sqrt(max(meas * (1 - meas), 1e-12)
                               / max(keep_i.sum(), 1)))
            p = np.array([[field.R0 + r, 0.0, 0.0]])
            xi_c = float(np.sqrt(1 - 1 / np.ravel(mirror_ratio(field, p))[0]))
            an = trapped_fraction_analytic(r / field.R0)
            rows.append(dict(field=label, r=r, measured=meas, stderr=se,
                             analytic=an, xi_crit_field=xi_c,
                             xi_crit_expansion=an,
                             ratio_to_analytic=meas / an,
                             n_used=int(keep_i.sum()),
                             n_lost=int(s.lost.sum())))
            print("  %-8s r=%.2f  measured %.4f+-%.4f  expansion %.4f  "
                  "ratio %.3f   field xi_crit/expansion %.4f"
                  % (label, r, meas, se, an, meas / an, xi_c / an), flush=True)
    _dump("equilibrium_trapped.json",
          dict(rows=rows,
               gs_residual_max=float(np.abs(gs).max()),
               gs_residual_median=float(np.median(np.abs(gs))),
               solovev=dict(R_inboard=eq.r_inboard, R_outboard=eq.r_outboard,
                            q_axis=eq.q_axis, q_edge=eq.q_edge,
                            kappa=eq.kappa)))


# --------------------------------------------------------------------------
def exp_thermalisation(quick):
    """[O] Energy operator: right equilibrium, right rate, right in the pusher.

    Four independent checks, in increasing order of what they can catch:

    1. A Maxwellian is stationary, and the residual bias converges away with
       the collision timestep. This is the test the drag was *derived* to
       pass, so failing it means the derivation is not what the code does.
    2. A cold beam and a hot beam both relax to the *same* Maxwellian --
       tested on the shape, not just the mean, via a KS statistic against the
       analytic speed CDF.
    3. The implemented drift matches the analytic drift, and the analytic
       drift approaches the NRL speed drag ``(nu_s - nu_perp/2) v`` at high
       energy. This is the only place an outside number enters.
    4. The same operator, run inside the Boris pusher in the tokamak field,
       reproduces the 0-D slowing-down curve. A gyro-orbit does not change
       the speed statistics, so any disagreement is an interfacing bug.
    """
    from scipy.special import erf
    from tokamak_orbits.collisions_full import (
        MaxwellianBackground, MaxwellianCollisions, relax_speeds,
        scatter_speed, speed_drift,
    )

    print("\n[O] energy scattering: thermalisation and slowing down", flush=True)
    mass, charge = SPECIES["D"]
    bg = MaxwellianBackground(density=1e19, temperature_ev=10e3)
    nu_ref = float(bg.nu_par(bg.v_th, mass))
    out = dict(density=bg.n_b, temperature_ev=bg.T_b_ev,
               coulomb_log=bg.ln_lambda, v_th=bg.v_th, nu_par_at_vth=nu_ref)

    def maxwellian_sample(n, seed):
        rng = np.random.default_rng(seed)
        return np.linalg.norm(
            rng.normal(0.0, np.sqrt(bg.T_b / mass), size=(n, 3)), axis=1)

    def maxwellian_cdf(v):
        s = np.sqrt(bg.T_b / mass)
        y = np.asarray(v, float) / s
        return erf(y / np.sqrt(2)) - np.sqrt(2 / np.pi) * y * np.exp(-y**2 / 2)

    # -- 1. stationarity and its convergence -------------------------------
    n = 20_000 if quick else 80_000
    steps_tau = 5.0
    v0 = maxwellian_sample(n, seed=101)
    fracs = (0.02, 0.005) if quick else (0.02, 0.01, 0.005, 0.0025)
    rows = []
    for f in fracs:
        dt = f / nu_ref
        v, _ = relax_speeds(bg, v0, mass, dt, int(round(steps_tau / f)), seed=202)
        mean_e = 0.5 * mass * float(np.mean(v**2)) / bg.T_b
        # binomial-free standard error on the mean of v^2
        se = float(np.std(0.5 * mass * v**2 / bg.T_b, ddof=1) / np.sqrt(n))
        rows.append(dict(nu_dt=f, mean_energy_over_T=mean_e, stderr=se,
                         bias_percent=100 * (mean_e / 1.5 - 1)))
        print("  nu*dt=%.4f   <E>/T = %.4f +- %.4f   bias %+.2f%%"
              % (f, mean_e, se, 100 * (mean_e / 1.5 - 1)), flush=True)
    out["stationarity"] = rows

    # -- 2. same equilibrium from both sides, and it converges -----------
    # A mean energy of 3T/2 is necessary but weak: many wrong distributions
    # have the right second moment. This tests the *shape* with a KS
    # statistic against the analytic Maxwellian speed CDF, from a cold start
    # and a hot start, and shows the discretisation error converging away.
    #
    # The two starts use *different* seeds. With a shared seed they are driven
    # by the identical sign stream, the dynamics contracts under common random
    # numbers, and the two runs synchronise particle-for-particle -- 99% of
    # speeds bitwise equal, and identical KS statistics. That looks like
    # independent confirmation and is not. Caught in review; see finding 26.
    n2 = 20_000 if quick else 60_000
    ks_crit = 1.36 / np.sqrt(n2)
    shape = []
    for label, start, tau, seed in (("cold 0.4 v_th", 0.4, 25.0, 303),
                                    ("hot 3.0 v_th", 3.0, 25.0, 90210)):
        for f in ((0.02, 0.005) if quick else (0.02, 0.01, 0.005, 0.0025)):
            dt = f / nu_ref
            v, _ = relax_speeds(bg, np.full(n2, start * bg.v_th), mass, dt,
                                int(round(tau / f)), seed=seed)
            sig = np.sqrt(bg.T_b / mass)
            moments = {}
            for k, exact in ((1, 2 * np.sqrt(2 / np.pi)), (2, 3.0),
                             (3, 8 * np.sqrt(2 / np.pi)), (4, 15.0)):
                moments[f"m{k}_over_maxwellian"] = float(
                    np.mean((v / sig) ** k) / exact)
            u = np.sort(v)
            ks = float(np.max(np.abs(
                maxwellian_cdf(u) - np.arange(1, len(u) + 1) / len(u))))
            e = float(0.5 * mass * np.mean(v**2) / bg.T_b)
            shape.append(dict(start=label, nu_dt=f, n=int(n2),
                              collision_times=tau, ks=ks, ks_95=ks_crit,
                              passes_ks=bool(ks < ks_crit),
                              mean_energy_over_T=e,
                              bias_percent=100 * (e / 1.5 - 1),
                              moments=moments))
            print("  %-14s nu*dt=%.4f  <E>/T=%.4f (%+.2f%%)  KS=%.5f  "
                  "%s (95%% crit %.5f)"
                  % (label, f, e, 100 * (e / 1.5 - 1), ks,
                     "pass" if ks < ks_crit else "FAIL", ks_crit), flush=True)
    out["equilibrium_shape"] = shape

    # -- 3. the drift: implemented = analytic = NRL speed drag -------------
    # The analytic drift came out of detailed balance with no collision-
    # frequency table in front of it. It turns out to equal
    # -(nu_s - nu_perp/2) v identically, for every mass ratio and every
    # speed -- so the derivation reproduces the standard result rather than
    # approximating it. Both sides are built from the same psi(x), so this
    # checks the *combination*, not psi itself; psi is checked separately
    # against its series and its derivative in the unit tests.
    grid = np.geomspace(0.02, 20.0, 200) * bg.v_th
    ident = []
    for tm in ("H", "D", "T", "He4"):
        m_t = SPECIES[tm][0]
        a = speed_drift(bg, grid, m_t)
        nrl = -(bg.nu_slowing(grid, m_t) - 0.5 * bg.nu_perp(grid, m_t)) * grid
        ident.append(dict(test_species=tm, mass_ratio=float(m_t / bg.m_b),
                          max_abs_rel_dev=float(np.max(np.abs(a / nrl - 1)))))
    print("  drift == -(nu_s - nu_perp/2) v :  max deviation over 0.02-20 v_th"
          " = %.2e (H,D,T,He4)"
          % max(r["max_abs_rel_dev"] for r in ident), flush=True)
    out["drift_identity"] = ident

    # NOTE ON WHAT THIS CAN AND CANNOT CATCH. `scatter_speed` is
    # v + mu*dt + sign*sqrt(B*dt) with a symmetric sign, so the sample mean
    # *is* mu plus zero-mean noise: the ratio below is 1 by construction and
    # the only defect it can detect is a biased sign draw or a broken floor.
    # It is a closure check on the sampler, not a validation of the drag --
    # the drag is validated by the identity above and by §9.2's equilibrium.
    rng = np.random.default_rng(404)
    n3 = 200_000 if quick else 2_000_000
    drift_rows = []
    for f in (1.0, 1.5, 2.0, 3.0, 4.5, 6.0):
        v = f * bg.v_th
        dtc = 1e-3 / float(bg.nu_par(v, mass))
        sample = scatter_speed(bg, np.full(n3, v), mass, dtc, rng)
        measured = float(np.mean(sample - v) / dtc)
        se = float(np.std(sample, ddof=1) / np.sqrt(n3) / dtc)
        analytic = float(speed_drift(bg, v, mass))
        drift_rows.append(dict(
            v_over_vth=f, measured=measured, stderr=se, analytic=analytic,
            nu_dt=1e-3,
            measured_over_analytic=measured / analytic,
            sigmas=abs(measured - analytic) / se))
        print("  v/vth=%.1f  measured %+.4e +- %.1e   analytic %+.4e   "
              "ratio %.4f  (%.1f sigma)"
              % (f, measured, se, analytic, measured / analytic,
                 abs(measured - analytic) / se), flush=True)
    out["drift_validation"] = drift_rows

    # -- 4. the same operator inside the pusher ----------------------------
    # The density is raised so that a full slowing-down fits in an affordable
    # number of gyro-orbits. The operator is linear in n_b, so this rescales
    # the time axis by exactly that factor and changes nothing else. The
    # launch radius is small enough that no particle reaches the wall, so
    # the comparison is not biased by survivor selection.
    field = TokamakField()
    dense = MaxwellianBackground(density=1e22, temperature_ev=10e3)
    e0_ev = 100e3
    v_fast = float(np.sqrt(2 * ev_to_joule(e0_ev) / mass))
    tau_s = v_fast / abs(float(speed_drift(dense, v_fast, mass)))
    dt = gyroperiod(mass, charge, field.b_max_in_domain) / 40
    span = 0.15 if quick else 1.0
    n_steps = int(round(span * tau_s / dt))
    collide_every = 200
    n_part = 24 if quick else 64

    pitches = sample_pitch(n_part, rng=7, mode="uniform_xi")
    x0, v0p, m2, q2 = initialise(field, energy_ev=e0_ev, r_start=0.04,
                                 pitch=pitches)
    # Pitch scattering is OFF for this comparison on purpose. With it on, the
    # collisions also drive radial transport, particles reach the wall, and
    # the survivors are a biased sample -- which is a different (and real)
    # effect, but it contaminates the one thing this subsection is for: a
    # gyro-orbit must not change speed statistics, so any disagreement with
    # the 0-D curve is an interfacing bug and nothing else.
    #
    # Some particles are still lost, and that IS a speed-dependent selection:
    # the banana width goes as m v_perp / (q B_pol), so a slowing ion has a
    # narrowing orbit. The bias is therefore real but one-directional and
    # small -- it removes the widest orbits, which are the *fastest*
    # survivors, so it biases the pusher mean DOWN relative to 0-D. The
    # measured ratio is above 1, i.e. the discrepancy is not in the direction
    # this bias would produce. An earlier comment here claimed no particle
    # reached the wall, which the run's own n_lost contradicted.
    col = MaxwellianCollisions(dense, seed=11, pitch=False, energy=True,
                               max_nu_dt=0.005).for_species("D")
    print("  pusher run: n=%.0e m^-3, tau_s=%.3e s, %d steps x %d particles"
          % (dense.n_b, tau_s, n_steps, n_part), flush=True)
    traj = integrate(x0, v0p, dt, n_steps, mass, charge, field.b_field,
                     collision_op=col, collide_every=collide_every, field=field,
                     sample_every=max(1, n_steps // 400),
                     loss_func=make_loss_func(field))
    n_lost = int(np.isfinite(traj.loss_time).sum())
    ke = traj.kinetic_energy(mass)
    keep = ~np.isfinite(traj.loss_time)
    e_hist = np.array([np.mean(row[keep]) for row in ke]) / ev_to_joule(1.0)

    # the 0-D prediction: same operator, same collision step, same sub-cycling
    # policy, no orbit
    n_coll = n_steps // collide_every
    _, trace = relax_speeds(
        dense, np.full(20_000, v_fast), mass, dt * collide_every, n_coll,
        seed=12, record_every=max(1, n_coll // 400), max_nu_dt=0.005)
    zero_t = np.array([t for t, _ in trace])
    zero_e = np.array([e for _, e in trace]) / ev_to_joule(1.0)

    e_end_pusher = float(e_hist[-1])
    e_end_zero = float(zero_e[-1])
    # sampling error on the pusher mean, which is the limiting uncertainty
    se_p = float(np.std(ke[-1][keep], ddof=1) / np.sqrt(max(keep.sum(), 1))
                 / ev_to_joule(1.0))
    print("  final <E>: pusher %.2f +- %.2f keV   0-D %.2f keV   ratio %.4f"
          "   lost %d/%d" % (e_end_pusher / 1e3, se_p / 1e3, e_end_zero / 1e3,
                             e_end_pusher / e_end_zero, n_lost, n_part),
          flush=True)
    n_particle_steps = (n_steps // collide_every) * n_part
    print("  collision step: raw nu_par*dt reached %.2f (the low-speed tail, "
          "not the bulk: the bulk ends at %.2e), sub-cycled to <= %.4f with "
          "up to %d sub-steps"
          % (col.max_nu_dt_seen,
             float(dense.nu_par(np.sqrt(2 * ev_to_joule(e_end_pusher) / mass),
                                mass) * dt * collide_every),
             col.target_nu_dt, col.max_substeps_used), flush=True)
    print("  silent-modification counters: substep cap %d, speed floor %d, "
          "of %d particle-steps"
          % (col.substep_cap_hits, col.speed_floor_hits, n_particle_steps),
          flush=True)
    out["pusher_slowing_down"] = dict(
        density=dense.n_b, energy_ev=e0_ev, tau_s=tau_s, n_steps=n_steps,
        n_particles=n_part, dt=dt, collide_every=collide_every,
        max_nu_dt_raw=float(col.max_nu_dt_seen),
        target_nu_dt=float(col.target_nu_dt),
        max_substeps_used=int(col.max_substeps_used),
        substep_cap_hits=int(col.substep_cap_hits),
        speed_floor_hits=int(col.speed_floor_hits),
        n_particle_steps=int(n_particle_steps),
        bulk_final_nu_dt=float(dense.nu_par(
            np.sqrt(2 * ev_to_joule(e_end_pusher) / mass), mass)
            * dt * collide_every),
        pitch_scattering=False, n_lost=n_lost,
        t=traj.t.tolist(), mean_energy_ev=e_hist.tolist(),
        stderr_final_ev=se_p,
        zero_d_t=zero_t.tolist(), zero_d_energy_ev=zero_e.tolist(),
        final_pusher_ev=e_end_pusher, final_zero_d_ev=e_end_zero,
        ratio=e_end_pusher / e_end_zero,
        background_thermal_ev=1.5 * dense.T_b_ev)
    _dump("thermalisation.json", out)


def exp_ripple(quick):
    """[P] Field ripple: the first non-axisymmetric field, and its losses.

    Everything before this point is axisymmetric, so the canonical toroidal
    momentum is exactly conserved and no collisionless orbit can wander in
    radius. Finite coil number breaks that. The experiment does four things:

    1. Shows the ripple field is still divergence-free -- the same standard
       that finding 2 failed -- and identifies exactly what the naive
       hand-written version gets wrong.
    2. Maps the ripple-well region, where a particle can be trapped between
       coils and drift straight out.
    3. Measures the loss it causes, against an axisymmetric control that is
       identical in every other respect.
    4. Asks whether any of it reaches the bulk population.
    """
    from tokamak_orbits.ripple import (
        RippledField, gwb_threshold, ripple_amplitude_measured,
        ripple_well_fraction,
    )

    print("\n[P] field ripple: losses from broken axisymmetry", flush=True)
    base = TokamakField()
    n_coils = 16
    out = dict(n_coils=n_coils, machine=repr(base))

    # -- 1. the field is still solenoidal ----------------------------------
    rng = np.random.default_rng(0)
    n_pts = 300 if quick else 1500
    r = base.a * np.sqrt(rng.uniform(0, 0.9, n_pts))
    th = rng.uniform(0, 2 * np.pi, n_pts)
    ph = rng.uniform(0, 2 * np.pi, n_pts)
    Rc = base.R0 + r * np.cos(th)
    pts = np.stack([Rc * np.cos(ph), Rc * np.sin(ph), r * np.sin(th)], axis=-1)

    def divergence(bfun):
        h = 1e-7
        d = np.zeros(len(pts))
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h
            pm[:, k] -= h
            d += (bfun(pp)[:, k] - bfun(pm)[:, k]) / (2 * h)
        return d / (np.linalg.norm(bfun(pts), axis=-1) / base.a)

    rip = RippledField(base, n_coils, 0.01)

    def naive(pos):
        """The wrong version: modulate B_phi and stop."""
        pos = np.atleast_2d(pos)
        R = np.hypot(pos[:, 0], pos[:, 1])
        phi = np.arctan2(pos[:, 1], pos[:, 0])
        b = base.b_field(pos).copy()
        fac = rip.delta(R) * base.B0 * base.R0 / R * np.cos(n_coils * phi)
        b[:, 0] += -np.sin(phi) * fac
        b[:, 1] += np.cos(phi) * fac
        return b

    d_base = divergence(base.b_field)
    d_rip = divergence(rip.b_field)
    d_naive = divergence(naive)
    print("  div B / (|B|/a):  axisymmetric max %.2e | potential ripple max "
          "%.2e | naive ripple max %.2e (median %.2e)"
          % (np.abs(d_base).max(), np.abs(d_rip).max(),
             np.abs(d_naive).max(), np.median(np.abs(d_naive))), flush=True)
    out["divergence"] = dict(
        axisymmetric_max=float(np.abs(d_base).max()),
        ripple_max=float(np.abs(d_rip).max()),
        ripple_median=float(np.median(np.abs(d_rip))),
        naive_max=float(np.abs(d_naive).max()),
        naive_median=float(np.median(np.abs(d_naive))),
        ratio=float(np.abs(d_naive).max() / np.abs(d_rip).max()))

    # measured amplitude vs the model, and the poloidal dilution
    amp = []
    for r_i in (0.05, 0.15, 0.25, 0.30):
        R = base.R0 + r_i
        meas = ripple_amplitude_measured(rip, R)
        model = float(rip.delta(R))
        b_pol = float(base.b_poloidal(r_i, R))
        b_tor = float(base.b_toroidal_at(R))
        dilution = 1.0 / (1.0 + (b_pol / b_tor) ** 2)
        amp.append(dict(r=r_i, R=R, measured=meas, model_delta=model,
                        poloidal_dilution=dilution,
                        measured_over_model=meas / model,
                        measured_over_diluted=meas / (model * dilution)))
        print("  r=%.2f  |B| ripple measured %.3e   B_phi model %.3e   "
              "x dilution %.4f -> ratio %.5f"
              % (r_i, meas, model, dilution, meas / (model * dilution)),
              flush=True)
    out["amplitude"] = amp

    # -- 2. the ripple-well region -----------------------------------------
    wells = []
    for delta_edge in (0.0025, 0.005, 0.01, 0.02):
        f = RippledField(base, n_coils, delta_edge)
        row = dict(delta_edge=delta_edge, n_coils=n_coils,
                   fractions={"%.2f" % rr: ripple_well_fraction(f, rr)
                              for rr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)},
                   angle_fractions={
                       "%.2f" % rr: ripple_well_fraction(f, rr, weight="angle")
                       for rr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)})
        wells.append(row)
        print("  delta_edge=%.4f  well fraction by r: %s" % (
            delta_edge, "  ".join("%s:%.3f" % kv
                                  for kv in row["fractions"].items())),
              flush=True)
    out["well_fraction"] = wells

    n_scan = []
    for n_c in (8, 12, 16, 20, 24, 32):
        f = RippledField(base, n_c, 0.01)
        n_scan.append(dict(n_coils=n_c, fraction=ripple_well_fraction(f, 0.25),
                           delta_at_R125=float(f.delta(1.25))))
    peak = max(n_scan, key=lambda d: d["fraction"])
    print("  at fixed delta_edge=1%%, well fraction peaks at N=%d (%.4f): %s"
          % (peak["n_coils"], peak["fraction"],
             " ".join("%d:%.3f" % (d["n_coils"], d["fraction"])
                      for d in n_scan)), flush=True)
    out["coil_number_scan"] = n_scan

    # -- 3. losses, against an identical axisymmetric control --------------
    # Launched at theta = 0 on the r = 0.25 surface: that is where the
    # ripple wells are, and where a small-|xi| particle turns.
    #
    # Sample size matters here in a way it does not elsewhere. Ripple loss is
    # a stochastic mechanism, so a ten-particle scan produces a non-monotonic
    # sequence that says nothing -- the first version of this experiment did
    # exactly that (20%, 60%, 50%, 20%, 70%, binomial error +-15%). The scan
    # below uses 40 launches with *shared* pitches and gyrophases across every
    # amplitude, which turns it into a paired comparison: the question is not
    # "what fraction is lost" but "which particles does ripple lose that the
    # axisymmetric field confines", and pairing removes the launch-condition
    # variance entirely.
    n_edge = 10 if quick else 40
    rng_e = np.random.default_rng(31)
    xis = np.sort(rng_e.uniform(-0.22, 0.22, n_edge))
    gyro = rng_e.uniform(0.0, 2 * np.pi, n_edge)
    t_max = 1.0e-4 if quick else 3.0e-4
    deltas = (0.0, 0.005, 0.02) if quick else (0.0, 0.0025, 0.005, 0.01, 0.02)
    scan = []
    control_lost = None
    for delta_edge in deltas:
        f = base if delta_edge == 0.0 else RippledField(base, n_coils, delta_edge)
        x0, v0, mass, charge = initialise(
            f, energy_ev=10e3, r_start=0.25, theta_start=0.0, pitch=xis,
            gyrophase=gyro)
        dt = gyroperiod(mass, charge, f.b_max_in_domain) / 40
        n_steps = int(np.ceil(t_max / dt))
        traj = integrate(x0, v0, dt, n_steps, mass, charge, f.b_field,
                         sample_every=max(1, n_steps // 400),
                         loss_func=make_loss_func(f))
        lost = np.isfinite(traj.loss_time)
        ke = traj.kinetic_energy(mass)
        if control_lost is None:
            control_lost = lost.copy()
        newly = int((lost & ~control_lost).sum())
        rescued = int((~lost & control_lost).sum())
        prompt = traj.loss_time[lost & control_lost]
        ripple_only = traj.loss_time[lost & ~control_lost]
        row = dict(delta_edge=delta_edge, n_particles=int(len(xis)),
                   pitches=xis.tolist(), gyrophases=gyro.tolist(),
                   n_lost=int(lost.sum()), lost_fraction=float(lost.mean()),
                   loss_times=[None if not np.isfinite(t) else float(t)
                               for t in traj.loss_time],
                   newly_lost=newly, rescued=rescued,
                   newly_lost_fraction=newly / max(int((~control_lost).sum()), 1),
                   median_prompt_loss=(float(np.median(prompt))
                                       if prompt.size else None),
                   median_ripple_loss=(float(np.median(ripple_only))
                                       if ripple_only.size else None),
                   median_loss_time=(float(np.median(traj.loss_time[lost]))
                                     if lost.any() else None),
                   well_fraction_at_r=(ripple_well_fraction(f, 0.25)
                                       if delta_edge else 0.0),
                   max_energy_error=float(np.abs(ke / ke[0] - 1).max()),
                   t_max=t_max)
        scan.append(row)
        print("  delta_edge=%.4f  lost %2d/%2d = %5.1f%%   newly lost %2d "
              "(%4.1f%% of the %d the control confines)   rescued %d   "
              "median: prompt %s ripple %s   well frac %.3f   dE/E %.1e"
              % (delta_edge, row["n_lost"], len(xis),
                 100 * row["lost_fraction"], newly,
                 100 * row["newly_lost_fraction"],
                 int((~control_lost).sum()), rescued,
                 ("%.0fus" % (row["median_prompt_loss"] * 1e6)
                  if row["median_prompt_loss"] else "  -- "),
                 ("%.0fus" % (row["median_ripple_loss"] * 1e6)
                  if row["median_ripple_loss"] else "  -- "),
                 row["well_fraction_at_r"], row["max_energy_error"]),
              flush=True)
    # Adjacent-amplitude comparison. "rescued" above is measured against the
    # delta = 0 control, where it is ZERO BY CONSTRUCTION: the control's
    # losses are prompt (~12 us) counter-going orbits that no ripple can
    # save. Comparing consecutive amplitudes instead shows the effect is not
    # one-directional -- particles lost at 0.5% are confined at 1% and 2%.
    # The first write-up of this section claimed one-directionality on the
    # strength of the zero-by-construction number; see finding 26.
    moves = []
    for a, b in zip(scan[:-1], scan[1:]):
        la = np.array([t is not None for t in a["loss_times"]])
        lb = np.array([t is not None for t in b["loss_times"]])
        moves.append(dict(
            from_delta=a["delta_edge"], to_delta=b["delta_edge"],
            lost_gained=int((lb & ~la).sum()), lost_given_back=int((la & ~lb).sum()),
            indices_gained=np.flatnonzero(lb & ~la).tolist(),
            indices_given_back=np.flatnonzero(la & ~lb).tolist()))
        print("  %.4f -> %.4f: %d newly lost, %d newly confined"
              % (a["delta_edge"], b["delta_edge"], moves[-1]["lost_gained"],
                 moves[-1]["lost_given_back"]), flush=True)
    out["adjacent_moves"] = moves
    out["loss_scan"] = scan

    # -- 4. does it reach the bulk? ----------------------------------------
    # The edge population above is the worst case by construction. This is
    # the fair question: an isotropic ensemble launched where the main scans
    # launch it, with and without ripple.
    n_bulk = 16 if quick else 40
    t_bulk = 1.0e-4 if quick else 2.0e-4
    pitches = sample_pitch(n_bulk, rng=21, mode="uniform_xi")
    bulk = []
    for delta_edge in ((0.0, 0.02) if quick else (0.0, 0.01, 0.02)):
        f = base if delta_edge == 0.0 else RippledField(base, n_coils, delta_edge)
        x0, v0, mass, charge = initialise(
            f, energy_ev=10e3, r_start=0.15, pitch=pitches)
        dt = gyroperiod(mass, charge, f.b_max_in_domain) / 40
        n_steps = int(np.ceil(t_bulk / dt))
        traj = integrate(x0, v0, dt, n_steps, mass, charge, f.b_field,
                         sample_every=max(1, n_steps // 300),
                         loss_func=make_loss_func(f))
        lost = np.isfinite(traj.loss_time)
        se = float(np.sqrt(max(lost.mean() * (1 - lost.mean()), 1e-12) / n_bulk))
        bulk.append(dict(delta_edge=delta_edge, n_particles=n_bulk,
                         n_lost=int(lost.sum()),
                         lost_fraction=float(lost.mean()), stderr=se,
                         t_max=t_bulk))
        print("  bulk r=0.15 isotropic  delta_edge=%.4f  lost %d/%d = "
              "%.1f%% +- %.1f%%" % (delta_edge, int(lost.sum()), n_bulk,
                                    100 * lost.mean(), 100 * se), flush=True)
    out["bulk"] = bulk

    # GWB stochasticity diagnostic at the same conditions
    # sigma is dimensionless: an earlier version was not, and reported
    # numbers ~4x these. See finding 26.
    v = float(np.sqrt(2 * ev_to_joule(10e3) / SPECIES["D"][0]))
    rho = SPECIES["D"][0] * v / (SPECIES["D"][1] * base.B0)
    out["gwb"] = [dict(delta_edge=d, r=0.25,
                       sigma=float(gwb_threshold(
                           RippledField(base, n_coils, d), 0.25, 0.0, rho)))
                  for d in (0.0025, 0.005, 0.01, 0.02)]
    print("  GWB sigma at r=0.25, rho=%.3f m: %s  (stochastic where < 1)"
          % (rho, " ".join("%.4g:%.2f" % (d["delta_edge"], d["sigma"])
                           for d in out["gwb"])), flush=True)
    _dump("ripple.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse, fast version")
    ap.add_argument("--only", default=None, help="comma-separated subset: A..P")
    args = ap.parse_args()

    table = dict(A=exp_ensemble_scan, B=exp_fixed_pitch_scans,
                 C=exp_trapped_fraction, D=exp_banana_width,
                 E=exp_convergence, F=exp_drift_validation,
                 G=exp_orbit_topology, H=exp_exb, I=exp_no_poloidal,
                 J=exp_collision_operator, K=exp_detrapping,
                 L=exp_collisional_smearing, M=exp_cogoing_refine,
                 N=exp_equilibrium, O=exp_thermalisation,
                 P=exp_ripple)
    keys = args.only.split(",") if args.only else list(table)
    t0 = time.time()
    for k in keys:
        table[k.strip().upper()](args.quick)
    print(f"\nall done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
