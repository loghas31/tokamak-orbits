#!/usr/bin/env python3
"""Is the detrapping time converged in the COLLISION timestep?

Finding 22 recorded that the measured detrapping exponent is -0.779 against a
predicted -1, cause unknown. This probes the most likely cause.

The collision operator is validated (section 7.1) against the Legendre decay
rates, which are a *bulk* property of the distribution. The detrapping time is a
*first-passage* quantity, and first passage is far more sensitive to step size:
each application of the operator moves the pitch by
``sqrt((1 - xi^2) nu dt)``, and if that is comparable to the distance from the
launch pitch to the trapped/passing boundary, the particle arrives in two or
three discrete jumps rather than by diffusion. The measured first-passage time
is then quantised and biased, and the bias grows with nu because the production
policy holds ``nu dt`` fixed at 0.02 rather than holding the *step size* fixed.

At nu_d = 1e5 the step is dxi ~ 0.135 against a distance to the boundary of
0.21 -- fewer than three steps. This script refines ``nu dt`` at fixed physics
and asks whether the answer moves.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokamak_orbits import TokamakField, ev_to_joule
from tokamak_orbits.collisions import PitchAngleCollisions
from tokamak_orbits.constants import SPECIES
from tokamak_orbits.diagnostics import is_trapped, make_loss_func
from tokamak_orbits.particles import initialise, trapping_boundary_pitch
from tokamak_orbits.pusher import gyroperiod, integrate


def median_detrap(field, nu_d, nu_dt, n=200, xi0=0.30, r0=0.15,
                  t_max=1.5e-4, seed=5, sample_per_tau=200):
    """Median first-detrapping time at a chosen collision step size."""
    rng = np.random.default_rng(seed)
    x0, v0, mass, charge = initialise(
        field, energy_ev=10e3, r_start=r0, theta_start=0.0,
        pitch=np.full(n, xi0), gyrophase=rng.uniform(0, 2 * np.pi, n))
    b_max = field.b_toroidal_at(field.R0 - field.a) * np.sqrt(
        1.0 + (field.b_poloidal_reference(field.a) / field.B0) ** 2)
    dt = gyroperiod(mass, charge, b_max) / 40
    n_steps = int(np.ceil(t_max / dt))
    collide_every = max(1, int(round(nu_dt / (nu_d * dt))))
    # sample finely enough that the sampling interval is not itself the limit
    tau_guess = 0.15 / nu_d
    sample_every = max(1, min(n_steps // 200,
                              int(tau_guess / (sample_per_tau * dt))))
    traj = integrate(x0, v0, dt, n_steps, mass, charge, field.b_field,
                     sample_every=sample_every,
                     loss_func=make_loss_func(field),
                     collision_op=PitchAngleCollisions(nu_d, seed=seed),
                     collide_every=collide_every, field=field)
    det = ~is_trapped(field, traj.x, traj.v)
    times = []
    for i in range(n):
        idx = np.flatnonzero(det[:, i] & traj.alive[:, i])
        times.append(traj.t[idx[0]] if idx.size else np.inf)
    times = np.array(times)
    frac = float(np.mean(np.isfinite(times)))
    med = float(np.median(times)) if frac > 0.5 else float("nan")
    step = float(np.sqrt((1 - xi0**2) * nu_dt))
    return dict(nu_d=nu_d, nu_dt=nu_dt, collide_every=collide_every,
                median=med, fraction=frac, xi_step=step,
                steps_to_boundary=float(
                    ((trapping_boundary_pitch(field, r0) - xi0) / step) ** 2),
                samples_per_tau=float(med / (traj.t[1] - traj.t[0]))
                if np.isfinite(med) else float("nan"))


def main():
    field = TokamakField()
    xi_c = trapping_boundary_pitch(field, 0.15)
    print(f"launch pitch 0.30, boundary {xi_c:.4f}, "
          f"distance to boundary {xi_c - 0.30:.4f}\n")
    print("  nu_d      nu*dt   collide_every  dxi per step  ~steps  "
          "median tau      frac   samples/tau")
    print("  " + "-" * 88)
    out = {}
    for nu_d in (3e3, 1e4, 3e4, 1e5):
        out[nu_d] = []
        for nu_dt in (0.02, 0.005, 0.00125):
            r = median_detrap(field, nu_d, nu_dt)
            out[nu_d].append(r)
            print("  %8.3g  %7.5f  %11d  %11.4f  %6.1f  %11.4g  %5.1f%%  %10.0f"
                  % (nu_d, nu_dt, r["collide_every"], r["xi_step"],
                     r["steps_to_boundary"], r["median"], 100 * r["fraction"],
                     r["samples_per_tau"]), flush=True)
        print()

    print("  exponent fitted at each collision step size:")
    for nu_dt in (0.02, 0.005, 0.00125):
        xs, ys = [], []
        for nu_d, rows in out.items():
            r = [q for q in rows if q["nu_dt"] == nu_dt][0]
            if np.isfinite(r["median"]):
                xs.append(np.log(nu_d))
                ys.append(np.log(r["median"]))
        if len(xs) >= 3:
            print("    nu*dt = %.5f  ->  tau ~ nu^%.3f" % (
                nu_dt, np.polyfit(xs, ys, 1)[0]))


if __name__ == "__main__":
    main()
