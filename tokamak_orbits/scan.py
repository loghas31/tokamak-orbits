"""Parameter scans: confinement versus poloidal field strength."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import numpy as np

from .fields import TokamakField
from .particles import initialise, sample_pitch
from .pusher import integrate, gyroperiod
from .diagnostics import classify, make_loss_func


@dataclass
class ScanPoint:
    """Outcome of one ensemble at one plasma current."""

    plasma_current: float
    q_axis: float
    q_edge: float
    n_particles: int
    n_lost: int
    lost_fraction: float
    n_trapped: int
    trapped_fraction: float
    median_loss_time: float
    mean_confinement_time: float
    max_energy_error: float
    max_mu_error: float
    t_max: float

    def as_dict(self):
        return asdict(self)


def run_ensemble(
    field: TokamakField,
    pitches,
    energy_ev: float = 10e3,
    r_start=0.15,
    theta_start=0.0,
    t_max: float = 2.0e-4,
    steps_per_gyroperiod: int = 40,
    sample_every: int | None = None,
    species: str = "D",
    r_limit=None,
):
    """Integrate one ensemble and return ``(trajectory, summary, mass, charge)``.

    The timestep is set from the gyroperiod at the *strongest* field the
    particles can reach (the inboard edge), not the axis value, so that the
    stated steps-per-gyroperiod is a guarantee rather than an average.
    """
    x0, v0, mass, charge = initialise(
        field, energy_ev=energy_ev, species=species,
        r_start=r_start, theta_start=theta_start, pitch=pitches,
    )
    dt = gyroperiod(mass, charge, field.b_max_in_domain) / steps_per_gyroperiod
    n_steps = int(np.ceil(t_max / dt))
    if sample_every is None:
        sample_every = max(1, n_steps // 4000)

    traj = integrate(
        x0, v0, dt, n_steps, mass, charge, field.b_field,
        sample_every=sample_every, loss_func=make_loss_func(field, r_limit),
    )
    summary = classify(field, traj, mass, charge)
    return traj, summary, mass, charge


def scan_plasma_current(
    currents,
    n_particles: int = 200,
    energy_ev: float = 10e3,
    r_start=0.15,
    t_max: float = 2.0e-4,
    seed: int = 12345,
    fixed_pitch: float | None = None,
    machine: dict | None = None,
    verbose: bool = True,
    steps_per_gyroperiod: int = 40,
):
    """Scan confinement against plasma current (hence poloidal field).

    Parameters
    ----------
    fixed_pitch : float, optional
        If given, every particle in every ensemble is launched with this pitch
        and only the gyrophase is randomised. This isolates the *single
        particle* response, which is where a sharp threshold is expected. If
        ``None``, pitches are sampled uniformly in ``xi`` (isotropic), which is
        the ensemble case.

    Notes
    -----
    The same random seed is used at every scan point, so the pitch sample is
    identical across currents and the scan measures the response to the field
    alone rather than sampling noise.
    """
    machine = dict(machine or {})
    machine.pop("plasma_current", None)
    rng = np.random.default_rng(seed)
    if fixed_pitch is None:
        pitches = sample_pitch(n_particles, rng=seed, mode="uniform_xi")
    else:
        pitches = np.full(n_particles, float(fixed_pitch))
    gyrophases = rng.uniform(0.0, 2.0 * np.pi, n_particles)

    points = []
    for ip in np.atleast_1d(currents):
        field = TokamakField(plasma_current=float(ip), **machine)
        x0, v0, mass, charge = initialise(
            field, energy_ev=energy_ev, r_start=r_start,
            theta_start=0.0, pitch=pitches, gyrophase=gyrophases,
        )
        dt = gyroperiod(mass, charge, field.b_max_in_domain) / steps_per_gyroperiod
        n_steps = int(np.ceil(t_max / dt))
        traj = integrate(
            x0, v0, dt, n_steps, mass, charge, field.b_field,
            sample_every=max(1, n_steps // 2000),
            loss_func=make_loss_func(field),
        )
        s = classify(field, traj, mass, charge)
        lt = traj.loss_time
        confined = np.where(np.isfinite(lt), lt, t_max)
        pt = ScanPoint(
            plasma_current=float(ip),
            q_axis=float(field.q_axis),
            q_edge=float(field.q_edge),
            n_particles=int(n_particles),
            n_lost=int(s.lost.sum()),
            lost_fraction=float(s.lost.mean()),
            n_trapped=int(s.trapped.sum()),
            trapped_fraction=float(s.trapped.mean()),
            median_loss_time=float(np.median(lt[np.isfinite(lt)])) if s.lost.any() else float("inf"),
            mean_confinement_time=float(confined.mean()),
            max_energy_error=float(s.energy_error.max()),
            max_mu_error=float(s.mu_error.max()),
            t_max=float(t_max),
        )
        points.append(pt)
        if verbose:
            print(
                f"Ip={ip/1e3:7.1f} kA  q0={pt.q_axis:5.2f} qa={pt.q_edge:6.2f}  "
                f"lost={pt.lost_fraction:6.1%}  trapped={pt.trapped_fraction:6.1%}  "
                f"<tau>={pt.mean_confinement_time*1e6:7.2f} us  "
                f"dE/E<{pt.max_energy_error:.1e}",
                flush=True,
            )
    return points


def save_scan(points, path):
    """Write scan points to JSON."""
    with open(path, "w") as fh:
        json.dump([p.as_dict() for p in points], fh, indent=2)


def load_scan(path):
    with open(path) as fh:
        return [ScanPoint(**d) for d in json.load(fh)]
