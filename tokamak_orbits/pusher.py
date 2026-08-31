"""Particle pushers.

Two integrators are provided:

``boris_push``
    The Boris algorithm (Boris 1970). Velocity-Verlet-like leapfrog in which
    the magnetic rotation is applied exactly as a rotation, so the kinetic
    energy is *exactly* preserved by the magnetic part of the step, to
    round-off. Its phase-space volume preservation means the energy error
    stays bounded for arbitrarily long integrations instead of accumulating.
    This is the production integrator.

``rk45_push``
    ``scipy.integrate.solve_ivp`` with RK45 on the same equations of motion.
    Retained *only* as a control: it is used in ``scripts/fig_energy.py`` and
    ``tests/test_conservation.py`` to demonstrate the secular energy drift
    that motivates using Boris in the first place. Do not use it for
    confinement studies.

The equations of motion are the Lorentz force law,

.. math:: m \\frac{d\\mathbf{v}}{dt} = q(\\mathbf{E} + \\mathbf{v}\\times\\mathbf{B}),
          \\qquad \\frac{d\\mathbf{x}}{dt} = \\mathbf{v}

in the non-relativistic limit. At 10 keV a deuteron has v/c ~ 3e-3, so the
relativistic correction to the mass is ~5e-6 -- below the level of anything
claimed here, but see ``docs/DOC_STATUS.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Callable, Optional
import numpy as np


@dataclass
class Trajectory:
    """Sampled output of an integration.

    Attributes
    ----------
    t : (S,) array
        Sample times (s).
    x : (S, N, 3) array
        Positions at the sample times (m).
    v : (S, N, 3) array
        Velocities at the sample times (m/s).
    alive : (S, N) bool array
        Whether each particle was still inside the vessel at that sample.
    loss_time : (N,) array
        Time at which each particle left the vessel, ``np.inf`` if it never did.
    """

    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    alive: np.ndarray
    loss_time: np.ndarray
    dt: float = 0.0
    n_steps: int = 0

    @property
    def n_particles(self) -> int:
        return self.x.shape[1]

    def kinetic_energy(self, mass: float) -> np.ndarray:
        """(S, N) kinetic energy in joules."""
        return 0.5 * mass * np.sum(self.v**2, axis=-1)

    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.v, axis=-1)


def gyroperiod(mass: float, charge: float, b_magnitude: float) -> float:
    """Cyclotron period ``2 pi m / (|q| B)`` in seconds."""
    return 2.0 * np.pi * mass / (abs(charge) * b_magnitude)


def gyroradius(mass: float, charge: float, v_perp: float, b_magnitude: float) -> float:
    """Larmor radius ``m v_perp / (|q| B)`` in metres."""
    return mass * v_perp / (abs(charge) * b_magnitude)


def boris_step(x, v, dt, mass, charge, b_func, e_func=None):
    """Advance one Boris step. ``x`` and ``v`` are ``(N, 3)``; returns new ones.

    The scheme is the standard three-stage form: half electric kick, exact
    rotation about **B**, half electric kick, then a full position drift.
    With ``E = 0`` the two kicks vanish and the step is a pure rotation, which
    is why ``|v|`` is preserved to round-off.
    """
    qm = charge / mass
    E = np.zeros_like(x) if e_func is None else e_func(x)
    B = b_func(x)

    v_minus = v + qm * E * (0.5 * dt)
    t_vec = qm * B * (0.5 * dt)
    t2 = np.sum(t_vec * t_vec, axis=-1, keepdims=True)
    s_vec = 2.0 * t_vec / (1.0 + t2)
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)
    v_new = v_plus + qm * E * (0.5 * dt)
    x_new = x + v_new * dt
    return x_new, v_new


def integrate(
    x0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    mass: float,
    charge: float,
    b_func: Callable[[np.ndarray], np.ndarray],
    e_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    sample_every: int = 1,
    loss_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    stop_when_all_lost: bool = True,
    collision_op=None,
    collide_every: int = 1,
    field=None,
) -> Trajectory:
    """Integrate ``N`` particles with the Boris pusher.

    Parameters
    ----------
    x0, v0 : (N, 3) arrays
        Initial positions (m) and velocities (m/s).
    dt : float
        Timestep (s). Must resolve the gyroperiod; ~40 steps per gyro-orbit is
        the default used throughout (see ``docs/NUMERICS.md`` for the
        convergence study).
    n_steps : int
        Number of steps to take.
    b_func, e_func : callable
        Field functions mapping ``(N, 3)`` positions to ``(N, 3)`` fields.
    sample_every : int
        Store every ``sample_every``-th step. Storage, not accuracy.
    loss_func : callable, optional
        Maps ``(N, 3)`` positions to a boolean "this particle is lost" array.
        Once lost, a particle is frozen and no longer advanced.
    stop_when_all_lost : bool
        Return early if every particle has been lost.
    collision_op : callable, optional
        ``collision_op(field, x, v, dt_coll) -> v`` applied every
        ``collide_every`` steps with ``dt_coll = collide_every * dt``. Used for
        Monte Carlo pitch-angle scattering; see :mod:`tokamak_orbits.collisions`.
        Requires ``field``. Only active particles are scattered.
    collide_every : int
        Apply ``collision_op`` every this many steps. Collisions act on a much
        slower timescale than the gyration, so applying the operator every step
        would be wasted work; what matters is that ``nu_d * collide_every * dt``
        stays well below 1, which the operator itself checks.
    field : Field, optional
        Passed to ``collision_op``. Required if one is given.

    Notes
    -----
    The initial velocity is treated as being defined at ``t = 0`` rather than
    at ``t = -dt/2``. This is a half-step inconsistency of formal order ``dt``
    in the *phase* of the gyro-orbit; it does not affect the energy or any
    orbit-averaged quantity, and is documented in ``docs/DOC_SELF_REVIEW.md``
    finding 5.
    """
    if collision_op is not None and field is None:
        raise ValueError("collision_op requires field")
    if collide_every < 1:
        raise ValueError("collide_every must be >= 1")

    x = np.array(x0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    if x.ndim == 1:
        x = x[None, :]
        v = v[None, :]
    n = x.shape[0]

    lost = np.zeros(n, dtype=bool)
    loss_time = np.full(n, np.inf)

    n_samples = n_steps // sample_every + 1
    ts = np.empty(n_samples)
    xs = np.empty((n_samples, n, 3))
    vs = np.empty((n_samples, n, 3))
    al = np.empty((n_samples, n), dtype=bool)

    ts[0], xs[0], vs[0], al[0] = 0.0, x, v, ~lost
    s = 1
    step = 0
    for step in range(1, n_steps + 1):
        active = ~lost
        if active.any():
            xa, va = boris_step(
                x[active], v[active], dt, mass, charge, b_func, e_func
            )
            x[active] = xa
            v[active] = va
            if collision_op is not None and step % collide_every == 0:
                v[active] = collision_op(
                    field, x[active], v[active], collide_every * dt)
        if loss_func is not None:
            newly = (~lost) & loss_func(x)
            if newly.any():
                loss_time[newly] = step * dt
                lost |= newly
                if stop_when_all_lost and lost.all():
                    if s < n_samples:
                        ts[s], xs[s], vs[s], al[s] = step * dt, x, v, ~lost
                        s += 1
                    break
        if step % sample_every == 0 and s < n_samples:
            ts[s], xs[s], vs[s], al[s] = step * dt, x, v, ~lost
            s += 1

    return Trajectory(
        t=ts[:s], x=xs[:s], v=vs[:s], alive=al[:s],
        loss_time=loss_time, dt=dt, n_steps=step,
    )


def rk45_push(x0, v0, t_end, mass, charge, b_func, e_func=None,
              rtol=1e-8, atol=1e-10, n_out=2000):
    """Integrate a *single* particle with scipy RK45. Control experiment only.

    Returns ``(t, x, v)`` with shapes ``(S,)``, ``(S, 3)``, ``(S, 3)``.
    """
    from scipy.integrate import solve_ivp

    qm = charge / mass
    x0 = np.asarray(x0, float).reshape(3)
    v0 = np.asarray(v0, float).reshape(3)

    def rhs(t, y):
        pos = y[:3][None, :]
        vel = y[3:]
        B = b_func(pos)[0]
        E = np.zeros(3) if e_func is None else e_func(pos)[0]
        return np.concatenate([vel, qm * (E + np.cross(vel, B))])

    sol = solve_ivp(
        rhs, (0.0, t_end), np.concatenate([x0, v0]),
        method="RK45", rtol=rtol, atol=atol,
        t_eval=np.linspace(0.0, t_end, n_out), dense_output=False,
    )
    return sol.t, sol.y[:3].T, sol.y[3:].T
