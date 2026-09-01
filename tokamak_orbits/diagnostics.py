"""Orbit diagnostics: invariants, classification, banana width, losses."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------
def parallel_perp_split(field, x, v):
    """Split velocities into ``(v_par, v_perp)`` given positions.

    ``x`` and ``v`` may be ``(..., 3)``; the leading axes are preserved.
    ``v_par`` is signed (positive means along **B**), ``v_perp`` is positive.
    """
    shape = np.shape(x)[:-1]
    xf = np.asarray(x, float).reshape(-1, 3)
    vf = np.asarray(v, float).reshape(-1, 3)
    B = field.b_field(xf)
    Bmag = np.linalg.norm(B, axis=-1)
    b_hat = B / Bmag[:, None]
    v_par = np.sum(vf * b_hat, axis=-1)
    v2 = np.sum(vf * vf, axis=-1)
    v_perp = np.sqrt(np.clip(v2 - v_par**2, 0.0, None))
    return v_par.reshape(shape), v_perp.reshape(shape)


def magnetic_moment(field, x, v, mass):
    """First adiabatic invariant ``mu = m v_perp^2 / (2 B)`` (J/T).

    ``mu`` is conserved only to all orders in an *adiabatic* expansion; it is
    not an exact constant of the motion. Its measured variation is therefore a
    physical diagnostic of how well the guiding-centre picture applies here,
    not purely a numerical error. Both contributions are separated in
    ``docs/NUMERICS.md`` by refining ``dt`` at fixed field.
    """
    xf = np.asarray(x, float)
    B = np.linalg.norm(field.b_field(xf.reshape(-1, 3)), axis=-1).reshape(xf.shape[:-1])
    _, v_perp = parallel_perp_split(field, x, v)
    return 0.5 * mass * v_perp**2 / B


def _magnetic_rotate(v, B, dt, mass, charge):
    """Rotate ``v`` about ``B`` by exactly the Boris angle for a step ``dt``."""
    t_vec = (charge / mass) * B * (0.5 * dt)
    t2 = np.sum(t_vec * t_vec, axis=-1, keepdims=True)
    s_vec = 2.0 * t_vec / (1.0 + t2)
    v_prime = v + np.cross(v, t_vec)
    return v + np.cross(v_prime, s_vec)


def guiding_centre(field, x, v, mass, charge, dt=None):
    """First-order guiding centre ``X_gc = x + m (v x B) / (q B^2)``.

    Removes the gyro-motion so that radial excursions measure the drift orbit
    rather than the Larmor circle. Accurate to ``O(rho/L)``; at 10 keV in this
    machine ``rho/a ~ 0.034``.

    Parameters
    ----------
    dt : float, optional
        The integration timestep. **Pass this whenever the input came from the
        Boris pusher.** Boris is a leapfrog: ``x_n = x_{n-1} + v_n dt`` makes
        the stored ``v_n`` the velocity over ``[t_{n-1}, t_n]``, i.e. centred
        at ``t_{n-1/2}``, while ``x_n`` sits at ``t_n``. Combining them naively
        leaves a spurious guiding-centre wobble of amplitude exactly
        ``omega dt * rho`` -- first order in dt, and 15% of a gyroradius at the
        production dt = T_c/40. Supplying ``dt`` rotates the velocity forward
        by half a step so that the two are time-centred, which removes it.
        Measured effect on banana width: 1.5% at T_c/40. See
        ``docs/DOC_SELF_REVIEW.md`` finding 5.
    """
    xf = np.asarray(x, float)
    vf = np.asarray(v, float)
    shape = xf.shape
    xr = xf.reshape(-1, 3)
    vr = vf.reshape(-1, 3)
    B = field.b_field(xr)
    if dt is not None:
        vr = _magnetic_rotate(vr, B, 0.5 * dt, mass, charge)
    B2 = np.sum(B * B, axis=-1)[:, None]
    corr = (mass / charge) * np.cross(vr, B) / B2
    return (xr + corr).reshape(shape)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
@dataclass
class OrbitSummary:
    """Per-particle summary of one integration."""

    kind: np.ndarray          # (N,) of 'trapped' | 'passing' | 'lost'
    trapped: np.ndarray       # (N,) bool, among confined particles
    lost: np.ndarray          # (N,) bool
    loss_time: np.ndarray     # (N,) s, inf if confined
    r_min: np.ndarray         # (N,) m, guiding-centre minor radius
    r_max: np.ndarray         # (N,) m
    radial_width: np.ndarray  # (N,) m, r_max - r_min
    energy_error: np.ndarray  # (N,) max |dE/E|
    mu_error: np.ndarray      # (N,) max |d(mu)/mu|
    n_bounces: np.ndarray     # (N,) sign changes of gyro-averaged v_par


def classify(field, traj, mass, charge, smooth_gyroperiods: float = 1.0,
             gyroperiod: float | None = None):
    """Classify orbits and measure their radial excursion.

    A particle is **trapped** if the gyro-averaged parallel velocity changes
    sign at least twice (a full bounce), **passing** otherwise. Averaging over
    a gyroperiod matters: the instantaneous ``v . b_hat`` carries an
    ``O(rho/L)`` ripple that produces spurious sign changes for particles
    launched close to the trapped/passing boundary.

    Only samples while the particle is still confined are used, so a passing
    particle that is later lost is not misclassified from its post-loss frozen
    state.
    """
    x, v = traj.x, traj.v           # (S, N, 3)
    alive = traj.alive              # (S, N)
    n = x.shape[1]

    v_par, v_perp = parallel_perp_split(field, x, v)
    gc = guiding_centre(field, x, v, mass, charge, dt=traj.dt)
    r_gc = np.asarray(field.surface_label(gc.reshape(-1, 3)))
    r_gc = r_gc.reshape(x.shape[:-1])

    # gyro-average window in samples
    if gyroperiod is None:
        Bmag = np.linalg.norm(field.b_field(x[0]), axis=-1).mean()
        gyroperiod = 2.0 * np.pi * mass / (abs(charge) * Bmag)
    sample_dt = traj.t[1] - traj.t[0] if len(traj.t) > 1 else traj.dt
    win = max(1, int(round(smooth_gyroperiods * gyroperiod / sample_dt)))
    if win > 1:
        kern = np.ones(win) / win
        vp_s = np.apply_along_axis(
            lambda a: np.convolve(a, kern, mode="same"), 0, v_par
        )
    else:
        vp_s = v_par

    kinds = np.empty(n, dtype=object)
    trapped = np.zeros(n, bool)
    n_bounce = np.zeros(n, int)
    r_min = np.empty(n); r_max = np.empty(n)
    e_err = np.empty(n); mu_err = np.empty(n)

    ke = 0.5 * mass * np.sum(v * v, axis=-1)
    mu = magnetic_moment(field, x, v, mass)

    lost = np.isfinite(traj.loss_time)
    edge = max(win, 2)   # discard convolution edge effects

    for i in range(n):
        m_ok = alive[:, i]
        if m_ok.sum() > 2 * edge:
            m_ok = m_ok.copy()
            m_ok[:edge] = False
            m_ok[max(0, m_ok.size - edge):] = False
        idx = np.flatnonzero(m_ok)
        if idx.size < 3:
            idx = np.flatnonzero(alive[:, i])
        s = np.sign(vp_s[idx, i])
        s = s[s != 0]
        n_bounce[i] = int(np.count_nonzero(np.diff(s) != 0)) if s.size > 1 else 0
        trapped[i] = n_bounce[i] >= 2
        r_min[i] = r_gc[idx, i].min()
        r_max[i] = r_gc[idx, i].max()
        e0 = ke[0, i]
        e_err[i] = np.abs(ke[idx, i] - e0).max() / e0
        mu_err[i] = np.abs(mu[idx, i] - mu[0, i]).max() / mu[0, i]
        kinds[i] = "lost" if lost[i] else ("trapped" if trapped[i] else "passing")

    return OrbitSummary(
        kind=kinds, trapped=trapped & ~lost, lost=lost,
        loss_time=traj.loss_time, r_min=r_min, r_max=r_max,
        radial_width=r_max - r_min, energy_error=e_err, mu_error=mu_err,
        n_bounces=n_bounce,
    )


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------
def banana_width_analytic(field, r, energy_j, mass, charge, xi=0.0):
    """Leading-order banana full width.

    .. math:: w_b \\simeq 2\\,\\rho_\\theta\\,\\sqrt{\\epsilon}
              = 2\\,\\frac{m v \\sqrt{\\epsilon}}{q B_\\theta(r)}

    with ``v`` the total speed and ``B_theta`` the poloidal field at ``r``.
    Different textbooks differ by O(1) prefactors and by whether ``v`` or
    ``v_par`` at the midplane is used, so the **prefactor is not treated as
    predictive here**. What is tested is the *scaling*: ``w_b`` proportional to
    ``1/B_theta`` (equivalently to ``q``) and to ``sqrt(E)``. The fitted
    prefactor is reported in ``RESULTS.md`` rather than asserted.
    """
    v = np.sqrt(2.0 * energy_j / mass)
    b_theta = field.b_poloidal_reference(r)
    rho_theta = mass * v / (abs(charge) * b_theta)
    return 2.0 * rho_theta * np.sqrt(r / field.R0)


def mirror_ratio(field, x):
    """``B_max / B`` where ``B_max`` is the largest field on the same flux surface.

    Delegates to ``field.b_extrema_on_surface``, which is analytic for the
    circular model and computed by sampling the surface for a Solov'ev
    equilibrium. Working through that interface rather than assuming the
    maximum sits at ``R = R0 - r`` is what lets the same diagnostics run
    against a field whose surfaces are not circles.
    """
    x = np.atleast_2d(np.asarray(x, float))
    _, b_max = field.b_extrema_on_surface(x)
    b_local = np.linalg.norm(field.b_field(x), axis=-1)
    return b_max / b_local


def is_trapped(field, x, v):
    r"""Poloidal-angle-independent trapped/passing test from mu conservation.

    A particle mirrors before completing a poloidal circuit iff its turning
    point field ``B_turn = m v^2 / (2 mu)`` lies below the maximum field on its
    flux surface, which rearranges to

    .. math:: (1 - \xi^2)\,\frac{B_{\max}}{B} > 1

    with everything evaluated at the particle's current position. Because
    :math:`\mu` and the energy are (adiabatically) conserved, this expression
    is independent of where on the orbit it is evaluated -- unlike the
    midplane form :math:`|\xi| < \sqrt{2\epsilon/(1+\epsilon)}`, which is
    only valid at the outboard midplane and misclassifies elsewhere. See
    ``docs/DOC_SELF_REVIEW.md`` finding 20.
    """
    shape = np.shape(x)[:-1]
    xf = np.asarray(x, float).reshape(-1, 3)
    vf = np.asarray(v, float).reshape(-1, 3)
    v_par, _ = parallel_perp_split(field, xf, vf)
    speed = np.linalg.norm(vf, axis=-1)
    xi = np.divide(v_par, speed, out=np.zeros_like(v_par), where=speed > 0)
    return ((1.0 - xi**2) * mirror_ratio(field, xf) > 1.0).reshape(shape)


def make_loss_func(field, r_limit=None):
    """Return a callable flagging particles outside ``r_limit`` (default ``a``).

    The test is on the **particle** position, not the guiding centre, so a
    particle whose guiding centre sits inside the boundary but whose Larmor
    circle crosses it counts as lost. At 10 keV the gyroradius is ~1 cm
    against a 30 cm minor radius, so this shifts the effective boundary by
    ~3%; the alternative convention is compared in ``RESULTS.md``.
    """
    r_limit = field.boundary_label if r_limit is None else r_limit

    def loss(pos):
        return field.surface_label(pos) > r_limit

    return loss
