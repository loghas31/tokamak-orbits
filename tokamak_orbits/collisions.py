"""Monte Carlo pitch-angle scattering (the Lorentz collision operator).

Physics
-------
Collisions with a background species deflect a test particle's velocity
direction without, to leading order, changing its speed. The corresponding
kinetic operator is the **Lorentz operator**

.. math:: C[f] = \\frac{\\nu_d}{2}\\,\\frac{\\partial}{\\partial\\xi}
          \\left[(1-\\xi^2)\\frac{\\partial f}{\\partial\\xi}\\right]

acting on the pitch :math:`\\xi = v_\\parallel/v`, with :math:`\\nu_d` the
deflection frequency. Energy scattering and drag are **not** included: this is
pure pitch-angle diffusion at fixed :math:`|v|`. That is the standard reduced
model for neoclassical transport, where the physics of interest is trapped
particles being scattered across the trapped/passing boundary.

Monte Carlo form
----------------
The equivalent single-particle update (Boozer & Kuo-Petravic 1981) is

.. math:: \\xi' = \\xi\\,(1 - \\nu_d \\Delta t)
          \\pm \\sqrt{(1-\\xi^2)\\,\\nu_d \\Delta t}

with the sign drawn with equal probability. Its first two moments,

.. math:: \\langle\\Delta\\xi\\rangle = -\\nu_d \\Delta t\\, \\xi, \\qquad
          \\langle(\\Delta\\xi)^2\\rangle = (1-\\xi^2)\\,\\nu_d \\Delta t
          + O(\\Delta t^2)

are exactly the drift and diffusion coefficients of the operator above. The
scheme is therefore first-order accurate in :math:`\\nu_d\\Delta t` and requires
:math:`\\nu_d\\Delta t \\ll 1`; :func:`scatter_pitch` enforces this.

What is exactly true, and what is not
-------------------------------------
- :math:`|v|` is preserved to round-off, so the operator adds **no** energy
  error on top of the Boris pusher. This is asserted by a test.
- :math:`\\xi` **cannot** leave :math:`[-1, 1]`, so the clip in
  :func:`scatter_pitch` never fires. By Cauchy-Schwarz, with
  :math:`a = \\nu_d\\Delta t`,

  .. math:: \\xi(1-a) + \\sqrt{a}\\sqrt{1-\\xi^2}
            \\le \\sqrt{(1-a)^2 + a}\\,\\sqrt{\\xi^2 + (1-\\xi^2)}
            = \\sqrt{1 - a + a^2}

  and :math:`1 - a + a^2 < 1` for every :math:`0 < a < 1`. Verified over
  :math:`5\\times10^5` samples at :math:`a` up to 0.1: zero overshoots, largest
  post-update :math:`|\\xi'|` of 0.99751. The clip is a round-off guard only.
  An earlier version of this docstring asserted an overshoot artefact and
  quoted a bias for it; see ``docs/DOC_SELF_REVIEW.md`` finding 19.
- The background is a fixed Maxwellian at rest. There is no back-reaction, no
  momentum conservation between species, and therefore **no bootstrap current**.
"""
from __future__ import annotations

import numpy as np


def scatter_pitch(xi, nu_dt, rng, clip=True):
    """Advance pitch ``xi`` by one Monte Carlo pitch-angle scattering step.

    Parameters
    ----------
    xi : array_like
        Pitch cosine(s) ``v_par / v``, each in ``[-1, 1]``.
    nu_dt : float
        The product ``nu_d * dt`` for this step. Must be small; a value above
        0.1 raises, because the scheme's moments are only correct to
        ``O(nu dt)``. Measured accuracy of the Legendre decay rates: within
        0.2% at ``nu dt = 0.01``, degrading to 8% for ``l = 2`` at 0.02.
    rng : numpy.random.Generator
    clip : bool
        Clip the result into ``[-1, 1]``. A round-off guard only: the update
        provably cannot leave the interval for ``nu_dt < 1`` (see the module
        docstring). Pass ``False`` to check that claim rather than trust it.

    Returns
    -------
    ndarray
        The new pitch cosines.
    """
    nu_dt = float(nu_dt)
    if nu_dt < 0:
        raise ValueError("nu_dt must be non-negative")
    if nu_dt > 0.1:
        raise ValueError(
            f"nu_dt = {nu_dt:.3g} is too large; the Monte Carlo pitch-angle "
            "operator is only first-order accurate and needs nu_d*dt << 1. "
            "Reduce collide_every or nu_d."
        )
    xi = np.asarray(xi, float)
    sign = rng.integers(0, 2, size=xi.shape) * 2 - 1
    xi_new = xi * (1.0 - nu_dt) + sign * np.sqrt(
        np.clip(1.0 - xi**2, 0.0, None) * nu_dt)
    return np.clip(xi_new, -1.0, 1.0) if clip else xi_new


def apply_to_velocity(field, x, v, nu_dt, rng):
    """Scatter the pitch of ``v`` about the local field direction.

    Decomposes each velocity into components parallel and perpendicular to
    **B**, scatters the pitch, and rebuilds the velocity keeping ``|v|`` and
    the gyrophase fixed. Positions are untouched.

    Returns a new ``(N, 3)`` velocity array.
    """
    x = np.atleast_2d(np.asarray(x, float))
    v = np.atleast_2d(np.asarray(v, float))
    B = field.b_field(x)
    b_hat = B / np.linalg.norm(B, axis=-1, keepdims=True)

    speed = np.linalg.norm(v, axis=-1)
    v_par = np.sum(v * b_hat, axis=-1)
    v_perp_vec = v - v_par[:, None] * b_hat
    v_perp = np.linalg.norm(v_perp_vec, axis=-1)

    # unit vector along the existing perpendicular velocity; for a particle
    # with no perpendicular component any perpendicular direction will do, so
    # fall back to an arbitrary one built from the least-aligned global axis
    tiny = v_perp < 1e-12 * np.maximum(speed, 1e-30)
    e_perp = np.empty_like(v_perp_vec)
    ok = ~tiny
    e_perp[ok] = v_perp_vec[ok] / v_perp[ok, None]
    if tiny.any():
        axes = np.eye(3)
        ref = axes[np.argmin(np.abs(b_hat[tiny] @ axes.T), axis=-1)]
        tmp = np.cross(b_hat[tiny], ref)
        e_perp[tiny] = tmp / np.linalg.norm(tmp, axis=-1, keepdims=True)

    xi = np.divide(v_par, speed, out=np.zeros_like(v_par), where=speed > 0)
    xi_new = scatter_pitch(xi, nu_dt, rng)
    return (speed * xi_new)[:, None] * b_hat + (
        speed * np.sqrt(np.clip(1.0 - xi_new**2, 0.0, None)))[:, None] * e_perp


class PitchAngleCollisions:
    """Callable collision operator for :func:`tokamak_orbits.pusher.integrate`.

    Parameters
    ----------
    nu_d : float
        Deflection frequency (s^-1).
    seed : int
        Seed for the internal generator, so runs are reproducible.

    Notes
    -----
    The operator is applied every ``collide_every`` pusher steps with an
    effective ``nu_dt = nu_d * collide_every * dt``. Applying it less often
    with a proportionally larger ``nu_dt`` is the standard way to keep the cost
    negligible next to the orbit integration, and is exact in the same
    ``O(nu dt)`` sense provided ``nu_dt`` stays small.
    """

    def __init__(self, nu_d: float, seed: int = 0):
        if nu_d < 0:
            raise ValueError("nu_d must be non-negative")
        self.nu_d = float(nu_d)
        self.rng = np.random.default_rng(seed)
        self.n_calls = 0

    def __call__(self, field, x, v, dt_coll):
        self.n_calls += 1
        if self.nu_d == 0.0:
            return v
        return apply_to_velocity(field, x, v, self.nu_d * dt_coll, self.rng)

    def __repr__(self):
        return f"PitchAngleCollisions(nu_d={self.nu_d:.4g} s^-1)"


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------
def legendre_decay_rate(l: int, nu_d: float) -> float:
    """Decay rate of the ``l``-th Legendre moment under the Lorentz operator.

    Legendre polynomials are the eigenfunctions:
    ``C[P_l] = -(nu_d/2) l(l+1) P_l``, so

    .. math:: \\langle P_l(\\xi)\\rangle(t) =
              \\langle P_l(\\xi)\\rangle(0)\\,e^{-l(l+1)\\nu_d t/2}

    This is exact and has no free parameters, which makes it the sharpest
    available test of the collision operator.
    """
    return 0.5 * l * (l + 1) * nu_d


def bounce_frequency(field, energy_j, mass, r, xi_midplane=0.5):
    """Estimate the bounce angular frequency of a trapped particle.

    .. math:: \\omega_b \\simeq \\frac{v}{qR_0}\\sqrt{\\frac{\\epsilon}{2}}

    the standard large-aspect-ratio estimate. Used only to normalise
    collisionality; the measured bounce frequency from the orbits themselves is
    reported alongside it in ``RESULTS.md``.
    """
    v = np.sqrt(2.0 * energy_j / mass)
    q = float(field.safety_factor(r))
    eps = r / field.R0
    return v / (q * field.R0) * np.sqrt(eps / 2.0)


def effective_detrapping_frequency(nu_d: float, epsilon: float) -> float:
    """``nu_eff = nu_d / epsilon``.

    A trapped particle only needs to be scattered through a pitch angle
    ``~sqrt(epsilon)`` to cross the trapped/passing boundary, and pitch-angle
    diffusion covers that in a time ``epsilon/nu_d`` rather than ``1/nu_d``.
    The banana picture survives while ``nu_eff < omega_b``.
    """
    return nu_d / epsilon
