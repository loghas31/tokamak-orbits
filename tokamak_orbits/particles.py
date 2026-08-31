"""Particle initialisation and pitch-angle sampling."""
from __future__ import annotations

import numpy as np

from .constants import SPECIES, ev_to_joule
from .fields import TokamakField


def speed_from_energy(energy_ev: float, mass: float) -> float:
    """Non-relativistic speed for a given kinetic energy in eV."""
    return float(np.sqrt(2.0 * ev_to_joule(energy_ev) / mass))


def local_basis(field, pos: np.ndarray):
    """Return orthonormal ``(b_hat, e1, e2)`` at each position, ``(N, 3)`` each.

    ``b_hat`` is along **B**; ``e1`` and ``e2`` span the perpendicular plane.
    ``e1`` is built from whichever global axis is least aligned with ``b_hat``,
    so the construction never degenerates.
    """
    pos = np.atleast_2d(pos)
    B = field.b_field(pos)
    b_hat = B / np.linalg.norm(B, axis=-1, keepdims=True)

    axes = np.eye(3)
    proj = np.abs(b_hat @ axes.T)              # (N, 3)
    ref = axes[np.argmin(proj, axis=-1)]       # (N, 3)

    e1 = np.cross(b_hat, ref)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(b_hat, e1)
    return b_hat, e1, e2


def initialise(
    field,
    energy_ev: float = 10e3,
    species: str = "D",
    r_start=0.15,
    theta_start=0.0,
    phi_start=0.0,
    pitch=0.5,
    gyrophase=0.0,
):
    """Build ``(x0, v0, mass, charge)`` for one or more particles.

    Parameters
    ----------
    r_start, theta_start, phi_start : array_like
        Launch position in flux/toroidal coordinates. Broadcast together.
    pitch : array_like
        ``xi = v_parallel / v``, in ``[-1, 1]``. ``xi = 0`` is purely
        perpendicular (deeply trapped); ``|xi| = 1`` is purely parallel
        (co- or counter-passing).
    gyrophase : array_like
        Initial phase of the perpendicular velocity in the ``(e1, e2)`` plane.

    Notes
    -----
    The pitch is defined *at the launch point*, so two particles launched with
    the same ``xi`` at different poloidal angles are not equivalent: what is
    conserved is ``mu``, and the trapping condition depends on the launch
    ``B``. All ensembles in this project launch on the outboard midplane
    (``theta = 0``), the weakest-field point on a flux surface, which is the
    conventional choice because it makes ``xi`` map monotonically onto the
    trapped/passing boundary.
    """
    mass, charge = SPECIES[species]
    r_start, theta_start, phi_start, pitch, gyrophase = np.broadcast_arrays(
        np.asarray(r_start, float), np.asarray(theta_start, float),
        np.asarray(phi_start, float), np.asarray(pitch, float),
        np.asarray(gyrophase, float),
    )
    r_start = r_start.ravel(); theta_start = theta_start.ravel()
    phi_start = phi_start.ravel(); pitch = pitch.ravel(); gyrophase = gyrophase.ravel()

    if np.any(np.abs(pitch) > 1.0):
        raise ValueError("pitch xi = v_par/v must lie in [-1, 1]")

    R = field.R0 + r_start * np.cos(theta_start)
    Z = r_start * np.sin(theta_start)
    x0 = np.stack([R * np.cos(phi_start), R * np.sin(phi_start), Z], axis=-1)

    v = speed_from_energy(energy_ev, mass)
    b_hat, e1, e2 = local_basis(field, x0)
    v_par = v * pitch
    v_perp = v * np.sqrt(np.clip(1.0 - pitch**2, 0.0, None))
    v0 = (
        v_par[:, None] * b_hat
        + v_perp[:, None] * (np.cos(gyrophase)[:, None] * e1
                             + np.sin(gyrophase)[:, None] * e2)
    )
    return x0, v0, mass, charge


def sample_pitch(n: int, rng=None, mode: str = "uniform_xi"):
    """Sample ``n`` pitch values ``xi = v_par / v``.

    ``mode='uniform_xi'``
        Uniform in ``xi`` on ``[-1, 1]``. This is the correct sampling for an
        **isotropic** velocity distribution: the solid-angle element is
        ``d(cos alpha) dphi`` and ``xi = cos alpha``, so uniform in ``xi`` is
        uniform on the sphere. Used for all trapped-fraction work.
    ``mode='uniform_angle'``
        Uniform in the pitch *angle* ``alpha = arccos(xi)``. This over-weights
        near-perpendicular particles and is **not** isotropic. Provided only so
        the difference can be demonstrated; see ``docs/DOC_SELF_REVIEW.md``
        finding 3.
    """
    rng = np.random.default_rng(rng)
    if mode == "uniform_xi":
        return rng.uniform(-1.0, 1.0, n)
    if mode == "uniform_angle":
        return np.cos(rng.uniform(0.0, np.pi, n))
    raise ValueError(f"unknown sampling mode {mode!r}")


def trapped_fraction_analytic(epsilon: float) -> float:
    """Standard large-aspect-ratio estimate ``sqrt(2 eps / (1 + eps))``.

    The fraction of an isotropic distribution that is trapped on a flux
    surface of local inverse aspect ratio ``eps = r / R0``, from the trapping
    condition ``|xi_0| < sqrt(2 eps / (1 + eps))`` for particles launched at
    the outboard midplane. Leading order only; used as the reference the
    simulation is checked against, not as ground truth. See ``docs/PHYSICS.md``.
    """
    return float(np.sqrt(2.0 * epsilon / (1.0 + epsilon)))


def trapping_boundary_pitch(field, r: float) -> float:
    """Critical launch pitch separating trapped from passing, from |B| alone.

    A particle launched at the outboard midplane with pitch ``xi_0`` mirrors if
    ``1 - xi_0^2 > B_min / B_max`` evaluated over its flux surface, giving
    ``|xi_0| < sqrt(1 - B_min/B_max)``. This uses the *actual* numerical field
    rather than the ``sqrt(2 eps/(1+eps))`` expansion, so it remains valid at
    finite aspect ratio and with the poloidal field included.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    R = field.R0 + r * np.cos(theta)
    Z = r * np.sin(theta)
    pos = np.stack([R, np.zeros_like(R), Z], axis=-1)
    B = np.linalg.norm(field.b_field(pos), axis=-1)
    return float(np.sqrt(np.clip(1.0 - B.min() / B.max(), 0.0, 1.0)))
