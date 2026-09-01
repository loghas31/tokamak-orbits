r"""A Solov'ev equilibrium: an exact analytic solution of Grad-Shafranov.

Why this exists
---------------
:class:`~tokamak_orbits.fields.TokamakField` is divergence-free and has
tokamak-like topology, but it is **not an equilibrium**: its flux surfaces are
concentric circles with no Shafranov shift, which do not solve the
Grad-Shafranov equation. Two consequences are documented in
``docs/DOC_SELF_REVIEW.md`` finding 4 and 13:

- :math:`|B|` is *exactly* proportional to :math:`1/R` on a flux surface, so
  the trapped fraction :math:`\sqrt{2\epsilon/(1+\epsilon)}` is exact **by
  construction** rather than approximate, and comparing against it tests the
  integrator rather than the field.
- The implied toroidal current density is not a flux function, and the nominal
  ``plasma_current`` is not the current the field actually encloses.

This module removes both objections by solving Grad-Shafranov exactly.

The equation
------------
In axisymmetry the equilibrium condition :math:`\nabla p = \mathbf{j}\times\mathbf{B}`
reduces to

.. math:: \Delta^{*}\psi \equiv R\frac{\partial}{\partial R}
          \left(\frac{1}{R}\frac{\partial\psi}{\partial R}\right)
          + \frac{\partial^{2}\psi}{\partial Z^{2}}
          = -\mu_{0}R^{2}p'(\psi) - FF'(\psi)

with the field reconstructed as
:math:`\mathbf{B} = \nabla\psi\times\nabla\phi + F(\psi)\nabla\phi`, which is
divergence-free identically.

The Solov'ev choice
-------------------
Taking both source terms constant -- :math:`\mu_{0}p' = -2c` and
:math:`FF' = -2ckR_{0}^{2}` -- admits the closed-form solution

.. math:: \psi(R,Z) = c\left[\frac{(R^{2}-R_{0}^{2})^{2}}{4}
          + k R_{0}^{2} Z^{2}\right]

Substituting gives :math:`\Delta^{*}\psi = 2cR^{2} + 2ckR_{0}^{2}` exactly, so
the equation is satisfied identically rather than approximately. Integrating
:math:`FF'` from the axis gives :math:`F^{2} = F_{0}^{2} - 4ckR_{0}^{2}\psi`
with :math:`F_{0} = B_{0}R_{0}`.

The resulting flux surfaces are **not** concentric circles. Near the magnetic
axis they are ellipses of elongation :math:`\kappa = 1/\sqrt{k}`; further out
the quartic term compresses them on the outboard side, so the *surface centres*
move inward as the surface grows -- equivalently, the magnetic axis sits
outboard of the boundary centre. That is the Shafranov shift, appearing as a
consequence of the equation rather than being imposed. (An earlier version of
this docstring said the surfaces shift outward, which is the same physics
described backwards; the midplane half-widths are :math:`A \mp A^2/2R_0`
outboard and inboard, so the centre shift is :math:`-A^2/2R_0`.) On the default
machine the boundary surface reaches :math:`R = 1.300` m outboard but
:math:`R = 0.557` m inboard, against :math:`R_{0} \pm a = 1.3 / 0.7` for the
circular model -- a large difference, and precisely the thing the circular
model cannot represent.

What this buys
--------------
:math:`|B|` is no longer proportional to :math:`1/R` on a surface, so the
mirror ratio, the trapping boundary and the trapped fraction all have to be
computed from the field rather than read off a formula. Comparing a measured
trapped fraction against :math:`\sqrt{2\epsilon/(1+\epsilon)}` becomes a real
test of a large-aspect-ratio approximation instead of an identity.
"""
from __future__ import annotations

import numpy as np

from .constants import MU_0


class SolovevField:
    """Exact Grad-Shafranov equilibrium, Solov'ev form.

    Parameters
    ----------
    major_radius : float
        ``R0``, the magnetic axis position (m). For this solution the axis sits
        exactly at ``R = R0``, ``Z = 0``, since ``dpsi/dR`` vanishes there.
    minor_radius : float
        ``a`` (m). Defines the plasma boundary as the flux surface passing
        through ``R = R0 + a`` on the midplane. Note the boundary is **not** a
        circle of this radius: see :attr:`r_inboard`.
    b_toroidal : float
        ``B0``, the vacuum toroidal field on the magnetic axis (T).
    q_axis : float
        Target safety factor on axis. Sets the Solov'ev constant via
        ``c = B0 / (2 q0 R0^2)``. The realised on-axis safety factor is
        **exactly** ``kappa * q_axis``, not ``q_axis``, so for an elongated
        equilibrium this parameter is a handle rather than a target -- at
        ``kappa = 1.8`` the realised ``q0`` is 1.8x the number passed in.
        :meth:`safety_factor_exact` and :attr:`q_axis` report what is actually
        realised, and the tests assert the ``kappa`` factor rather than
        assuming it away.
    elongation : float
        ``kappa``, the near-axis elongation. ``k = 1/kappa^2`` in the formula
        above. ``kappa = 1`` gives circular near-axis surfaces.
    """

    def __init__(self, major_radius=1.0, minor_radius=0.3, b_toroidal=2.0,
                 q_axis=1.0, elongation=1.0):
        if minor_radius <= 0 or major_radius <= 0:
            raise ValueError("major_radius and minor_radius must be positive")
        if minor_radius >= major_radius:
            raise ValueError("minor_radius must be smaller than major_radius")
        if q_axis <= 0:
            raise ValueError("q_axis must be positive")
        if elongation <= 0:
            raise ValueError("elongation must be positive")
        self.R0 = float(major_radius)
        self.a = float(minor_radius)
        self.B0 = float(b_toroidal)
        self.q0_target = float(q_axis)
        self.kappa = float(elongation)
        self.k = 1.0 / self.kappa**2
        self.c = self.B0 / (2.0 * self.q0_target * self.R0**2)
        self.F0 = self.B0 * self.R0
        self.psi_b = float(self.psi_of(self.R0 + self.a, 0.0))
        if self.f_squared(self.psi_b) <= 0:
            raise ValueError(
                "F^2 goes negative before the boundary: the toroidal field "
                "reverses. Reduce minor_radius, raise q_axis, or raise "
                "elongation.")

    # -- the flux function -------------------------------------------------
    def psi_of(self, R, Z):
        """Poloidal flux function, zero on the magnetic axis."""
        R = np.asarray(R, float)
        Z = np.asarray(Z, float)
        return self.c * ((R**2 - self.R0**2) ** 2 / 4.0
                         + self.k * self.R0**2 * Z**2)

    def f_squared(self, psi):
        """``F^2(psi) = F0^2 - 4 c k R0^2 psi``."""
        return self.F0**2 - 4.0 * self.c * self.k * self.R0**2 * np.asarray(psi, float)

    def pressure_gradient(self):
        """``dp/dpsi``, constant for a Solov'ev equilibrium."""
        return -2.0 * self.c / MU_0

    def ff_prime(self):
        """``F dF/dpsi``, constant for a Solov'ev equilibrium."""
        return -2.0 * self.c * self.k * self.R0**2

    # -- geometry ----------------------------------------------------------
    @property
    def epsilon(self):
        """Nominal inverse aspect ratio ``a / R0`` (outboard midplane)."""
        return self.a / self.R0

    @property
    def r_outboard(self):
        """Midplane major radius of the plasma boundary, outboard side."""
        return float(np.sqrt(self.R0**2 + 2.0 * np.sqrt(self.psi_b / self.c)))

    @property
    def r_inboard(self):
        """Midplane major radius of the plasma boundary, inboard side.

        This is **not** ``R0 - a``. The Shafranov shift built into the
        equilibrium pushes the outer surfaces outward, so the inboard leg sits
        further in than a circle of the same outboard radius would.
        """
        inner = self.R0**2 - 2.0 * np.sqrt(self.psi_b / self.c)
        if inner <= 0:
            raise ValueError("boundary surface encloses the machine axis")
        return float(np.sqrt(inner))

    def midplane_extent(self, psi):
        """``(R_inner, R_outer)`` where the surface ``psi`` crosses ``Z = 0``."""
        d = 2.0 * np.sqrt(np.maximum(psi, 0.0) / self.c)
        return (float(np.sqrt(max(self.R0**2 - d, 1e-12))),
                float(np.sqrt(self.R0**2 + d)))

    def surface_points(self, psi, n=721):
        """Sample ``(R, Z)`` around the closed flux surface ``psi``.

        The contour is parametrised by ``R`` rather than by a poloidal angle,
        because the surfaces are not star-shaped about ``R0`` at large ``psi``
        and an angle parametrisation misses parts of them.
        """
        r_in, r_out = self.midplane_extent(psi)
        R = np.linspace(r_in, r_out, n // 2)
        arg = (psi / self.c - (R**2 - self.R0**2) ** 2 / 4.0) / (
            self.k * self.R0**2)
        Z = np.sqrt(np.clip(arg, 0.0, None))
        return np.concatenate([R, R[::-1]]), np.concatenate([Z, -Z[::-1]])

    # -- the field ---------------------------------------------------------
    def b_field(self, pos):
        """Magnetic field at ``(N, 3)`` Cartesian positions."""
        pos = np.atleast_2d(np.asarray(pos, float))
        x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
        R = np.hypot(x, y)
        R = np.where(R < 1e-12, 1e-12, R)

        # B = grad(psi) x grad(phi) + F grad(phi)
        b_R = -(2.0 * self.c * self.k * self.R0**2 * z) / R
        b_Z = self.c * (R**2 - self.R0**2)
        psi = self.psi_of(R, z)
        f2 = self.f_squared(psi)
        b_phi = np.sqrt(np.clip(f2, 0.0, None)) / R

        cos_p, sin_p = x / R, y / R
        return np.stack([b_R * cos_p - b_phi * sin_p,
                         b_R * sin_p + b_phi * cos_p,
                         b_Z], axis=-1)

    def b_magnitude(self, pos):
        return np.linalg.norm(self.b_field(pos), axis=-1)

    @staticmethod
    def cylindrical(pos):
        pos = np.atleast_2d(pos)
        return np.hypot(pos[..., 0], pos[..., 1]), pos[..., 2]

    def flux_coords(self, pos):
        """``(r_eff, theta)`` -- the effective minor radius and a poloidal angle.

        The angle is the geometric one about the magnetic axis. It is a label
        for plotting, not a straight-field-line coordinate.
        """
        R, Z = self.cylindrical(np.atleast_2d(np.asarray(pos, float)))
        return self.surface_label(pos), np.arctan2(Z, R - self.R0)

    # -- interface shared with TokamakField --------------------------------
    def effective_minor_radius(self, psi):
        """Outboard-midplane distance from the magnetic axis to surface ``psi``.

        Used as the radial coordinate everywhere, in place of ``psi`` itself,
        so that the same diagnostics work against this field and the circular
        one without unit changes: it is in metres, zero on axis, and equal to
        ``a`` at the plasma boundary by construction. It is *not* the distance
        to the inboard leg of the same surface -- the Shafranov shift makes
        those differ, which is the physics this field exists to carry.
        """
        psi = np.asarray(psi, float)
        return np.sqrt(self.R0**2 + 2.0 * np.sqrt(np.maximum(psi, 0.0) / self.c)) - self.R0

    def surface_label(self, pos):
        """Effective minor radius (m) of the flux surface through ``pos``."""
        pos = np.atleast_2d(np.asarray(pos, float))
        R = np.hypot(pos[..., 0], pos[..., 1])
        return self.effective_minor_radius(self.psi_of(R, pos[..., 2]))

    def psi_from_effective_radius(self, r_eff):
        """Inverse of :meth:`effective_minor_radius`."""
        r_eff = np.asarray(r_eff, float)
        return self.c * ((r_eff + self.R0) ** 2 - self.R0**2) ** 2 / 4.0

    @property
    def boundary_label(self):
        return self.a

    def b_extrema_on_surface(self, pos):
        """``(B_min, B_max)`` on the flux surface through each position.

        Computed by sampling the surface, because for this equilibrium ``|B|``
        is *not* a simple function of ``R`` alone -- which is the whole point
        of using it.
        """
        r_eff = np.atleast_1d(self.surface_label(pos))
        psi = np.atleast_1d(self.psi_from_effective_radius(r_eff))
        lo = np.empty(psi.shape)
        hi = np.empty(psi.shape)
        for i, p in enumerate(psi.ravel()):
            R, Z = self.surface_points(max(float(p), 0.0), n=241)
            pts = np.stack([R, np.zeros_like(R), Z], axis=-1)
            b = self.b_magnitude(pts)
            lo.ravel()[i], hi.ravel()[i] = b.min(), b.max()
        return lo, hi

    @property
    def b_max_in_domain(self):
        """Largest ``|B|`` inside the plasma boundary (T).

        Found by sampling the boundary surface, which is where the field is
        strongest: ``|B|`` grows inward, and the boundary reaches furthest in.
        Not analytic here, because for this equilibrium ``|B|`` is not a
        function of ``R`` alone.
        """
        R, Z = self.surface_points(self.psi_b, 801)
        pts = np.stack([R, np.zeros_like(R), Z], axis=-1)
        return float(self.b_magnitude(pts).max())

    def safety_factor_exact(self, psi, n_theta=4001):
        r"""Safety factor by integrating around the flux surface.

        .. math:: q = \frac{1}{2\pi}\oint \frac{F}{R\,|\nabla\psi|}\,d\ell

        which follows from :math:`d\phi/d\ell = B_\phi/(R B_{\rm pol})` and
        :math:`B_{\rm pol} = |\nabla\psi|/R`.
        """
        psi = float(psi)
        if psi <= 0:
            return self._q_axis_limit()
        R, Z = self.surface_points(psi, n=n_theta)
        # close the contour and integrate along arclength
        R = np.append(R, R[0])
        Z = np.append(Z, Z[0])
        dl = np.hypot(np.diff(R), np.diff(Z))
        Rm = 0.5 * (R[1:] + R[:-1])
        Zm = 0.5 * (Z[1:] + Z[:-1])
        dpsi_dR = self.c * Rm * (Rm**2 - self.R0**2)
        dpsi_dZ = 2.0 * self.c * self.k * self.R0**2 * Zm
        grad = np.hypot(dpsi_dR, dpsi_dZ)
        good = grad > 1e-30
        F = np.sqrt(np.clip(self.f_squared(psi), 0.0, None))
        return float(np.sum((F / (Rm[good] * grad[good])) * dl[good])
                     / (2.0 * np.pi))

    def _q_axis_limit(self):
        """Analytic on-axis limit ``q0 = B0 kappa / (2 c R0^2)``... measured.

        Rather than trust an expansion, evaluate ``safety_factor_exact`` on a
        surface close to the axis. For ``kappa = 1`` this reproduces
        ``B0 / (2 c R0^2) = q0_target`` exactly.
        """
        return self.safety_factor_exact(1e-6 * self.psi_b)

    @property
    def q_axis(self):
        return self._q_axis_limit()

    @property
    def q_edge(self):
        return self.safety_factor_exact(self.psi_b)

    def __repr__(self):
        return (f"SolovevField(R0={self.R0:g} m, a={self.a:g} m, "
                f"B0={self.B0:g} T, kappa={self.kappa:g}, "
                f"q0={self.q_axis:.3f}, qa={self.q_edge:.3f}, "
                f"R_in={self.r_inboard:.3f}, R_out={self.r_outboard:.3f})")


def grad_shafranov_residual(field, R, Z, h=1e-5):
    r"""Numerical residual of the Grad-Shafranov equation, normalised.

    Returns
    :math:`\left(\Delta^{*}\psi + \mu_{0}R^{2}p' + FF'\right) /
    |\Delta^{*}\psi|`, which is zero for a true equilibrium.

    This is the test the circular model in :mod:`tokamak_orbits.fields` cannot
    pass: it is divergence-free, but it does not solve this equation.
    """
    R = np.atleast_1d(np.asarray(R, float))
    Z = np.atleast_1d(np.asarray(Z, float))
    psi = field.psi_of

    d2psi_dZ2 = (psi(R, Z + h) - 2 * psi(R, Z) + psi(R, Z - h)) / h**2
    # R d/dR ( (1/R) dpsi/dR ) by central differences
    def inner(Rq):
        return (psi(Rq + h, Z) - psi(Rq - h, Z)) / (2 * h) / Rq
    d_inner = (inner(R + h) - inner(R - h)) / (2 * h)
    delta_star = R * d_inner + d2psi_dZ2

    rhs = -MU_0 * R**2 * field.pressure_gradient() - field.ff_prime()
    return (delta_star - rhs) / np.maximum(np.abs(delta_star), 1e-30)


def force_balance_residual(field, R, Z, h=1e-5):
    r"""Residual of :math:`\mathbf{J}\times\mathbf{B} = \nabla p`, from ``b_field``.

    Why this exists as well as :func:`grad_shafranov_residual`
    ---------------------------------------------------------
    The Grad-Shafranov residual is computed from ``psi_of`` and the two
    constant source terms, and for a Solov'ev solution
    :math:`\Delta^*\psi = 2cR^2 + 2ckR_0^2` holds **algebraically** -- both
    central differences in that function are exact on a quartic, so what it
    returns is floating-point cancellation, not physics. It scales as
    :math:`1/h^2` and can be made arbitrarily small by changing ``h``. Worse,
    it never touches ``b_field``: mutating the :math:`F^2(\psi)` integration
    so that the toroidal field no longer solves the equation leaves that
    residual, and every field-structure test, completely unchanged.

    This function closes that hole. It builds :math:`\mathbf{J} = \nabla\times
    \mathbf{B}/\mu_0` by differencing ``b_field`` itself, forms
    :math:`\mathbf{J}\times\mathbf{B}`, and compares it against
    :math:`\nabla p = p'(\psi)\nabla\psi` with the same :math:`p'` the
    equilibrium claims. It therefore tests the field the *particles* see, and
    it fails if :math:`F^2(\psi)` is inconsistent with :math:`FF'`.

    Returns the residual normalised by :math:`|\mathbf{J}||\mathbf{B}|`.
    """
    R = np.atleast_1d(np.asarray(R, float))
    Z = np.atleast_1d(np.asarray(Z, float))
    pts = np.stack([R, np.zeros_like(R), Z], axis=-1)

    b = field.b_field(pts)
    jac = np.zeros((len(pts), 3, 3))
    for k in range(3):
        pp, pm = pts.copy(), pts.copy()
        pp[:, k] += h
        pm[:, k] -= h
        jac[:, :, k] = (field.b_field(pp) - field.b_field(pm)) / (2 * h)
    curl = np.stack([jac[:, 2, 1] - jac[:, 1, 2],
                     jac[:, 0, 2] - jac[:, 2, 0],
                     jac[:, 1, 0] - jac[:, 0, 1]], axis=-1)
    j = curl / MU_0
    jxb = np.cross(j, b)

    # grad p = p'(psi) grad psi, in Cartesian at y = 0 (so R is along x)
    dpsi_dR = (field.psi_of(R + h, Z) - field.psi_of(R - h, Z)) / (2 * h)
    dpsi_dZ = (field.psi_of(R, Z + h) - field.psi_of(R, Z - h)) / (2 * h)
    grad_p = np.zeros_like(pts)
    grad_p[:, 0] = field.pressure_gradient() * dpsi_dR
    grad_p[:, 2] = field.pressure_gradient() * dpsi_dZ

    scale = (np.linalg.norm(j, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.linalg.norm(jxb - grad_p, axis=-1) / np.maximum(scale, 1e-30)
