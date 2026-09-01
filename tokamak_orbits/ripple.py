r"""Toroidal field ripple: breaking axisymmetry without breaking ``div B = 0``.

Why this module exists
----------------------
Every field in this package so far is axisymmetric, which makes the canonical
toroidal momentum an exact invariant and guarantees that no collisionless
orbit can wander in radius indefinitely. Real tokamaks have a **finite number
of toroidal field coils**, so :math:`B_\phi` is periodic in :math:`\phi`
rather than constant, the invariant is destroyed, and a class of loss
mechanisms appears that axisymmetric models cannot represent at all:
particles trapped in the local wells *between* coils, which then drift
vertically out of the plasma.

The trap this module was written to avoid
-----------------------------------------
The obvious implementation is to write

.. math:: B_\phi = \frac{B_0R_0}{R}\bigl[1 + \delta(R)\cos N\phi\bigr]

and stop there. That field is **not** divergence-free: it has a
:math:`\phi`-derivative with nothing to cancel it. This is exactly the error
that produced finding 2 (a 4.8% divergence in the poloidal field), and the
same fix applies -- derive the field from a potential rather than writing the
components down by hand.

The ripple is a **vacuum** perturbation: it is produced by coils outside the
plasma, so it carries no current and can be written as the gradient of a
scalar potential, :math:`\mathbf{B}_r = \nabla\chi`. Then
:math:`\nabla\cdot\mathbf{B}_r = \nabla^2\chi`, so the field is
divergence-free **iff** :math:`\chi` is harmonic. Taking

.. math:: \chi = \frac{B_0R_0\delta_0}{N}
                 \left(\frac{R}{R_0}\right)^{N}\sin N\phi

and checking :math:`\nabla^2\chi = \chi_{RR} + \chi_R/R + \chi_{\phi\phi}/R^2
+ \chi_{ZZ}`: the first three terms give
:math:`\left[N(N-1) + N - N^2\right]R^{N-2} = 0` identically, and the last
vanishes because :math:`\chi` is independent of :math:`Z`. So this potential
is harmonic for *any* :math:`N`, and the resulting components

.. math:: B_R = g(R)\sin N\phi, \quad
          B_\phi = g(R)\cos N\phi, \quad
          B_Z = 0, \qquad
          g(R) = \frac{B_0R_0\delta_0}{R}\left(\frac{R}{R_0}\right)^{N}

are divergence-free to machine precision, by construction rather than by
tuning. The :math:`(R/R_0)^N` profile is not an ad-hoc choice either: the
:math:`Z`-independent solutions of that Euler equation are
:math:`R^{\pm N}`, and :math:`R^{-N}` is singular on the machine axis, so
:math:`R^{+N}` is the only one **regular in the plasma**. It reproduces the
observed behaviour of real machines, where the ripple is negligible on axis
and concentrated at the outboard edge. With :math:`N = 16` and
:math:`a/R_0 = 0.3` it is a factor :math:`1.3^{16} = 66.5` larger at the
outboard boundary than on axis.

Because :math:`\nabla\times\nabla\chi = 0`, adding this to any
divergence-free field leaves it divergence-free, so the same wrapper works on
:class:`~tokamak_orbits.fields.TokamakField` and on
:class:`~tokamak_orbits.equilibrium.SolovevField` without modification.

What ripple does to orbits
--------------------------
Two distinct mechanisms, and they have different thresholds:

**Ripple trapping.** Where the ripple modulation beats the :math:`1/R`
variation along a field line, :math:`|B|` acquires local minima between
coils. A particle with small enough :math:`v_\parallel` is trapped in one of
these wells, no longer samples the poloidal circuit that averages its
vertical drift to zero, and drifts straight out. The standard criterion is

.. math:: \alpha_{\rm rt} =
          \frac{\epsilon\,|\sin\theta|}{N q\,\delta} < 1

with :math:`\epsilon = r/R_0`; :func:`ripple_well_parameter` evaluates it and
:func:`ripple_well_fraction` measures the fraction of a flux surface that
satisfies it.

**Stochastic ripple diffusion.** Banana-trapped particles whose turning
points sit in the ripple field get a small random radial kick each bounce.
Above the Goldston-White-Boozer threshold these kicks decorrelate and the
banana tips diffuse. :func:`gwb_threshold` evaluates that criterion.
"""
from __future__ import annotations

import numpy as np

from .fields import Field, TokamakField


class RippledField(Field):
    r"""Any axisymmetric field plus a divergence-free vacuum ripple.

    Parameters
    ----------
    base : Field
        The axisymmetric field to perturb. Must expose ``R0``, ``a``,
        ``b_field`` and the surface interface used by the diagnostics.
    n_coils : int
        Number of toroidal field coils, :math:`N`.
    delta_edge : float
        Peak ripple amplitude :math:`\delta` at the **outboard midplane
        boundary**, :math:`R = R_0 + a`. Real machines sit around 0.1-2%
        there; the default 1% is deliberately on the pessimistic side so
        that the effect is measurable in a short run.

    Notes
    -----
    Every attribute not defined here is delegated to ``base``, so a
    ``RippledField`` is a drop-in replacement anywhere a field is taken --
    including :func:`~tokamak_orbits.diagnostics.make_loss_func`, which uses
    the base surface label unchanged because the ripple does not move the
    wall.
    """

    def __init__(self, base=None, n_coils: int = 16, delta_edge: float = 0.01):
        self.base = TokamakField() if base is None else base
        if int(n_coils) < 2:
            raise ValueError("n_coils must be at least 2")
        if delta_edge < 0:
            raise ValueError("delta_edge must be non-negative")
        self.n_coils = int(n_coils)
        self.delta_edge = float(delta_edge)
        self.R0 = float(self.base.R0)
        self.a = float(self.base.a)
        # B0 is the on-axis toroidal field of the base machine
        self.B0 = float(self.base.B0)
        # delta_0 is normalised so that delta(R0 + a) = delta_edge
        self.delta_0 = self.delta_edge / ((self.R0 + self.a) / self.R0) ** self.n_coils

    # -- delegation --------------------------------------------------------
    def __getattr__(self, name):
        # only reached for attributes not found on self; guard against
        # recursion during __init__, before ``base`` is bound
        try:
            base = self.__dict__["base"]
        except KeyError:                                    # pragma: no cover
            raise AttributeError(name) from None
        return getattr(base, name)

    # -- the ripple itself -------------------------------------------------
    def delta(self, R):
        r"""Local ripple amplitude :math:`\delta(R) = \delta_0 (R/R_0)^N`."""
        return self.delta_0 * (np.asarray(R, float) / self.R0) ** self.n_coils

    def ripple_potential(self, pos):
        r""":math:`\chi`, the harmonic scalar potential the ripple comes from.

        Exposed so the tests can check :math:`\nabla^2\chi = 0` and
        :math:`\nabla\chi = \mathbf{B}_{\rm ripple}` independently, rather
        than trusting the hand-differentiated components below.
        """
        pos = np.atleast_2d(np.asarray(pos, float))
        R = np.hypot(pos[:, 0], pos[:, 1])
        phi = np.arctan2(pos[:, 1], pos[:, 0])
        return (self.B0 * self.R0 * self.delta_0 / self.n_coils
                * (R / self.R0) ** self.n_coils * np.sin(self.n_coils * phi))

    def ripple_field(self, pos):
        """The ripple contribution alone, in Cartesian components."""
        pos = np.atleast_2d(np.asarray(pos, float))
        R = np.hypot(pos[:, 0], pos[:, 1])
        Rs = np.where(R < 1e-12, 1e-12, R)
        phi = np.arctan2(pos[:, 1], pos[:, 0])
        g = self.B0 * self.R0 * self.delta(Rs) / Rs
        s, c = np.sin(self.n_coils * phi), np.cos(self.n_coils * phi)
        b_r, b_phi = g * s, g * c
        cp, sp = np.cos(phi), np.sin(phi)
        out = np.zeros_like(pos)
        out[:, 0] = b_r * cp - b_phi * sp
        out[:, 1] = b_r * sp + b_phi * cp
        return out

    def b_field(self, pos):
        return self.base.b_field(pos) + self.ripple_field(pos)

    # -- interface the diagnostics need ------------------------------------
    @property
    def b_max_in_domain(self):
        """Base maximum, widened by the ripple at the inboard edge."""
        return float(self.base.b_max_in_domain
                     * (1.0 + float(self.delta(self.R0 - self.a))))

    def b_extrema_on_surface(self, pos):
        """Base extrema, widened by the local ripple at each extremum.

        The global maximum of ``|B|`` on a surface is still the inboard
        midplane: the ``1/R`` variation is a factor of ``(R0+a)/(R0-a)``
        (1.86 for the default machine) against a ripple of order 1%.
        """
        lo, hi = self.base.b_extrema_on_surface(pos)
        r = np.asarray(self.base.surface_label(pos), float)
        d_out = self.delta(self.R0 + r)
        d_in = self.delta(self.R0 - r)
        return lo * (1.0 - d_out), hi * (1.0 + d_in)

    def safety_factor(self, r):
        """``q`` at effective minor radius ``r``, whichever the base provides.

        `TokamakField` exposes the cylindrical `safety_factor(r)`;
        `SolovevField` exposes only `safety_factor_exact(psi)`. Without this
        shim `ripple_well_parameter` and `gwb_threshold` raise
        `AttributeError` on a Solov'ev base -- which they did, and only the
        divergence test covered that combination.
        """
        base = self.base
        if hasattr(base, "safety_factor"):
            return base.safety_factor(r)
        psi = base.psi_from_effective_radius(r)
        return np.vectorize(base.safety_factor_exact)(psi)

    def __repr__(self):
        return (f"RippledField({self.base!r}, n_coils={self.n_coils}, "
                f"delta_edge={self.delta_edge:.3%})")


# --------------------------------------------------------------------------
# ripple-loss criteria
# --------------------------------------------------------------------------
def ripple_well_parameter(field: RippledField, r, theta):
    r"""Ripple-well parameter :math:`\alpha_{\rm rt}`.

    .. math:: \alpha_{\rm rt} = \frac{\epsilon\,|\sin\theta|}{N q(r)\,\delta}

    A local magnetic well exists between coils where
    :math:`\alpha_{\rm rt} < 1`: there the ripple modulation along the field
    line beats the poloidal variation of :math:`|B|`. The midplane
    (:math:`\theta = 0`) is always in a well, because that is where the
    ripple is largest and the poloidal gradient vanishes.

    With :math:`\delta = 0` the expression is :math:`0/0` on the midplane, so
    it is defined to be :math:`+\infty` there: no ripple, no wells.
    """
    r = np.asarray(r, float)
    theta = np.asarray(theta, float)
    q = np.asarray(field.safety_factor(r), float)
    R = field.R0 + r * np.cos(theta)
    d = np.asarray(field.delta(R), float)
    eps = r / field.R0
    den = field.n_coils * q * d
    out = np.full(np.broadcast(eps * theta, den).shape, np.inf)
    good = den > 0
    num = np.broadcast_to(eps * np.abs(np.sin(theta)), out.shape)
    np.divide(num, np.broadcast_to(den, out.shape),
              out=out, where=good)
    return out if out.ndim else float(out)


def ripple_well_fraction(field: RippledField, r, n_theta: int = 20001,
                         weight="area"):
    """Fraction of the flux surface at ``r`` that contains a ripple well.

    ``weight="area"`` (the default) weights each poloidal angle by the surface
    element :math:`\propto R_0 + r\cos\theta`. This matters: the well region
    sits at :math:`\theta \approx 0`, which is exactly where that weight is
    largest, so a plain average over :math:`\theta` **understates** the area
    fraction -- 0.131 against 0.163 at :math:`\delta_{\rm edge} = 1\%`,
    :math:`r = 0.25` m, a 24% error. ``weight="angle"`` gives the unweighted
    poloidal-angle fraction, kept so the difference can be quoted.
    """
    th = np.linspace(-np.pi, np.pi, int(n_theta))
    inside = (ripple_well_parameter(field, r, th) < 1.0).astype(float)
    if weight == "angle":
        return float(np.mean(inside))
    if weight != "area":
        raise ValueError("weight must be 'area' or 'angle'")
    w = field.R0 + np.asarray(r, float) * np.cos(th)
    return float(np.sum(inside * w) / np.sum(w))


def gwb_threshold(field: RippledField, r, theta_tip, rho, dq_dr=None):
    r"""Goldston-White-Boozer stochasticity parameter (dimensionless).

    .. math:: \sigma_{\rm GWB} =
              \frac{1}{\pi N q'\rho}
              \left(\frac{\epsilon}{N q}\right)^{3/2}
              \frac{1}{\delta}

    Banana tips diffuse stochastically where :math:`\sigma < 1`. Every factor
    here is dimensionless except :math:`q'\rho`, which is
    :math:`(dq/dr)\,\rho` and therefore dimensionless too.

    An earlier version of this function substituted
    :math:`\partial\ln\delta/\partial r = N/R` for :math:`q'` and then
    multiplied by :math:`R_0`, which left :math:`\sigma` with units of inverse
    length -- invisible on the default machine because :math:`R_0 = 1` m, and
    caught only by scaling the machine: :math:`\sigma \propto R_0^{-3/2}` at
    fixed :math:`\epsilon, \delta, N, q, \rho`. It now uses the real
    :math:`dq/dr`, differenced from the base field's own safety factor unless
    one is supplied.

    The numerical prefactor differs between references by order unity, so this
    is reported as a **diagnostic with its inputs**, not as a calibrated
    threshold, and nothing in the package branches on it.
    """
    r = np.asarray(r, float)
    q = np.asarray(field.safety_factor(r), float)
    if dq_dr is None:
        h = 1e-4 * field.a
        dq_dr = ((np.asarray(field.safety_factor(r + h), float)
                  - np.asarray(field.safety_factor(np.maximum(r - h, 1e-9)),
                               float)) / (2 * h))
    dq_dr = np.asarray(dq_dr, float)
    eps = r / field.R0
    R = field.R0 + r * np.cos(np.asarray(theta_tip, float))
    d = field.delta(R)
    denom = np.pi * field.n_coils * np.abs(dq_dr) * rho * d
    return (eps / (field.n_coils * q)) ** 1.5 / np.maximum(denom, 1e-300)


def ripple_amplitude_measured(field, R, Z=0.0, n_phi: int = 721):
    """Measure ``(Bmax - Bmin) / (Bmax + Bmin)`` around one toroidal circuit.

    Used to check that the ripple the orbits actually see is the ripple the
    model claims, rather than trusting :meth:`RippledField.delta`.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi))
    R = float(R)
    pts = np.stack([R * np.cos(phi), R * np.sin(phi),
                    np.full_like(phi, float(Z))], axis=-1)
    b = np.linalg.norm(field.b_field(pts), axis=-1)
    return float((b.max() - b.min()) / (b.max() + b.min()))
