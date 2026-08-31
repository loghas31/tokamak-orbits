"""Magnetic field models.

Geometry and conventions
------------------------
Cartesian coordinates ``(x, y, z)`` are used for integration because the Boris
pusher is cleanest there. Cylindrical coordinates are derived on the fly::

    R = sqrt(x^2 + y^2)          major radius
    Z = z                        height above the midplane
    phi = atan2(y, x)            toroidal angle
    r = sqrt((R - R0)^2 + Z^2)   minor radius from the magnetic axis
    theta = atan2(Z, R - R0)     poloidal angle

The unit vectors form a right-handed triad in the order ``(R, phi, Z)``::

    R_hat   = ( x/R,  y/R, 0)
    phi_hat = (-y/R,  x/R, 0)
    Z_hat   = (   0,    0, 1)      with  R_hat x phi_hat = Z_hat

Sign conventions
----------------
``b_toroidal > 0`` puts the toroidal field along ``+phi_hat``. The plasma
current is taken parallel to the toroidal field (the standard "co-current"
tokamak configuration), so by Ampere's law the poloidal field direction is
``phi_hat x r_hat_pol``, which equals ``-theta_hat``. This is a convention,
not physics: flipping the sign of ``plasma_current`` mirrors every orbit in
the midplane and changes nothing measurable about confinement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import MU_0, DEFAULT_MACHINE


class Field:
    """Base class. Subclasses implement :meth:`b_field` and optionally
    :meth:`e_field`, both taking ``(N, 3)`` positions and returning ``(N, 3)``."""

    def b_field(self, pos: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def e_field(self, pos: np.ndarray) -> np.ndarray:
        return np.zeros_like(pos)

    def b_magnitude(self, pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.b_field(pos), axis=-1)


@dataclass
class UniformField(Field):
    """Spatially uniform B (and optionally E). Used for validation tests."""

    b_vec: tuple = (0.0, 0.0, 1.0)
    e_vec: tuple = (0.0, 0.0, 0.0)

    def b_field(self, pos: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.b_vec, float), np.shape(pos)).copy()

    def e_field(self, pos: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.e_vec, float), np.shape(pos)).copy()


class TokamakField(Field):
    """Axisymmetric tokamak field: 1/R toroidal plus a poloidal field from a
    toy plasma-current profile.

    The current density is taken as

    .. math:: j_\\phi(r) = j_0 \\left(1 - (r/a)^2\\right)^\\nu, \\quad r < a

    which integrates to an enclosed current

    .. math:: I(r) = \\frac{\\pi j_0 a^2}{\\nu + 1}
                     \\left[1 - (1 - (r/a)^2)^{\\nu+1}\\right]

    and hence, by Ampere's law, a poloidal field
    :math:`B_\\theta(r) = \\mu_0 I(r) / (2\\pi r)`. Outside the plasma the
    enclosed current saturates at :math:`I_p` and :math:`B_\\theta \\propto 1/r`.

    Parameters
    ----------
    major_radius : float
        ``R0``, radius of the magnetic axis (m).
    minor_radius : float
        ``a``, plasma edge minor radius (m).
    b_toroidal : float
        ``B0``, toroidal field strength *on the magnetic axis* (T).
    plasma_current : float
        ``Ip``, total toroidal plasma current (A). Set to zero to recover a
        pure vacuum toroidal field.
    current_peaking : float
        ``nu``, peaking exponent of the current profile. ``nu = 0`` is a flat
        current, ``nu = 1`` a parabolic one.
    """

    def __init__(
        self,
        major_radius: float = DEFAULT_MACHINE["major_radius"],
        minor_radius: float = DEFAULT_MACHINE["minor_radius"],
        b_toroidal: float = DEFAULT_MACHINE["b_toroidal"],
        plasma_current: float = DEFAULT_MACHINE["plasma_current"],
        current_peaking: float = DEFAULT_MACHINE["current_peaking"],
    ):
        if minor_radius <= 0 or major_radius <= 0:
            raise ValueError("major_radius and minor_radius must be positive")
        if minor_radius >= major_radius:
            raise ValueError("minor_radius must be smaller than major_radius")
        if current_peaking < 0:
            raise ValueError("current_peaking (nu) must be non-negative")
        self.R0 = float(major_radius)
        self.a = float(minor_radius)
        self.B0 = float(b_toroidal)
        self.Ip = float(plasma_current)
        self.nu = float(current_peaking)

    # -- geometry ---------------------------------------------------------
    @property
    def epsilon(self) -> float:
        """Inverse aspect ratio ``a / R0``."""
        return self.a / self.R0

    @staticmethod
    def cylindrical(pos: np.ndarray):
        """Return ``(R, Z)`` for ``(N, 3)`` Cartesian positions."""
        pos = np.atleast_2d(pos)
        R = np.hypot(pos[..., 0], pos[..., 1])
        return R, pos[..., 2]

    def flux_coords(self, pos: np.ndarray):
        """Return ``(r, theta)``: minor radius and poloidal angle."""
        R, Z = self.cylindrical(pos)
        dR = R - self.R0
        return np.hypot(dR, Z), np.arctan2(Z, dR)

    # -- profiles ---------------------------------------------------------
    def enclosed_current(self, r):
        """Toroidal current enclosed within minor radius ``r`` (A)."""
        r = np.asarray(r, float)
        x = np.clip((r / self.a) ** 2, 0.0, 1.0)
        return self.Ip * (1.0 - (1.0 - x) ** (self.nu + 1.0))

    def b_poloidal_reference(self, r):
        """Reference poloidal field profile ``mu0 I(r) / (2 pi r)`` (T).

        This is the *straight-cylinder* Ampere result. It is the value the
        actual poloidal field takes on the magnetic axis circle ``R = R0``;
        elsewhere the true field carries an extra ``R0/R`` factor (see
        :meth:`b_poloidal`). Regularised on axis, where ``I(r) ~ r^2`` so
        ``B_theta ~ r`` and the naive ``I/r`` is 0/0.
        """
        r = np.asarray(r, float)
        small = r < 1e-12 * self.a
        r_safe = np.where(small, 1e-12 * self.a, r)
        b = MU_0 * self.enclosed_current(r_safe) / (2.0 * np.pi * r_safe)
        return np.where(small, 0.0, b)

    def b_poloidal(self, r, R=None):
        """Poloidal field magnitude at minor radius ``r``, major radius ``R`` (T).

        .. math:: |B_\theta|(r, R) = \frac{R_0}{R}\,
                  \frac{\mu_0 I(r)}{2 \pi r}

        The ``R0/R`` factor is *not* cosmetic. Without it the field is not
        divergence-free in toroidal geometry: for a poloidal field
        :math:`B_\theta(r)\hat\theta` one finds
        :math:`\nabla\cdot\mathbf{B} = B_R / R`, which for this machine is
        ~10% of :math:`|B|`. See :meth:`b_field` and
        ``docs/DOC_SELF_REVIEW.md`` finding 2.

        If ``R`` is omitted the magnetic-axis value ``R = R0`` is returned,
        i.e. the same thing as :meth:`b_poloidal_reference`.
        """
        ref = self.b_poloidal_reference(r)
        if R is None:
            return ref
        R = np.asarray(R, float)
        return ref * self.R0 / np.where(np.abs(R) < 1e-12, 1e-12, R)

    def b_toroidal_at(self, R):
        """Toroidal field magnitude at major radius ``R`` (T)."""
        return self.B0 * self.R0 / np.asarray(R, float)

    def safety_factor(self, r):
        """Cylindrical-approximation safety factor ``q(r) = r B0 / (R0 B_theta)``.

        This is the large-aspect-ratio form. It is used for reporting and for
        choosing scan points, never inside the integrator.
        """
        r = np.asarray(r, float)
        bp = self.b_poloidal_reference(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = r * self.B0 / (self.R0 * bp)
        return q

    def safety_factor_exact(self, r, n_theta: int = 20001):
        """Field-line safety factor obtained by integrating around a flux surface.

        .. math:: q(r) = \frac{1}{2\pi}\oint \frac{r B_\phi}{R B_\theta}\,
                  d\theta = \frac{r B_0}{B_{\theta,\mathrm{ref}}(r)
                  \sqrt{R_0^2 - r^2}}

        which is the cylindrical value :meth:`safety_factor` multiplied by
        :math:`R_0/\sqrt{R_0^2 - r^2}`. For the default machine that factor is
        1.048 at the edge, so the conventional label ``q(a) = 2.00`` understates
        the true field-line ``q(a) = 2.097`` by 4.8%. Use this, not
        :meth:`safety_factor`, whenever the number is a physical claim rather
        than a label. See ``docs/DOC_SELF_REVIEW.md`` finding 13.
        """
        r = float(r)
        if r <= 0:
            return float(self.q_axis)
        th = np.linspace(0.0, 2.0 * np.pi, n_theta)
        R = self.R0 + r * np.cos(th)
        integ = np.trapezoid(
            r * self.b_toroidal_at(R) / (R * self.b_poloidal(r, R)), th)
        return float(integ / (2.0 * np.pi))

    def enclosed_current_actual(self, r):
        """Current actually enclosed by the flux surface ``r``, from Ampere's law.

        The ``R0/R`` factor that makes the field solenoidal also means the true
        enclosed current is not :meth:`enclosed_current` but

        .. math:: I_{\mathrm{enc}}(r) = I(r)\,\frac{R_0}{\sqrt{R_0^2 - r^2}}

        At ``r = a`` on the default machine this is 471.7 kA against the nominal
        450 kA -- 4.8% high. ``plasma_current`` is therefore a *profile
        parameter*, not the physical total current. See
        ``docs/DOC_SELF_REVIEW.md`` finding 13.
        """
        r = np.asarray(r, float)
        return self.enclosed_current(r) * self.R0 / np.sqrt(self.R0**2 - r**2)

    @property
    def q_axis(self) -> float:
        """Analytic on-axis safety factor, ``q0 = 2 B0 / (mu0 j0 R0)``.

        Note this is the *cylindrical* value; the exact on-axis limit coincides
        with it because ``R0/sqrt(R0^2 - r^2) -> 1`` as ``r -> 0``.
        """
        if self.Ip == 0:
            return np.inf
        j0 = self.Ip * (self.nu + 1.0) / (np.pi * self.a**2)
        return 2.0 * self.B0 / (self.R0 * MU_0 * j0)

    @property
    def q_edge(self) -> float:
        """Safety factor at ``r = a``."""
        return float(self.safety_factor(self.a))

    # -- the field itself -------------------------------------------------
    def b_field(self, pos: np.ndarray) -> np.ndarray:
        pos = np.atleast_2d(np.asarray(pos, float))
        x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
        R = np.hypot(x, y)
        R_safe = np.where(R < 1e-12, 1e-12, R)

        # toroidal: B0 R0 / R along phi_hat
        phi_hat = np.stack([-y / R_safe, x / R_safe, np.zeros_like(z)], axis=-1)
        b = (self.B0 * self.R0 / R_safe)[..., None] * phi_hat

        if self.Ip != 0.0:
            dR = R - self.R0
            r = np.hypot(dR, z)
            r_safe = np.where(r < 1e-12, 1e-12, r)
            # outward unit vector in the poloidal plane
            R_hat = np.stack([x / R_safe, y / R_safe, np.zeros_like(z)], axis=-1)
            Z_hat = np.zeros_like(R_hat)
            Z_hat[..., 2] = 1.0
            r_hat = (dR / r_safe)[..., None] * R_hat + (z / r_safe)[..., None] * Z_hat
            # Ampere: B_pol direction is I_hat x r_hat, with I along phi_hat.
            # Magnitude carries the R0/R factor that makes the total field
            # solenoidal -- equivalently, this is grad(psi) x grad(phi) for a
            # flux function psi(r) with dpsi/dr = -R0 mu0 I(r) / (2 pi r).
            pol_hat = np.cross(phi_hat, r_hat)
            b = b + self.b_poloidal(r, R_safe)[..., None] * pol_hat
        return b

    def __repr__(self) -> str:
        return (
            f"TokamakField(R0={self.R0:g} m, a={self.a:g} m, B0={self.B0:g} T, "
            f"Ip={self.Ip:.3g} A, nu={self.nu:g}, eps={self.epsilon:.3f}, "
            f"q0={self.q_axis:.2f}, qa={self.q_edge:.2f})"
        )


def grad_b_curvature_drift(field: TokamakField, pos, v_par, v_perp, mass, charge):
    """Analytic vertical drift in a *pure* 1/R toroidal field.

    .. math:: v_D = \\frac{m}{q B R}\\left(v_\\parallel^2
                    + \\tfrac{1}{2} v_\\perp^2\\right) \\hat{z}

    Returns the signed vertical drift speed (m/s), positive meaning ``+z``.
    This is the sum of the grad-B and curvature drifts and is the reference
    value used by :mod:`tests.test_drifts`.
    """
    R, _ = TokamakField.cylindrical(np.atleast_2d(pos))
    B = field.b_toroidal_at(R)
    return mass * (v_par**2 + 0.5 * v_perp**2) / (charge * B * R)


def e_cross_b_drift(e_vec, b_vec):
    """``E x B / |B|^2``, the drift velocity independent of mass and charge."""
    e_vec = np.asarray(e_vec, float)
    b_vec = np.asarray(b_vec, float)
    return np.cross(e_vec, b_vec) / np.dot(b_vec, b_vec)
