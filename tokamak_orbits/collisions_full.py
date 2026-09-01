r"""Full Coulomb collisions: pitch-angle scattering *and* energy exchange.

What this adds over :mod:`tokamak_orbits.collisions`
---------------------------------------------------
That module implements the Lorentz operator: pitch-angle scattering at a
**single, constant** deflection frequency and **fixed speed**. It is the right
tool for a parameter scan in collisionality, and every result in ``RESULTS.md``
section 7 uses it. But it cannot slow a particle down, cannot thermalise one,
and has no plasma behind it -- ``nu_d`` is a dial, not a temperature.

This module replaces the dial with a plasma. Given a background density and
temperature it computes velocity-dependent collision frequencies from the
Coulomb logarithm, and adds an energy operator so that speed changes too.

Collision frequencies
---------------------
With :math:`x = m_b v^2 / 2T_b` and the NRL Formulary definitions,

.. math:: \nu_0(v) = \frac{n_b Z^2 Z_b^2 e^4 \ln\Lambda}
                          {4\pi \varepsilon_0^2 m^2 v^3}

.. math:: \psi(x) = \operatorname{erf}(\sqrt{x})
                    - \frac{2}{\sqrt{\pi}}\sqrt{x}\,e^{-x}

the deflection and parallel (speed-diffusion) frequencies are

.. math:: \nu_\perp = 2\nu_0\left[\left(1-\tfrac{1}{2x}\right)\psi(x)
                                  + \psi'(x)\right], \qquad
          \nu_\parallel = \nu_0\,\frac{\psi(x)}{x}

Both fall off as :math:`v^{-3}` at high energy, which is why fast ions are
weakly collisional and why the slowing-down time is long.

Where the drag comes from
-------------------------
The drag is **not** taken from a table. Writing the speed as an Itô process
:math:`dv = \mu(v)\,dt + \sigma(v)\,dW` with
:math:`\sigma^2 = B(v) = \nu_\parallel v^2`, the Fokker-Planck equation for the
speed density :math:`p(v)` is

.. math:: \partial_t p = -\partial_v(\mu p) + \tfrac12\partial_v^2(\sigma^2 p)

and demanding that a Maxwellian be **exactly stationary** -- zero probability
flux, :math:`p_M \propto v^2 e^{-mv^2/2T_b}` -- fixes

.. math:: \mu(v) = \tfrac12\frac{dB}{dv}
          + \frac{B}{2}\left(\frac{2}{v} - \frac{m v}{T_b}\right)

So the drag is derived from the diffusion by detailed balance rather than
quoted, which removes an entire class of convention error: there is no way to
get the drag "nearly right" and end up with a distribution that relaxes to the
wrong temperature. The measured slowing-down rate is then compared against the
independent NRL expression for :math:`\nu_s` as a check
(see ``RESULTS.md`` §9).

What is still missing
---------------------
The background is a fixed Maxwellian at rest with no back-reaction, so momentum
is not conserved between species and there is still **no bootstrap current**.
This is a test-particle operator, not a full field-particle one.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

from .constants import ELEMENTARY_CHARGE, SPECIES, ev_to_joule

EPS0 = 8.8541878128e-12


# --------------------------------------------------------------------------
# background plasma and collision frequencies
# --------------------------------------------------------------------------
def coulomb_logarithm(n_e, T_e_ev, Z=1, Z_b=1, mu=2.0, mu_b=2.0, T_b_ev=None):
    """NRL Formulary Coulomb logarithm for ion-ion collisions.

    ``n_e`` in m^-3, temperatures in eV, ``mu`` the atomic mass numbers. The
    formula is quoted for cgs densities, so the conversion is done here.
    """
    T_b_ev = T_e_ev if T_b_ev is None else T_b_ev
    n_cgs = n_e * 1e-6
    return float(
        23.0 - np.log(
            Z * Z_b * (mu + mu_b) / (mu * T_b_ev + mu_b * T_e_ev)
            * np.sqrt(n_cgs * Z**2 / T_e_ev + n_cgs * Z_b**2 / T_b_ev)))


def psi_chandrasekhar(x):
    r"""``psi(x) = erf(sqrt(x)) - (2/sqrt(pi)) sqrt(x) exp(-x)``."""
    x = np.asarray(x, float)
    sx = np.sqrt(np.maximum(x, 0.0))
    return erf(sx) - (2.0 / np.sqrt(np.pi)) * sx * np.exp(-x)


def psi_prime(x):
    r"""``d psi / dx = (2/sqrt(pi)) sqrt(x) exp(-x)``.

    Obtained by differentiating the integral form directly, not numerically.
    """
    x = np.asarray(x, float)
    return (2.0 / np.sqrt(np.pi)) * np.sqrt(np.maximum(x, 0.0)) * np.exp(-x)


class MaxwellianBackground:
    """A fixed Maxwellian background species that a test ion collides with.

    Parameters
    ----------
    density : float
        Background number density (m^-3).
    temperature_ev : float
        Background temperature (eV).
    species : str
        Background species key from :data:`tokamak_orbits.constants.SPECIES`.
    charge_number, background_charge_number : int
        ``Z`` of the test particle and of the background.
    coulomb_log : float, optional
        Override the computed Coulomb logarithm.
    """

    def __init__(self, density=1e19, temperature_ev=10e3, species="D",
                 charge_number=1, background_charge_number=1,
                 coulomb_log=None):
        if density <= 0 or temperature_ev <= 0:
            raise ValueError("density and temperature must be positive")
        self.n_b = float(density)
        self.T_b = ev_to_joule(temperature_ev)
        self.T_b_ev = float(temperature_ev)
        self.m_b, _ = SPECIES[species]
        self.Z = int(charge_number)
        self.Z_b = int(background_charge_number)
        self.ln_lambda = (float(coulomb_log) if coulomb_log is not None
                          else coulomb_logarithm(density, temperature_ev,
                                                 self.Z, self.Z_b))
        self.v_th = float(np.sqrt(2.0 * self.T_b / self.m_b))

    def x_of(self, v):
        """``x = m_b v^2 / (2 T_b)``, the squared speed in thermal units."""
        return self.m_b * np.asarray(v, float) ** 2 / (2.0 * self.T_b)

    def nu_0(self, v, mass):
        """Basic collision frequency, ``~ v^-3``."""
        v = np.maximum(np.asarray(v, float), 1e-30)
        e = ELEMENTARY_CHARGE
        return (self.n_b * (self.Z * self.Z_b) ** 2 * e**4 * self.ln_lambda
                / (4.0 * np.pi * EPS0**2 * mass**2 * v**3))

    def nu_perp(self, v, mass):
        """Deflection (pitch-angle) frequency."""
        x = self.x_of(v)
        x = np.maximum(x, 1e-12)
        return 2.0 * self.nu_0(v, mass) * (
            (1.0 - 1.0 / (2.0 * x)) * psi_chandrasekhar(x) + psi_prime(x))

    def nu_par(self, v, mass):
        """Parallel (speed-diffusion) frequency."""
        x = np.maximum(self.x_of(v), 1e-12)
        return self.nu_0(v, mass) * psi_chandrasekhar(x) / x

    def nu_slowing(self, v, mass):
        """NRL slowing-down frequency, used only as an independent check."""
        x = self.x_of(v)
        return (1.0 + mass / self.m_b) * psi_chandrasekhar(x) * self.nu_0(v, mass)

    def maxwellian_speed_pdf(self, v, mass=None):
        """``p(v) ~ v^2 exp(-m v^2 / 2T)`` for the *test* particle mass."""
        m = self.m_b if mass is None else mass
        v = np.asarray(v, float)
        p = v**2 * np.exp(-m * v**2 / (2.0 * self.T_b))
        return p / np.trapezoid(p, v)


# --------------------------------------------------------------------------
# the energy operator
# --------------------------------------------------------------------------
def _diffusion_constant(bg, mass):
    """``C`` in ``B(v) = C psi(x) / (alpha v^3)`` with ``alpha = m_b/2T_b``."""
    e = ELEMENTARY_CHARGE
    return (bg.n_b * (bg.Z * bg.Z_b) ** 2 * e**4 * bg.ln_lambda
            / (4.0 * np.pi * EPS0**2 * mass**2))


def speed_diffusion(bg, v, mass):
    r"""``B(v) = nu_par(v) v^2 = C psi(x) / (alpha v^3)``.

    Written in closed form rather than as ``nu_par * v**2`` so that its
    derivative can be taken analytically -- see :func:`speed_drift`.
    """
    v = np.maximum(np.asarray(v, float), 1e-30)
    alpha = bg.m_b / (2.0 * bg.T_b)
    return _diffusion_constant(bg, mass) * psi_chandrasekhar(alpha * v**2) / (
        alpha * v**3)


def speed_drift(bg, v, mass):
    r"""Drift fixed by requiring a Maxwellian to be exactly stationary.

    .. math:: \mu(v) = \tfrac12 B'(v)
              + \frac{B(v)}{2}\left(\frac{2}{v} - \frac{m v}{T_b}\right)

    ``B'`` is **analytic**, not finite-differenced:

    .. math:: \frac{dB}{dv} = C\left[\frac{2\psi'(x)}{v^{2}}
              - \frac{3\psi(x)}{\alpha v^{4}}\right]

    using :math:`x = \alpha v^{2}` and :math:`d\psi/dv = 2\alpha v\,\psi'(x)`.
    An earlier version differenced it numerically, which cost three evaluations
    of the Chandrasekhar function per particle per step and dominated the
    runtime of every relaxation run.
    """
    v = np.maximum(np.asarray(v, float), 1e-30)
    alpha = bg.m_b / (2.0 * bg.T_b)
    x = alpha * v**2
    C = _diffusion_constant(bg, mass)
    dB = C * (2.0 * psi_prime(x) / v**2
              - 3.0 * psi_chandrasekhar(x) / (alpha * v**4))
    B = C * psi_chandrasekhar(x) / (alpha * v**3)
    return 0.5 * dB + 0.5 * B * (2.0 / v - mass * v / bg.T_b)


def scatter_speed(bg, v, mass, dt, rng, floor_frac=1e-3, max_nu_dt=None,
                  max_substeps=256, report=False):
    """One Monte Carlo step of the energy operator.

    ``v' = v + mu(v) dt + sqrt(B(v) dt) R`` with ``R = +-1``.

    The speed is floored at a small fraction of the background thermal speed:
    the drift diverges as ``1/v`` at the origin (a geometric effect of the
    ``v^2`` measure, not a physical one), and without a floor a particle that
    random-walks close to zero takes an unphysically large step.

    Sub-cycling
    -----------
    The collision frequencies go as ``v^-3``, so a step that is small for a
    fast ion is not small for the same ion once it has slowed down. In a
    slowing-down run the *same* ``dt`` that starts at ``nu dt ~ 1e-3`` ends at
    ``nu dt ~ 2``, where the update is meaningless -- the first version of the
    §9.3 experiment did exactly that and reported a number built on it.

    If ``max_nu_dt`` is given, each particle's step is split into
    ``ceil(nu_par dt / max_nu_dt)`` sub-steps, so the accuracy condition is
    enforced per particle rather than assumed. Sub-stepping is capped at
    ``max_substeps``; particles that would need more are stepped at the cap and
    the shortfall is reported by the caller, never silently absorbed.
    """
    v = np.asarray(v, float)
    floor = floor_frac * bg.v_th
    if max_nu_dt is None:
        mu = speed_drift(bg, v, mass)
        B = speed_diffusion(bg, v, mass)
        sign = rng.integers(0, 2, size=v.shape) * 2 - 1
        raw = v + mu * dt + sign * np.sqrt(np.maximum(B, 0.0) * dt)
        out = np.maximum(raw, floor)
        if report:
            return out, dict(cap_hits=0, max_substeps=1,
                             floor_hits=int((raw < floor).sum()))
        return out

    want = np.ceil(bg.nu_par(v, mass) * dt / float(max_nu_dt))
    cap_hits = int((want > int(max_substeps)).sum())
    n_sub = np.clip(want, 1, int(max_substeps)).astype(np.int64)
    out = v.copy()
    floor_hits = 0
    remaining = n_sub.copy()
    # every particle takes at least one sub-step; the deepest sub-cycler sets
    # the loop length, and particles drop out as their budget is spent
    for _ in range(int(n_sub.max())):
        active = remaining > 0
        if not active.any():
            break
        sub_dt = dt / n_sub[active]
        va = out[active]
        mu = speed_drift(bg, va, mass)
        B = speed_diffusion(bg, va, mass)
        sign = rng.integers(0, 2, size=va.shape) * 2 - 1
        raw = va + mu * sub_dt + sign * np.sqrt(np.maximum(B, 0.0) * sub_dt)
        floor_hits += int((raw < floor).sum())
        out[active] = np.maximum(raw, floor)
        remaining[active] -= 1
    if report:
        return out, dict(cap_hits=cap_hits, max_substeps=int(n_sub.max()),
                         floor_hits=floor_hits)
    return out


class MaxwellianCollisions:
    """Pitch-angle **and** energy collisions against a Maxwellian background.

    Drop-in replacement for
    :class:`tokamak_orbits.collisions.PitchAngleCollisions` in
    :func:`tokamak_orbits.pusher.integrate`, but the frequencies come from a
    real plasma rather than a single dial, and the particle's speed changes.

    Parameters
    ----------
    background : MaxwellianBackground
    seed : int
    pitch, energy : bool
        Enable each half of the operator independently, which is what makes
        it possible to attribute an effect to one or the other.
    max_nu_dt : float or None
        Accuracy target for the collision sub-step. Both halves of the
        operator are sub-cycled per particle so that ``nu dt`` stays below
        this. ``None`` disables sub-cycling and reproduces the single-step
        behaviour. The default is 0.005, which is what §9.2 actually measures
        as sufficient for distribution-shape work -- an earlier default of
        0.02 cited that section as authority while being the value that
        section marks as *failing* its KS test. Set it smaller still for
        first-passage quantities (see finding 22).

    Attributes
    ----------
    max_nu_dt_seen : float
        Largest ``nu dt`` presented to the operator before sub-cycling, over
        the whole run. This is the number that tells you whether the caller's
        ``collide_every`` was reasonable.
    max_substeps_used : int
        Largest sub-step count actually taken.
    substep_cap_hits : int
        Number of particle-steps that wanted more sub-steps than the cap
        allowed. Anything above zero means part of the run was integrated at
        a coarser collisionality than requested, and must be reported.
    speed_floor_hits, pitch_clip_hits : int
        Counts of the two places the operator silently modifies a value: the
        speed floor at ``1e-3 v_th`` and the ``[-1, 1]`` clip on the pitch.
        Both are counted rather than absorbed, on the same principle as the
        cap above -- a run that leaned on either is not the run that was
        requested.
    """

    def __init__(self, background, seed=0, pitch=True, energy=True,
                 max_nu_dt=0.005, max_substeps=256):
        self.bg = background
        self.rng = np.random.default_rng(seed)
        self.do_pitch = bool(pitch)
        self.do_energy = bool(energy)
        self.target_nu_dt = None if max_nu_dt is None else float(max_nu_dt)
        self.max_substeps = int(max_substeps)
        self.n_calls = 0
        self.max_nu_dt = 0.0            # kept for backward compatibility
        self.max_nu_dt_seen = 0.0
        self.max_substeps_used = 1
        self.substep_cap_hits = 0
        self.pitch_clip_hits = 0
        self.speed_floor_hits = 0

    def __call__(self, field, x, v, dt_coll):
        from .collisions import scatter_pitch

        self.n_calls += 1
        x = np.atleast_2d(np.asarray(x, float))
        v = np.atleast_2d(np.asarray(v, float))
        mass = self._mass
        speed = np.linalg.norm(v, axis=-1)

        B = field.b_field(x)
        b_hat = B / np.linalg.norm(B, axis=-1, keepdims=True)
        v_par = np.sum(v * b_hat, axis=-1)
        perp = v - v_par[:, None] * b_hat
        pn = np.linalg.norm(perp, axis=-1)
        safe = pn > 1e-12 * np.maximum(speed, 1e-30)
        e_perp = np.zeros_like(perp)
        e_perp[safe] = perp[safe] / pn[safe, None]
        if (~safe).any():
            axes = np.eye(3)
            ref = axes[np.argmin(np.abs(b_hat[~safe] @ axes.T), axis=-1)]
            tmp = np.cross(b_hat[~safe], ref)
            e_perp[~safe] = tmp / np.linalg.norm(tmp, axis=-1, keepdims=True)

        xi = np.divide(v_par, speed, out=np.zeros_like(v_par), where=speed > 0)

        # report the frequency of whichever half is actually enabled: an
        # earlier version took max(nu_perp, nu_par) unconditionally and so
        # quoted nu_perp for a run with pitch scattering switched off, where
        # nu_perp is up to 2x nu_par and governs nothing.
        active_nu = []
        if self.do_pitch:
            active_nu.append(self.bg.nu_perp(speed, mass))
        if self.do_energy:
            active_nu.append(self.bg.nu_par(speed, mass))
        if active_nu:
            nu_dt_seen = float(np.max(np.maximum.reduce(active_nu)) * dt_coll)
            self.max_nu_dt = max(self.max_nu_dt, nu_dt_seen)
            self.max_nu_dt_seen = self.max_nu_dt

        if self.do_pitch:
            nu_dt = self.bg.nu_perp(speed, mass) * dt_coll
            if self.target_nu_dt is None:
                n_sub = np.ones_like(nu_dt, dtype=np.int64)
            else:
                want = np.ceil(nu_dt / self.target_nu_dt)
                self.substep_cap_hits += int((want > self.max_substeps).sum())
                n_sub = np.clip(want, 1, self.max_substeps).astype(np.int64)
            self.max_substeps_used = max(self.max_substeps_used,
                                         int(n_sub.max()))
            step = nu_dt / n_sub
            # scatter_pitch takes a scalar nu*dt; here it is per particle, so
            # the same update is applied elementwise, sub-cycled per particle
            remaining = n_sub.copy()
            for _ in range(int(n_sub.max())):
                act = remaining > 0
                if not act.any():
                    break
                s = step[act]
                sign = self.rng.integers(0, 2, size=s.shape) * 2 - 1
                xa = xi[act]
                prop = (xa * (1.0 - s)
                        + sign * np.sqrt(np.clip(1 - xa**2, 0, None) * s))
                self.pitch_clip_hits += int((np.abs(prop) > 1.0).sum())
                xi[act] = np.clip(prop, -1.0, 1.0)
                remaining[act] -= 1

        if self.do_energy:
            # the counters come from scatter_speed itself: an earlier version
            # recomputed them from the *post*-step speed, which is not the
            # speed the sub-step decision was made from, so the cap-hit count
            # it reported was not the count of steps that were capped.
            speed, rep = scatter_speed(
                self.bg, speed, mass, dt_coll, self.rng,
                max_nu_dt=self.target_nu_dt,
                max_substeps=self.max_substeps, report=True)
            self.substep_cap_hits += rep["cap_hits"]
            self.speed_floor_hits += rep["floor_hits"]
            self.max_substeps_used = max(self.max_substeps_used,
                                         rep["max_substeps"])

        return (speed * xi)[:, None] * b_hat + (
            speed * np.sqrt(np.clip(1 - xi**2, 0.0, None)))[:, None] * e_perp

    # the pusher does not pass the mass, so it is bound once at first use
    _mass = SPECIES["D"][0]

    def for_species(self, species="D"):
        self._mass = SPECIES[species][0]
        return self

    def __repr__(self):
        return (f"MaxwellianCollisions(n={self.bg.n_b:.2e} m^-3, "
                f"T={self.bg.T_b_ev/1e3:.1f} keV, lnL={self.bg.ln_lambda:.1f}, "
                f"pitch={self.do_pitch}, energy={self.do_energy})")


def relax_speeds(bg, v0, mass, dt, n_steps, seed=0, record_every=None,
                 max_nu_dt=None, max_substeps=256):
    """Evolve a population of speeds under the energy operator alone.

    Used by the validation in ``RESULTS.md`` §9: whatever the initial
    distribution, the result must be a Maxwellian at the background
    temperature, with mean energy ``3T_b/2``.

    ``max_nu_dt`` is passed through to :func:`scatter_speed`. It is left at
    ``None`` for the convergence study in §9.2 -- the whole point there is to
    watch the raw single-step bias converge -- and set to match the pusher's
    policy whenever the two are being compared.
    """
    rng = np.random.default_rng(seed)
    v = np.array(v0, dtype=float, copy=True)
    trace = []
    for k in range(1, n_steps + 1):
        v = scatter_speed(bg, v, mass, dt, rng, max_nu_dt=max_nu_dt,
                          max_substeps=max_substeps)
        if record_every and k % record_every == 0:
            trace.append((k * dt, 0.5 * mass * float(np.mean(v**2))))
    return v, trace
