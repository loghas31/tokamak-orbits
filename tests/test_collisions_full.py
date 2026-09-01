"""Energy scattering and drag: does it thermalise to the right temperature?"""
import numpy as np
import pytest

from tokamak_orbits import TokamakField
from tokamak_orbits.collisions_full import (
    MaxwellianBackground, MaxwellianCollisions, coulomb_logarithm,
    psi_chandrasekhar, psi_prime, relax_speeds, speed_diffusion, speed_drift,
    _diffusion_constant,
)
from tokamak_orbits.constants import SPECIES
from tokamak_orbits.particles import initialise
from tokamak_orbits.pusher import gyroperiod, integrate

M, Q = SPECIES["D"]


@pytest.fixture
def bg():
    return MaxwellianBackground(density=1e19, temperature_ev=10e3)


def maxwellian_speeds(bg, n, mass, seed=0):
    """Speeds drawn from a Maxwellian at the background temperature."""
    rng = np.random.default_rng(seed)
    return np.linalg.norm(
        rng.normal(0.0, np.sqrt(bg.T_b / mass), size=(n, 3)), axis=1)


# --------------------------------------------------------------------------
class TestChandrasekhar:
    def test_psi_limits(self):
        assert psi_chandrasekhar(0.0) == pytest.approx(0.0)
        assert psi_chandrasekhar(50.0) == pytest.approx(1.0, abs=1e-12)

    def test_psi_prime_is_the_derivative(self):
        x = np.array([0.05, 0.4, 1.0, 3.0, 8.0])
        h = 1e-6
        fd = (psi_chandrasekhar(x + h) - psi_chandrasekhar(x - h)) / (2 * h)
        assert np.allclose(psi_prime(x), fd, rtol=1e-6)

    def test_psi_small_x_series(self):
        """psi(x) -> (4/3 sqrt(pi)) x^(3/2) as x -> 0."""
        x = 1e-4
        assert psi_chandrasekhar(x) == pytest.approx(
            4.0 / (3.0 * np.sqrt(np.pi)) * x**1.5, rel=1e-3)

    def test_psi_monotonic(self):
        x = np.geomspace(1e-3, 30, 60)
        assert np.all(np.diff(psi_chandrasekhar(x)) > 0)


class TestFrequencies:
    def test_nu0_scales_as_v_minus_3(self, bg):
        v = bg.v_th
        assert bg.nu_0(2 * v, M) / bg.nu_0(v, M) == pytest.approx(1 / 8)

    def test_nu0_scales_with_density(self):
        a = MaxwellianBackground(1e19, 10e3, coulomb_log=17.0)
        b = MaxwellianBackground(2e19, 10e3, coulomb_log=17.0)
        assert b.nu_0(a.v_th, M) / a.nu_0(a.v_th, M) == pytest.approx(2.0)

    def test_frequencies_positive(self, bg):
        v = np.geomspace(0.05, 8, 40) * bg.v_th
        assert np.all(bg.nu_perp(v, M) > 0)
        assert np.all(bg.nu_par(v, M) > 0)
        assert np.all(bg.nu_slowing(v, M) > 0)

    def test_coulomb_log_is_plausible(self):
        """Fusion-relevant plasmas sit around 15-25."""
        assert 10.0 < coulomb_logarithm(1e19, 10e3) < 30.0
        assert 10.0 < coulomb_logarithm(1e20, 1e3) < 30.0

    def test_rejects_bad_background(self):
        with pytest.raises(ValueError):
            MaxwellianBackground(density=-1.0)
        with pytest.raises(ValueError):
            MaxwellianBackground(temperature_ev=0.0)


class TestDrag:
    """The drag is derived, not quoted. These pin down that derivation."""

    def test_diffusion_derivative_is_analytic_and_correct(self, bg):
        v = np.array([0.3, 1.0, 2.5, 5.0]) * bg.v_th
        h = 1e-6 * v
        fd = (speed_diffusion(bg, v + h, M) - speed_diffusion(bg, v - h, M)) / (2 * h)
        alpha = bg.m_b / (2 * bg.T_b)
        x = alpha * v**2
        C = _diffusion_constant(bg, M)
        analytic = C * (2 * psi_prime(x) / v**2
                        - 3 * psi_chandrasekhar(x) / (alpha * v**4))
        assert np.allclose(analytic, fd, rtol=1e-6)

    def test_drift_is_exactly_the_nrl_speed_drag(self, bg):
        """The derived drift equals -(nu_s - nu_perp/2) v identically.

        This is the payoff of deriving the drag instead of quoting one. The
        drift came out of a zero-flux condition on a Maxwellian and never saw
        a collision-frequency table; it nonetheless reproduces the standard
        NRL *speed* drag to machine precision, over three decades of speed
        and both signs. NRL's nu_s alone is the *momentum* drag, which is
        larger, because a pure deflection carries momentum away without
        slowing the particle -- comparing against that is the trap this test
        exists to document.
        """
        v = np.geomspace(0.02, 20.0, 200) * bg.v_th
        nrl = -(bg.nu_slowing(v, M) - 0.5 * bg.nu_perp(v, M)) * v
        assert np.allclose(speed_drift(bg, v, M), nrl, rtol=1e-12)

    @pytest.mark.parametrize("species", ["H", "D", "T", "He4"])
    def test_the_identity_holds_for_every_mass_ratio(self, bg, species):
        """Not an accident of m = m_b: it holds for 0.25 <= m/m_b <= 4."""
        m = SPECIES[species][0]
        v = np.geomspace(0.05, 10.0, 80) * bg.v_th
        nrl = -(bg.nu_slowing(v, m) - 0.5 * bg.nu_perp(v, m)) * v
        assert np.allclose(speed_drift(bg, v, m), nrl, rtol=1e-12)

    def test_momentum_drag_is_the_larger_one(self, bg):
        """Guard on the distinction above: nu_s v is bigger than the speed
        drag wherever the particle is actually slowing down."""
        v = np.geomspace(1.2, 8.0, 20) * bg.v_th
        assert np.all(bg.nu_slowing(v, M) * v > -speed_drift(bg, v, M))

    def test_drag_points_inward_above_thermal(self, bg):
        """A fast particle must slow down."""
        v = 3.0 * bg.v_th
        assert speed_drift(bg, v, M) < 0

    def test_drift_pushes_slow_particles_up(self, bg):
        """And a very slow one must be heated, or there is no equilibrium."""
        v = 0.1 * bg.v_th
        assert speed_drift(bg, v, M) > 0


class TestThermalisation:
    """The operator must have the right equilibrium, not merely a stable one."""

    def test_maxwellian_is_stationary(self, bg):
        n = 60_000
        v0 = maxwellian_speeds(bg, n, M, seed=0)
        nu = bg.nu_par(bg.v_th, M)
        dt = 0.005 / nu
        v, _ = relax_speeds(bg, v0, M, dt, int(round(5 / (nu * dt))), seed=3)
        mean_e = 0.5 * M * np.mean(v**2) / bg.T_b
        assert mean_e == pytest.approx(1.5, abs=0.02)

    def test_stationarity_converges_in_the_collision_timestep(self, bg):
        """The bias is discretisation and must shrink with dt.

        At nu*dt = 0.02 the mean energy drifts up by ~1.3%; by 0.005 it is
        inside the sampling error. Same lesson as finding 22 -- a first-order
        operator needs its step checked against the quantity being measured.
        """
        n = 40_000
        v0 = maxwellian_speeds(bg, n, M, seed=1)
        nu = bg.nu_par(bg.v_th, M)
        devs = []
        for f in (0.02, 0.005):
            dt = f / nu
            v, _ = relax_speeds(bg, v0, M, dt, int(round(5 / (nu * dt))), seed=4)
            devs.append(abs(0.5 * M * np.mean(v**2) / bg.T_b - 1.5))
        assert devs[1] < devs[0]

    def test_cold_population_is_heated_to_the_background(self, bg):
        """Relaxation from below, where the frequencies are fast."""
        n = 40_000
        v0 = np.full(n, 0.4 * bg.v_th)
        nu = bg.nu_par(bg.v_th, M)
        dt = 0.005 / nu
        v, _ = relax_speeds(bg, v0, M, dt, int(round(40 / (nu * dt))), seed=5)
        assert 0.5 * M * np.mean(v**2) / bg.T_b == pytest.approx(1.5, abs=0.12)

    def test_hot_population_slows_down(self, bg):
        n = 20_000
        v0 = np.full(n, 3.0 * bg.v_th)
        nu = bg.nu_par(bg.v_th, M)
        dt = 0.005 / nu
        v, _ = relax_speeds(bg, v0, M, dt, int(round(20 / (nu * dt))), seed=6)
        assert np.mean(v) < 3.0 * bg.v_th

    def test_speed_stays_positive(self, bg):
        n = 20_000
        v0 = maxwellian_speeds(bg, n, M, seed=7)
        nu = bg.nu_par(bg.v_th, M)
        v, _ = relax_speeds(bg, v0, M, 0.01 / nu, 400, seed=8)
        assert np.all(v > 0)


class TestOperatorInThePusher:
    def test_energy_changes_when_the_energy_operator_is_on(self):
        f = TokamakField()
        bg = MaxwellianBackground(1e20, 2e3)
        x0, v0, m, q = initialise(f, energy_ev=10e3, r_start=0.15, pitch=[0.4])
        dt = gyroperiod(m, q, f.b_max_in_domain) / 40
        col = MaxwellianCollisions(bg, seed=1, pitch=False, energy=True).for_species("D")
        tr = integrate(x0, v0, dt, 40000, m, q, f.b_field,
                       collision_op=col, collide_every=400, field=f)
        ke = tr.kinetic_energy(m)[:, 0]
        assert np.abs(ke[-1] / ke[0] - 1) > 1e-6

    def test_pitch_only_conserves_energy_exactly(self):
        """With the energy half disabled it must reduce to the Lorentz case."""
        f = TokamakField()
        bg = MaxwellianBackground(1e20, 2e3)
        x0, v0, m, q = initialise(f, energy_ev=10e3, r_start=0.15, pitch=[0.4])
        dt = gyroperiod(m, q, f.b_max_in_domain) / 40
        col = MaxwellianCollisions(bg, seed=1, pitch=True, energy=False).for_species("D")
        tr = integrate(x0, v0, dt, 20000, m, q, f.b_field,
                       collision_op=col, collide_every=200, field=f)
        ke = tr.kinetic_energy(m)[:, 0]
        assert np.abs(ke - ke[0]).max() / ke[0] < 1e-12

    def test_both_halves_off_is_a_no_op(self):
        f = TokamakField()
        bg = MaxwellianBackground(1e20, 2e3)
        x0, v0, m, q = initialise(f, energy_ev=10e3, r_start=0.15, pitch=[0.4])
        dt = gyroperiod(m, q, f.b_max_in_domain) / 40
        col = MaxwellianCollisions(bg, seed=1, pitch=False, energy=False).for_species("D")
        a = integrate(x0, v0, dt, 4000, m, q, f.b_field)
        b = integrate(x0, v0, dt, 4000, m, q, f.b_field,
                      collision_op=col, collide_every=100, field=f)
        assert np.allclose(a.x, b.x)

    def test_repr_reports_the_plasma(self, bg):
        s = repr(MaxwellianCollisions(bg))
        assert "m^-3" in s and "keV" in s and "lnL" in s
