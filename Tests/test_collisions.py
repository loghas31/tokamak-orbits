"""Pitch-angle collision operator: exactness, bounds, and eigenmode rates."""
import numpy as np
import pytest
from numpy.polynomial import legendre as _leg

from tokamak_orbits import TokamakField, UniformField
from tokamak_orbits.collisions import (
    PitchAngleCollisions, apply_to_velocity, bounce_frequency,
    effective_detrapping_frequency, legendre_decay_rate, scatter_pitch,
)
from tokamak_orbits.constants import SPECIES, ev_to_joule
from tokamak_orbits.particles import initialise
from tokamak_orbits.pusher import gyroperiod, integrate

M, Q = SPECIES["D"]
V = float(np.sqrt(2 * ev_to_joule(10e3) / M))
TC = gyroperiod(M, Q, 2.0)


def Pl(l, x):
    c = np.zeros(l + 1)
    c[l] = 1.0
    return _leg.legval(x, c)


@pytest.fixture
def field():
    return TokamakField()


# --------------------------------------------------------------------------
class TestPitchBound:
    """The update provably cannot leave [-1, 1]; the clip never fires.

    By Cauchy-Schwarz, xi(1-a) + sqrt(a)sqrt(1-xi^2) <= sqrt(1 - a + a^2),
    which is < 1 for every 0 < a < 1.
    """

    @pytest.mark.parametrize("a", [0.001, 0.005, 0.02, 0.05, 0.1])
    def test_never_overshoots_without_clipping(self, a):
        rng = np.random.default_rng(0)
        xi = rng.uniform(-1, 1, 200_000)
        raw = scatter_pitch(xi, a, rng, clip=False)
        assert np.all(np.abs(raw) <= 1.0)

    @pytest.mark.parametrize("a", [0.005, 0.02, 0.05, 0.1])
    def test_attains_the_cauchy_schwarz_bound(self, a):
        """The bound is tight: the extremiser lies inside [-1, 1]."""
        rng = np.random.default_rng(1)
        xi = rng.uniform(-1, 1, 400_000)
        raw = scatter_pitch(xi, a, rng, clip=False)
        assert np.abs(raw).max() == pytest.approx(np.sqrt(1 - a + a * a),
                                                  rel=1e-4)

    def test_clip_is_a_noop_in_practice(self):
        rng = np.random.default_rng(2)
        xi = rng.uniform(-1, 1, 100_000)
        a = 0.02
        r1 = scatter_pitch(xi, a, np.random.default_rng(3), clip=True)
        r2 = scatter_pitch(xi, a, np.random.default_rng(3), clip=False)
        assert np.array_equal(r1, r2)

    def test_endpoints_are_stable(self):
        """xi = +-1 has no random kick, only drag toward zero."""
        rng = np.random.default_rng(4)
        out = scatter_pitch(np.array([1.0, -1.0]), 0.02, rng)
        assert out[0] == pytest.approx(0.98)
        assert out[1] == pytest.approx(-0.98)


class TestLegendreEigenmodes:
    """<P_l(xi)> must decay at exactly l(l+1)nu/2. No free parameters."""

    @pytest.mark.parametrize("l,tol", [(1, 0.02), (2, 0.03)])
    def test_decay_rate(self, l, tol):
        rng = np.random.default_rng(11)
        n, nu, nu_dt, xi0 = 400_000, 1.0, 0.005, 0.9
        noise = 1.0 / np.sqrt(n)
        xi = np.full(n, xi0)
        ts, ys = [0.0], [float(Pl(l, xi0))]
        for step in range(1, 601):
            xi = scatter_pitch(xi, nu_dt, rng)
            if step % 10 == 0:
                ts.append(step * nu_dt / nu)
                ys.append(float(np.mean(Pl(l, xi))))
        t, y = np.array(ts), np.array(ys)
        ok = (y > 10 * noise) & (t > 0)
        assert ok.sum() >= 4
        rate = -np.polyfit(t[ok], np.log(y[ok]), 1)[0]
        assert rate == pytest.approx(legendre_decay_rate(l, nu), rel=tol)

    def test_rate_formula(self):
        assert legendre_decay_rate(1, 2.0) == pytest.approx(2.0)
        assert legendre_decay_rate(2, 1.0) == pytest.approx(3.0)
        assert legendre_decay_rate(3, 1.0) == pytest.approx(6.0)

    def test_relaxes_to_isotropy(self):
        """A beam becomes uniform in xi, which is the isotropic distribution."""
        rng = np.random.default_rng(5)
        xi = np.full(200_000, 1.0)
        for _ in range(4000):
            xi = scatter_pitch(xi, 0.01, rng)
        assert abs(np.mean(xi)) < 0.01              # <xi> -> 0
        assert np.mean(xi**2) == pytest.approx(1 / 3, rel=0.05)   # uniform


class TestOperatorValidation:
    def test_rejects_large_nu_dt(self):
        with pytest.raises(ValueError, match="too large"):
            scatter_pitch(np.array([0.5]), 0.5, np.random.default_rng(0))

    def test_rejects_negative_nu_dt(self):
        with pytest.raises(ValueError):
            scatter_pitch(np.array([0.5]), -0.01, np.random.default_rng(0))

    def test_rejects_negative_nu(self):
        with pytest.raises(ValueError):
            PitchAngleCollisions(-1.0)

    def test_zero_nu_is_identity(self, field):
        col = PitchAngleCollisions(0.0)
        x = np.array([[1.15, 0.0, 0.0]])
        v = np.array([[0.0, 0.5 * V, 0.866 * V]])
        assert np.array_equal(col(field, x, v, 1e-6), v)

    def test_reproducible_from_seed(self, field):
        x = np.tile([[1.15, 0.0, 0.0]], (50, 1))
        v = np.tile([[0.0, 0.5 * V, 0.866 * V]], (50, 1))
        a = PitchAngleCollisions(1e5, seed=7)(field, x, v, 1e-7)
        b = PitchAngleCollisions(1e5, seed=7)(field, x, v, 1e-7)
        assert np.array_equal(a, b)


class TestEnergyExactness:
    """Pitch-angle scattering changes direction only. |v| must be untouched."""

    def test_speed_preserved_by_operator(self, field):
        rng = np.random.default_rng(9)
        x = np.tile([[1.15, 0.0, 0.0]], (500, 1))
        v = rng.normal(size=(500, 3))
        v *= V / np.linalg.norm(v, axis=-1, keepdims=True)
        v2 = apply_to_velocity(field, x, v, 0.02, rng)
        assert np.allclose(np.linalg.norm(v2, axis=-1), V, rtol=1e-13)

    def test_speed_preserved_for_purely_parallel_velocity(self, field):
        """The degenerate case: no perpendicular component to build a basis on."""
        x = np.array([[1.15, 0.0, 0.0]])
        B = field.b_field(x)
        b_hat = B / np.linalg.norm(B)
        v = V * b_hat
        v2 = apply_to_velocity(field, x, v, 0.02, np.random.default_rng(0))
        assert np.linalg.norm(v2) == pytest.approx(V, rel=1e-13)

    def test_energy_conserved_through_the_pusher(self, field):
        x0, v0, m, q = initialise(field, r_start=0.15, pitch=[0.5, -0.3, 0.9])
        dt = TC / 40
        col = PitchAngleCollisions(1e5, seed=1)
        tr = integrate(x0, v0, dt, 6000, m, q, field.b_field,
                       collision_op=col, collide_every=50, field=field)
        ke = tr.kinetic_energy(m)
        assert np.abs(ke - ke[0]).max() / ke[0].max() < 1e-12

    def test_collisions_actually_fired(self, field):
        x0, v0, m, q = initialise(field, r_start=0.15, pitch=[0.5])
        col = PitchAngleCollisions(1e5, seed=1)
        integrate(x0, v0, TC / 40, 1000, m, q, field.b_field,
                  collision_op=col, collide_every=50, field=field)
        assert col.n_calls == 20

    def test_collisions_change_the_orbit(self, field):
        """Guard against a silently inert operator."""
        x0, v0, m, q = initialise(field, r_start=0.15, pitch=[0.5])
        args = dict(mass=m, charge=q, b_func=field.b_field)
        a = integrate(x0, v0, TC / 40, 8000, m, q, field.b_field)
        b = integrate(x0, v0, TC / 40, 8000, m, q, field.b_field,
                      collision_op=PitchAngleCollisions(1e6, seed=2),
                      collide_every=20, field=field)
        assert np.linalg.norm(a.x[-1] - b.x[-1]) > 1e-4


class TestPusherIntegration:
    def test_requires_field(self, field):
        x0, v0, m, q = initialise(field, r_start=0.15, pitch=[0.5])
        with pytest.raises(ValueError, match="requires field"):
            integrate(x0, v0, TC / 40, 10, m, q, field.b_field,
                      collision_op=PitchAngleCollisions(1e5))

    def test_rejects_bad_stride(self, field):
        x0, v0, m, q = initialise(field, r_start=0.15, pitch=[0.5])
        with pytest.raises(ValueError, match="collide_every"):
            integrate(x0, v0, TC / 40, 10, m, q, field.b_field,
                      collide_every=0)

    def test_no_collisions_matches_collisionless(self, field):
        """collision_op=None must leave the integrator bit-identical."""
        x0, v0, m, q = initialise(field, r_start=0.15, pitch=[0.45])
        a = integrate(x0, v0, TC / 40, 2000, m, q, field.b_field)
        b = integrate(x0, v0, TC / 40, 2000, m, q, field.b_field,
                      collision_op=None, collide_every=10, field=field)
        assert np.array_equal(a.x, b.x)

    def test_lost_particles_are_not_scattered(self):
        """A frozen particle must stay frozen."""
        f = TokamakField(plasma_current=0.0)
        x0, v0, m, q = initialise(f, r_start=0.10, pitch=[0.0])
        from tokamak_orbits.diagnostics import make_loss_func
        tr = integrate(x0, v0, TC / 40, 40 * 3000, m, q, f.b_field,
                       loss_func=make_loss_func(f),
                       collision_op=PitchAngleCollisions(1e6, seed=3),
                       collide_every=20, field=f, sample_every=40)
        assert np.isfinite(tr.loss_time[0])
        after = tr.alive[:, 0].argmin()
        assert np.allclose(tr.v[after:, 0], tr.v[after, 0])


class TestReferenceFormulas:
    def test_effective_detrapping_frequency(self):
        assert effective_detrapping_frequency(1e4, 0.15) == pytest.approx(1e4 / 0.15)

    def test_bounce_frequency_scales_with_speed(self, field):
        e1 = ev_to_joule(10e3)
        w1 = bounce_frequency(field, e1, M, 0.15)
        w2 = bounce_frequency(field, 4 * e1, M, 0.15)
        assert w2 / w1 == pytest.approx(2.0, rel=1e-10)

    def test_bounce_frequency_near_measured_value(self, field):
        """Leading-order estimate, measured 2.09e5 rad/s at r=0.15."""
        w = bounce_frequency(field, ev_to_joule(10e3), M, 0.15)
        assert 1.5e5 < w < 3.0e5


class TestFirstPassageConvergence:
    """A first-passage statistic needs a far finer step than a bulk one.

    Finding 22: the collision operator was validated on the Legendre decay
    rates, which are a bulk property of the distribution and converge quickly,
    and that validation was wrongly assumed to cover the detrapping time. It
    does not. These tests pin the distinction down so it cannot be forgotten.
    """

    @staticmethod
    def _first_passage(nu_dt, target=0.51, xi0=0.30, n=20_000, seed=3):
        """Median steps x nu_dt for |xi| to first reach `target`."""
        rng = np.random.default_rng(seed)
        xi = np.full(n, xi0)
        hit = np.full(n, np.inf)
        for k in range(1, 4001):
            xi = scatter_pitch(xi, nu_dt, rng)
            newly = np.isinf(hit) & (np.abs(xi) >= target)
            hit[newly] = k * nu_dt
            if np.isfinite(hit).all():
                break
        return float(np.median(hit))

    def test_coarse_step_overestimates_first_passage(self):
        """The bias that produced the -0.779 exponent, in miniature."""
        coarse = self._first_passage(0.02)
        fine = self._first_passage(0.00125)
        assert coarse > fine
        assert coarse / fine > 1.05      # a real bias, not noise

    def test_first_passage_converges_under_refinement(self):
        a = self._first_passage(0.005)
        b = self._first_passage(0.00125)
        assert abs(a - b) / b < 0.20     # much tighter than coarse-vs-fine

    def test_bulk_rate_is_already_converged_where_first_passage_is_not(self):
        """The point of finding 22: the l=1 rate is fine at nu_dt = 0.02,
        which is exactly why the first-passage error went unnoticed."""
        rng = np.random.default_rng(21)
        n, xi0 = 300_000, 0.9
        rates = []
        for nu_dt in (0.02, 0.005):
            xi = np.full(n, xi0)
            ts, ys = [0.0], [xi0]
            for step in range(1, 401):
                xi = scatter_pitch(xi, nu_dt, rng)
                if step % 10 == 0:
                    ts.append(step * nu_dt)
                    ys.append(float(np.mean(xi)))
            t, y = np.array(ts), np.array(ys)
            ok = y > 10 / np.sqrt(n)
            rates.append(-np.polyfit(t[ok], np.log(y[ok]), 1)[0])
        assert abs(rates[0] - rates[1]) / rates[1] < 0.02
