"""Diagnostics: invariants, guiding centre, classification, loss bookkeeping."""
import numpy as np
import pytest

from tokamak_orbits import TokamakField, UniformField
from tokamak_orbits.constants import SPECIES, ev_to_joule
from tokamak_orbits.diagnostics import (
    banana_width_analytic, classify, guiding_centre, magnetic_moment,
    make_loss_func, parallel_perp_split,
)
from tokamak_orbits.particles import initialise
from tokamak_orbits.pusher import gyroperiod, gyroradius, integrate

M, Q = SPECIES["D"]
V = float(np.sqrt(2 * ev_to_joule(10e3) / M))
TC = gyroperiod(M, Q, 2.0)


@pytest.fixture
def field():
    return TokamakField()


class TestParallelPerpSplit:
    def test_pure_parallel(self):
        f = UniformField(b_vec=(0, 0, 2.0))
        vp, vt = parallel_perp_split(f, np.zeros((1, 3)), np.array([[0, 0, V]]))
        assert vp[0] == pytest.approx(V)
        assert vt[0] == pytest.approx(0.0, abs=1e-6)

    def test_pure_perpendicular(self):
        f = UniformField(b_vec=(0, 0, 2.0))
        vp, vt = parallel_perp_split(f, np.zeros((1, 3)), np.array([[V, 0, 0]]))
        assert vp[0] == pytest.approx(0.0, abs=1e-6)
        assert vt[0] == pytest.approx(V)

    def test_antiparallel_is_negative(self):
        f = UniformField(b_vec=(0, 0, 2.0))
        vp, _ = parallel_perp_split(f, np.zeros((1, 3)), np.array([[0, 0, -V]]))
        assert vp[0] == pytest.approx(-V)

    def test_pythagoras(self, field):
        rng = np.random.default_rng(11)
        x = np.tile([[1.15, 0.0, 0.0]], (20, 1))
        v = rng.normal(size=(20, 3)) * V
        vp, vt = parallel_perp_split(field, x, v)
        assert np.allclose(vp**2 + vt**2, np.sum(v * v, axis=-1))

    def test_perp_is_non_negative(self, field):
        rng = np.random.default_rng(12)
        x = np.tile([[1.15, 0.0, 0.0]], (20, 1))
        v = rng.normal(size=(20, 3)) * V
        _, vt = parallel_perp_split(field, x, v)
        assert np.all(vt >= 0)

    def test_preserves_leading_shape(self, field):
        x = np.zeros((7, 4, 3)) + np.array([1.15, 0.0, 0.0])
        v = np.zeros((7, 4, 3)) + np.array([0.0, 0.0, V])
        vp, vt = parallel_perp_split(field, x, v)
        assert vp.shape == (7, 4) and vt.shape == (7, 4)


class TestMagneticMoment:
    def test_formula(self):
        f = UniformField(b_vec=(0, 0, 2.0))
        mu = magnetic_moment(f, np.zeros((1, 3)), np.array([[V, 0, 0]]), M)
        assert mu[0] == pytest.approx(0.5 * M * V**2 / 2.0)

    def test_zero_for_pure_parallel(self):
        f = UniformField(b_vec=(0, 0, 2.0))
        mu = magnetic_moment(f, np.zeros((1, 3)), np.array([[0, 0, V]]), M)
        assert mu[0] == pytest.approx(0.0, abs=1e-30)

    def test_exactly_conserved_in_uniform_field(self):
        f = UniformField(b_vec=(0, 0, 2.0))
        tr = integrate(np.zeros((1, 3)), np.array([[0.6 * V, 0, 0.8 * V]]),
                       TC / 40, 40 * 200, M, Q, f.b_field)
        mu = magnetic_moment(f, tr.x, tr.v, M)[:, 0]
        assert np.abs(mu - mu[0]).max() / mu[0] < 1e-12

    def test_error_is_physical_not_numerical(self, field):
        """mu error must NOT shrink when the timestep is refined.

        This is the test that distinguishes finite-Larmor-radius physics from
        integration error. If someone introduces a genuine numerical bug in
        the pusher, this test starts failing because refining dt would then
        change the answer.
        """
        errs = []
        for nper in (20, 80, 320):
            x0, v0, m, q = initialise(field, r_start=0.15, pitch=[0.95])
            tr = integrate(x0, v0, TC / nper, int(6e-6 / (TC / nper)),
                           m, q, field.b_field, sample_every=max(1, nper // 20))
            mu = magnetic_moment(field, tr.x, tr.v, m)[:, 0]
            errs.append(np.abs(mu - mu[0]).max() / mu[0])
        assert max(errs) - min(errs) < 0.02 * max(errs)
        assert min(errs) > 0.05      # and it is a big, real effect

    def test_error_grows_linearly_with_gyroradius(self, field):
        """delta mu / mu is first order in rho/L."""
        errs, rhos = [], []
        for e_kev in (2.5, 10.0, 40.0):
            x0, v0, m, q = initialise(field, energy_ev=e_kev * 1e3,
                                      r_start=0.15, pitch=[0.95])
            v = np.linalg.norm(v0[0])
            tr = integrate(x0, v0, TC / 40, int(6e-6 / (TC / 40)),
                           m, q, field.b_field, sample_every=2)
            mu = magnetic_moment(field, tr.x, tr.v, m)[:, 0]
            errs.append(np.abs(mu - mu[0]).max() / mu[0])
            rhos.append(gyroradius(m, q, v, 2.0))
        ratios = [e / r for e, r in zip(errs, rhos)]
        assert max(ratios) / min(ratios) < 1.15


class TestGuidingCentre:
    def test_removes_gyration_in_uniform_field(self):
        """With the half-step correction the guiding centre is stationary."""
        f = UniformField(b_vec=(0, 0, 2.0))
        dt = TC / 200
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       dt, 200 * 4, M, Q, f.b_field)
        gc = guiding_centre(f, tr.x, tr.v, M, Q, dt=dt)
        spread = np.linalg.norm(gc[:, 0, :2] - gc[0, 0, :2], axis=-1).max()
        assert spread < 1e-3 * gyroradius(M, Q, V, 2.0)

    def test_uncorrected_wobble_is_first_order_in_dt(self):
        """Without dt the leapfrog staggering leaves a wobble of omega*dt*rho.

        Documents the size of the error the correction removes, and fails if
        someone silently changes the pusher's time-centring convention.
        """
        f = UniformField(b_vec=(0, 0, 2.0))
        rho = gyroradius(M, Q, V, 2.0)
        for n in (100, 200, 400):
            tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                           TC / n, n * 4, M, Q, f.b_field)
            gc = guiding_centre(f, tr.x, tr.v, M, Q)      # no dt
            spread = np.linalg.norm(gc[:, 0, :2] - gc[0, 0, :2], axis=-1).max()
            assert spread / rho == pytest.approx(2 * np.pi / n, rel=1e-3)

    def test_correction_is_second_order(self):
        """The corrected residual falls by 4x per halving of dt."""
        f = UniformField(b_vec=(0, 0, 2.0))
        errs = []
        for n in (100, 200, 400):
            dt = TC / n
            tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                           dt, n * 4, M, Q, f.b_field)
            gc = guiding_centre(f, tr.x, tr.v, M, Q, dt=dt)
            errs.append(np.linalg.norm(
                gc[:, 0, :2] - gc[0, 0, :2], axis=-1).max())
        for i in range(len(errs) - 1):
            assert errs[i] / errs[i + 1] == pytest.approx(4.0, rel=0.1)

    def test_banana_width_is_converged_at_production_timestep(self):
        """Measured width must not depend on dt once the correction is in.

        Before the half-step fix this varied by 3% between T_c/20 and T_c/320,
        which would have been reported as physics.
        """
        tf = TokamakField()
        widths = []
        for n in (20, 40, 160):
            x0, v0, m, q = initialise(tf, r_start=0.15, pitch=[0.45])
            tr = integrate(x0, v0, TC / n, int(6e-5 / (TC / n)), m, q,
                           tf.b_field, sample_every=max(1, n // 10))
            widths.append(classify(tf, tr, m, q).radial_width[0])
        assert (max(widths) - min(widths)) / np.mean(widths) < 5e-3

    def test_offset_is_one_gyroradius(self):
        f = UniformField(b_vec=(0, 0, 2.0))
        gc = guiding_centre(f, np.zeros((1, 3)), np.array([[V, 0, 0]]), M, Q)
        # no dt: raw first-order formula
        assert np.linalg.norm(gc[0]) == pytest.approx(
            gyroradius(M, Q, V, 2.0), rel=1e-12)

    def test_no_offset_for_pure_parallel(self):
        f = UniformField(b_vec=(0, 0, 2.0))
        gc = guiding_centre(f, np.zeros((1, 3)), np.array([[0, 0, V]]), M, Q)
        assert np.linalg.norm(gc[0]) < 1e-12

    def test_shape_preserved(self, field):
        x = np.zeros((5, 2, 3)) + np.array([1.15, 0.0, 0.0])
        v = np.zeros((5, 2, 3)) + np.array([0.0, 0.0, V])
        assert guiding_centre(field, x, v, M, Q).shape == (5, 2, 3)


class TestClassification:
    @pytest.fixture(scope="class")
    def run(self):
        f = TokamakField()
        pitches = np.array([0.95, 0.70, 0.55, 0.45, 0.20, -0.45, -0.95])
        x0, v0, m, q = initialise(f, r_start=0.15, pitch=pitches)
        tr = integrate(x0, v0, TC / 40, int(6e-5 / (TC / 40)), m, q,
                       f.b_field, sample_every=10,
                       loss_func=make_loss_func(f))
        return f, tr, classify(f, tr, m, q), pitches

    def test_boundary_matches_prediction(self, run):
        """xi_crit = sqrt(2 eps/(1+eps)) = 0.511 at r = 0.15."""
        f, tr, s, pitches = run
        assert not s.trapped[pitches == 0.55][0]     # just outside -> passing
        assert s.trapped[pitches == 0.45][0]         # just inside  -> trapped

    def test_deeply_passing_are_passing(self, run):
        f, tr, s, pitches = run
        assert s.kind[0] == "passing" and s.kind[-1] == "passing"

    def test_trapped_have_bounces(self, run):
        f, tr, s, pitches = run
        assert np.all(s.n_bounces[s.trapped] >= 2)

    def test_passing_have_no_bounces(self, run):
        f, tr, s, pitches = run
        passing = np.array([k == "passing" for k in s.kind])
        assert np.all(s.n_bounces[passing] == 0)

    def test_energy_conserved_throughout(self, run):
        f, tr, s, pitches = run
        assert s.energy_error.max() < 1e-12

    def test_widest_orbit_is_near_the_boundary(self, run):
        """Banana width peaks at the trapped/passing boundary, not at xi = 0."""
        f, tr, s, pitches = run
        assert s.radial_width[pitches == 0.45][0] > s.radial_width[pitches == 0.20][0]

    def test_co_and_counter_orbits_shift_oppositely(self, run):
        """The classic result: co-going orbits sit inboard, counter outboard."""
        f, tr, s, pitches = run
        co = s.r_min[pitches == 0.45][0], s.r_max[pitches == 0.45][0]
        ctr = s.r_min[pitches == -0.45][0], s.r_max[pitches == -0.45][0]
        assert co[0] < 0.15 < ctr[1]
        assert np.mean(co) < np.mean(ctr)

    def test_r_min_le_r_max(self, run):
        f, tr, s, pitches = run
        assert np.all(s.r_min <= s.r_max)
        assert np.all(s.radial_width >= 0)

    def test_kind_is_consistent_with_flags(self, run):
        f, tr, s, pitches = run
        for i, k in enumerate(s.kind):
            assert k == ("lost" if s.lost[i] else
                         ("trapped" if s.trapped[i] else "passing"))


class TestLossFunction:
    def test_flags_outside_only(self, field):
        loss = make_loss_func(field)
        inside = np.array([[field.R0 + 0.1, 0.0, 0.0]])
        outside = np.array([[field.R0 + 0.5, 0.0, 0.0]])
        assert not loss(inside)[0]
        assert loss(outside)[0]

    def test_custom_limit(self, field):
        loss = make_loss_func(field, r_limit=0.1)
        assert loss(np.array([[field.R0 + 0.15, 0.0, 0.0]]))[0]

    def test_boundary_is_the_particle_not_the_guiding_centre(self, field):
        """Documented convention: the test is on the particle position."""
        loss = make_loss_func(field)
        just_outside = np.array([[field.R0 + field.a + 1e-6, 0.0, 0.0]])
        assert loss(just_outside)[0]


class TestBananaWidthAnalytic:
    def test_scales_inversely_with_poloidal_field(self):
        f1 = TokamakField(plasma_current=4.5e5)
        f2 = TokamakField(plasma_current=9.0e5)
        w1 = banana_width_analytic(f1, 0.15, ev_to_joule(10e3), M, Q)
        w2 = banana_width_analytic(f2, 0.15, ev_to_joule(10e3), M, Q)
        assert w1 / w2 == pytest.approx(2.0, rel=1e-10)

    def test_scales_as_sqrt_energy(self, field):
        w1 = banana_width_analytic(field, 0.15, ev_to_joule(10e3), M, Q)
        w2 = banana_width_analytic(field, 0.15, ev_to_joule(40e3), M, Q)
        assert w2 / w1 == pytest.approx(2.0, rel=1e-10)

    def test_scales_as_sqrt_epsilon(self, field):
        w1 = banana_width_analytic(field, 0.10, ev_to_joule(10e3), M, Q)
        w2 = banana_width_analytic(field, 0.40, ev_to_joule(10e3), M, Q)
        bt1 = field.b_poloidal_reference(0.10)
        bt2 = field.b_poloidal_reference(0.40)
        assert (w2 / w1) * (bt2 / bt1) == pytest.approx(2.0, rel=1e-10)

    def test_positive(self, field):
        assert banana_width_analytic(field, 0.15, ev_to_joule(10e3), M, Q) > 0
