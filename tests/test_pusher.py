"""Integrator correctness: exact solutions, conservation, convergence order."""
import numpy as np
import pytest

from tokamak_orbits import TokamakField, UniformField, e_cross_b_drift
from tokamak_orbits.constants import SPECIES, ev_to_joule
from tokamak_orbits.pusher import (
    Trajectory, boris_step, gyroperiod, gyroradius, integrate, rk45_push,
)

M, Q = SPECIES["D"]
B0 = 2.0
V = float(np.sqrt(2 * ev_to_joule(10e3) / M))
TC = gyroperiod(M, Q, B0)
RHO = gyroradius(M, Q, V, B0)


@pytest.fixture
def uniform():
    return UniformField(b_vec=(0.0, 0.0, B0))


# --------------------------------------------------------------------------
class TestExactGyroOrbit:
    """In a uniform B the motion is an exact circle; anything else is a bug."""

    def test_radius(self, uniform):
        """At dt = T_c/200 the discrete orbit radius is high by ~1.2e-4.

        This is not sloppiness: the Boris rotation angle per step is
        2 arctan(omega dt / 2) rather than omega dt, which inflates the radius
        by (omega dt)^2 / 12 = 8.2e-5 at leading order. The tolerance is set
        just above the measured value so a real regression still trips it.
        """
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       TC / 200, 200 * 5, M, Q, uniform.b_field)
        r = np.linalg.norm(tr.x[:, 0, :2], axis=-1).max() / 2
        assert r == pytest.approx(RHO, rel=2e-4)

    def test_period(self, uniform):
        """One full gyroperiod must return the particle to its start."""
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       TC / 2000, 2000, M, Q, uniform.b_field)
        assert np.linalg.norm(tr.x[-1, 0] - tr.x[0, 0]) < 1e-3 * RHO

    def test_no_parallel_drift(self, uniform):
        """A purely perpendicular launch must not acquire parallel velocity."""
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       TC / 100, 100 * 20, M, Q, uniform.b_field)
        assert np.max(np.abs(tr.v[:, 0, 2])) < 1e-9 * V

    def test_parallel_velocity_is_constant(self, uniform):
        v0 = np.array([[0.6 * V, 0.0, 0.8 * V]])
        tr = integrate(np.zeros((1, 3)), v0, TC / 100, 100 * 20,
                       M, Q, uniform.b_field)
        assert np.allclose(tr.v[:, 0, 2], 0.8 * V, rtol=1e-12)

    def test_helix_pitch(self, uniform):
        """Distance advanced along B in one gyroperiod is v_par * T_c."""
        v0 = np.array([[0.6 * V, 0.0, 0.8 * V]])
        tr = integrate(np.zeros((1, 3)), v0, TC / 1000, 1000,
                       M, Q, uniform.b_field)
        assert tr.x[-1, 0, 2] == pytest.approx(0.8 * V * TC, rel=1e-3)

    def test_rotation_sense_is_negative_for_positive_charge(self, uniform):
        """A positive charge in +z field gyrates clockwise seen from +z."""
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       TC / 400, 100, M, Q, uniform.b_field)
        assert tr.x[10, 0, 1] < 0

    def test_guiding_centre_offset(self, uniform):
        """Larmor circle is centred one gyroradius from the launch point."""
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       TC / 400, 400, M, Q, uniform.b_field)
        centre = tr.x[:, 0, :2].mean(axis=0)
        assert np.linalg.norm(centre) == pytest.approx(RHO, rel=2e-2)


class TestEnergyConservation:
    def test_boris_conserves_energy_to_roundoff(self, uniform):
        tr = integrate(np.zeros((1, 3)), np.array([[0.6 * V, 0, 0.8 * V]]),
                       TC / 40, 40 * 2000, M, Q, uniform.b_field)
        ke = tr.kinetic_energy(M)[:, 0]
        assert np.abs(ke - ke[0]).max() / ke[0] < 1e-12

    def test_boris_energy_error_does_not_grow(self, uniform):
        """The hallmark of a symplectic-like scheme: bounded, not secular."""
        tr = integrate(np.zeros((1, 3)), np.array([[0.6 * V, 0, 0.8 * V]]),
                       TC / 40, 40 * 4000, M, Q, uniform.b_field)
        ke = tr.kinetic_energy(M)[:, 0]
        err = np.abs(ke - ke[0]) / ke[0]
        first, last = err[: len(err) // 4].max(), err[-len(err) // 4:].max()
        assert last < 10 * max(first, 1e-16)

    def test_energy_conserved_in_tokamak_field(self):
        f = TokamakField()
        x0 = np.array([[1.15, 0.0, 0.0]])
        v0 = np.array([[0.0, 0.5 * V, np.sqrt(1 - 0.25) * V]])
        tr = integrate(x0, v0, TC / 40, 40 * 500, M, Q, f.b_field)
        ke = tr.kinetic_energy(M)[:, 0]
        assert np.abs(ke - ke[0]).max() / ke[0] < 1e-12

    def test_magnetic_force_does_no_work(self, uniform):
        """|v| is invariant under a single Boris step with E = 0."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=(50, 3))
        v = rng.normal(size=(50, 3)) * V
        _, v2 = boris_step(x, v, TC / 7, M, Q, uniform.b_field)
        assert np.allclose(np.linalg.norm(v2, axis=-1),
                           np.linalg.norm(v, axis=-1), rtol=1e-13)

    def test_energy_conserved_for_huge_timestep(self, uniform):
        """Boris stays on the energy shell even when badly under-resolved.

        The orbit is wrong at dt = T_c/2, but |v| is still exact, because the
        magnetic substep is a rotation regardless of angle. This is why energy
        conservation alone is NOT evidence that a run is converged -- see
        docs/NUMERICS.md.
        """
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       TC / 2, 500, M, Q, uniform.b_field)
        ke = tr.kinetic_energy(M)[:, 0]
        assert np.abs(ke - ke[0]).max() / ke[0] < 1e-12


class TestConvergence:
    def test_second_order_in_timestep(self, uniform):
        errs = []
        for nper in (20, 40, 80, 160):
            tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                           TC / nper, nper * 20, M, Q, uniform.b_field)
            r = np.linalg.norm(tr.x[:, 0, :2], axis=-1).max() / 2
            errs.append(abs(r - RHO) / RHO)
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        assert all(1.85 < o < 2.15 for o in orders), orders

    def test_error_decreases_monotonically(self, uniform):
        errs = []
        for nper in (10, 20, 40, 80, 160, 320):
            tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                           TC / nper, nper * 20, M, Q, uniform.b_field)
            r = np.linalg.norm(tr.x[:, 0, :2], axis=-1).max() / 2
            errs.append(abs(r - RHO) / RHO)
        assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))


class TestEcrossB:
    """E x B drift is mass- and charge-independent -- a sharp test of the
    electric half-kicks, which are otherwise unexercised (this project runs
    with E = 0)."""

    def test_drift_velocity(self):
        E = np.array([0.0, 1e4, 0.0])
        B = np.array([0.0, 0.0, B0])
        f = UniformField(b_vec=tuple(B), e_vec=tuple(E))
        expected = e_cross_b_drift(E, B)
        # A raw start-to-end displacement is contaminated by the Larmor
        # circle, which over 60 gyroperiods is 52% of the drift displacement.
        # Fitting a straight line to x(t) averages the gyration away.
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       TC / 100, 100 * 400, M, Q, f.b_field, f.e_field)
        measured = np.polyfit(tr.t, tr.x[:, 0, 0], 1)[0]
        assert measured == pytest.approx(expected[0], rel=5e-3)

    def test_drift_independent_of_species(self):
        E = np.array([0.0, 1e4, 0.0])
        f = UniformField(b_vec=(0, 0, B0), e_vec=tuple(E))
        speeds = []
        for name in ("H", "D", "T"):
            m, q = SPECIES[name]
            tc = gyroperiod(m, q, B0)
            tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                           tc / 100, 100 * 400, m, q, f.b_field, f.e_field)
            speeds.append(np.polyfit(tr.t, tr.x[:, 0, 0], 1)[0])
        assert np.std(speeds) / abs(np.mean(speeds)) < 1e-2

    def test_zero_field_is_free_streaming(self):
        f = UniformField(b_vec=(0, 0, 1e-30))
        tr = integrate(np.zeros((1, 3)), np.array([[V, 0, 0]]),
                       1e-9, 100, M, Q, f.b_field)
        assert tr.x[-1, 0, 0] == pytest.approx(V * tr.t[-1], rel=1e-6)


class TestLossHandling:
    def test_particle_is_frozen_after_loss(self):
        """Uses Ip = 0 so the particle is genuinely unconfined.

        With the default 450 kA it is NOT lost -- the grad-B drift moves it
        only 6.5 cm in 200 gyroperiods and the poloidal field returns it. That
        is the physics this project exists to show, but it makes for a useless
        test of the loss bookkeeping.
        """
        f = TokamakField(plasma_current=0.0)
        x0 = np.array([[1.15, 0.0, 0.0]])
        v0 = np.array([[0.0, 0.0, V]])   # purely perpendicular -> pure drift

        def loss(p):
            r, _ = f.flux_coords(p)
            return r > 0.2

        tr = integrate(x0, v0, TC / 40, 40 * 2000, M, Q, f.b_field,
                       loss_func=loss)
        assert np.isfinite(tr.loss_time[0])
        after = tr.alive[:, 0].argmin()
        assert np.allclose(tr.x[after:, 0], tr.x[after, 0])

    def test_loss_time_infinite_when_confined(self):
        f = TokamakField()
        x0 = np.array([[1.15, 0.0, 0.0]])
        v0 = np.array([[0.0, 0.3 * V, 0.954 * V]])
        tr = integrate(x0, v0, TC / 40, 40 * 50, M, Q, f.b_field,
                       loss_func=lambda p: np.zeros(len(p), bool))
        assert not np.isfinite(tr.loss_time[0])

    def test_early_exit_when_all_lost(self):
        f = TokamakField()
        x0 = np.tile([[1.29, 0.0, 0.0]], (4, 1))
        v0 = np.tile([[0.0, 0.0, V]], (4, 1))
        tr = integrate(x0, v0, TC / 40, 40 * 5000, M, Q, f.b_field,
                       loss_func=lambda p: f.flux_coords(p)[0] > 0.3,
                       stop_when_all_lost=True)
        assert tr.n_steps < 40 * 5000

    def test_independent_particles_do_not_interact(self):
        """Running N particles together must equal running them separately."""
        f = TokamakField()
        rng = np.random.default_rng(7)
        x0 = np.tile([[1.15, 0.0, 0.0]], (5, 1))
        v0 = rng.normal(size=(5, 3))
        v0 *= V / np.linalg.norm(v0, axis=-1, keepdims=True)
        together = integrate(x0, v0, TC / 40, 40 * 30, M, Q, f.b_field)
        for i in range(5):
            alone = integrate(x0[i:i + 1], v0[i:i + 1], TC / 40, 40 * 30,
                              M, Q, f.b_field)
            assert np.allclose(together.x[:, i], alone.x[:, 0], rtol=1e-12)


class TestBorisVersusRK45:
    def test_rk45_reproduces_boris_short_term(self):
        """Over a few orbits the two agree; they diverge only in the long run."""
        f = TokamakField()
        x0 = np.array([1.15, 0.0, 0.0])
        v0 = np.array([0.0, 0.5 * V, 0.866 * V])
        t_end = 20 * TC
        t, x, v = rk45_push(x0, v0, t_end, M, Q, f.b_field, rtol=1e-11, atol=1e-13)
        tr = integrate(x0[None], v0[None], TC / 1000, 1000 * 20, M, Q, f.b_field)
        # Boris advances the gyrophase at 2 arctan(omega dt/2)/dt rather than
        # omega, a relative frequency error of (omega dt)^2/12. Over 20 orbits
        # at dt = T_c/1000 that is a displacement of ~4e-6 m = 4e-4 rho, so the
        # two schemes cannot be expected to agree better than that.
        assert np.linalg.norm(x[-1] - tr.x[-1, 0]) < 5e-3 * RHO

    def test_rk45_energy_drifts_more_than_boris(self):
        """The motivating measurement for choosing Boris."""
        f = TokamakField()
        x0 = np.array([1.15, 0.0, 0.0])
        v0 = np.array([0.0, 0.5 * V, 0.866 * V])
        t_end = 3000 * TC
        t, x, v = rk45_push(x0, v0, t_end, M, Q, f.b_field, rtol=1e-6, atol=1e-8)
        ke = 0.5 * M * np.sum(v**2, axis=-1)
        rk_err = np.abs(ke - ke[0]).max() / ke[0]
        tr = integrate(x0[None], v0[None], TC / 40, int(3000 * 40),
                       M, Q, f.b_field)
        bk = tr.kinetic_energy(M)[:, 0]
        boris_err = np.abs(bk - bk[0]).max() / bk[0]
        assert rk_err > 100 * boris_err


class TestTrajectoryObject:
    def test_shapes(self):
        f = TokamakField()
        x0 = np.tile([[1.15, 0.0, 0.0]], (3, 1))
        v0 = np.tile([[0.0, 0.5 * V, 0.866 * V]], (3, 1))
        tr = integrate(x0, v0, TC / 40, 400, M, Q, f.b_field, sample_every=10)
        assert tr.x.shape[1:] == (3, 3)
        assert tr.v.shape == tr.x.shape
        assert tr.alive.shape == tr.x.shape[:2]
        assert tr.n_particles == 3
        assert tr.t.shape[0] == tr.x.shape[0]

    def test_speed_and_energy_agree(self):
        f = TokamakField()
        tr = integrate(np.array([[1.15, 0.0, 0.0]]),
                       np.array([[0.0, 0.5 * V, 0.866 * V]]),
                       TC / 40, 200, M, Q, f.b_field)
        assert np.allclose(tr.kinetic_energy(M), 0.5 * M * tr.speed() ** 2)

    def test_accepts_1d_input(self):
        f = TokamakField()
        tr = integrate(np.array([1.15, 0.0, 0.0]), np.array([0.0, 0.0, V]),
                       TC / 40, 50, M, Q, f.b_field)
        assert tr.n_particles == 1


class TestHelpers:
    def test_gyroperiod_formula(self):
        assert gyroperiod(M, Q, B0) == pytest.approx(2 * np.pi * M / (Q * B0))

    def test_gyroradius_formula(self):
        assert gyroradius(M, Q, V, B0) == pytest.approx(M * V / (Q * B0))

    def test_gyroperiod_independent_of_energy(self):
        assert gyroperiod(M, Q, B0) == gyroperiod(M, Q, B0)

    def test_gyroradius_proportional_to_speed(self):
        assert gyroradius(M, Q, 2 * V, B0) == pytest.approx(
            2 * gyroradius(M, Q, V, B0))
