"""Solov'ev equilibrium: does it actually solve Grad-Shafranov?"""
import numpy as np
import pytest

from tokamak_orbits import TokamakField
from tokamak_orbits.constants import MU_0, SPECIES, ev_to_joule
from tokamak_orbits.diagnostics import (
    is_trapped, make_loss_func, mirror_ratio,
)
from tokamak_orbits.equilibrium import (
    SolovevField, force_balance_residual, grad_shafranov_residual,
)
from tokamak_orbits.particles import initialise, trapped_fraction_analytic
from tokamak_orbits.pusher import gyroperiod, integrate

M, Q = SPECIES["D"]
V = float(np.sqrt(2 * ev_to_joule(10e3) / M))


@pytest.fixture
def eq():
    return SolovevField()


def interior(field, n=300, seed=0):
    """Random points inside the plasma boundary."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(field.r_inboard + 0.02, field.r_outboard - 0.02, 4 * n)
    Z = rng.uniform(-0.28, 0.28, 4 * n)
    keep = field.psi_of(R, Z) < 0.95 * field.psi_b
    return R[keep][:n], Z[keep][:n]


# --------------------------------------------------------------------------
class TestGradShafranov:
    """The whole point of this field: it solves the equilibrium equation."""

    def test_residual_is_zero(self, eq):
        """Necessary but weak -- see TestForceBalance for why.

        `psi_of` is quartic in R and quadratic in Z, so both central
        differences in `grad_shafranov_residual` are exact and the residual
        is analytically zero for any (c, k, R0). What this measures is
        floating-point cancellation, which scales as 1/h^2: 7e-12 at
        h = 1e-3, 8.9e-6 at h = 1e-6. It confirms the algebra was typed in
        correctly and nothing more.
        """
        R, Z = interior(eq, 200)
        res = grad_shafranov_residual(eq, R, Z)
        assert np.abs(res).max() < 1e-5

    def test_residual_is_roundoff_and_scales_like_one(self, eq):
        """Asserted so nobody quotes it as a physics result.

        A genuine discretisation error falls as h^2. This one *rises* as
        1/h^2, which is the signature of pure cancellation.
        """
        R, Z = interior(eq, 40)
        big = np.abs(grad_shafranov_residual(eq, R, Z, h=1e-3)).max()
        small = np.abs(grad_shafranov_residual(eq, R, Z, h=1e-6)).max()
        assert small > 1e3 * big

    def test_delta_star_matches_the_analytic_form(self, eq):
        """Delta* psi = 2cR^2 + 2ckR0^2 exactly, by construction."""
        R, Z = interior(eq, 50)
        h = 1e-5
        d2 = (eq.psi_of(R, Z + h) - 2 * eq.psi_of(R, Z)
              + eq.psi_of(R, Z - h)) / h**2

        def inner(Rq):
            return (eq.psi_of(Rq + h, Z) - eq.psi_of(Rq - h, Z)) / (2 * h) / Rq

        delta_star = R * (inner(R + h) - inner(R - h)) / (2 * h) + d2
        expected = 2 * eq.c * R**2 + 2 * eq.c * eq.k * eq.R0**2
        assert np.allclose(delta_star, expected, rtol=1e-5)

    def test_source_terms_are_constant(self, eq):
        """Solov'ev means p' and FF' are both flux-independent."""
        assert eq.pressure_gradient() == pytest.approx(-2 * eq.c / MU_0)
        assert eq.ff_prime() == pytest.approx(-2 * eq.c * eq.k * eq.R0**2)

    def test_circular_field_does_not_solve_it(self):
        """Guard: the circular model is divergence-free but NOT an equilibrium.

        Grad-Shafranov requires Delta* psi to be an exact affine function of
        R^2 on each flux surface. For the circular model it is not, and the
        violation grows with radius -- 0.4% at r = 0.10 m, 5.7% at r = 0.25 m.
        If this ever starts passing, someone has changed the circular field
        into something it is not.
        """
        from scipy.integrate import quad
        tf = TokamakField()

        def psi(R, Z):
            r = np.hypot(np.asarray(R) - tf.R0, np.asarray(Z))
            out = np.empty(r.shape)
            for i, rr in enumerate(r.ravel()):
                out.ravel()[i] = -tf.R0 * quad(
                    lambda s: float(tf.b_poloidal_reference(s)), 0, rr,
                    limit=60)[0]
            return out

        h = 1e-4
        th = np.linspace(0.15, 2 * np.pi - 0.15, 40)
        r = 0.25
        R = tf.R0 + r * np.cos(th)
        Z = r * np.sin(th)
        d2 = (psi(R, Z + h) - 2 * psi(R, Z) + psi(R, Z - h)) / h**2
        inner = lambda Rq: (psi(Rq + h, Z) - psi(Rq - h, Z)) / (2 * h) / Rq
        ds = R * (inner(R + h) - inner(R - h)) / (2 * h) + d2
        A = np.stack([R**2, np.ones_like(R)], axis=-1)
        coef, *_ = np.linalg.lstsq(A, ds, rcond=None)
        rel = np.abs(ds - A @ coef).max() / np.abs(ds).mean()
        assert rel > 0.01


class TestForceBalance:
    """The check that actually touches `b_field`.

    `grad_shafranov_residual` is computed from `psi_of` and the two constant
    source terms and never evaluates the field, so an `F^2(psi)` that does
    not integrate `FF'` correctly leaves it untouched. This class closes that
    hole by testing J x B = grad p on the field the particles see.
    """

    def interior_RZ(self, eq, n=120, seed=11):
        rng = np.random.default_rng(seed)
        R = rng.uniform(eq.R0 - 0.22, eq.R0 + 0.22, 4 * n)
        Z = rng.uniform(-0.18, 0.18, 4 * n)
        keep = eq.psi_of(R, Z) < 0.85 * eq.psi_b
        return R[keep][:n], Z[keep][:n]

    def test_force_balance_holds(self, eq):
        R, Z = self.interior_RZ(eq)
        res = force_balance_residual(eq, R, Z)
        assert np.abs(res).max() < 1e-6

    def test_it_catches_an_inconsistent_toroidal_field(self, eq):
        """The mutation the Grad-Shafranov residual cannot see.

        Replacing F^2 = F0^2 - 4ckR0^2 psi with a -12 coefficient breaks the
        equilibrium: the toroidal field no longer integrates the FF' the
        source term claims. The GS residual stays at ~5e-8 (it never looks at
        F), while force balance jumps to ~0.14 -- nine orders of magnitude.
        """
        R, Z = self.interior_RZ(eq, 60, seed=12)
        good = np.abs(force_balance_residual(eq, R, Z)).max()
        gs_good = np.abs(grad_shafranov_residual(eq, R, Z)).max()

        broken = SolovevField()
        broken.f_squared = lambda psi: (
            broken.F0**2 - 12.0 * broken.c * broken.k * broken.R0**2
            * np.asarray(psi, float))
        bad = np.abs(force_balance_residual(broken, R, Z)).max()
        gs_bad = np.abs(grad_shafranov_residual(broken, R, Z)).max()

        assert bad > 1e-3 and bad > 1e6 * good        # force balance catches it
        assert gs_bad < 1e-5 and gs_good < 1e-5       # GS residual does not


class TestSolenoidal:
    def test_divergence_is_zero(self, eq):
        R, Z = interior(eq, 150)
        pts = np.stack([R, np.zeros_like(R), Z], axis=-1)
        h = 1e-7
        div = np.zeros(len(pts))
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h
            pm[:, k] -= h
            div += (eq.b_field(pp)[:, k] - eq.b_field(pm)[:, k]) / (2 * h)
        scale = eq.b_magnitude(pts) / eq.a
        assert np.max(np.abs(div) / scale) < 1e-6

    def test_axisymmetric(self, eq):
        mags = []
        for ph in np.linspace(0, 2 * np.pi, 13):
            p = np.array([[1.15 * np.cos(ph), 1.15 * np.sin(ph), 0.05]])
            mags.append(float(eq.b_magnitude(p)[0]))
        assert np.std(mags) / np.mean(mags) < 1e-13


class TestNotCircular:
    """These are the properties the circular model cannot have."""

    def test_b_times_R_is_not_constant_on_a_surface(self, eq):
        """The identity that made the trapped fraction true by construction."""
        R, Z = eq.surface_points(0.5 * eq.psi_b, 401)
        pts = np.stack([R, np.zeros_like(R), Z], axis=-1)
        b = eq.b_magnitude(pts)
        assert np.std(b * R) / np.mean(b * R) > 1e-4

    def test_circular_field_does_have_that_identity(self):
        """The contrast, asserted so the comparison stays honest."""
        tf = TokamakField()
        th = np.linspace(0, 2 * np.pi, 401)
        R = tf.R0 + 0.2 * np.cos(th)
        Z = 0.2 * np.sin(th)
        pts = np.stack([R, np.zeros_like(R), Z], axis=-1)
        b = np.linalg.norm(tf.b_field(pts), axis=-1)
        assert np.std(b * R) / np.mean(b * R) < 1e-12

    def test_shafranov_shift_exists(self, eq):
        """The boundary is not a circle: the inboard leg sits inside R0 - a."""
        assert eq.r_outboard == pytest.approx(eq.R0 + eq.a, rel=1e-9)
        assert eq.r_inboard < eq.R0 - eq.a

    def test_trapping_boundary_deviates_from_the_expansion(self, eq):
        """And the deviation grows with radius, as an expansion error should."""
        ratios = []
        for r in (0.10, 0.15, 0.20, 0.25):
            p = np.array([[eq.R0 + r, 0.0, 0.0]])
            xi_c = np.sqrt(1 - 1 / np.ravel(mirror_ratio(eq, p))[0])
            ratios.append(xi_c / trapped_fraction_analytic(r / eq.R0))
        assert all(x > 1.01 for x in ratios)
        assert all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))


class TestSurfaceInterface:
    def test_surface_label_is_the_launch_radius(self, eq):
        for r in (0.0, 0.05, 0.15, 0.30):
            p = np.array([[eq.R0 + r, 0.0, 0.0]])
            assert np.ravel(eq.surface_label(p))[0] == pytest.approx(r, abs=1e-9)

    def test_effective_radius_round_trip(self, eq):
        r = np.array([0.05, 0.15, 0.29])
        psi = eq.psi_from_effective_radius(r)
        assert np.allclose(eq.effective_minor_radius(psi), r)

    def test_boundary_label_is_a(self, eq):
        assert eq.boundary_label == pytest.approx(eq.a)

    def test_loss_function_brackets_the_boundary(self, eq):
        loss = make_loss_func(eq)
        assert not np.ravel(loss(np.array([[eq.R0 + 0.99 * eq.a, 0, 0]])))[0]
        assert np.ravel(loss(np.array([[eq.R0 + 1.01 * eq.a, 0, 0]])))[0]

    def test_b_extrema_bracket_the_local_field(self, eq):
        p = np.array([[eq.R0 + 0.2, 0.0, 0.0]])
        lo, hi = eq.b_extrema_on_surface(p)
        b = eq.b_magnitude(p)[0]
        assert lo[0] <= b + 1e-9
        assert hi[0] >= b - 1e-9

    def test_psi_zero_on_axis_and_psi_b_at_edge(self, eq):
        assert float(eq.psi_of(eq.R0, 0.0)) == pytest.approx(0.0, abs=1e-30)
        assert float(eq.psi_of(eq.R0 + eq.a, 0.0)) == pytest.approx(eq.psi_b)


class TestSafetyFactorAndCurrent:
    def test_q_axis_matches_target_for_circular_near_axis(self, eq):
        assert eq.q_axis == pytest.approx(eq.q0_target, rel=2e-3)

    @pytest.mark.parametrize("kappa", [1.0, 1.5, 1.8])
    def test_realised_q_axis_is_kappa_times_the_target(self, kappa):
        """`q_axis` is a handle, not a target, and by an exact factor.

        The constructor sets c = B0/(2 q0 R0^2), which realises q0 exactly
        only at kappa = 1; elongating multiplies it by kappa. Asserted rather
        than tolerated, because an earlier docstring called the discrepancy
        "approximate" when it is exact and can reach 80%.
        """
        eq = SolovevField(q_axis=1.0, elongation=kappa)
        assert eq.q_axis == pytest.approx(kappa, rel=2e-3)

    def test_q_rises_outward(self, eq):
        qs = [eq.safety_factor_exact(f * eq.psi_b) for f in (0.1, 0.4, 0.8, 1.0)]
        assert all(qs[i] < qs[i + 1] for i in range(len(qs) - 1))

    def test_f_squared_stays_positive(self, eq):
        assert eq.f_squared(eq.psi_b) > 0

    @pytest.mark.parametrize("bad", [
        dict(minor_radius=0.0), dict(major_radius=-1.0),
        dict(minor_radius=2.0), dict(q_axis=0.0), dict(elongation=-1.0),
    ])
    def test_invalid_parameters_rejected(self, bad):
        with pytest.raises(ValueError):
            SolovevField(**bad)

    def test_elongation_changes_the_shape(self):
        tall = SolovevField(elongation=1.8)
        R, Z = tall.surface_points(0.6 * tall.psi_b, 401)
        assert (Z.max() - Z.min()) / (R.max() - R.min()) > 1.3


class TestOrbitsInTheEquilibrium:
    """The integrator must work against this field with no changes."""

    def test_energy_conserved(self, eq):
        x0, v0, m, q = initialise(eq, r_start=0.15, pitch=[0.5, -0.3])
        dt = gyroperiod(m, q, 3.5) / 40
        tr = integrate(x0, v0, dt, 20000, m, q, eq.b_field)
        ke = tr.kinetic_energy(m)
        assert np.abs(ke - ke[0]).max() / ke[0].max() < 1e-12

    def test_trapped_classification_is_constant_along_an_orbit(self, eq):
        """mu conservation demands it, and it is a strong check on the field."""
        for xi, expected in ((0.20, True), (0.90, False)):
            x0, v0, m, q = initialise(eq, r_start=0.15, pitch=[xi])
            dt = gyroperiod(m, q, 3.5) / 40
            tr = integrate(x0, v0, dt, 30000, m, q, eq.b_field, sample_every=20)
            flag = is_trapped(eq, tr.x, tr.v)[:, 0]
            assert flag.mean() > 0.98 if expected else flag.mean() < 0.02

    def test_particles_are_confined_at_full_current(self, eq):
        from tokamak_orbits.particles import sample_pitch
        pitches = sample_pitch(40, rng=1, mode="uniform_xi")
        x0, v0, m, q = initialise(eq, r_start=0.12, pitch=pitches)
        dt = gyroperiod(m, q, 3.5) / 40
        tr = integrate(x0, v0, dt, 30000, m, q, eq.b_field,
                       loss_func=make_loss_func(eq), sample_every=50)
        assert np.isfinite(tr.loss_time).mean() < 0.2
