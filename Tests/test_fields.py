"""Field model: geometry, profiles, and the solenoidal constraint."""
import numpy as np
import pytest

from tokamak_orbits import TokamakField, UniformField
from tokamak_orbits.constants import MU_0


@pytest.fixture
def field():
    return TokamakField()


def sample_points(field, n=200, seed=0, r_frac=(0.02, 0.98)):
    """Random points inside the plasma volume."""
    rng = np.random.default_rng(seed)
    r = field.a * np.sqrt(rng.uniform(*r_frac, n))
    th = rng.uniform(0, 2 * np.pi, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    R = field.R0 + r * np.cos(th)
    Z = r * np.sin(th)
    return np.stack([R * np.cos(ph), R * np.sin(ph), Z], axis=-1)


def divergence(field, pts, h=1e-7):
    div = np.zeros(len(pts))
    for k in range(3):
        pp, pm = pts.copy(), pts.copy()
        pp[:, k] += h
        pm[:, k] -= h
        div += (field.b_field(pp)[:, k] - field.b_field(pm)[:, k]) / (2 * h)
    return div


# --------------------------------------------------------------------------
class TestSolenoidal:
    """div B = 0 is the single most important property of the field model.

    A poloidal field written as B_theta(r) theta_hat is divergence-free in a
    straight cylinder but NOT on a torus: div B = B_R / R. The R0/R factor in
    TokamakField.b_poloidal is what fixes this.
    """

    def test_divergence_is_zero(self, field):
        pts = sample_points(field, 300)
        scale = np.linalg.norm(field.b_field(pts), axis=-1) / field.a
        assert np.max(np.abs(divergence(field, pts)) / scale) < 1e-6

    @pytest.mark.parametrize("ip", [0.0, 1e5, 4.5e5, 1.2e6])
    def test_divergence_zero_for_all_currents(self, ip):
        f = TokamakField(plasma_current=ip)
        pts = sample_points(f, 100, seed=int(ip) % 97)
        scale = np.linalg.norm(f.b_field(pts), axis=-1) / f.a
        assert np.max(np.abs(divergence(f, pts)) / scale) < 1e-6

    @pytest.mark.parametrize("nu", [0.0, 0.5, 1.0, 2.0, 3.0])
    def test_divergence_zero_for_all_peaking(self, nu):
        f = TokamakField(current_peaking=nu)
        pts = sample_points(f, 100, seed=int(nu * 10))
        scale = np.linalg.norm(f.b_field(pts), axis=-1) / f.a
        assert np.max(np.abs(divergence(f, pts)) / scale) < 1e-6

    def test_naive_cylindrical_field_would_fail(self, field):
        """Guard against someone 'simplifying away' the R0/R factor.

        Rebuilds the field without it and asserts the divergence is large, so
        that this test fails loudly if the correction is ever removed.
        """
        pts = sample_points(field, 100)
        x, y, z = pts.T
        R = np.hypot(x, y)
        phi_hat = np.stack([-y / R, x / R, np.zeros_like(z)], axis=-1)

        def naive(p):
            xx, yy, zz = p[:, 0], p[:, 1], p[:, 2]
            RR = np.hypot(xx, yy)
            ph = np.stack([-yy / RR, xx / RR, np.zeros_like(zz)], axis=-1)
            b = (field.B0 * field.R0 / RR)[:, None] * ph
            dR = RR - field.R0
            r = np.hypot(dR, zz)
            Rh = np.stack([xx / RR, yy / RR, np.zeros_like(zz)], axis=-1)
            Zh = np.zeros_like(Rh); Zh[:, 2] = 1.0
            rh = (dR / r)[:, None] * Rh + (zz / r)[:, None] * Zh
            # NOTE: no R0/R factor -- this is the buggy version
            return b + field.b_poloidal_reference(r)[:, None] * np.cross(ph, rh)

        div = np.zeros(len(pts))
        h = 1e-7
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h; pm[:, k] -= h
            div += (naive(pp)[:, k] - naive(pm)[:, k]) / (2 * h)
        scale = np.linalg.norm(naive(pts), axis=-1) / field.a
        assert np.max(np.abs(div) / scale) > 1e-3


class TestToroidalField:
    def test_one_over_r(self, field):
        for R in (0.75, 1.0, 1.25):
            assert field.b_toroidal_at(R) == pytest.approx(field.B0 * field.R0 / R)

    def test_axis_value(self, field):
        b = field.b_field(np.array([[field.R0, 0.0, 0.0]]))
        assert np.linalg.norm(b) == pytest.approx(field.B0, rel=1e-12)

    def test_toroidal_direction_is_phi_hat(self):
        f = TokamakField(plasma_current=0.0)
        b = f.b_field(np.array([[1.0, 0.0, 0.0]]))[0]
        assert b[1] > 0 and abs(b[0]) < 1e-12 and abs(b[2]) < 1e-12

    def test_axisymmetry(self, field):
        """|B| must depend only on (R, Z), not on toroidal angle."""
        r, z = 0.12, 0.04
        mags = []
        for ph in np.linspace(0, 2 * np.pi, 17):
            R = field.R0 + r
            p = np.array([[R * np.cos(ph), R * np.sin(ph), z]])
            mags.append(np.linalg.norm(field.b_field(p)))
        assert np.std(mags) / np.mean(mags) < 1e-13


class TestCurrentProfile:
    def test_enclosed_current_saturates(self, field):
        assert field.enclosed_current(field.a) == pytest.approx(field.Ip)
        assert field.enclosed_current(2 * field.a) == pytest.approx(field.Ip)

    def test_enclosed_current_zero_on_axis(self, field):
        assert field.enclosed_current(0.0) == pytest.approx(0.0)

    def test_enclosed_current_monotonic(self, field):
        r = np.linspace(0, field.a, 50)
        assert np.all(np.diff(field.enclosed_current(r)) >= 0)

    def test_parabolic_profile_half_radius(self):
        """For nu=1, I(a/2)/Ip = 1 - (1 - 1/4)^2 = 7/16."""
        f = TokamakField(current_peaking=1.0)
        assert f.enclosed_current(f.a / 2) / f.Ip == pytest.approx(7 / 16)

    def test_flat_profile_scales_as_r_squared(self):
        f = TokamakField(current_peaking=0.0)
        assert f.enclosed_current(f.a / 2) / f.Ip == pytest.approx(0.25)

    def test_ampere_law_outside_plasma(self, field):
        """Outside the plasma B_theta must be mu0 Ip / (2 pi r)."""
        r = 1.5 * field.a
        assert field.b_poloidal_reference(r) == pytest.approx(
            MU_0 * field.Ip / (2 * np.pi * r))

    def test_b_poloidal_vanishes_on_axis(self, field):
        assert field.b_poloidal_reference(0.0) == pytest.approx(0.0)

    def test_b_poloidal_linear_near_axis(self, field):
        """I ~ r^2 near the axis, so B_theta ~ r."""
        r = np.array([1e-4, 2e-4, 4e-4])
        b = field.b_poloidal_reference(r)
        assert b[1] / b[0] == pytest.approx(2.0, rel=1e-3)
        assert b[2] / b[1] == pytest.approx(2.0, rel=1e-3)

    def test_r0_over_r_factor(self, field):
        """b_poloidal at R0 equals the straight-cylinder reference value."""
        r = 0.2
        assert field.b_poloidal(r, field.R0) == pytest.approx(
            field.b_poloidal_reference(r))
        assert field.b_poloidal(r, field.R0 / 2) == pytest.approx(
            2 * field.b_poloidal_reference(r))


class TestSafetyFactor:
    def test_default_machine_q_values(self, field):
        """The default machine is tuned to q0 = 1, q(a) = 2."""
        assert field.q_axis == pytest.approx(1.0, rel=2e-2)
        assert field.q_edge == pytest.approx(2.0, rel=2e-2)

    def test_q_axis_matches_limit_of_profile(self, field):
        assert field.safety_factor(1e-6) == pytest.approx(field.q_axis, rel=1e-4)

    def test_q_scales_inversely_with_current(self):
        f1 = TokamakField(plasma_current=4.5e5)
        f2 = TokamakField(plasma_current=9.0e5)
        assert f1.q_edge / f2.q_edge == pytest.approx(2.0, rel=1e-10)

    def test_q_infinite_without_current(self):
        assert np.isinf(TokamakField(plasma_current=0.0).q_axis)


class TestGeometry:
    def test_flux_coords_on_axis(self, field):
        r, th = field.flux_coords(np.array([[field.R0, 0.0, 0.0]]))
        assert r[0] == pytest.approx(0.0)

    def test_flux_coords_outboard_midplane(self, field):
        r, th = field.flux_coords(np.array([[field.R0 + 0.1, 0.0, 0.0]]))
        assert r[0] == pytest.approx(0.1)
        assert th[0] == pytest.approx(0.0)

    def test_flux_coords_top(self, field):
        r, th = field.flux_coords(np.array([[field.R0, 0.0, 0.1]]))
        assert r[0] == pytest.approx(0.1)
        assert th[0] == pytest.approx(np.pi / 2)

    def test_epsilon(self, field):
        assert field.epsilon == pytest.approx(0.3)

    @pytest.mark.parametrize("bad", [
        dict(minor_radius=0.0), dict(major_radius=-1.0),
        dict(minor_radius=2.0), dict(current_peaking=-1.0),
    ])
    def test_invalid_geometry_rejected(self, bad):
        with pytest.raises(ValueError):
            TokamakField(**bad)


class TestUniformField:
    def test_constant_everywhere(self):
        f = UniformField(b_vec=(0.1, 0.2, 0.3))
        pts = np.random.default_rng(0).normal(size=(20, 3))
        assert np.allclose(f.b_field(pts), np.array([0.1, 0.2, 0.3]))

    def test_zero_e_by_default(self):
        f = UniformField(b_vec=(0, 0, 1))
        assert np.allclose(f.e_field(np.zeros((5, 3))), 0.0)

    def test_divergence_free(self):
        f = UniformField(b_vec=(0.3, -0.2, 1.0))
        pts = np.random.default_rng(1).normal(size=(10, 3))
        assert np.max(np.abs(divergence(f, pts))) < 1e-6


class TestFieldShape:
    def test_accepts_single_point(self, field):
        assert field.b_field(np.array([1.0, 0.0, 0.0])).shape == (1, 3)

    def test_accepts_many_points(self, field):
        pts = sample_points(field, 37)
        assert field.b_field(pts).shape == (37, 3)

    def test_b_magnitude_matches_norm(self, field):
        pts = sample_points(field, 20)
        assert np.allclose(field.b_magnitude(pts),
                           np.linalg.norm(field.b_field(pts), axis=-1))

    def test_repr_is_informative(self, field):
        s = repr(field)
        for token in ("R0", "a=", "B0", "Ip", "q0", "qa"):
            assert token in s


class TestExactSafetyFactorAndCurrent:
    """The R0/R factor of finding 2 also changes the derived quantities.

    These tests pin down finding 13: the nominal ``plasma_current`` and the
    cylindrical ``q`` are labels, and the true values differ by exactly
    ``R0 / sqrt(R0^2 - r^2)``.
    """

    def test_exact_q_exceeds_cylindrical_by_known_factor(self, field):
        for r in (0.05, 0.15, 0.30):
            factor = field.R0 / np.sqrt(field.R0**2 - r**2)
            assert field.safety_factor_exact(r) == pytest.approx(
                float(field.safety_factor(r)) * factor, rel=1e-4)

    def test_exact_q_at_edge(self, field):
        """The machine labelled q(a) = 2.00 really has q(a) = 2.097."""
        assert field.safety_factor_exact(field.a) == pytest.approx(2.097, rel=2e-3)

    def test_exact_q_matches_axis_value_on_axis(self, field):
        assert field.safety_factor_exact(1e-6) == pytest.approx(
            field.q_axis, rel=1e-3)

    def test_enclosed_current_actual_matches_ampere_loop(self, field):
        """Integrate B.dl around a poloidal circuit and compare."""
        for r in (0.10, 0.20, 0.30):
            th = np.linspace(0, 2 * np.pi, 20001)[:-1]
            dth = th[1] - th[0]
            R = field.R0 + r * np.cos(th)
            Z = r * np.sin(th)
            pos = np.stack([R, np.zeros_like(R), Z], axis=-1)
            B = field.b_field(pos)
            # theta_hat = -sin(th) R_hat + cos(th) Z_hat, with R_hat = x_hat here
            that = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], axis=-1)
            circulation = np.sum(np.sum(B * that, axis=-1)) * r * dth
            i_enc = abs(circulation) / MU_0
            assert i_enc == pytest.approx(
                float(field.enclosed_current_actual(r)), rel=1e-3)

    def test_actual_enclosed_current_exceeds_nominal(self, field):
        """The headline Ip understates the enclosed current by 4.8% at r = a."""
        ratio = field.enclosed_current_actual(field.a) / field.Ip
        assert ratio == pytest.approx(1.0483, rel=1e-3)

    def test_nominal_and_actual_agree_on_axis(self, field):
        r = 1e-4
        assert field.enclosed_current_actual(r) == pytest.approx(
            field.enclosed_current(r), rel=1e-6)
