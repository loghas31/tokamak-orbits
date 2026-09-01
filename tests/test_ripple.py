"""Field ripple: does breaking axisymmetry break the divergence too?"""
import numpy as np
import pytest

from tokamak_orbits import SolovevField, TokamakField
from tokamak_orbits.constants import SPECIES
from tokamak_orbits.diagnostics import make_loss_func
from tokamak_orbits.particles import initialise
from tokamak_orbits.pusher import gyroperiod, integrate
from tokamak_orbits.ripple import (
    RippledField, gwb_threshold, ripple_amplitude_measured,
    ripple_well_fraction, ripple_well_parameter,
)

M, Q = SPECIES["D"]


@pytest.fixture
def rip():
    return RippledField(TokamakField(), n_coils=16, delta_edge=0.01)


def interior(field, n=300, seed=0):
    rng = np.random.default_rng(seed)
    r = field.a * np.sqrt(rng.uniform(0, 0.9, n))
    th = rng.uniform(0, 2 * np.pi, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    R = field.R0 + r * np.cos(th)
    return np.stack([R * np.cos(ph), R * np.sin(ph), r * np.sin(th)], axis=-1)


# --------------------------------------------------------------------------
class TestStillSolenoidal:
    """The whole point. Finding 2 was a hand-written non-solenoidal field;
    this is the same trap with an extra dimension to get wrong."""

    def test_divergence_is_zero(self, rip):
        pts = interior(rip, 300)
        h = 1e-7
        div = np.zeros(len(pts))
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h
            pm[:, k] -= h
            div += (rip.b_field(pp)[:, k] - rip.b_field(pm)[:, k]) / (2 * h)
        scale = np.linalg.norm(rip.b_field(pts), axis=-1) / rip.a
        assert np.max(np.abs(div) / scale) < 1e-6

    def test_the_naive_version_would_have_failed(self, rip):
        """Guard: writing B_phi *= (1 + delta cos N phi) and stopping there
        leaves an uncancelled phi-derivative.

        The test does not just assert "big"; it identifies the leftover
        exactly as the missing term,

            div B = -(N delta B_phi / R) sin(N phi) / R,

        which is what makes it a diagnosis rather than an alarm. Asserted so
        that nobody "simplifies" the potential away.
        """
        base = rip.base

        def naive(pos):
            pos = np.atleast_2d(pos)
            R = np.hypot(pos[:, 0], pos[:, 1])
            phi = np.arctan2(pos[:, 1], pos[:, 0])
            b = base.b_field(pos).copy()
            fac = rip.delta(R) * np.cos(rip.n_coils * phi)
            # add delta*B_phi in the phi direction only
            b[:, 0] += -np.sin(phi) * base.B0 * base.R0 / R * fac
            b[:, 1] += np.cos(phi) * base.B0 * base.R0 / R * fac
            return b

        pts = interior(rip, 60, seed=3)
        h = 1e-7
        div = np.zeros(len(pts))
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h
            pm[:, k] -= h
            div += (naive(pp)[:, k] - naive(pm)[:, k]) / (2 * h)
        R = np.hypot(pts[:, 0], pts[:, 1])
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        missing = (-rip.n_coils * rip.delta(R) * base.B0 * base.R0 / R
                   * np.sin(rip.n_coils * phi) / R)
        assert np.max(np.abs(div - missing)) / np.max(np.abs(missing)) < 1e-6
        scale = np.linalg.norm(naive(pts), axis=-1) / rip.a
        assert np.max(np.abs(div) / scale) > 1e-3

    def test_potential_is_harmonic(self, rip):
        """div B = laplacian(chi), checked on chi directly."""
        pts = interior(rip, 80, seed=1)
        h = 1e-5
        lap = np.zeros(len(pts))
        c0 = rip.ripple_potential(pts)
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h
            pm[:, k] -= h
            lap += (rip.ripple_potential(pp) - 2 * c0
                    + rip.ripple_potential(pm)) / h**2
        scale = np.abs(c0).max() / rip.a**2
        assert np.max(np.abs(lap)) / scale < 1e-4

    def test_ripple_field_is_the_gradient_of_the_potential(self, rip):
        """The components are hand-differentiated; this checks the hand."""
        pts = interior(rip, 60, seed=2)
        h = 1e-6
        grad = np.zeros_like(pts)
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h
            pm[:, k] -= h
            grad[:, k] = (rip.ripple_potential(pp)
                          - rip.ripple_potential(pm)) / (2 * h)
        assert np.allclose(grad, rip.ripple_field(pts), rtol=1e-5,
                           atol=1e-9 * np.abs(rip.ripple_field(pts)).max())

    def test_ripple_is_curl_free(self, rip):
        """It is a vacuum field: no coil current inside the plasma."""
        pts = interior(rip, 60, seed=4)
        h = 1e-6
        jac = np.zeros((len(pts), 3, 3))
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h
            pm[:, k] -= h
            jac[:, :, k] = (rip.ripple_field(pp) - rip.ripple_field(pm)) / (2 * h)
        curl = np.stack([jac[:, 2, 1] - jac[:, 1, 2],
                         jac[:, 0, 2] - jac[:, 2, 0],
                         jac[:, 1, 0] - jac[:, 0, 1]], axis=-1)
        scale = np.linalg.norm(rip.ripple_field(pts), axis=-1).max() / rip.a
        assert np.max(np.abs(curl)) / scale < 1e-5

    def test_works_on_the_solovev_equilibrium_too(self):
        """A curl-free addition preserves any divergence-free field."""
        rip = RippledField(SolovevField(), n_coils=18, delta_edge=0.01)
        pts = interior(rip, 120, seed=5)
        h = 1e-7
        div = np.zeros(len(pts))
        for k in range(3):
            pp, pm = pts.copy(), pts.copy()
            pp[:, k] += h
            pm[:, k] -= h
            div += (rip.b_field(pp)[:, k] - rip.b_field(pm)[:, k]) / (2 * h)
        scale = np.linalg.norm(rip.b_field(pts), axis=-1) / rip.a
        assert np.max(np.abs(div) / scale) < 1e-6


class TestAmplitudeAndShape:
    def test_axisymmetry_is_actually_broken(self, rip):
        mags = []
        for ph in np.linspace(0, 2 * np.pi / rip.n_coils, 9):
            p = np.array([[1.28 * np.cos(ph), 1.28 * np.sin(ph), 0.0]])
            mags.append(float(np.linalg.norm(rip.b_field(p))))
        assert np.std(mags) / np.mean(mags) > 1e-4

    def test_n_fold_periodic_in_phi(self, rip):
        ph = np.linspace(0, 2 * np.pi, 97)
        p = np.stack([1.25 * np.cos(ph), 1.25 * np.sin(ph),
                      np.zeros_like(ph)], axis=-1)
        b = np.linalg.norm(rip.b_field(p), axis=-1)
        spec = np.abs(np.fft.rfft(b[:-1]))
        assert int(np.argmax(spec[1:]) + 1) == rip.n_coils

    def test_measured_amplitude_matches_the_model(self, rip):
        """|B| ripple is the *toroidal field* ripple diluted by B_pol.

        The construction sets the ripple on B_phi exactly. What an orbit
        feels is |B| = sqrt(B_phi^2 + B_pol^2), whose modulation is smaller
        by 1/(1 + (B_pol/B_phi)^2). On the outboard edge that is 2.2%, and
        it is a real physical dilution rather than an error.
        """
        R = rip.R0 + rip.a
        meas = ripple_amplitude_measured(rip, R)
        b_pol = float(rip.base.b_poloidal(rip.a, R))
        b_tor = float(rip.base.b_toroidal_at(R))
        expected = float(rip.delta(R)) / (1 + (b_pol / b_tor) ** 2)
        assert meas == pytest.approx(expected, rel=2e-3)

    def test_ripple_grows_steeply_outward(self, rip):
        assert rip.delta(rip.R0 + rip.a) / rip.delta(rip.R0) == pytest.approx(
            1.3**16, rel=1e-9)
        assert rip.delta(rip.R0 - rip.a) < rip.delta(rip.R0)

    def test_delta_edge_is_what_it_says(self, rip):
        assert float(rip.delta(rip.R0 + rip.a)) == pytest.approx(0.01)

    def test_zero_ripple_reduces_to_the_base_field(self):
        base = TokamakField()
        rip = RippledField(base, n_coils=16, delta_edge=0.0)
        pts = interior(base, 50, seed=6)
        assert np.allclose(rip.b_field(pts), base.b_field(pts))

    @pytest.mark.parametrize("bad", [dict(n_coils=1), dict(delta_edge=-0.1)])
    def test_invalid_parameters_rejected(self, bad):
        with pytest.raises(ValueError):
            RippledField(TokamakField(), **bad)


class TestWellCriterion:
    def test_midplane_is_always_in_a_well(self, rip):
        assert ripple_well_parameter(rip, 0.2, 0.0) < 1.0

    def test_no_well_near_the_poloidal_extremes(self, rip):
        assert ripple_well_parameter(rip, 0.2, np.pi / 2) > 1.0

    def test_well_region_grows_with_ripple(self):
        base = TokamakField()
        fr = [ripple_well_fraction(RippledField(base, 16, d), 0.25)
              for d in (0.002, 0.005, 0.01, 0.02)]
        assert all(fr[i] < fr[i + 1] for i in range(len(fr) - 1))

    def test_area_weighting_matters(self):
        """The wells sit where the surface element is largest.

        A plain average over poloidal angle understates the area fraction,
        because the well region is at theta ~ 0 and the surface element goes
        as (R0 + r cos theta). 24% at the default settings, which is larger
        than most effects this project reports.
        """
        rip = RippledField(TokamakField(), 16, 0.01)
        area = ripple_well_fraction(rip, 0.25)
        angle = ripple_well_fraction(rip, 0.25, weight="angle")
        assert area > angle
        assert area / angle == pytest.approx(1.24, rel=0.05)

    def test_gwb_is_dimensionless(self):
        """Scale the whole machine: a dimensionless number must not move.

        The first version of `gwb_threshold` substituted dln(delta)/dr for
        q' and multiplied by R0, leaving sigma with units of inverse length.
        That is invisible at R0 = 1 m, which is why this test scales it.
        """
        vals = []
        for R0 in (1.0, 2.0, 4.0):
            m = TokamakField(major_radius=R0, minor_radius=0.3 * R0,
                             plasma_current=4.5e5 * R0)
            g = RippledField(m, 16, 0.01)
            vals.append(gwb_threshold(g, 0.25 * R0, 0.0, 0.0102 * R0))
        assert np.allclose(vals, vals[0], rtol=1e-9)

    def test_gwb_falls_with_amplitude_and_gyroradius(self):
        rip = lambda d: RippledField(TokamakField(), 16, d)
        assert (gwb_threshold(rip(0.0025), 0.2, 0.0, 0.01)
                > gwb_threshold(rip(0.02), 0.2, 0.0, 0.01))
        assert (gwb_threshold(rip(0.01), 0.2, 0.0, 0.005)
                > gwb_threshold(rip(0.01), 0.2, 0.0, 0.02))

    def test_criteria_work_on_the_solovev_base(self):
        """These raised AttributeError before: SolovevField has no
        `safety_factor`, only `safety_factor_exact`, and only the divergence
        test covered the Solov'ev + ripple combination."""
        sv = RippledField(SolovevField(), 18, 0.01)
        assert 0.0 < ripple_well_fraction(sv, 0.2) < 1.0
        assert gwb_threshold(sv, 0.2, 0.0, 0.0102) > 0.0

    def test_well_region_is_non_monotonic_in_coil_number(self):
        """Two effects fight, and neither wins outright.

        More coils sharpen the modulation along a field line, which favours
        wells (the ``N`` in the denominator of ``alpha``). But at a *fixed
        edge* ripple more coils also concentrate the ripple further outboard,
        since ``delta(R) = delta_edge (R/R_edge)^N`` falls faster everywhere
        inside the boundary. The well fraction therefore peaks -- near
        ``N = 16-20`` for this machine -- and falls off on both sides. The
        first version of this test asserted monotonic growth and was wrong.
        """
        base = TokamakField()
        ns = (8, 12, 16, 20, 24, 32)
        fr = [ripple_well_fraction(RippledField(base, n, 0.01), 0.25)
              for n in ns]
        peak = int(np.argmax(fr))
        assert 0 < peak < len(ns) - 1
        assert ns[peak] in (16, 20)
        assert fr[0] < fr[peak] and fr[-1] < fr[peak]

    def test_no_wells_without_ripple(self):
        rip = RippledField(TokamakField(), 16, 0.0)
        assert ripple_well_fraction(rip, 0.25) == 0.0




class TestOrbits:
    def test_energy_is_still_conserved(self, rip):
        """The ripple is static, so the Boris pusher must not care."""
        x0, v0, m, q = initialise(rip, r_start=0.2, pitch=[0.1, 0.6, -0.4])
        dt = gyroperiod(m, q, rip.b_max_in_domain) / 40
        tr = integrate(x0, v0, dt, 20000, m, q, rip.b_field)
        ke = tr.kinetic_energy(m)
        assert np.abs(ke - ke[0]).max() / ke[0].max() < 1e-12

    def test_ripple_loses_a_particle_the_axisymmetric_field_confines(self):
        """The headline: a deeply trapped edge ion at theta = 0.

        xi = -0.12 at r = 0.25 is well inside the trapped cone (xi_crit =
        0.63 there) -- small parallel velocity is exactly what ripple
        trapping needs, and an earlier version of this docstring called it
        "barely passing", which is the opposite.

        Without ripple it is confined for the whole run; with 2% ripple it
        is ripple-trapped and drifts vertically to the wall. Same initial
        condition, same integrator, same timestep.
        """
        base = TokamakField()
        for field, expect_loss in ((base, False),
                                   (RippledField(base, 16, 0.02), True)):
            x0, v0, m, q = initialise(field, r_start=0.25, theta_start=0.0,
                                      pitch=[-0.12])
            dt = gyroperiod(m, q, field.b_max_in_domain) / 40
            n = int(np.ceil(1.0e-4 / dt))
            tr = integrate(x0, v0, dt, n, m, q, field.b_field,
                           sample_every=200, loss_func=make_loss_func(field))
            assert bool(np.isfinite(tr.loss_time[0])) is expect_loss
