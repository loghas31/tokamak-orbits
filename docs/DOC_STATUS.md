# Status: what is proven, what is assumed

A deliberate boundary between claims backed by a measurement in this repository
and claims that are inherited, approximate, or simply not tested. Read this
before quoting any number from `RESULTS.md`.

## Proven here

Each of these is measured by code in this repository and asserted by a test.

| Claim | Evidence |
|---|---|
| The field is divergence-free | `\|div B\|/(\|B\|/a) ~ 1e-9` at 300 random interior points, for all currents and peaking factors tested |
| The integrator is second-order accurate | measured slopes 1.974, 1.993, 1.998, 2.000, 2.000 against the exact uniform-field orbit |
| The integrator conserves energy | `\|dE/E\| < 1e-13` over 4000 gyro-orbits, bounded and non-growing |
| Grad-B + curvature drift matches theory | 0.11-0.69% against `v_D = m(v_par^2 + v_perp^2/2)/(qBR)` |
| E x B drift matches theory, species-independent | 0.5% against `E x B / B^2`; H, D, T agree to 1% |
| Ions are unconfined without a poloidal field | every test particle reaches the wall at `Ip = 0` |
| Orbits split into passing and trapped | classified for pitch angles either side of the predicted boundary |
| Trapped orbits are wider near the boundary | width at `xi = 0.45` exceeds that at `xi = 0.20` |
| Co- and counter-going orbits shift oppositely | co spans r in [0.070, 0.161], counter r in [0.160, 0.235] |
| Banana width scales as `1/B_theta` and `sqrt(E)` | see `RESULTS.md` |
| `mu` non-conservation is physical, not numerical | flat under 16x timestep refinement; linear in `rho/a` |
| Ensemble loss vs plasma current is gradual, not sharp | 90%-to-10% width is 157% of the midpoint current |
| Collision operator reproduces the Lorentz eigenvalue spectrum | Legendre l=1,2,3 rates within 0.2% of the exact l(l+1)nu/2, no fitted parameters |
| The pitch update cannot leave [-1,1] | proved by Cauchy-Schwarz; measured maximum equals the analytic bound to 5 decimals |
| Collisions destroy the single-particle loss threshold | the same ion is still lost at 3.9x its collisionless threshold at nu_d = 1e5 |
| The Solov'ev field is in force balance | `J x B = grad p` with `J` differenced from `b_field` itself: 2.7e-10 max. This is the load-bearing check -- the finite-differenced Grad-Shafranov residual is algebraically zero (roundoff only, scaling as 1/h^2) and never evaluates the field; an F^2 mutation that breaks the equilibrium leaves it and all 28 tests untouched, while force balance moves nine orders of magnitude. See finding 26. |
| The circular field does *not* solve Grad-Shafranov | the same measurement on the circular model: 0.4% violation at r = 0.10 m rising to 5.7% at r = 0.25 m. Asserted by a test, so the distinction cannot quietly vanish |
| The energy-operator drag equals the NRL speed drag | identical to 2.6e-13 over 0.02 <= v/v_th <= 20, through the sign change, for every mass ratio H/D/T/He4. Derived from detailed balance, not quoted |
| Both a cold and a hot beam relax to the same Maxwellian | KS statistic against the analytic speed CDF below the 95% critical value at `nu dt <= 0.005`; the two starts agree to three digits |
| The ripple field is divergence-free | 1e-9 relative, unchanged from the axisymmetric case, while the naive component-wise version sits at 1.6e-2 -- and the naive leftover is identified exactly as the missing term to 1e-7 |
| Ripple loses particles the axisymmetric field confines | 40 launches, paired across amplitudes: 17.1% of the 35 the axisymmetric field confines are lost at `delta_edge` = 2%. Not one-directional -- three particles lost at 0.5% are confined at 1% and 2%. See `RESULTS.md` §10.3 |

## Assumed, inherited, or approximate

| Item | Status |
|---|---|
| **The field is an equilibrium** | **False, and important.** Circular concentric flux surfaces with no Shafranov shift do not solve Grad-Shafranov. This is a divergence-free toy field with tokamak-like topology. Nothing here should be read as a prediction about a real machine. |
| Trapped fraction `sqrt(2 eps/(1+eps))` | Agreement is partly true *by construction*: `\|B\|` is exactly proportional to `1/R` on a flux surface in this model, so the formula is exact rather than approximate. Validates the integrator and classifier, not the field. See finding 4. |
| Banana width prefactor `2 rho_theta sqrt(eps)` | Order-unity convention, not predictive. Only the *scaling* is tested; the fitted prefactor is reported, not asserted. |
| Guiding-centre reconstruction | First order in `rho/L`; `rho/a = 0.034` here, so it is a percent-level diagnostic approximation. The orbits themselves are full-orbit and assume nothing. |
| Safety factor `q(r)` | Large-aspect-ratio cylindrical form. Used for labelling only, never inside the integrator. |
| The current profile | An arbitrary `(1-(r/a)^2)^nu` toy, not measured or self-consistent. The plasma current is imposed; the particles do not carry it. |
| "Confinement time" | Time for a test particle to reach `r = a`, with no sources or sinks. It is an *orbit loss time*, not an energy or particle confinement time in the experimental sense. |
| `tau_detrap = eps / nu_d` | Magnitude confirmed to 4%, scaling exponent -0.962 against -1 once the collision step is converged. The first measurement (-0.779) was an under-resolved timestep, not a failure of theory; see finding 22. |
| Collision frequencies scanned | `nu_eff/omega_b` spans 0.1 to 32, banana regime to well past plateau. A real 10 keV deuterium plasma at 1e19 m^-3 sits near the bottom. The upper rows are a parameter scan, not a machine. |
| Loss boundary | Defined on the particle position, not the guiding centre. A convention worth `rho/a = 3.4%`. See finding 10. |
| The Solov'ev *measured* trapped fractions | Sampling-limited: 100 launches per point gives +-0.05 absolute, which does not separate the two field models at better than ~2 sigma. The deterministic `xi_crit` comparison is the result of that section; the measured column is not. |
| The main scans on the Solov'ev field | **Not done.** Sections 1-7 and 10 all run on the circular model, so the headline thresholds carry an unquantified geometry error. |
| Ripple amplitude | `delta_edge` = 1-2% is at or above the high end for a real machine. Chosen so the effect is measurable in a 3e-4 s run, not because it is typical. |
| Goldston-White-Boozer threshold | Reported as a diagnostic with its inputs, not as a calibrated threshold: the numerical prefactor varies between references and was not pinned down here. In the scan run it is below 1 at *every* amplitude, so it does not discriminate between the amplitudes that lose particles and the one that does not. |
| The ripple well fraction | An **area** fraction, weighted by the surface element. The angle-averaged number is 24% smaller; both are recorded in `results/ripple.json`. |
| Test count as a coverage measure | It is not one. About a dozen of the 234 tests assert an implementation back to itself (see finding 26f). They are kept as change detectors; the mutation tests are the honest coverage measure. |

## Not modelled at all

Stated so that absence is not mistaken for a null result.

- **Momentum conservation between species.** The background is a fixed
  Maxwellian at rest with no back-reaction, in *both* collision operators, so
  there is **no bootstrap current** and no complete neoclassical calculation.
- **The section 7 scans with energy scattering on.** Detrapping and collisional
  smearing were measured with the fixed-speed Lorentz operator only. Repeating
  them with the full operator would change the answers by an amount nobody here
  has measured. Do not read section 7 as "the collisional result" and section 9
  as a refinement of it; they are two operators.
- **A neoclassical diffusion coefficient.** Not measured. The radial statistics
  over a 150 us run are too poor to fit one worth reporting.
- **Electric fields.** `E = 0` everywhere. No radial electric field, no sheath,
  no `E x B` rotation. The `E x B` machinery exists only to validate the pusher.
- **Time-dependent fields.** Every field here is static, so there is no
  induction, no MHD activity, no sawteeth.
- **Ripple transport coefficients.** Ripple *is* modelled (`ripple.py`), and it
  loses particles. What is not measured is a diffusion coefficient for it.
- **Self-consistency.** Test particles only: no current, no pressure, no
  back-reaction on the field.
- **Electrons.** Deuterons, protons, tritons and alphas are available; electron
  dynamics would need a far smaller timestep and are not attempted.
- **A wall.** "Lost" means crossing `r = a`. No reflection, recycling, or
  material interaction.

## Known open questions

- The `mu` error at `xi -> 1` is amplified because `v_perp^2` is a small
  difference of large numbers. The `4.1 (rho/a)` coefficient is measured at
  `xi = 0.95` and is not claimed to be universal across pitch.
- Whether the headline current thresholds survive on a real equilibrium. The
  Solov'ev field moves the trapping boundary by 2.3-5.7%; the main scan was not
  re-run on it, and the size of the resulting shift in the thresholds is
  unknown.
- Whether ripple loss reaches the bulk population at realistic amplitudes, or
  only the edge population it was measured on.
- The highest-collisionality smearing point is truncated by the scan range, so
  "6x the collisionless threshold" is a lower bound.
- The confinement scan is censored at the integration limit: above ~320 kA no
  particle is lost within 150 us, so the mean confinement time saturates at
  `t_max` and the curve there is a lower bound, not a measurement.
- Orbit classification counts sign changes of a gyro-averaged `v_par`. For
  deeply trapped particles (`xi ~ 0`) `v_par` hovers near zero and the bounce
  *count* becomes unreliable, though the trapped/passing *classification* does
  not.
