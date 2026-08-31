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
| `tau_detrap = eps / nu_d` | Magnitude confirmed to 4%; **scaling exponent is not** -- measured -0.779 against -1, unexplained. See finding 22. Do not use this repository as evidence for the -1 scaling. |
| Collision frequencies scanned | `nu_eff/omega_b` spans 0.1 to 32, banana regime to well past plateau. A real 10 keV deuterium plasma at 1e19 m^-3 sits near the bottom. The upper rows are a parameter scan, not a machine. |
| Loss boundary | Defined on the particle position, not the guiding centre. A convention worth `rho/a = 3.4%`. See finding 10. |

## Not modelled at all

Stated so that absence is not mistaken for a null result.

- **Energy scattering, drag and slowing-down.** Collisions are pitch-angle only
  (the Lorentz operator). Speed is exactly conserved by construction, so there is
  no thermalisation and no fast-ion slowing-down.
- **Momentum conservation between species.** The background is a fixed
  Maxwellian at rest with no back-reaction, so there is **no bootstrap current**
  and no complete neoclassical calculation. What the collision operator adds is
  trapped-particle detrapping, not transport coefficients.
- **A neoclassical diffusion coefficient.** Not measured. The radial statistics
  over a 150 us run are too poor to fit one worth reporting.
- **Electric fields.** `E = 0` everywhere. No radial electric field, no sheath,
  no `E x B` rotation. The `E x B` machinery exists only to validate the pusher.
- **Time-dependent fields.** The field is static, so there is no induction, no
  ripple, no MHD activity, no sawteeth.
- **Field ripple.** Perfect axisymmetry. Real tokamaks have discrete coils and
  the resulting ripple is a leading loss channel for fast ions.
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
- The detrapping scaling exponent disagrees with theory by 22% and the cause is
  not established (finding 22).
- The highest-collisionality smearing point is truncated by the scan range, so
  "6x the collisionless threshold" is a lower bound.
- The confinement scan is censored at the integration limit: above ~320 kA no
  particle is lost within 150 us, so the mean confinement time saturates at
  `t_max` and the curve there is a lower bound, not a measurement.
- Orbit classification counts sign changes of a gyro-averaged `v_par`. For
  deeply trapped particles (`xi ~ 0`) `v_par` hovers near zero and the bounce
  *count* becomes unreliable, though the trapped/passing *classification* does
  not.
