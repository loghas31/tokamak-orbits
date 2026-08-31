# Self-review: findings against my own work

A running register of errors found, claims retracted, and things that turned out
to be weaker than they first looked. Findings are numbered in the order they were
found, not by importance. Nothing here is removed once written; when a finding is
superseded, the correction is appended to it.

The point of this file is that a result is only worth as much as the search for
reasons it might be wrong. Several entries below would have silently changed the
headline numbers.

**Legend** — `BUG` code was wrong · `RETRACTED` a claim I made and withdrew ·
`WEAKER` a result that is real but proves less than advertised ·
`TRAP` a mistake avoided, recorded so it is not made later.

---

### 1. `BUG` — deuteron mass had an electron subtracted from it

`DEUTERON_MASS` was written as `2.013553212745 u - m_e`. The CODATA value
`2.013553212745 u` is already the *deuteron* (bare nucleus); the neutral
deuterium **atom** is `2.0141017778 u`. Subtracting an electron from the nucleus
mass gave `3.34267e-27 kg` against the true `3.34358e-27 kg`.

The same error was present for tritium and, doubled, for helium-4, where
`4.002603254 u` is the atom and `4.001506179127 u` the alpha particle.

Size: 0.027% in `m_D`. Small, but it enters the gyroradius and hence the banana
width linearly, and it would have been an error nobody could see. Caught before
any physics run. Fixed in `constants.py`; masses are now explicitly documented as
bare-nucleus values.

---

### 2. `BUG` — the poloidal field was not divergence-free

**The most serious error in this project.** The first field model set the
poloidal field to `B_theta(r)` directed along the poloidal unit vector, which is
the textbook straight-cylinder result. On a **torus** that field is not
solenoidal. In cylindrical coordinates

```
div B = (1/R) d(R B_R)/dR + (1/R) dB_phi/dphi + dB_Z/dZ
```

and for a purely poloidal field of that form the first and third terms leave a
residual `B_R / R`. Measured on the default machine over 3000 interior points:
`|div B| / (|B|/a)` has max **4.83%**, median **2.76%**. A magnetic field with a
divergence is not a magnetic field; it
implies magnetic monopoles distributed through the plasma, does work that a real
`v x B` force cannot, and corrupts every orbit at `O(eps)`.

Found by a finite-difference divergence check written before any orbit was
integrated, because "is this actually a magnetic field" seemed like the first
thing to establish.

**Fix.** Derive the field from a poloidal flux function, which is solenoidal by
construction:

```
B = grad(psi) x grad(phi) + F(psi) grad(phi)
```

For flux surfaces that are circles of radius `r` about the magnetic axis, taking
`dpsi/dr = -R0 mu0 I(r) / (2 pi r)` gives a poloidal field equal to the
straight-cylinder value multiplied by `R0/R`. That single factor is the whole
correction. Verified: relative `|div B|` fell from `1.0e-1` to `1.0e-9`, which
is the finite-difference floor.

`tests/test_fields.py::TestSolenoidal::test_naive_cylindrical_field_would_fail`
rebuilds the buggy version and asserts its divergence is large, so the
correction cannot be quietly removed later.

**Correction to this entry.** It originally quoted the error as "10% of |B|".
That figure came from `|div B| / |B|`, which has units of inverse length
(max 0.161 m^-1) and is not a percentage of anything. Normalised properly by
`|B|/a` to give a dimensionless number, it is 4.83% max / 2.76% median. The
conclusion is unchanged -- a 5% divergence is still fatal -- but the number was
being quoted as roughly twice its real size through a units error.

**Caveat that must be stated with the fix:** circular concentric flux surfaces
with no Shafranov shift are *not* a solution of the Grad-Shafranov equation.
This is a divergence-free toy field, not an MHD equilibrium. See
`DOC_STATUS.md`.

---

### 3. `TRAP` — sampling uniformly in pitch angle is not isotropic

For an isotropic velocity distribution the solid-angle element is
`d(cos alpha) dphi`, so an isotropic ensemble is uniform in `xi = cos alpha`,
**not** uniform in the pitch angle `alpha`. Sampling `alpha` uniformly
over-weights near-perpendicular particles, which are exactly the ones that are
trapped.

Measured, at `r = 0.15` where the true trapped fraction is 0.5108:

| sampling | measured trapped fraction | bias |
|---|---|---|
| uniform in `xi` | 0.5112 | +0.1% |
| uniform in `alpha` | 0.3416 | **-33.1%** |

A 33% error in the headline trapped fraction, from one line of sampling code.
Both modes are kept in `particles.sample_pitch` so the difference is
demonstrable; `uniform_xi` is the default and is used for every result.

---

### 4. `WEAKER` — the trapped-fraction agreement is partly true by construction

Reported as "measured trapped fraction agrees with `sqrt(2 eps / (1+eps))`". That
is true, and it is a real check of the integrator and the orbit classifier. It is
**not** an independent check of the field model, and the original wording implied
it was.

Reason: in this field both components carry the same `1/R` factor, so on a flux
surface

```
|B| = (R0/R) sqrt(B0^2 + B_theta,ref(r)^2)
```

which is *exactly* proportional to `1/R`, whatever the plasma current. Hence
`B_min/B_max = R_min/R_max = (R0-r)/(R0+r)` identically, and the mirror-ratio
trapping condition `|xi_0| < sqrt(1 - B_min/B_max)` reduces algebraically to
`sqrt(2 eps/(1+eps))`. Verified numerically: `std(|B| * R) / mean = 2e-16` on a
flux surface, for `Ip` = 0, 450 kA and 900 kA alike.

So the comparison tests that the code reproduces a property its own field model
guarantees. What it genuinely validates: orbit integration, the trapped/passing
classifier, and the pitch sampling. What it does not validate: the field model.
A real equilibrium with a Shafranov shift would break the exact `1/R` and the
formula would then only be approximate.

The claim in `RESULTS.md` is scoped accordingly.

---

### 5. `BUG` — leapfrog staggering biased every banana width by 1.5%

The Boris pusher is a leapfrog: `x_n = x_{n-1} + v_n dt` means the stored `v_n`
is the velocity over `[t_{n-1}, t_n]`, centred at `t_{n-1/2}`, while `x_n` sits
at `t_n`. The guiding-centre reconstruction `X_gc = x + m (v x B)/(q B^2)`
combines the two, so it inherited a half-step inconsistency.

Found by testing the guiding centre in a **uniform** field, where the exact
answer is a stationary point. It was not stationary: it traced a circle of
radius `omega dt * rho`, first order in `dt` and 15% of a gyroradius at the
production `dt = T_c/40`.

Effect on the headline number — measured banana width at `xi = 0.45`:

| `dt` | before fix | after fix |
|---|---|---|
| `T_c/20` | 0.092125 m | 0.089453 m |
| `T_c/40` | 0.090679 m | 0.089377 m |
| `T_c/80` | 0.089959 m | 0.089366 m |
| `T_c/160` | 0.089614 m | 0.089368 m |
| `T_c/320` | 0.089462 m | 0.089362 m |

Before: 3.0% spread across the range and only first-order convergence, so the
production value was 1.5% high — a systematic that would have been reported as
physics. After: 0.1% spread, converged at `T_c/20`.

**Fix.** `guiding_centre` now takes the timestep and rotates the velocity
forward by half a step about **B**, using the same rotation the pusher uses. The
residual wobble drops from `0.126 rho` to `0.004 rho` and becomes second order
(verified: falls by exactly 4x per halving of `dt`).

---

### 6. `RETRACTED` — "the magnetic moment is not being conserved well enough"

Initial reaction to seeing `|delta mu / mu|` reach 14% for `xi = 0.95` was that
the integrator was wrong, since energy was conserved to `1e-14` at the same time.

It is not an integrator error. Two measurements settle it:

**Refining the timestep 16x changes nothing** (fixed physics, `t_end = 6 us`):

| steps per gyro-period | `xi = 0.95` | `xi = 0.45` |
|---|---|---|
| 20 | 0.139311 | 0.026553 |
| 40 | 0.139081 | 0.026469 |
| 80 | 0.139003 | 0.026449 |
| 160 | 0.138961 | 0.026444 |
| 320 | 0.138969 | 0.026442 |

*(These are the values in `results/convergence.json`. An earlier version of this
entry carried numbers from a scratch run made before the production timestep
policy was fixed; they differed in the fourth digit. Findings must quote the
shipped data, not the exploration that led to them.)*

**Varying the gyroradius changes it linearly** (fixed `dt`):

| `E` (keV) | `rho/a` | `delta mu / mu` | ratio |
|---|---|---|---|
| 0.5 | 0.0076 | 0.0306 | 4.03 |
| 2 | 0.0152 | 0.0615 | 4.05 |
| 10 | 0.0340 | 0.1391 | 4.09 |
| 40 | 0.0681 | 0.2838 | 4.17 |

So `delta mu / mu ~ 4.1 (rho/a)`: first order in the finite-Larmor-radius
parameter, independent of `dt`. `mu` is an *adiabatic* invariant, conserved only
asymptotically as `rho/L -> 0`; at 10 keV in a 30 cm plasma `rho/a = 0.034` is
not small enough for `mu` to hold to better than ~10%.

This is now an asserted property rather than a worry:
`test_error_is_physical_not_numerical` requires that refining `dt` does **not**
change the answer, so a genuine pusher bug would make it fail.

The `xi`-dependence has the same origin: at `xi = 0.95`, `v_perp^2` is only 9.75%
of `v^2`, so `mu` is computed as a small difference of large numbers and the same
absolute error is a larger relative one.

---

### 7. `RETRACTED` — the original plan to integrate with `scipy` RK45

The project was specified around `scipy.integrate` RK45. That is the wrong tool
here. RK45 is not symplectic and its energy error accumulates secularly; over the
`~10^5` gyro-orbits a confinement study needs, an ion silently gains or loses
energy and the measured confinement time becomes an artefact of the integrator
rather than a property of the field.

Replaced with the Boris pusher, whose magnetic substep is an exact rotation, so
`|v|` is preserved to round-off regardless of step size. Measured over 4000
gyro-orbits in the tokamak field: Boris `|dE/E| < 1e-14`, bounded and
non-growing; RK45 at `rtol=1e-6` drifts by orders of magnitude more and keeps
growing. `test_rk45_energy_drifts_more_than_boris` asserts a factor of 100.

RK45 is kept in `pusher.rk45_push` purely as the control that demonstrates this,
and to cross-check Boris over short times where both are accurate.

---

### 8. `TRAP` — energy conservation is not evidence of convergence

Corollary of finding 7 that is easy to get backwards. Boris conserves `|v|`
exactly *because* the magnetic substep is a rotation — and a rotation by the
wrong angle is still a rotation. At `dt = T_c/2` the orbit is badly wrong and the
energy error is still `1e-14`.

So a flat energy trace says nothing about whether a run is resolved. The
resolution claim rests on the separate convergence study
(`docs/NUMERICS.md`): the gyroradius error is second order, measured slopes
1.974, 1.993, 1.998, 2.000, 2.000. `test_energy_conserved_for_huge_timestep`
pins this down deliberately.

---

### 9. `RETRACTED` — "confinement collapses sharply once the banana width crosses
the plasma boundary"

This was the project's stated expectation, written before any code existed. For
an **ensemble** it is wrong.

Measured: 300 deuterons at 10 keV, launched at `r = 0.15 m` on the outboard
midplane with pitch sampled uniformly in `xi`, integrated to 150 us, over 24
plasma currents from 60 kA to 900 kA. The lost fraction falls from 90% to 10% of
its maximum between `Ip = 78 kA` and `Ip = 273 kA` — a width of 195 kA, or
**157% of the midpoint current**. That is a broad, gradual rolloff, not a
threshold.

The reason is structural, and it is the interesting part. A threshold does exist,
but it is a property of a *single* pitch angle: a particle is lost when its own
orbit width exceeds its own distance to the wall, and both depend on `xi`.
Particles near the trapped/passing boundary have the widest orbits and are lost
first; deeply trapped and deeply passing particles have narrow orbits and survive
to much lower current. An ensemble spans all of them at once, so the sharp
single-particle thresholds are smeared into a gradual ensemble curve — each pitch
angle crossing its own threshold at a different `Ip`.

The retraction and the mechanism are both reported. This is the project's main
result.

*(Appended after run B: the fixed-pitch scans quantify the single-particle side
of this. See `RESULTS.md`.)*

---

### 10. `TRAP` — the loss boundary is a convention, and it moves the answer

"Lost" is defined as the **particle** position exceeding `r = a`, not the
guiding centre. The two differ by up to a gyroradius, which at 10 keV is
1.02 cm against `a = 30 cm` — about 3.4% of the minor radius. Particles whose
guiding centre sits just inside the boundary but whose Larmor circle crosses it
count as lost under this convention and would not under the other.

This is not a physical statement; a real machine has a material limiter and the
particle convention is the closer analogue. It is recorded because the
confinement numbers shift if it changes, and because "confinement time" sounds
like a measured quantity when part of it is a choice.

---

### 11. `BUG` — the banana-width scan was measuring orbits that were not bananas

The width scaling study launched its test particle at `xi = 0.45`. At `r = 0.15`
the trapped/passing boundary is `xi_crit = 0.511`, so that is only **12% inside
the trapped region**.

Banana width grows as the plasma current falls. Wide orbits sample a range of
`|B|` far from their launch surface, and a particle launched marginally inside
the boundary can end up on the passing side entirely. That is what happened: the
measured "banana widths" were non-monotonic in both plasma current and energy —

| `Ip` (kA) | width (m) | bounces | trapped? |
|---|---|---|---|
| 250 | 0.08548 | **0** | **no** |
| 350 | 0.12597 | 2 | yes |
| 450 | 0.08937 | 3 | yes |
| 600 | 0.06375 | 4 | yes |
| 800 | 0.04637 | 5 | yes |

At 250 kA the particle has zero bounces and is not trapped, so the number in the
width column is the radial excursion of a *passing* orbit. The 350 kA point is a
badly distorted near-boundary orbit reaching `r_min = 0.0345`, almost to the
magnetic axis. Both were being reported as banana widths. The same effect
corrupted the 40 keV point of the energy scan.

**Fix.** Two changes. The scan now launches at `xi = 0.30`, comfortably inside
the trapped region, which gives a monotonic sequence (0.08550, 0.06457, 0.05125,
0.03898, 0.02946 for the same currents). And a guard rejects any point where the
particle is lost, is not classified trapped, or records fewer than two bounces,
printing `SKIPPED` with the reason rather than silently contributing a number.
The guard immediately caught a further point, `E = 1.25` keV, which has only one
bounce in the integration window.

Only after this did the scaling exponents come out sensibly:
`w_b ~ B_theta^-0.950` against an ideal `-1`, and `w_b ~ E^+0.464` against an
ideal `+1/2`.

The general lesson, which applies to the whole project: a diagnostic that assumes
an orbit is of a particular type must **verify** that it is, not assume the
launch condition guarantees it.

---

### 12. `WEAKER` — the trapped fraction only matches theory in a narrow window

Following finding 4, the remaining question was how well the measurement tracks
`sqrt(2 eps/(1+eps))` across minor radius. Answer: only in the middle.

| `r` (m) | `w_b/r` | `w_b/(a-r)` | lost/600 | measured/analytic |
|---|---|---|---|---|
| 0.05 | **1.85** | 0.37 | 0 | 0.621 |
| 0.10 | 0.68 | 0.34 | 0 | 0.922 |
| 0.15 | 0.40 | 0.40 | 0 | **0.963** |
| 0.20 | 0.29 | 0.59 | 54 | 0.907 |
| 0.25 | 0.25 | **1.25** | 216 | 0.786 |

The deviation is 7.3 sigma at `r = 0.05` and 5.3 sigma at `r = 0.25`, so it is
not statistics. Two distinct causes:

- **Small `r`:** the banana is 1.85x wider than the flux-surface radius itself.
  The particle never samples the local `eps` the formula is built on; it sweeps
  across most of the plasma. The thin-orbit assumption does not merely degrade,
  it becomes meaningless.
- **Large `r`:** the banana exceeds the distance to the wall, 36% of the sample
  is lost before classification, and the lost ones are preferentially the wide
  near-boundary orbits the trapped fraction is most sensitive to. The surviving
  sample is biased.

Agreement is best (3.7%, under 1 sigma) exactly where both ratios are smallest
and comparable. The honest statement is not "the trapped fraction agrees with
theory" but "it agrees to 4% in the one radial window where the assumptions
behind the formula hold, and deviates by up to 38% outside it, in the direction
and by roughly the amount finite orbit width predicts."

The earlier, broader claim is retracted.

---

### 13. `BUG` — the stated plasma current is not the current the field actually carries

The `R0/R` factor of finding 2 fixes the divergence, but it also changes what
current the field implies. `plasma_current` was documented and reported as "total
toroidal plasma current". It is not.

Computing `j = curl(B)/mu0` for the actual field and integrating Ampere loops
around flux surfaces:

| `r` (m) | `I_enc` from the field (kA) | nominal `I(r)` (kA) | ratio |
|---|---|---|---|
| 0.10 | 94.9 | 94.4 | 1.0050 |
| 0.20 | 317.5 | 311.1 | 1.0206 |
| 0.30 (= a) | **471.7** | **450.0** | **1.0483** |
| 0.40 | 491.0 | 450.0 | 1.0911 |
| 0.50 | 519.6 | 450.0 | 1.1547 |

The ratio is exactly `R0 / sqrt(R0^2 - r^2)`. Three consequences, all of which
were being stated wrongly:

1. **The headline "Ip = 450 kA" is 4.8% low.** The field enclosed by the plasma
   edge carries 471.7 kA. `plasma_current` is a *profile parameter* that sets
   `B_theta,ref`, not the physical total current.
2. **`q(a) = 2.00` is a label, not a measurement.** Integrating `dphi/dtheta`
   around a flux surface gives the true field-line safety factor
   `q(r) = r B0 / (B_theta,ref(r) sqrt(R0^2 - r^2))`, which is the cylindrical
   value times the same `R0/sqrt(R0^2-r^2)`. So `q(a) = 2.097`, not 2.000 —
   4.8% high. `q0 = 1.000` is exact because the factor tends to 1 on axis.
3. **`j_phi` is not a flux function.** It varies around a flux surface: at
   `r = 0.15` m it runs 1.918e6 A/m^2 outboard, 2.387e6 at the top, 3.098e6
   inboard — a 62% variation — following
   `j_phi = (R0/R) j(r) - R0 cos(theta) f(r)/(mu0 R^2)`. The docstring claim
   that the enclosed current "saturates at Ip" outside the plasma is also false:
   `j_phi` is non-zero for `r > a`.

None of this affects a single orbit — the field is what it is, it is
divergence-free, and the particles never see `plasma_current` except through
`B_theta`. What it affects is every *label*. Fixed by adding
`safety_factor_exact` and `enclosed_current_actual`, documenting
`plasma_current` as a profile parameter, and quoting `q(a) = 2.097` wherever the
number is a physical claim rather than a scan label.

The general shape of this mistake: a correction was applied to the field (finding
2) and the derived quantities were not re-derived to match.

---

### 14. `BUG` — the drift validation used an uncorrected guiding centre

`exp_drift_validation` reconstructed the guiding centre with an inline copy of
`X_gc = x + m(v x B)/(qB^2)` instead of calling `diagnostics.guiding_centre`. The
inline copy predated finding 5 and omitted the half-step rotation, so the single
measurement offered as "the strongest validation in the project" was contaminated
by the exact bug that finding 5 had already found and fixed elsewhere.

| `E`, `xi` | reported | corrected |
|---|---|---|
| 10 keV, 0.00 | 0.327% | **0.005%** |
| 10 keV, +0.50 | 0.113% | 0.347% |
| 2 keV, +0.50 | 0.270% | 0.129% |

Individual rows were wrong by up to 65x. Worse, `RESULTS.md` had claimed "the
error grows with energy, as it must" while its own table showed 2 keV (0.270%)
worse than 10 keV (0.113%) — a stated trend contradicted by the numbers directly
beneath it, which nobody had checked.

With the corrected diagnostic the trend does hold: 0.129% at 2 keV, 0.347% at
10 keV, 0.698% at 40 keV, close to the expected `sqrt(E)`. So the physics claim
was right and the evidence offered for it was wrong, which is the more
embarrassing of the two.

Fixed by deleting the inline reconstruction and calling the shared function.
The lesson is narrow and practical: a bug fixed in one place is not fixed;
duplicated formulas must be removed, not corrected twice.

---

### 15. `BUG` — the Boris radius-error formula was mis-derived, and the mismatch explained away

`docs/NUMERICS.md` derived the discretisation error in the gyroradius as
`(omega dt)^2 / 12`, predicted `2.06e-3` at `dt = T_c/40` against a measured
`3.08e-3`, and called that "the right size and the right scaling".

It is not the right size. It is 50% low, and the discrepancy should have been a
signal rather than something to wave at. The correct result is exact, not
asymptotic: Boris rotates by `theta` with `tan(theta/2) = omega dt/2`, so the
discrete orbit radius is

```
rho_discrete = rho * sqrt(1 + (omega dt / 2)^2)
```

giving a leading coefficient of `(omega dt)^2 / 8`.

| n/gyro | measured | exact form | `/8` | `/12` (as published) |
|---|---|---|---|---|
| 40 | 3.079483e-3 | 3.07951e-3 | 3.0843e-3 | 2.056e-3 |
| 160 | 1.927456e-4 | 1.92747e-4 | 1.92766e-4 | 1.285e-4 |
| 320 | 4.818899e-5 | 4.81903e-5 | 4.81914e-5 | 3.213e-5 |

The exact form matches to five significant figures. The corrected version is a
much stronger statement than the original claimed to be — the discretisation
error is fully accounted for — and it was available the whole time.

---

### 16. `BUG` — several results tables could not be regenerated by the shipped code

`RESULTS.md` opened with "Every number here comes from
`scripts/run_experiments.py`". That was false for four tables: the orbit-topology
table (section 2), the `E x B` figures (1.2), the "no poloidal field" statement
(1.3), and the banana-width-versus-radius table in section 3. All four came from
interactive exploration during development and were transcribed by hand.

They had drifted. Re-running section 2 through the shipped code path gives widths
of 0.0894 / 0.0309 / 0.0045 m at `xi` = +0.45 / +0.20 / 0.00, against the
published 0.0906 / 0.0324 / 0.0060 — up to **25% off** for the `xi = 0` row —
because the exploration predated the guiding-centre fix of finding 5. The
published table also had "—" for the `xi = 0` bounce count where the shipped code
reports 5.

Fixed by adding experiments **G** (orbit topology and width versus radius),
**H** (`E x B`), and **I** (no poloidal field) to `run_experiments.py`, and
regenerating every affected table from their JSON output.

A reproducibility claim that has never been exercised is not a reproducibility
claim. This one was written in good faith and was wrong within a day.

---

### 17. `WEAKER` — the "sharp threshold" width was resolution-limited, not measured

Reported: the `xi = -0.45` loss transition is 18% of the midpoint current wide,
"nine times sharper than the ensemble".

The fixed-pitch scans used a uniform 30-point grid over 60–900 kA, i.e. a spacing
of 28.97 kA. For `xi = -0.45` the loss fraction reads 1.000 at 234 kA, **0.333 at
263 kA**, 0.000 at 292 kA. The entire transition is described by *one*
intermediate sample, and the quoted 238→283 kA width of 45 kA is 1.55 grid cells
produced by linear interpolation through that single point. The same is true of
`xi = 0.00` (1.000, 0.625, 0.000).

So 18% was an *upper bound set by the grid*, not a measurement — the true width
could have been anything smaller, including zero. The ensemble width, by
contrast, spans 5.3 grid cells and was genuinely resolved, so the *comparison*
was sound in direction while the ratio was not.

Fixed by making the fixed-pitch scan two-stage: a coarse pass brackets the
transition, then 22 points are placed across it. Also the sample was raised from
24 to 48 particles per point, since 24 gives a ~10% binomial standard error and
the transition is being read off fractions.

**Result of the re-run.** The refined widths are far *smaller* than the ones
originally published, so the original figures were not merely unresolved, they
were badly wrong in size:

| pitch | published width | refined width | grid points inside |
|---|---|---|---|
| `xi = -0.45` | 18% | **10.5%** | 4 |
| `xi = 0.00` | 48% | **5.9%** | 4 |
| `xi = +0.45` | "never lost" | **<= 2.7%** | 0 |
| `xi = +0.80` | "never lost" | **<= 3.5%** | 0 |

The two co-going rows remain unresolved even at the finer spacing: all 48
particles flip between adjacent grid points 1.27 kA apart. They are reported as
**upper bounds** with the interior-point count shown, so a reader can see which
numbers are measurements and which are limits.

This makes the project's central comparison stronger, not weaker — the ratio
between the ensemble width and the sharpest resolved single-pitch width goes from
9x to **27x** — but that is incidental. The point is that the original 18% was
not a measurement of anything, and it was published as one.

Related overstatement in the same section: "identical setup" to the ensemble scan
was wrong — the fixed-pitch runs used 24 particles rather than 300, 30 currents
rather than 24, and seed 999 rather than 12345.

---

### 18. `RETRACTED` — "co-going ions are never lost"

Stated in the README abstract, and used to argue that "roughly half of any
isotropic population is not participating in the transition at all".

False. The scans simply started at 60 kA, above the co-going thresholds. The
re-run with the floor lowered to 15 kA puts them at **37.0 kA** (`xi = +0.45`)
and **30.6 kA** (`xi = +0.80`) — both entirely off the bottom of the original
plot. Both are 100% lost below their threshold and 0% above it, with the flip
occurring between adjacent grid points 1.27 kA apart.

`RESULTS.md` had scoped this correctly ("do not fail anywhere in the range
scanned"); the README had not, and the README is what people read.

The mechanism survives and is in fact stronger. The point was never that co-going
ions are immune, but that different pitch angles cross their thresholds at widely
separated currents, so a population's curve is the smeared superposition of many
sharp ones. With the co-going thresholds located, the spread of thresholds across
pitch angle is **8.3x** in current (30.6 kA to 253.5 kA at the 50%-loss
crossing), not the 2.7x originally quoted from the truncated range.

Both the retraction and the strengthened claim are in `RESULTS.md` section 4.2.
The scan floor was lowered to 15 kA so the thresholds are inside the data.

---

### 19. `RETRACTED` — an artefact of the collision operator that cannot occur

The first version of `collisions.py` documented that the Monte Carlo
pitch-angle update "can overshoot for `|xi|` near 1, which is a known artefact
of the scheme", and cited a clipping bias "below 0.5% of `<xi^2>`".

Both halves were invented. The update

```
xi' = xi (1 - a) +/- sqrt((1 - xi^2) a),      a = nu_d dt
```

**cannot** leave `[-1, 1]`. By Cauchy-Schwarz,

```
xi (1-a) + sqrt(a) sqrt(1 - xi^2)  <=  sqrt((1-a)^2 + a) sqrt(xi^2 + 1 - xi^2)
                                    =  sqrt(1 - a + a^2)
```

and `1 - a + a^2 < 1` for every `0 < a < 1`. Measured over 5e5 samples with
`xi` uniform on `[-1,1]`: zero overshoots at `a` = 0.005, 0.02, 0.05 and 0.1,
with the largest post-update `|xi|` being 0.99751.

The 0.5% figure was never measured. It was written into a docstring alongside a
forward reference to a finding that did not yet exist, which is how a plausible
number gets into a repository: it was asserted while describing the code rather
than while testing it.

The clip is kept as a round-off guard and `scatter_pitch(clip=False)` exists so
the claim can be checked. `tests/test_collisions.py` asserts the bound.

The general lesson is narrower than "don't guess": it is that **prose written
at the same time as the code inherits none of the code's testing**. Every
quantitative claim in a docstring needs the same treatment as a claim in
`RESULTS.md`.

---

### 20. `BUG` — the detrapping criterion used a midplane formula everywhere

The collisional detrapping experiment classified a particle as detrapped when
`|xi| > sqrt(2 eps_local / (1 + eps_local))`, evaluated at the particle's
instantaneous position.

That formula is the trapping boundary **at the outboard midplane only**. It is
derived by asking whether a particle launched at the weakest-field point mirrors
before reaching the strongest, so applying it at other poloidal angles compares
the local pitch against a boundary that does not belong there. For a trapped
particle `|xi|` falls as it climbs to higher field, so the test was
systematically lenient off the midplane.

**Fix.** `diagnostics.is_trapped` now uses the mirror condition directly. A
particle mirrors iff its turning-point field lies below the maximum field on its
flux surface, which rearranges to

```
(1 - xi^2) * B_max / B  >  1
```

with everything at the particle's current position. Because `mu` and the energy
are conserved this is independent of where on the orbit it is evaluated, which
is exactly the property the old test lacked.

Verified two ways: at the outboard midplane it reproduces the midplane boundary
exactly (flips between `xi = 0.50` and `0.52` against `xi_crit = 0.5108`), and
along a full orbit it is constant — 100% trapped for `xi = 0.30`, 0% for
`xi = 0.80`, over 2654 samples each.

**Size of the error.** Every detrapping time changed. The ratio
`tau / (eps/nu_d)` moved from 0.87 to 0.62 at `nu_d = 3e3`, from 1.27 to 0.95 at
1e4, and from 1.43 to 1.26 at 3e4 — a 20-30% shift, in the direction expected
from a lenient test. The fitted scaling exponent moved from -0.69 to -0.78.

---

### 21. `TRAP` — censoring and competing risks are not the same thing

The first detrapping statistic was the **mean of the observed** detrapping
times. That is biased low by construction: the particles that never detrapped
are exactly the slow ones, and dropping them removes the right-hand tail.
Switched to the **median**, which is unbiased under right-censoring as long as
more than half the sample detraps.

The filter for which points to fit then had the opposite error. It excluded any
point with *any* censoring at all, which threw away the `nu_d = 3e3` and `1e4`
points (94% and 99.5% detrapped) whose medians are perfectly sound, leaving only
two points and no fit. Relaxed to "more than 60% detrapped".

There is a third distinction that was being missed entirely. A particle can fail
to register a detrapping time for **two different reasons**: it stayed trapped
for the whole run, or it hit the wall while still trapped. The first is
censoring; the second is a *competing risk*, and pooling them makes the median
mean something other than what it appears to. They are now counted separately.
Measured: at most 1 particle in 200 was lost while trapped, at any
collisionality, so in this case the distinction did not change the answer.

It was still worth separating. "I checked and it does not matter here" is a
different statement from "I did not think about it", and only the first one is
safe to carry into a regime where it might.

---

### 22. `RETRACTED` — "the detrapping scaling does not reproduce theory"

**Original entry.** `tau_detrap = eps / nu_d` is the standard banana-regime
estimate. Measured exponent **-0.779** against -1, with `<tau nu/eps> = 1.04`.
Censoring, sampling resolution and the competing risk of wall loss were each
checked and excluded. The entry concluded: *"What remains is physics I have not
accounted for... This is reported as an unexplained discrepancy rather than as
agreement."*

**That conclusion was wrong.** It was not physics. It was the collision
timestep, and the three things I checked were the wrong three things.

The operator is validated (RESULTS §7.1) against the Legendre decay rates,
which are a **bulk** property of the distribution: they average over the whole
population and converge quickly in `nu dt`. A detrapping time is a
**first-passage** quantity, and first passage is far more sensitive to step
size. At the production `nu dt = 0.02` each application of the operator moves
the pitch by `dxi = sqrt((1 - xi^2) nu dt) = 0.135`, while the distance from the
launch pitch to the trapped/passing boundary is only `0.5108 - 0.30 = 0.211`.
The particle was arriving in roughly **two discrete jumps**, not by diffusion,
and a quantised first passage is biased high.

Refining `nu dt` at fixed physics (`scripts/probe_detrap_convergence.py`):

| `nu dt` | `dxi` per step | steps to boundary | fitted exponent |
|---|---|---|---|
| 0.02 | 0.135 | 2.4 | -0.876 |
| 0.005 | 0.0675 | 9.8 | **-0.959** |
| 0.00125 | 0.0337 | 39.0 | **-0.920** |

Rerunning production at `nu dt = 0.005`, with the sampling interval sized to the
detrapping time rather than to the run, and with all six collision frequencies
resolved:

```
tau ~ nu^-0.962   (ideal -1)      <tau nu/eps> = 0.88   (ideal 1)
```

Exponent within 4%, magnitude within 12%, over 333x in collision frequency.
**The textbook estimate was right and my measurement was under-resolved.**

Two things are worth taking from this beyond the number.

The first is that "I checked three explanations and excluded them" is not
evidence for a fourth. The original entry read as rigorous because it listed
what had been ruled out, and that list was persuasive enough to stop the search.
An honest unexplained result is still a result that has not been explained, and
the register should not let the appearance of diligence stand in for a cause.

The second is specific and reusable: **validating a scheme on a bulk statistic
does not validate it for an extreme-value or first-passage statistic.** Nothing
in the Legendre validation was wrong; it simply did not cover the quantity being
measured, and I never noticed that it did not. Every new *kind* of quantity
needs its own convergence study, not just every new piece of code.

---

### 23. `TRAP` — the run was sized to the physics, not to the memory

The first attempt at the converged detrapping rerun was killed by the OS. The
new sampling policy asked for 200 samples per detrapping time, which at
`nu_d = 3e5` means a sampling interval of two pusher steps across a full 150 us
run: about 66,000 samples of 400 particles, or **1.3 GB** of trajectory in
memory, for a measurement whose answer is complete after a few microseconds.

There was no bug in the physics and no error message worth the name -- the
process simply vanished, which is the least informative failure mode there is.

**Fix.** Two changes. The run length is now sized to the quantity being
measured: `t_run = min(150 us, 40 eps / nu_d)`, i.e. forty predicted detrapping
times, which is ample for a median and is 6 us rather than 150 us at the top of
the scan. And `_collisional_run` now enforces a hard cap of 8000 stored samples
regardless of what the sampling policy asks for, so a future policy change
cannot silently reintroduce the same failure.

The general shape: a sampling rule expressed as "resolve *this* timescale" and a
run length expressed as "cover *that* timescale" multiply, and nothing in either
rule knows about the product. Bounds belong on the product.
