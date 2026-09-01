# Results

Every number here comes from `scripts/run_experiments.py`, which writes raw JSON
into `results/`. Regenerate with:

```bash
python scripts/run_experiments.py      # ~2 h (A-P)
python scripts/run_experiments.py --only O,P    # one section
python scripts/make_figures.py
```

Read `docs/DOC_STATUS.md` before quoting any of this. The short version: §1–§7
and §10 run on a divergence-free toy field with tokamak-like topology, **not** an
MHD equilibrium. §8 adds a real Grad–Shafranov equilibrium, but the main scans
were not re-run on it. §7's collisions are pitch-angle only; §9's have a
temperature, and the two were not cross-checked against each other. Several
agreements below are weaker than they look, and where that is true it is said
here rather than left for the reader to find.

Machine: $R_0 = 1.0$ m, $a = 0.3$ m, $B_0 = 2.0$ T, parabolic current profile,
$\epsilon = 0.3$. Deuterium at 10 keV unless stated.

Two labels in that line need care. `plasma_current = 450 kA` is a **profile
parameter**, not the current the field actually encloses — Ampère loops around
the field give 471.7 kA at $r = a$, 4.8% higher. Likewise the conventional
labels $q_0 = 1.0$, $q(a) = 2.0$ are the cylindrical approximation; the true
field-line safety factors are $q_0 = 1.000$ and $q(a) = 2.097$. Both discrepancies
are the same factor $R_0/\sqrt{R_0^2 - r^2}$, introduced by the $R_0/R$ term that
makes the field solenoidal. Scan axes are labelled with the nominal values for
continuity; see `docs/DOC_SELF_REVIEW.md` finding 13.

---

## 1. Validation against analytic theory

### 1.1 Grad-B and curvature drift

In a pure toroidal field ($I_p = 0$) the vertical drift should be
$v_D = \frac{m}{qBR}\left(v_\parallel^2 + \tfrac12 v_\perp^2\right)$.
Measured by fitting a straight line to the guiding-centre height over 30
gyro-periods:

| $E$ (keV) | $\xi$ | measured (m/s) | predicted (m/s) | error |
|---|---|---|---|---|
| 10 | +0.50 | 6228.30 | 6250.00 | 0.347% |
| 10 | +0.00 | 5000.27 | 5000.00 | 0.005% |
| 10 | +0.90 | 9013.46 | 9050.00 | 0.404% |
| 2 | +0.50 | 1248.39 | 1250.00 | 0.129% |
| 40 | +0.50 | 24825.41 | 25000.00 | 0.698% |

Agreement 0.005–0.698%. At fixed pitch the error grows with energy, as it must:
the comparison uses a first-order guiding centre whose truncation is $O(\rho/L)$,
and $\rho \propto \sqrt{E}$. Along the $\xi = +0.50$ rows that is 0.129% →
0.347% → 0.698% for 2 → 10 → 40 keV, close to $\sqrt{E}$.

**This is the strongest validation in the project** — an absolute, prefactor-and-all
prediction from theory, tested against the raw integrator with no fitting.

*An earlier version of this table was produced with a guiding-centre
reconstruction that omitted the half-step correction of finding 5, making
individual rows wrong by up to 65× and inverting the energy trend the text
claimed. See finding 14.*

### 1.2 $E \times B$ drift

$\mathbf{v} = \mathbf{E}\times\mathbf{B}/B^2$ is independent of mass and charge.
With $E_y = 10^4$ V/m and $B_z = 2$ T the prediction is 5000.000 m/s:

| species | measured (m/s) | error |
|---|---|---|
| H | 4997.741 | 0.0452% |
| D | 4998.405 | 0.0319% |
| T | 4998.698 | 0.0260% |

Species-to-species spread: $8.0\times10^{-5}$ relative. This exercises the electric
half-kicks of the Boris step, which are otherwise dead code here
($\mathbf{E} = 0$ in all physics runs).

### 1.3 Confinement requires a poloidal field

120 deuterons, isotropic pitch, launched at $r = 0.10$ m, integrated 150 µs:

| $I_p$ | fraction lost |
|---|---|
| 0 | **100%** |
| 450 kA | **0%** |

Every particle is lost without a poloidal field; none are with one. This is the
whole point of the geometry and it is reproduced without being put in.

See `figures/no_poloidal_field.png`.

---

## 2. Orbit topology

Deuterons launched at $r = 0.15$ m on the outboard midplane, 10 keV, integrated
60 µs. Predicted trapped/passing boundary $\xi_c = 0.511$.

| $\xi$ | class | bounces | $r_{\min}$ (m) | $r_{\max}$ (m) | width (m) |
|---|---|---|---|---|---|
| +0.95 | passing | 0 | 0.1308 | 0.1536 | 0.0228 |
| +0.70 | passing | 0 | 0.1330 | 0.1583 | 0.0253 |
| +0.55 | passing | 0 | 0.1278 | 0.1597 | 0.0319 |
| +0.45 | trapped | 3 | 0.0710 | 0.1604 | 0.0894 |
| +0.20 | trapped | 4 | 0.1305 | 0.1614 | 0.0309 |
| +0.00 | trapped | 5 | 0.1616 | 0.1661 | 0.0045 |
| -0.45 | trapped | 3 | 0.1604 | 0.2340 | 0.0736 |
| -0.95 | passing | 0 | 0.1536 | 0.1778 | 0.0242 |

Three things worth reading off this table.

**The boundary is where theory says.** The class flips between $\xi = 0.55$
(passing) and $\xi = 0.45$ (trapped), bracketing the predicted 0.511.

**Width peaks at the boundary, not at $\xi = 0$.** The widest orbit is
$\xi = 0.45$, just inside the boundary. Deeply trapped particles ($\xi = 0$) have
the *narrowest* orbits of all, 4.5 mm, because $w_b \propto v_\parallel$ and
their parallel velocity is nearly zero everywhere.

**Co- and counter-going orbits sit on opposite sides of their flux surface.**
$\xi = +0.45$ spans $r \in [0.071, 0.160]$ — entirely inboard of the launch
surface. $\xi = -0.45$ spans $[0.160, 0.234]$ — entirely outboard. Same $|\xi|$,
same energy, orbits displaced in opposite directions. This turns out to dominate
the confinement result in §4.

See `figures/orbits_poloidal.png`.

---

## 3. Trapped fraction — a partial agreement, and why

600 deuterons per point, pitch sampled uniformly in $\xi$ (isotropic), classified
after 80 µs. Compared against $f_t = \sqrt{2\epsilon/(1+\epsilon)}$.

| $r$ (m) | $\epsilon$ | measured | analytic | ratio | deviation | ions lost |
|---|---|---|---|---|---|---|
| 0.05 | 0.050 | 0.1917 ± 0.0161 | 0.3086 | 0.621 | −37.9% (7.3σ) | 0 |
| 0.10 | 0.100 | 0.3933 ± 0.0199 | 0.4264 | 0.922 | −7.8% (1.7σ) | 0 |
| 0.15 | 0.150 | 0.4917 ± 0.0204 | 0.5108 | **0.963** | −3.7% (0.9σ) | 0 |
| 0.20 | 0.200 | 0.5238 ± 0.0214 | 0.5774 | 0.907 | −9.3% (2.5σ) | 54 |
| 0.25 | 0.250 | 0.4974 ± 0.0255 | 0.6325 | 0.786 | −21.4% (5.3σ) | 216 |

**This does not simply agree, and the disagreement is the interesting part.**
Agreement is good only at $r = 0.15$ and degrades in both directions — for two
different reasons.

$\sqrt{2\epsilon/(1+\epsilon)}$ assumes a *thin* orbit: a particle that stays on
its flux surface, so that "the local $\epsilon$" is well defined. Tabulating the
banana width against the two lengths it has to be small compared with:

| $r$ (m) | $w_b$ (m) | $w_b / r$ | $w_b/(a-r)$ | ratio to theory |
|---|---|---|---|---|
| 0.05 | 0.0927 | **1.85** | 0.37 | 0.621 |
| 0.10 | 0.0684 | 0.68 | 0.34 | 0.922 |
| 0.15 | 0.0603 | 0.40 | 0.40 | **0.963** |
| 0.20 | 0.0587 | 0.29 | 0.59 | 0.907 |
| 0.25 | 0.0626 | 0.25 | **1.25** | 0.786 |

At $r = 0.05$ the banana is 1.85× *wider than the flux surface radius itself*. The
particle does not sample $\epsilon = 0.05$; it sweeps across most of the plasma,
and a formula built on a local $\epsilon$ has nothing to describe.

At $r = 0.25$ the banana is 1.25× the distance to the wall, so 36% of the sample
is lost before it can be classified — and the ones lost are preferentially the
wide-orbit, near-boundary particles that the trapped fraction is most sensitive
to. What survives is a biased sample.

Agreement is best exactly where both ratios are smallest and comparable. The
usable window is narrow.

**A caveat that has to be stated with this table** (`DOC_SELF_REVIEW.md` finding
4): in this field model both components carry the same $1/R$ factor, so $|B|$ is
*exactly* proportional to $1/R$ on a flux surface — verified at
$\mathrm{std}(|B|R)/\mathrm{mean} = 2\times10^{-16}$ for $I_p$ = 0, 450 and 900 kA
alike. The trapping condition therefore reduces algebraically to
$\sqrt{2\epsilon/(1+\epsilon)}$ *by construction*. So this table validates the
integrator, the orbit classifier and the pitch sampling — it does **not**
validate the field model, and the residual deviations above are finite-orbit-width
effects rather than a test of the mirror ratio.

See `figures/trapped_fraction.png`.

---

## 4. The main result: confinement versus poloidal field strength

**The question.** As the plasma current (and hence the poloidal field) is reduced,
orbits widen, and eventually they are wide enough to intersect the wall. Does
confinement fail sharply at a threshold, or gradually?

**The expectation, written before any code existed:** a sharp transition once the
banana width crosses the plasma boundary, not a smooth decline.

**That expectation is wrong for an ensemble, and right for a single particle.**
Both were measured.

### 4.1 Isotropic ensemble

300 deuterons, 10 keV, launched at $r = 0.15$ m with pitch uniform in $\xi$,
integrated 150 µs, at 24 currents from 60 to 900 kA. The same seed at every scan
point, so this measures the response to the field and not sampling noise.

| $I_p$ (kA) | $q(a)$ | lost | trapped | $\langle\tau\rangle$ (µs) |
|---|---|---|---|---|
| 60.0 | 15.00 | 60.7% | 0.0% | 80.5 |
| 96.5 | 9.32 | 48.3% | 2.3% | 94.3 |
| 133.0 | 6.76 | 24.7% | 11.3% | 121.4 |
| 169.6 | 5.31 | 20.7% | 14.0% | 126.0 |
| 206.1 | 4.37 | 14.7% | 30.0% | 133.0 |
| 242.6 | 3.71 | 11.0% | 34.3% | 137.2 |
| 279.1 | 3.22 | 5.0% | 40.3% | 144.3 |
| 315.7 | 2.85 | 0.0% | 45.0% | 150.0 |
| 450–900 | 2.12–1.00 | 0.0% | 46–49% | 150.0 |

The lost fraction falls from 90% to 10% of its maximum between $I_p = 78$ kA and
$I_p = 273$ kA — a width of 195 kA, or **157% of the midpoint current (124 kA)**.
That is a broad rolloff. It is not a threshold.

### 4.2 The same scan at fixed pitch

48 particles per point, all launched at one pitch with only the gyrophase
randomised, seed 999. Each pitch gets its own **two-stage** grid: a coarse pass
over 15–900 kA brackets the transition, then 22 further currents are placed
across it. This matters — the first version of this scan used one uniform
30-point grid and put a single sample inside the $\xi = -0.45$ transition, so the
width it reported was set by the grid spacing rather than measured
(`DOC_SELF_REVIEW.md` finding 17).

| launch pitch | threshold $I_p$ | 90%→10% range | width | % of midpoint | grid points inside |
|---|---|---|---|---|---|
| $\xi = +0.80$ (co-passing) | 30.6 kA | 30.08 → 31.17 kA | ≤ 1.09 kA | **≤ 3.5%** | **0** |
| $\xi = +0.45$ (co-going) | 37.0 kA | 36.52 → 37.54 kA | ≤ 1.02 kA | **≤ 2.7%** | **0** |
| $\xi = 0.00$ (deeply trapped) | 90.2 kA | 88.27 → 93.55 kA | 5.28 kA | **5.8%** | 4 |
| $\xi = -0.45$ (counter-going) | 253.5 kA | 243.5 → 270.1 kA | 26.6 kA | **10.5%** | 4 |
| isotropic ensemble, 300 ions | 124 kA | 78 → 273 kA | 195 kA | **157%** | 5 |

**The single-particle transition is a near-step.** At the two co-going pitches all
48 particles flip from lost to confined between adjacent grid points 1.27 kA
apart, so those two rows are **upper bounds, not measurements** — the width is
below 1.1 kA and this scan cannot say how far below. The deeply-trapped and
counter-going rows are resolved with four interior points each and give genuine
widths of 5.8% and 10.5%.

**And each pitch crosses at a completely different current.** The thresholds span
30.6 kA to 253.5 kA (50%-loss crossings), a factor of **8.3**. A counter-going ion needs seven times
more plasma current to stay confined than a co-going one of identical energy and
$|\xi|$.

That is the whole mechanism. An ensemble contains every pitch at once, so its
curve is the superposition of step functions whose locations are spread over
nearly an order of magnitude in current. The superposition is smooth even though
every component of it is sharp. Dividing the ensemble width by the sharpest
*resolved* single-pitch width gives **27×**; against the counter-going row it is
15×; against the unresolved co-going rows it is larger and unknown.

**Why the thresholds sit where they do** is §2's asymmetry: co-going orbits are
displaced *inboard*, away from the wall, counter-going orbits of identical
$|\xi|$ and energy *outboard*, into it. Loss in this machine is overwhelmingly a
counter-going phenomenon — but it is not true that co-going ions are never lost.
They are, at 30–37 kA. An earlier version of this document said "co-going ones do
not fail anywhere in the range scanned", which was correct but only because the
scan started at 60 kA; the README repeated it without the qualifier and was
simply wrong. See finding 18.

So the original claim was not so much wrong as scale-confused: a true statement
about one particle applied to a population. Reporting only the ensemble curve
would have looked like a failed prediction; reporting only the fixed-pitch curve
would have looked like a confirmed one. Both are needed, and the gap between them
is the result.

See `figures/confinement_scan.png`.

### 4.3 Censoring — a limitation, not a result

Above ~320 kA no particle is lost within the 150 µs integration window, so the
mean confinement time saturates at $t_{\max}$ by construction. Every point on the
flat part of that curve is a **lower bound**, not a measurement. Longer runs
would push the flat region up; they would not change §4.1 or §4.2, which depend
on the lost *fraction* rather than the time.

---

## 5. Banana width scaling

Theory: $w_b \simeq 2\rho_\theta\sqrt{\epsilon}$, so $w_b \propto B_\theta^{-1}$
and $w_b \propto E^{+1/2}$. Measured at $\xi = 0.30$, $r = 0.15$ m, accepting only
cleanly trapped orbits.

| $I_p$ (kA) | $w_b$ measured (m) | $w_b$ analytic (m) |
|---|---|---|
| 250 | 0.08550 | 0.10851 |
| 300 | 0.07388 | 0.09043 |
| 350 | 0.06457 | 0.07751 |
| 450 | 0.05125 | 0.06029 |
| 600 | 0.03898 | 0.04521 |
| 800 | 0.02946 | 0.03391 |
| 1000 | 0.02356 | 0.02713 |
| 1300 | 0.01796 | 0.02087 |

| $E$ (keV) | $w_b$ measured (m) | $w_b$ analytic (m) |
|---|---|---|
| 2.5 | 0.02619 | 0.03014 |
| 5.0 | 0.03673 | 0.04263 |
| 10.0 | 0.05125 | 0.06029 |
| 20.0 | 0.07060 | 0.08526 |
| 40.0 | 0.09417 | 0.12057 |

Fitted exponents:

| quantity | measured | ideal | deviation |
|---|---|---|---|
| $\partial \ln w_b / \partial \ln B_\theta$ | **−0.950** | −1 | 5.0% |
| $\partial \ln w_b / \partial \ln E$ | **+0.464** | +0.5 | 7.3% |
| $w_b^{\text{meas}} / w_b^{\text{analytic}}$ | 0.843 | O(1) | — |

Both exponents come out slightly *soft*. That is the expected direction for
finite-orbit-width corrections: the derivation assumes the orbit is a thin
excursion about a flux surface, and here $w_b/r$ ranges from 0.12 to 0.57 across
the scan, so the assumption is being stretched at the wide end.

**The prefactor is not a result.** Textbook expressions for $w_b$ differ by
$O(1)$ depending on whether $v$, $v_\parallel$, or the midplane $v_\parallel$ is
used, and on where $\sqrt{\epsilon}$ is evaluated. The measured ratio 0.843 is
reported as an observation, not as agreement with anything. Only the exponents
are claims.

**This scan had to be redone.** The first version measured at $\xi = 0.45$, which
sits only 12% inside the trapped/passing boundary; as $I_p$ dropped, the growing
orbit width pushed the particle *across* the boundary, and at 250 kA it registered
zero bounces and was not trapped at all. The measured "widths" were non-monotonic
in both $I_p$ and $E$. See `DOC_SELF_REVIEW.md` finding 11.

See `figures/banana_width.png`.

---

## 6. Numerical validation

### 6.1 The field is a magnetic field

$|\nabla\cdot\mathbf{B}| / (|B|/a) \sim 10^{-9}$ at 300 random interior points —
the finite-difference floor. Holds for plasma currents 0 to 1.2 MA and current
peaking $\nu$ from 0 to 3.

The first version of this model had $\nabla\cdot\mathbf{B}$ at **4.83% max,
2.76% median** of $|B|/a$. See `DOC_SELF_REVIEW.md` findings 2 and 14; this is
the single most consequential fix in the project.

### 6.2 Order of accuracy

| steps per gyro-period | gyroradius error | observed order |
|---|---|---|
| 10 | 4.818e-2 | — |
| 20 | 1.226e-2 | 1.974 |
| 40 | 3.079e-3 | 1.993 |
| 80 | 7.707e-4 | 1.998 |
| 160 | 1.927e-4 | 2.000 |
| 320 | 4.819e-5 | 2.000 |

Second order, converging to exactly 2.000.

### 6.3 Boris versus RK45

Over 4000 gyro-orbits in the tokamak field, Boris holds $|\Delta E/E| < 10^{-13}$
with no secular growth; RK45 at `rtol=1e-6` drifts by orders of magnitude more
and keeps growing. The worst case across every production scan in `results/` is
$8.97\times10^{-14}$; short runs reach $10^{-14}$, but quoting that would be
quoting the best case. See `figures/energy_conservation.png`.

**But energy conservation proves nothing about resolution.** Boris preserves
$|v|$ because its magnetic substep is a rotation — and a rotation by the wrong
angle is still a rotation. At $\Delta t = T_c/2$ the orbit is badly wrong and the
energy error is still $10^{-14}$. The resolution claim rests entirely on §6.2.

### 6.4 The magnetic moment is not conserved, and that is physical

$|\delta\mu/\mu|$ reaches 14% for $\xi = 0.95$ at 10 keV. Two measurements show
this is not integration error.

Refining the timestep 16× changes nothing:

| steps per gyro-period | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|
| $\delta\mu/\mu$ | 0.13931 | 0.13908 | 0.13900 | 0.13896 | 0.13897 |

Varying the gyroradius changes it linearly — $\delta\mu/\mu$ divided by $\rho/a$,
across a 32× range of energy:

| 4.02 | 4.03 | 4.04 | 4.06 | 4.08 | 4.12 | 4.17 |
|---|---|---|---|---|---|---|

So $\delta\mu/\mu \approx 4.1\,(\rho/a)$: first order in the finite-Larmor-radius
parameter, independent of $\Delta t$. $\mu$ is an *adiabatic* invariant and at
$\rho/a = 0.034$ the expansion it rests on is simply not that accurate.

This is now an asserted property rather than a worry — a test requires that
refining $\Delta t$ does *not* change the answer, so a real pusher bug would make
it fail. See `figures/convergence.png`.

---

## 7. Collisions

Everything above is collisionless: trapped particles bounce forever, and nothing
in it describes transport. This section adds a Monte Carlo pitch-angle (Lorentz)
collision operator and asks what survives.

The operator scatters the pitch at fixed speed,
$\xi' = \xi(1-\nu_d\Delta t) \pm \sqrt{(1-\xi^2)\nu_d\Delta t}$, whose first two
moments are exactly the drift and diffusion coefficients of the Lorentz
operator. Energy scattering, drag, and momentum conservation between species are
**not** included, so there is still no bootstrap current here.

### 7.1 Validation: Legendre eigenmodes

Legendre polynomials are exact eigenfunctions of the Lorentz operator, with
$C[P_l] = -\tfrac12 l(l+1)\nu_d P_l$, so $\langle P_l(\xi)\rangle$ decays
exponentially at a rate with **no free parameters**. Fitted rate divided by the
exact rate, over the window where the signal exceeds ten times the Monte Carlo
noise floor:

| $\nu_d\Delta t$ | $l=1$ | $l=2$ | $l=3$ |
|---|---|---|---|
| 0.0200 | 1.0065 | 0.9716 | below noise |
| 0.0050 | 1.0028 | 0.9967 | 0.9949 |

Within 0.2% at $\nu_d\Delta t = 0.01$. The $l=2$ value degrades to 0.917 at
0.02, which is the expected first-order error of the scheme; production runs
keep $\nu_d\Delta t \leq 0.02$ and mostly well below.

This is the sharpest validation in the project — an exact eigenvalue spectrum,
tested mode by mode, against a scheme with nothing fitted.

### 7.2 The update provably cannot leave $[-1, 1]$

By Cauchy-Schwarz the update is bounded by $\sqrt{1-a+a^2}$ with
$a = \nu_d\Delta t$, which is below 1 for every $0<a<1$. Measured over
$2\times10^5$ samples per row:

| $a$ | overshoots | max $\lvert\xi'\rvert$ | bound $\sqrt{1-a+a^2}$ |
|---|---|---|---|
| 0.005 | 0.0000% | 0.99751 | 0.99751 |
| 0.020 | 0.0000% | 0.99015 | 0.99015 |
| 0.050 | 0.0000% | 0.97596 | 0.97596 |
| 0.100 | 0.0000% | 0.95394 | 0.95394 |

Zero overshoots, and the measured maximum equals the analytic bound to five
decimals — the bound is tight, not merely safe. The clip in `scatter_pitch` is a
round-off guard that never fires.

An earlier version of the module docstring asserted an overshoot artefact and
quoted a bias figure for it. Both were invented while describing the code rather
than testing it. See `DOC_SELF_REVIEW.md` finding 19.

### 7.3 Detrapping

A trapped particle need only be scattered through $\Delta\xi \sim \sqrt{\epsilon}$
to cross the trapped/passing boundary, so the standard estimate is
$\tau_{\rm detrap} = \epsilon/\nu_d$. 400 deuterons launched at $\xi=0.30$,
$r = 0.15$ m. The **median** first-detrapping time is used, because it is
unbiased under right-censoring while the mean of the observed times is not.

| $\nu_d$ (s$^{-1}$) | detrapped | median $\tau$ (s) | $\epsilon/\nu_d$ (s) | ratio | $\nu_{\rm eff}/\omega_b$ | samples per $\tau$ |
|---|---|---|---|---|---|---|
| 3e+03 | 93.0% | 3.64e-05 | 5e-05 | 0.73 | 0.10 | 146 |
| 1e+04 | 99.8% | 1.3e-05 | 1.5e-05 | 0.87 | 0.32 | 174 |
| 3e+04 | 100.0% | 4.64e-06 | 5e-06 | 0.93 | 0.95 | 186 |
| 1e+05 | 100.0% | 1.3e-06 | 1.5e-06 | 0.87 | 3.18 | 191 |
| 3e+05 | 100.0% | 4.74e-07 | 5e-07 | 0.95 | 9.55 | 209 |
| 1e+06 | 100.0% | 1.43e-07 | 1.5e-07 | 0.95 | 31.83 | 126 |

Fitted over all six points, every one of which is now resolved:

| quantity | measured | predicted |
|---|---|---|
| $\partial\ln\tau / \partial\ln\nu_d$ | **-0.962** | $-1$ |
| $\langle \tau\nu_d/\epsilon \rangle$ | **0.88** | $1$ |

Exponent within 4% of $-1$, magnitude within 12%, across 333× in collision
frequency and spanning the banana-plateau boundary at $\nu_{\rm eff} = \omega_b$.

**This did not come out right the first time, and the reason is worth stating.**
The first version of this measurement gave an exponent of $-0.779$ — a 22%
discrepancy that survived checks on censoring, on the competing risk of wall
loss, and on trajectory sampling, and was written up as an unexplained failure.

It was not physics. It was the **collision** timestep. The operator is validated
in §7.1 against the Legendre decay rates, which are a *bulk* property of the
distribution and converge quickly. A detrapping time is a **first-passage**
quantity, and first passage is far more sensitive to step size: at
$\nu_d\Delta t = 0.02$ each application moves the pitch by $\Delta\xi = 0.135$,
against a distance to the boundary of only 0.211. The particle was arriving in
roughly two discrete jumps rather than diffusing, and the resulting time is
quantised and biased high.

Refining at fixed physics:

| $\nu_d\Delta t$ | $\Delta\xi$ per step | steps to the boundary | fitted exponent |
|---|---|---|---|
| 0.02 | 0.135 | 2.4 | $-0.876$ |
| 0.005 | 0.0675 | 9.8 | $-0.959$ |
| 0.00125 | 0.0337 | 39.0 | $-0.920$ |

Production runs now use $\nu_d\Delta t = 0.005$ and size the sampling interval
to the detrapping time rather than to the run. The reproduction script is
`scripts/probe_detrap_convergence.py`. See `DOC_SELF_REVIEW.md` findings 22
and 23.

Measured bounce frequency $\omega_b = 2.09\times10^5$ rad/s against the
leading-order estimate $2.35\times10^5$ — 12% apart, the expected accuracy of
that formula.

**Competing risks were checked rather than assumed away.** A particle can fail
to register a detrapping time for two different reasons: it stayed trapped for
the whole run, or it hit the wall while still trapped. Those are censoring and a
competing risk respectively, and pooling them would make the median mean
something other than it appears to. Counted separately, at most 1 particle in
200 was lost while trapped at any collisionality, so the medians stand.

### 7.4 Collisions destroy the single-particle threshold

This is the direct extension of §4. Collisionlessly, a fixed pitch angle gives a
near-step loss transition while a population gives a broad one, because the
population contains many thresholds at once. Collisions let a **single** particle
wander across pitch angles within its own lifetime, so it should sample many
thresholds too.

It does, and the effect is large. Same ion ($\xi = -0.45$, $r = 0.15$ m, 150 µs),
scanning plasma current from 120 kA to 1.6 MA:

| $\nu_d$ (s$^{-1}$) | max loss | highest $I_p$ at which it is still lost | vs collisionless |
|---|---|---|---|
| collisionless | 100.0% | 266 kA | 1.0x |
| $10^{4}$ | 83.3% | 600 kA | 2.3x |
| $10^{5}$ | 85.4% | 1029 kA | 3.9x |
| $10^{6}$ | 100.0% | $\geq$1600 kA | 6.0x |

Collisionlessly this ion is perfectly confined above 266 kA. At
$\nu_d = 10^5$ s$^{-1}$ it is still being lost at 1029 kA — **3.9 times the
collisionless threshold** — and at $10^6$ s$^{-1}$ the scan runs out at 1.6 MA
with the ion still being lost, so that row is a lower bound and is drawn pale.

The mechanism is §4 read backwards. Loss is overwhelmingly a counter-going
phenomenon, because those orbits are displaced outward into the wall. A
collisionless ion keeps whatever pitch it was born with, so if that pitch is
safe it stays safe. A collisional one is walked across pitch angle until, at
some point in its life, it lands in the wide counter-going region — and then it
is lost, however large the plasma current. **The sharp threshold of §4.2 is not a
robust feature of the geometry; it is an artefact of forbidding the particle to
change pitch.**

**A caveat in the same breath.** The frequencies scanned span
$\nu_{\rm eff}/\omega_b$ from 0.1 to 32, from the banana regime to well past the
plateau. A real 10 keV deuterium plasma at $10^{19}$ m$^{-3}$ sits near the
*bottom* of that range. The upper rows are a parameter scan, not a description
of any machine.

See `figures/collision_validation.png` and `figures/collisional_smearing.png`.


---

## 8. A real equilibrium: Solov'ev

Everything above uses a field that is divergence-free and has tokamak topology
but is **not** a solution of the Grad–Shafranov equation. §3 flagged that as the
reason its trapped fraction agrees with the $\epsilon$ expansion suspiciously
well. This section closes that gap.

`tokamak_orbits/equilibrium.py` implements the Solov'ev solution — the analytic
case where $p'(\psi)$ and $FF'(\psi)$ are both constant, so

$$\Delta^*\psi \equiv R\,\partial_R\!\left(\frac{\partial_R\psi}{R}\right) + \partial_Z^2\psi = -\mu_0 R^2 p'(\psi) - FF'(\psi)$$

has the closed-form solution

$$\psi(R,Z) = c\left[\frac{(R^2 - R_0^2)^2}{4} + \frac{R_0^2 Z^2}{\kappa^2}\right].$$

### 8.1 It solves the equation it claims to — checked twice, for a reason

| quantity | value |
|---|---|
| **Force balance** $\|\mathbf{J}\times\mathbf{B} - \nabla p\| / (\|J\|\|B\|)$, max | $2.7\times10^{-10}$ |
| same, median | $1.2\times10^{-10}$ |
| Grad–Shafranov residual (see caveat below) | $\sim 10^{-7}$, at $h = 10^{-5}$ |
| $\nabla\cdot\mathbf{B}$ relative to $\|B\|/a$ | $3.3\times10^{-10}$ |
| $q_0$ (exact field-line integration) | 1.000 |
| $q(a)$ | 1.297 |

**The Grad–Shafranov residual proves much less than it appears to, and the first
version of this section said the opposite.** $\psi$ is quartic in $R$ and
quadratic in $Z$, so both central differences in `grad_shafranov_residual` are
*exact* and the residual is analytically zero for any $(c, k, R_0)$. What it
measures is floating-point cancellation, which **rises** as $h$ falls —
$7\times10^{-12}$ at $h = 10^{-3}$, $8.9\times10^{-6}$ at $h = 10^{-6}$. Quoting
it to three figures was quoting a property of an unstated parameter. A test now
asserts that $1/h^2$ scaling, so it cannot be mistaken for a discretisation
error again.

Worse, it never evaluates `b_field`. Mutating $F^2(\psi)$ so the toroidal field
no longer integrates the $FF'$ the source term claims — a genuine
equilibrium-breaking change — left **all 28 equilibrium tests passing**.

The force-balance row is the real check. It builds
$\mathbf{J} = \nabla\times\mathbf{B}/\mu_0$ by differencing `b_field` itself
and compares $\mathbf{J}\times\mathbf{B}$ against $p'(\psi)\nabla\psi$, so it
tests the field the particles actually see. On the correct field it returns
$2.7\times10^{-10}$; on the mutated one, $1.4\times10^{-1}$ — nine orders of
magnitude, while the Grad–Shafranov residual does not move at all. Both the
contrast and the mutation are now asserted by tests. See finding 26.

`test_equilibrium.py` also asserts the **converse**: the circular field of
§1–§7 does *not* solve Grad–Shafranov, and the violation grows from 0.4% at
$r = 0.10$ m to 5.7% at $r = 0.25$ m. That test exists so the distinction cannot
quietly disappear.

### 8.2 The Shafranov shift is real and large

The boundary is not a circle. On the midplane it runs from $R = 0.5568$ m to
$R = 1.3$ m, so its geometric centre is at $R = 0.928$ m while the magnetic axis
sits at $R_0 = 1.0$ m — the axis is displaced outboard by $0.072$ m, or $0.24a$.
That is the Shafranov shift, and it is what breaks the identity the circular
model relies on.

### 8.3 What that does to trapping

On the circular field, $|B| R$ is constant on a flux surface to
$2.2\times10^{-16}$ — machine precision, and at *every* radius. That identity is
exactly the assumption behind $\xi_{\rm crit} = \sqrt{2\epsilon/(1+\epsilon)}$,
which is why §3's agreement was partly circular. On the Solov'ev field the same
quantity varies, and by an amount that grows with radius:
$4.5\times10^{-4}$ at $r = 0.10$ m, $5.7\times10^{-3}$ at $r \approx 0.21$ m,
$8.7\times10^{-3}$ at $r = 0.25$ m, $1.6\times10^{-2}$ at the boundary. The
trapping boundary moves with it:

| $r$ (m) | $\xi_{\rm crit}$, circular | $\xi_{\rm crit}$, Solov'ev | deviation |
|---|---|---|---|
| 0.10 | 0.4264 | 0.4360 | +2.3% |
| 0.15 | 0.5108 | 0.5275 | +3.3% |
| 0.20 | 0.5774 | 0.6025 | +4.3% |
| 0.25 | 0.6325 | 0.6684 | +5.7% |

These are computed from the field's own $B_{\max}/B_{\min}$ on each surface, so
they carry no sampling error, and the deviation grows monotonically with radius
— the signature of an expansion error rather than a bug.

**The measured trapped fractions are much weaker evidence than that table.**
With $n = 100$ launches per point the binomial error is $\pm 0.05$ absolute, so
the measured ratios (0.87 to 1.11) do not separate the two fields at better than
about $2\sigma$. The circular row at $r = 0.25$ is worse still: 46 of 100
particles hit the wall, so its "0.761" is a censoring artefact and not a
measurement. Do not quote the measured column as the result of this section; the
$\xi_{\rm crit}$ column is the result.

The integrator needed no changes to run in this field — energy conserved to
$10^{-12}$, trapped classification constant along an orbit to better than 2% of
samples — which is the point of having the field expose a common surface
interface rather than special-casing geometry inside the diagnostics.

---

## 9. Energy scattering, drag, and slowing down

§7's collision operator scatters pitch angle at a fixed speed and a fixed
frequency. It is the right tool for a collisionality scan and the wrong tool for
anything involving energy: it cannot slow a particle down, cannot thermalise
one, and has no plasma behind it — $\nu_d$ is a dial, not a temperature.
`tokamak_orbits/collisions_full.py` replaces the dial with a background Maxwellian
at a given density and temperature, and adds a speed operator.

### 9.1 The drag is derived, not quoted

Writing the speed as an Itô process $dv = \mu(v)\,dt + \sqrt{B(v)}\,dW$ with
$B(v) = \nu_\parallel v^2$, and demanding that the Maxwellian
$p_M \propto v^2 e^{-mv^2/2T_b}$ carry **zero probability flux**, fixes

$$\mu(v) = \frac{1}{2}\frac{dB}{dv} + \frac{B}{2}\left(\frac{2}{v} - \frac{mv}{T_b}\right).$$

Nothing about a collision-frequency table enters that derivation. The reason to
do it this way is that it removes a whole class of convention error: there is no
way to get the drag "nearly right" and still relax to the wrong temperature.

**The result is that this drift equals the NRL speed drag identically:**

$$\mu(v) \;=\; -\left(\nu_s - \tfrac{1}{2}\nu_\perp\right)v$$

to $2.6\times10^{-13}$ relative, over $0.02 \le v/v_{\rm th} \le 20$, through the
sign change near $v_{\rm th}$, and for every mass ratio in
$\{$H, D, T, He$^4\} \times \{$H, D, T, He$^4\}$ — i.e. $0.25 \le m/m_b \le 4$.

Two caveats keep that honest. First, both sides are built from the same
Chandrasekhar function $\psi(x)$, so this validates the *combination*, not
$\psi$ itself; $\psi$ is checked separately against its small-$x$ series and
against a finite difference of its own derivative. Second, the naive comparison
is against $\nu_s$ alone: NRL's $\nu_s$ is the *momentum* drag, and a pure
deflection carries momentum away without slowing the particle. The two differ
by a factor approaching **two** at equal masses in the high-energy limit
$x\to\infty$ — it is not a general factor, and below $v = 0.80\,v_{\rm th}$ the
speed drag changes sign entirely while the momentum drag does not. That trap
cost time here and is now a named regression test.

The sampler is checked against the analytic drift by measuring
$\langle\Delta v\rangle/\Delta t$ over $2\times10^6$ samples. **This table is a
closure check, not a validation of the drag**: `scatter_speed` computes
`v + mu*dt + sign*sqrt(B*dt)` with a symmetric sign, so the sample mean *is*
`mu` plus zero-mean noise and the ratio is 1 by construction. The only defects it
can catch are a biased sign draw or a misfiring speed floor. It is included
because those are real failure modes, and labelled because the first version of
this section presented it as evidence for the drag itself (finding 26).

| $v/v_{\rm th}$ | measured (m s$^{-2}$) | analytic | ratio | $\sigma$ |
|---|---|---|---|---|
| 1.0 | $-3.214\times10^{6}$ | $-3.038\times10^{6}$ | 1.058 | 1.4 |
| 1.5 | $-4.713\times10^{6}$ | $-4.680\times10^{6}$ | 1.007 | 0.7 |
| 2.0 | $-3.457\times10^{6}$ | $-3.463\times10^{6}$ | 0.998 | 0.3 |
| 3.0 | $-1.574\times10^{6}$ | $-1.573\times10^{6}$ | 1.001 | 0.4 |
| 4.5 | $-6.787\times10^{5}$ | $-6.793\times10^{5}$ | 0.999 | 0.8 |
| 6.0 | $-3.779\times10^{5}$ | $-3.781\times10^{5}$ | 1.000 | 0.7 |

### 9.2 The equilibrium is a Maxwellian, and the bias converges

A mean energy of $3T_b/2$ is necessary but weak — many wrong distributions have
the right second moment. The test is on the **shape**: relax a cold beam
($0.4 v_{\rm th}$) and a hot beam ($3 v_{\rm th}$) for 25 collision times each
and compare against the analytic Maxwellian speed CDF with a Kolmogorov–Smirnov
statistic ($n = 60{,}000$, 95% critical value $0.00555$).

| $\nu\Delta t$ | cold start $\langle E\rangle/T_b$ | KS | hot start $\langle E\rangle/T_b$ | KS | verdict |
|---|---|---|---|---|---|
| 0.0200 | 1.5124 (+0.83%) | 0.01563 | 1.5288 (+1.92%) | 0.01831 | fail |
| 0.0100 | 1.5051 (+0.34%) | 0.00583 | 1.4997 (−0.02%) | 0.00660 | fail |
| 0.0050 | 1.5036 (+0.24%) | 0.00293 | 1.4912 (−0.59%) | 0.00373 | pass |
| 0.0025 | 1.4993 (−0.05%) | 0.00474 | 1.5030 (+0.20%) | 0.00385 | pass |

Both starts converge below the critical value at $\nu\Delta t \leq 0.005$,
from opposite directions and with **independent random seeds**. That last clause
is doing real work: the first version of this table ran both starts on the same
seed, under which the dynamics contracts and the two trajectories synchronise —
99.3% of the particles were bitwise identical and the KS statistics agreed to
all 17 digits. It read as independent confirmation and was one run reported
twice (finding 26). With independent seeds the two columns genuinely differ,
which is what makes their agreement on the *verdict* mean something.

The discrete operator's stationary distribution is not exactly Maxwellian — it
converges to one as $\Delta t \to 0$. Same lesson as finding 22: a first-order
operator needs its step checked against the quantity being measured, and
$\nu\Delta t \le 0.005$ is the requirement for distribution-shape work. That
is now the library default, replacing a 0.02 default that cited this section as
its authority while being the value this section marks as failing.

### 9.3 Inside the pusher

The last thing that can be wrong is the interfacing. A gyro-orbit does not
change speed statistics, so running the same operator inside the Boris pusher in
the tokamak field must reproduce the 0-D relaxation curve. 64 deuterons are
launched at 100 keV and followed through a full slowing-down time against a
10 keV background — $2.3\times10^{6}$ pusher steps.

| | final $\langle E\rangle$ |
|---|---|
| inside the pusher, tokamak field | $16.22 \pm 2.10$ keV |
| 0-D, same operator, same step | 15.56 keV |
| background $3T_b/2$ | 15.00 keV |

Agreement to 0.3σ, and both land on the background temperature. Three
qualifications:

**Pitch scattering is off for this comparison, deliberately.** With it on the
collisions also drive radial transport, particles reach the wall, and the
survivors are a badly biased sample — a real effect, but a different one.

11 of the 64 are still lost, and that *is* a speed-dependent selection: the
banana width goes as $m v_\perp/(qB_\theta)$, so a slowing ion has a narrowing
orbit and the ones that leave are the ones that were still fast. An earlier
version of this section argued the loss was independent of speed, which is
wrong — changing the speed is the operator's entire purpose. What can be said
is the *direction*: removing the fastest survivors biases the pusher mean
**down** relative to the 0-D curve, and the measured ratio is 1.04, i.e. above
1. The discrepancy is not in the direction this bias produces, so the bias is
not what the table is showing. The script's own comment claiming no particle
reached the wall — contradicted by its own `n_lost` in the same output — has
been corrected.

**The density is raised to $10^{22}$ m$^{-3}$** so $\tau_s$ fits in an affordable
number of gyro-orbits. The operator is linear in $n_b$, so this rescales the time
axis by exactly that factor and changes nothing else — but the *absolute* times
in this subsection are not physical fast-ion slowing-down times.

**Part of this run was integrated at a coarser collisionality than requested,
and the code says so.** The raw $\nu_\parallel\Delta t_{\rm coll}$ reaches
**19.6** somewhere in the run. That is *not* the bulk: the bulk population ends
at $\nu_\parallel\Delta t = 5.0\times10^{-4}$, four orders of magnitude below.
19.6 corresponds to a sub-eV particle in the low-speed tail — the frequencies go
as $v^{-2}$ down there, so a handful of random-walkers reach it even though the
population as a whole never comes close.

Sub-cycling brings the step back to $\leq 0.005$. 21 particle-steps out of
$7.4\times10^{5}$ (0.003%) wanted more than the 256-sub-step cap and were stepped
at the cap; the speed floor was never reached (0 hits). Both counts are returned
by the operator rather than inferred afterwards — an earlier version recomputed
the cap count from the speed *after* the step rather than the speed the
sub-stepping decision was made from, so it was not counting what it claimed to,
and it reported $\nu_\perp$ in a run where pitch scattering was switched off.
See findings 25 and 26.

![Energy operator validation](figures/thermalisation.png)

### 9.4 What is still missing

The background is a fixed Maxwellian at rest with no back-reaction, so momentum
is not conserved between species: this is a test-particle operator, not a
field-particle one, and there is still no bootstrap current.

---

## 10. Field ripple: the first non-axisymmetric field

Every field above is axisymmetric. That is a strong statement about orbits, not
just a modelling convenience: it makes the canonical toroidal momentum an exact
invariant, so no collisionless orbit can wander in radius indefinitely. Real
tokamaks have a finite number of toroidal field coils, and the loss mechanisms
that opens cannot be represented at all by anything in §1–§9.

### 10.1 Getting it divergence-free

The obvious implementation is to write $B_\phi = (B_0R_0/R)(1 + \delta\cos N\phi)$
and stop. **That field is not divergence-free** — the $\phi$-derivative has
nothing to cancel it. This is finding 2 again with an extra dimension to get
wrong, and the same fix applies: derive the field from a potential.

Ripple is a *vacuum* perturbation produced by coils outside the plasma, so it
carries no current and can be written $\mathbf{B}_r = \nabla\chi$, giving
$\nabla\cdot\mathbf{B}_r = \nabla^2\chi$ — divergence-free **iff** $\chi$ is
harmonic. The only $Z$-independent harmonic choice is

$$\chi = \frac{B_0R_0\delta_0}{N}\left(\frac{R}{R_0}\right)^{N}\sin N\phi
\quad\Longrightarrow\quad
\delta(R) = \delta_0\left(\frac{R}{R_0}\right)^{N},$$

which is also the profile real machines have: negligible on axis, concentrated
at the outboard edge. At $N = 16$ and $a/R_0 = 0.3$ it is $1.3^{16} = 66.5$
times larger at the outboard boundary than on the magnetic axis, and about
$2\times10^4$ times larger there than at the inboard edge.

$R^{+N}$ is not the *only* $Z$-independent harmonic solution — $R^{-N}$ solves
the same Euler equation — but it is the only one regular on the machine axis,
which is the actual reason to choose it. Because $\nabla\times\nabla\chi = 0$,
the same wrapper applies to the Solov'ev equilibrium without modification;
`b_field` always did, and the two ripple *criteria* now do as well, having
raised `AttributeError` on a Solov'ev base until review caught that only the
divergence test covered the combination.

| field | max $\|\nabla\cdot\mathbf{B}\|\,/\,(\|B\|/a)$ |
|---|---|
| axisymmetric | $\sim 10^{-9}$ |
| potential-derived ripple | $\sim 10^{-9}$ |
| naive $B_\phi$-only ripple | $2.65\times10^{-2}$ (median $3.4\times10^{-4}$) |

The naive version's leftover is not merely "large" — it is identified exactly as
the missing term $-(N\delta B_\phi/R)\sin N\phi\,/\,R$, to $10^{-7}$ relative.
That identification is what makes the test a diagnosis rather than an alarm.

**The ripple in $|B|$ is not the ripple in $B_\phi$.** The construction sets
$\delta$ on the toroidal field exactly; what an orbit feels is
$|B| = \sqrt{B_\phi^2 + B_{\rm pol}^2}$, whose modulation is smaller by
$1/(1 + (B_{\rm pol}/B_\phi)^2)$ — a 2.2% dilution at the outboard edge. Real
dilution, not an error, and asserted as such.

### 10.2 Ripple wells

Where the ripple modulation along a field line beats the $1/R$ variation, $|B|$
acquires local minima *between* coils. A particle with small enough
$v_\parallel$ is trapped in one, stops sampling the poloidal circuit that
averages its vertical drift to zero, and drifts straight out. The criterion is

$$\alpha_{\rm rt} = \frac{\epsilon\,|\sin\theta|}{N q(r)\,\delta(R)} < 1 .$$

The well region is confined to the outboard midplane, and at $\delta_{\rm edge}
= 1\%$, $N = 16$ it covers **16.3%** of the $r = 0.25$ m surface by area.

That is an area fraction, weighted by the surface element
$\propto R_0 + r\cos\theta$, and the weighting is not cosmetic: the wells sit
at $\theta \approx 0$, which is exactly where the element is largest, so a
plain average over poloidal angle gives 13.1% and understates it by 24%. The
first version of this section reported the angle average.

The dependence on coil number is **not monotonic**, which was a wrong prediction
here before it was a measurement. More coils sharpen the modulation along a
field line (favouring wells, the $N$ in the denominator), but at fixed *edge*
ripple more coils also concentrate the ripple further outboard, since
$\delta(R) = \delta_{\rm edge}(R/R_{\rm edge})^N$ falls faster everywhere inside
the boundary. The two compete and the well fraction peaks near $N = 16$–$20$:

| $N$ | 8 | 12 | 16 | 20 | 24 | 32 |
|---|---|---|---|---|---|---|
| well fraction at $r = 0.25$ m | 0.134 | 0.156 | **0.163** | 0.163 | 0.159 | 0.147 |

### 10.3 The loss it causes

40 ions launched at $\theta = 0$ on $r = 0.25$ m — where the wells are — with
$|\xi| \leq 0.22$ so they turn there. The **same** pitches and gyrophases are
used at every ripple amplitude, which makes this a paired comparison: the
question is not "what fraction is lost" but "which particles does ripple lose
that the axisymmetric field confines". Pairing removes the launch-condition
variance, and it is necessary rather than tidy — the first version of this scan
used 10 particles and produced 20%, 60%, 50%, 20%, 70%, which is pure noise at a
$\pm15\%$ binomial error.

Five of the 40 are lost in the **axisymmetric control**. Those are prompt
counter-going orbit losses — fat bananas at $r = 0.25$ m whose outboard
excursion crosses the wall — and they have nothing to do with ripple. They are
excluded from the denominator below rather than being allowed to inflate it.

| $\delta_{\rm edge}$ | newly lost, of the 35 the control confines | well fraction at $r=0.25$ m | ripple loss times (µs) |
|---|---|---|---|
| 0 (control) | — | 0 | — |
| 0.25% | 0 (0.0%) | 0.051 | — |
| 0.5% | 3 (8.6%) | 0.096 | 157, 247, 272 |
| 1% | 3 (8.6%) | 0.163 | 15, 17, 274 |
| 2% | **6 (17.1%)** | 0.242 | 15, 19, 45, 151, 192, 251 |

Four things have to be said about that table, three of which are limitations.

**It is not one-directional, and the first version of this section said it was.**
That claim rested on a "rescued" count measured against the $\delta = 0$
control, which is **zero by construction**: the control's five losses are prompt
counter-going orbits gone in ~12 µs, and no ripple can save those. Comparing
*adjacent* amplitudes instead:

| step | newly lost | newly confined |
|---|---|---|
| 0 → 0.25% | 0 | 0 |
| 0.25% → 0.5% | 3 | 0 |
| 0.5% → 1% | 3 | **3** |
| 1% → 2% | 4 | **1** |

Three particles lost at 0.5% are confined at 1% and again at 2%. Ripple moves
orbits both ways; the net loss grows, but the mechanism reshuffles. See
finding 26.

**The two loss channels overlap in time.** The prompt losses run 11.1–14.6 µs.
Two of the three ripple losses at $\delta = 1\%$ are at 15 and 17 µs — inside or
immediately adjacent to that window. They are separable by *which particle*
(the paired comparison), not by when it left, and an earlier claim that they
"stay separate" in time does not survive the loss-time list above.

**The prompt loss time is not unchanged by ripple.** Its median runs
12.42 → 12.04 → 11.78 → 11.54 → 11.13 µs across the scan: a 1.29 µs spread and
monotonically decreasing in $\delta$. Small, but a systematic trend, and it was
previously described as unchanged.

**Energy is conserved to $10^{-13}$ throughout.** This matters more here than
anywhere else in the document: a non-axisymmetric field is exactly where one
would suspect the integrator first, and a particle that gains energy will leave
the machine for reasons that have nothing to do with physics. It doesn't.

Finally, the ripple loss times are censored at $t_{\max} = 300$ µs and are
medians of 3–6 events. The 0.5% row's three losses are all in the last third of
the run, so its median is a lower bound. The *counts* are the result of this
table; the times are indicative.

### 10.4 It does not reach the bulk

The edge population above is the worst case by construction. The fair question
is what happens to an isotropic ensemble launched where the main scans launch
it — $r = 0.15$ m, 40 particles, 200 µs:

| $\delta_{\rm edge}$ | 0 | 1% | 2% |
|---|---|---|---|
| lost | 0/40 | 0/40 | 0/40 |

Nothing. At these amplitudes and this run length, ripple loss is an **edge
phenomenon** and the core population does not feel it. That is a null result and
is reported as one: it bounds the effect rather than demonstrating it, and 40
particles over 200 µs cannot exclude a slow ripple diffusion that would show up
over a confinement time.

![Field ripple](figures/ripple.png)

The Goldston–White–Boozer stochasticity parameter at $r = 0.25$ m for a 10 keV
deuteron falls from 0.23 at $\delta_{\rm edge} = 0.25\%$ to 0.03 at 2%, so banana
tips are predicted stochastic ($\sigma < 1$) at every amplitude in the scan —
including the one at which no particle was newly lost. The criterion therefore
does **not** discriminate here, and saying so is the honest reading; an earlier
version of this function returned values about 4× larger and, being
dimensionally inconsistent (units of inverse length, invisible because
$R_0 = 1$ m), happened to straddle 1 and appeared to discriminate. It is quoted
as a diagnostic with its inputs: the numerical prefactor varies between
references and nothing in the package branches on it.

---

## 11. Things that did not work, or are not settled

Recorded because a results file with no disappointments in it is not a results
file.

- **The banana-width scan had to be thrown away and redone** at a different pitch
  angle (§5). The first version produced non-monotonic widths and would have been
  reported as data.
- **The trapped-fraction comparison is much weaker than it looks** (§3), both
  because it is partly true by construction and because it only holds in a narrow
  window of minor radius.
- **The confinement-time curve is censored** above ~320 kA (§4.3) and is a lower
  bound there, not a measurement.
- **Bounce counting is unreliable for deeply trapped particles.** At $\xi \approx 0$
  the parallel velocity hovers near zero and the sign-change count becomes noise —
  25 "bounces" were recorded in 60 µs for a particle that physically executes far
  fewer. The trapped/passing *classification* is unaffected; the bounce *count* at
  small $|\xi|$ should not be used.
- **The $4.1(\rho/a)$ coefficient for $\mu$ error is measured at $\xi = 0.95$
  only** and is not claimed to be universal across pitch angle. The $\xi$
  dependence is understood in direction (small $v_\perp$ means $\mu$ is a small
  difference of large numbers) but was not mapped.
- **The detrapping exponent was wrong, and it was my discretisation.** The first
  measurement gave $-0.779$ against a predicted $-1$ and was written up as an
  unexplained failure. It was the collision timestep: a first-passage quantity
  needs a far finer step than the bulk Legendre rates the operator was validated
  against. Converged, it is $-0.962$ (§7.3). Recorded because the wrong version
  was published first and survived three rounds of the wrong checks.
- **The highest-collisionality smearing point is truncated.** At
  $\nu_d = 10^6$ s$^{-1}$ the ion is still being lost at the top of a 1.6 MA
  scan (§7.4), so "6x the collisionless threshold" is a lower bound. Extending
  the scan further was not attempted.
- **The §7 collisions are pitch-angle only** — that is now a property of §7
  rather than of the project, since §9 adds energy scattering and drag. What is
  still missing in both is momentum conservation between species: the background
  is a fixed Maxwellian at rest with no back-reaction, so there is no bootstrap
  current and nothing here is a complete neoclassical calculation.
- **The §7 scans were not re-run with the full operator.** Everything in §7 —
  the detrapping exponent, the collisional smearing of the threshold — uses the
  fixed-speed Lorentz operator. Repeating those scans with energy scattering on
  would change the answers by an amount nobody here has measured, and the cost
  was judged not worth it. Do not read §7 as "the collisional result" and §9 as
  a refinement of it; they are two operators, and only §9 has a temperature.
- **The Solov'ev trapped fractions are sampling-limited** (§8.3). With 100
  launches per point the binomial error is $\pm 0.05$ absolute, which does not
  separate the two field models. The deterministic $\xi_{\rm crit}$ comparison
  is the result of that section; the measured column is not.
- **The Solov'ev equilibrium is not the field used for the main results.**
  §1–§7 and §10 all run on the circular model. Re-running the headline
  current scan on the Solov'ev field is the obvious next step and was not done,
  so the threshold numbers in §4 carry an unquantified geometry error.
- **The ripple study measures loss, not diffusion.** §10 shows that ripple
  loses particles the axisymmetric field confines, and where the wells are. It
  does not measure a ripple transport coefficient, and the
  Goldston–White–Boozer stochasticity parameter is reported as a diagnostic with
  its inputs rather than as a threshold, because the numerical prefactor varies
  between references and no attempt was made to pin it down here.
- **The ripple amplitude is on the pessimistic side.** $\delta_{\rm edge} = 1$–2%
  is at or above the high end for a real machine; it was chosen so the effect is
  measurable in a $3\times10^{-4}$ s run rather than because it is typical.
- **No neoclassical diffusion coefficient was measured.** It was the obvious next
  step after §7 and was not attempted; the radial transport statistics over a
  150 µs run are too poor to fit a diffusion coefficient worth reporting.
