# Numerics

How the orbits are integrated, how the resolution was chosen, and what the
residual errors are.

## 1. Why the Boris pusher

The original plan was `scipy.integrate.solve_ivp` with RK45. That was replaced.

RK45 is an accurate but **non-symplectic** scheme: its error in a conserved
quantity accumulates secularly. A production run is $1.3\times10^5$ timesteps,
which at 40 steps per gyro-period is $3.3\times10^3$ gyro-orbits, and over that
span an RK45 ion silently gains or loses energy. Since "confinement time" is measured by watching a particle drift to the
wall, an integrator that changes the particle's energy changes its drift speed
and therefore manufactures the answer.

The Boris algorithm splits each step into a half electric kick, an exact rotation
about $\mathbf{B}$, and a second half kick. With $\mathbf{E} = 0$ the kicks vanish
and the step is a pure rotation, so $|\mathbf{v}|$ — and hence the kinetic
energy — is preserved to round-off at any step size. It is not formally
symplectic, but it is volume-preserving in phase space, which gives the same
practical consequence: the energy error is **bounded**, not growing.

Measured over 4000 gyro-orbits in the tokamak field:

| integrator | $\max\lvert\Delta E/E\rvert$ | behaviour |
|---|---|---|
| Boris, $\Delta t = T_c/40$ | $< 10^{-13}$ | bounded, non-growing |
| RK45, `rtol=1e-6` | orders of magnitude larger | grows without bound |

The Boris bound is $10^{-13}$, not $10^{-14}$: single short runs reach
$\sim10^{-14}$, but the worst case across every production scan in `results/` is
$8.97\times10^{-14}$, which accumulates over the longer integrations. Quoting
$10^{-14}$ would be quoting the best case.

`tests/test_pusher.py::TestBorisVersusRK45` asserts a factor of 100 separation,
and separately checks that the two agree to $5\times10^{-3}\rho$ over 20 orbits
where both are accurate.

RK45 is retained in `pusher.rk45_push` solely as this control.

## 2. Order of accuracy

Measured against the exact circular orbit in a uniform field:

| steps per gyro-period | relative error in gyroradius | observed order |
|---|---|---|
| 10 | 4.818e-2 | — |
| 20 | 1.226e-2 | 1.974 |
| 40 | 3.079e-3 | 1.993 |
| 80 | 7.707e-4 | 1.998 |
| 160 | 1.927e-4 | 2.000 |
| 320 | 4.819e-5 | 2.000 |

Second order, as required. The error has an exact closed form. Boris rotates by
an angle $\theta$ with $\tan(\theta/2) = \omega\Delta t/2$ rather than by
$\omega\Delta t$, and the resulting discrete orbit radius is

$$ \rho_{\text{discrete}} = \rho\sqrt{1 + \left(\tfrac{\omega\Delta t}{2}\right)^2} $$

so the relative error is $(\omega\Delta t)^2/8$ at leading order. Against the
measured values:

| steps per gyro-period | measured | exact $\sqrt{1+(\omega\Delta t/2)^2}-1$ | leading $(\omega\Delta t)^2/8$ |
|---|---|---|---|
| 40 | 3.079483e-3 | 3.07951e-3 | 3.0843e-3 |
| 160 | 1.927456e-4 | 1.92747e-4 | 1.92766e-4 |
| 320 | 4.818899e-5 | 4.81903e-5 | 4.81914e-5 |

The exact form matches to five significant figures — the discretisation error is
fully accounted for, not merely the right order of magnitude.

*(An earlier version of this document gave the coefficient as
$(\omega\Delta t)^2/12$, which is 50% low, and described the resulting mismatch
as "the right size". That was a mis-derivation rationalised rather than checked;
see `DOC_SELF_REVIEW.md` finding 15.)*

## 3. Energy conservation is *not* evidence of convergence

This deserves its own heading because it is the easiest mistake to make here.

Boris conserves $|\mathbf{v}|$ because the magnetic substep is a rotation — and a
rotation *by the wrong angle* is still a rotation. At $\Delta t = T_c/2$ the orbit
is completely wrong and the energy error is still $10^{-14}$.

So a flat energy trace proves nothing about resolution. The resolution claim rests
entirely on the convergence study in §2, and
`test_energy_conserved_for_huge_timestep` exists to make the point unmissable.

## 4. Choice of timestep

Production runs use **40 steps per gyro-period**, with the gyro-period evaluated
at the *strongest* field a particle can reach — the inboard plasma edge,
$B_{\max} = B_0R_0/(R_0-a)\sqrt{1 + (B_{\theta}(a)/B_0)^2} = 2.889$ T — rather
than the on-axis value. So "40 steps per gyro-period" is a guarantee everywhere in
the domain, not an average. This gives $\Delta t = 1.135$ ns.

At that resolution:

- gyroradius error $3.1\times10^{-3}$
- energy error $< 10^{-13}$
- banana width converged to 0.1% (see §5)

A 150 µs run is $1.32\times10^5$ steps, i.e. $3.3\times10^3$ gyro-orbits.
Particles are vectorised over NumPy and the time loop is in Python, so cost
scales with the number of steps and only weakly with the number of particles.

## 4b. Collision timestep

The pitch-angle operator is applied every `collide_every` pusher steps with an
effective $\nu_d\Delta t_{\rm coll} = \nu_d \cdot$ `collide_every` $\cdot \Delta t$.
Collisions act far more slowly than the gyration, so applying the operator every
step would be wasted work; what matters is only that $\nu_d\Delta t_{\rm coll}$
stays small, since the scheme is first-order in it.

`collide_every` is therefore **chosen per run**, as the largest stride keeping
$\nu_d\Delta t_{\rm coll} \leq 0.02$, capped at 500:

| $\nu_d$ (s$^{-1}$) | `collide_every` | $\nu_d\Delta t_{\rm coll}$ |
|---|---|---|
| $3\times10^3$ | 500 | 0.0017 |
| $3\times10^4$ | 500 | 0.0170 |
| $10^5$ | 176 | 0.0200 |
| $10^6$ | 17 | 0.0193 |

A fixed stride cannot work across a collisionality scan: 200 steps at
$\nu_d = 10^6$ gives $\nu_d\Delta t_{\rm coll} = 0.23$, well outside the
operator's validity — `scatter_pitch` raises rather than silently returning
a wrong answer. Measured accuracy at 0.02 is 0.5% on the $l=1$ Legendre rate and
8% on $l=2$; at 0.01 both are within 0.2%.

### 4c. Why a *fixed* stride is not enough for the full operator

Choosing `collide_every` once per run works when $\nu_d$ is a dial. It does not
work for `collisions_full.py`, where the frequencies come from a plasma and go
as $v^{-3}$: **the same $\Delta t_{\rm coll}$ that is small for a fast ion is
not small for that ion once it has slowed down.**

The §9.3 slowing-down run is the worst case. It starts at
$\nu\Delta t_{\rm coll} \approx 10^{-3}$ and, by the time the population has
thermalised, the low-speed tail reaches $\nu\Delta t_{\rm coll} = 2.0$. The
first version of that experiment ran exactly like that and reported a final
mean energy built on it. Measured, on a relaxation from a cold beam:

| raw $\nu\Delta t$ | $\langle E\rangle/T_b$, no sub-cycling | with sub-cycling to 0.005 |
|---|---|---|
| 0.02 | 1.526 | 1.499 |
| 0.10 | 1.644 | 1.493 |
| 0.50 | $7.4\times10^{3}$ | 1.485 |
| 2.00 | $3.8\times10^{6}$ | 1.522 |

The failure is not graceful. Past $\nu\Delta t \sim 0.5$ the diffusive term
$\sqrt{B\Delta t}$ exceeds the speed itself, the population runs away, and the
reported temperature is nonsense by six orders of magnitude — while every other
diagnostic in the run (energy error, divergence, orbit topology) stays perfect,
because none of them touch the collision operator.

The fix is **per-particle sub-cycling**: each particle's collision step is split
into $\lceil \nu_\parallel \Delta t_{\rm coll} / (\nu\Delta t)_{\max}\rceil$
sub-steps, so the accuracy condition is enforced rather than assumed.
`MaxwellianCollisions` does this for both halves of the operator and reports
three numbers the caller must check:

- `max_nu_dt_seen` — the largest raw $\nu\Delta t$ presented, which says whether
  `collide_every` was sane
- `max_substeps_used` — how deep the sub-cycling actually went
- `substep_cap_hits` — particle-steps that wanted more sub-steps than the cap
  allowed. **Non-zero means part of the run was integrated at a coarser
  collisionality than requested**, and it is counted rather than absorbed
  silently, so it can be reported.

This is the same lesson as finding 22, arriving for the third time: the step
that validates a *bulk* rate is not the step that a different quantity needs.
The difference is that the requirement is now enforced by the code instead of
being a note in a document.

## 5. The half-step staggering correction

Boris is a leapfrog: $x_n = x_{n-1} + v_n\Delta t$ makes the stored $v_n$ the
velocity over $[t_{n-1}, t_n]$, centred at $t_{n-1/2}$, while $x_n$ sits at $t_n$.
Any diagnostic that combines position and velocity inherits that half-step
inconsistency.

The guiding-centre reconstruction does exactly that. Tested in a uniform field,
where the exact guiding centre is a fixed point, the reconstructed one traced a
circle of radius exactly $\omega\Delta t\,\rho$ — first order in $\Delta t$, and
15% of a gyroradius at production resolution.

Fixed by rotating the velocity forward half a step about $\mathbf{B}$ with the
same rotation the pusher uses. Residual wobble, in units of $\rho$:

| $\Delta t$ | uncorrected | corrected |
|---|---|---|
| $T_c/50$ | 0.1257 | 3.95e-3 |
| $T_c/100$ | 0.0628 | 9.87e-4 |
| $T_c/200$ | 0.0314 | 2.47e-4 |
| $T_c/400$ | 0.0157 | 6.17e-5 |

Uncorrected: exactly first order. Corrected: exactly second order (falls 4× per
halving). Effect on the measured banana width: it went from 3.0% $\Delta t$-dependent
to 0.1%, removing a 1.5% systematic at production resolution. Full detail in
`DOC_SELF_REVIEW.md` finding 5.

## 6. Residual error budget at production settings

| source | size | numerical or physical |
|---|---|---|
| energy non-conservation | $<10^{-13}$ | numerical, negligible |
| gyroradius discretisation | $3.1\times10^{-3}$ | numerical |
| guiding-centre staggering | $6\times10^{-4}\,\rho$ | numerical, corrected |
| guiding-centre 1st-order truncation | $O(\rho/a) = 3.4\times10^{-2}$ | approximation of the *diagnostic*, not the orbit |
| $\mu$ non-conservation, $\xi = 0.95$ at 10 keV | $\approx 4.1\,(\rho/a) \approx 14\%$ | **physical** |
| $\mu$ non-conservation, worst case in production | **123%** (ensemble scan at 60 kA) | **physical** |
| loss-boundary convention | $\rho/a = 3.4\%$ | a choice, see finding 10 |
| ensemble sampling (N=300) | $\pm 2.9\%$ on a fraction | statistical |
| collision operator, $\nu_d\Delta t = 0.02$ | 0.5% ($l=1$), 8% ($l=2$) on decay rates | numerical, first order |

The dominant uncertainties are physical or statistical, not numerical. That is
the intended state of affairs.

The 14% figure is specific to $\xi = 0.95$ at 10 keV. Across the production
scans the recorded `max_mu_error` reaches 0.55–0.58 in the trapped-fraction
ensembles and **1.226** in the ensemble scan at 60 kA, where orbits are wide
enough that the adiabatic expansion has broken down entirely. Nothing in this
project relies on $\mu$ being conserved — the orbits are full-orbit — but any
guiding-centre *interpretation* of those runs is invalid.

## 7. Reproducibility

Every number in `RESULTS.md` comes from `scripts/run_experiments.py`, which uses
fixed seeds and a fixed timestep policy and writes raw JSON into `results/`.
Experiments **G**, **H** and **I** were added specifically because the
orbit-topology table, the $E\times B$ figures and the "no poloidal field"
statement had originally been produced by ad-hoc exploration and could not be
regenerated; see `DOC_SELF_REVIEW.md` finding 16.
`scripts/make_figures.py` reads only those JSON files plus short re-runs, so the
figures cannot disagree with the tables.

```bash
python scripts/run_experiments.py          # ~75 min, full
python scripts/run_experiments.py --quick  # ~5 min, coarse
python scripts/make_figures.py
```

The same random seed is used at every point of a scan, so a scan measures the
response to the field alone rather than sampling noise between points.
