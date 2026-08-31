# Physics

What the code solves, what field it solves it in, and which analytic results are
used as references.

## 1. Equations of motion

Single charged particles are pushed under the Lorentz force with no electric
field and no collisions:

$$ m\frac{d\mathbf{v}}{dt} = q\,\mathbf{v}\times\mathbf{B}(\mathbf{x}), \qquad
   \frac{d\mathbf{x}}{dt} = \mathbf{v} $$

There is no back-reaction: the particles do not generate fields, do not see each
other, and do not modify the plasma current that produces the field they move in.
This is a **test-particle** model. It is the right model for asking "given this
field geometry, is a particle confined?", and the wrong model for anything
involving transport, turbulence, or equilibrium.

At 10 keV a deuteron has $v/c \approx 3.3\times10^{-3}$, so the relativistic mass
correction is $\gamma - 1 \approx 5\times10^{-6}$ — below every effect claimed
here. The equations are solved non-relativistically.

## 2. Geometry

Cylindrical coordinates $(R, \phi, Z)$ with the right-handed triad
$\hat{R}\times\hat{\phi} = \hat{Z}$. Flux coordinates measured from the magnetic
axis at $R = R_0$, $Z = 0$:

$$ r = \sqrt{(R-R_0)^2 + Z^2}, \qquad \theta = \arctan_2(Z,\, R - R_0) $$

The inverse aspect ratio is $\epsilon = a/R_0$, and $\epsilon_{\text{local}} = r/R_0$.

## 3. The field

### Toroidal component

Produced by external coils. Ampère's law around the torus gives the exact $1/R$
dependence:

$$ B_\phi = \frac{B_0 R_0}{R} $$

This $1/R$ is the origin of everything interesting here. It makes the field
stronger on the inboard side, which is what traps particles, and it has a
gradient, which is what makes them drift.

### Poloidal component

Produced by toroidal current flowing in the plasma. The current density is a
toy profile

$$ j_\phi(r) = j_0\left(1 - (r/a)^2\right)^{\nu}, \qquad r < a $$

which integrates to an enclosed current

$$ I(r) = \frac{\pi j_0 a^2}{\nu+1}\left[1 - \left(1-(r/a)^2\right)^{\nu+1}\right]
        = I_p\left[1 - \left(1-(r/a)^2\right)^{\nu+1}\right] $$

and, by Ampère's law in the straight-cylinder limit,
$B_{\theta,\text{ref}}(r) = \mu_0 I(r)/(2\pi r)$.

**The actual poloidal field carries an extra factor $R_0/R$:**

$$ |B_\theta|(r, R) = \frac{R_0}{R}\,\frac{\mu_0 I(r)}{2\pi r} $$

This is not decoration. Without it the total field has
$\nabla\cdot\mathbf{B} = B_R/R$, measured at 4.83% max / 2.76% median of
$|B|/a$ — see `DOC_SELF_REVIEW.md` finding 2. With it, the field is exactly

$$ \mathbf{B} = \nabla\psi\times\nabla\phi + F(\psi)\nabla\phi $$

for a flux function with $d\psi/dr = -R_0\,\mu_0 I(r)/(2\pi r)$, which is
solenoidal by construction. Verified numerically at $|\nabla\cdot\mathbf{B}|
/(|B|/a) \sim 10^{-9}$.

### What this field is not

Circular concentric flux surfaces with no Shafranov shift are **not** a solution
of the Grad-Shafranov equilibrium equation. A real equilibrium has its flux
surfaces shifted outward by an amount growing with plasma pressure. This model
is a divergence-free *toy field with tokamak-like topology*, not an equilibrium,
and results should be read as "what this geometry does", not "what a tokamak
does". See `DOC_STATUS.md`.

### Safety factor

$$ q(r) \simeq \frac{r B_0}{R_0\, B_{\theta,\text{ref}}(r)} $$

is the large-aspect-ratio form, used only for labelling scan points. The machine
is *tuned* to $q_0 = 1.0$, $q(a) = 2.0$ in this measure.

The **true field-line safety factor** is obtained by integrating
$d\phi/d\theta = rB_\phi/(RB_\theta)$ around a flux surface:

$$ q(r) = \frac{r B_0}{B_{\theta,\text{ref}}(r)\sqrt{R_0^2 - r^2}}
        = q_{\text{cyl}}(r)\,\frac{R_0}{\sqrt{R_0^2 - r^2}} $$

so $q(a) = 2.097$, not 2.000 — 4.8% higher. The same factor applies to the
enclosed current: Ampère loops around the actual field give 471.7 kA at $r = a$
against the nominal $I_p = 450$ kA, so `plasma_current` is a **profile
parameter**, not the physical total current. Both discrepancies come from the
$R_0/R$ factor above; see `DOC_SELF_REVIEW.md` finding 13. Use
`safety_factor_exact` and `enclosed_current_actual` whenever the number is a
physical claim rather than a scan label.

## 4. Why a poloidal field is needed at all

In the toroidal field alone, the field gradient $\nabla B$ and the field-line
curvature both point inward along $-\hat{R}$, and both produce drifts in the same
vertical direction:

$$ \mathbf{v}_D = \frac{m}{qBR}\left(v_\parallel^2 + \tfrac{1}{2}v_\perp^2\right)\hat{z} $$

The direction depends on the sign of the charge, so ions drift up and electrons
down (or vice versa) — and, crucially, the drift **does not average to zero**,
because nothing returns the particle to where it started. Every ion marches
vertically out of the machine at a few km/s regardless of how strong the field
is. This is measured in `results/drift_validation.json`, agreeing with the
formula above to 0.005–0.698%.

Adding a poloidal component twists the field lines into helices. A particle
following one now spends half its time above the midplane and half below, so the
vertical drift is alternately outward and inward and largely cancels. Confinement
in a tokamak is a consequence of field-line **topology**, not field strength.

## 5. Trapped and passing orbits

Because $B \propto 1/R$, a particle moving along a field line from the outboard
side to the inboard side sees a rising field. With $\mu = mv_\perp^2/2B$ and the
energy both conserved, $v_\parallel^2 = v^2 - 2\mu B/m$ falls as $B$ rises, and
if it reaches zero the particle mirrors back. Launching at the outboard midplane
(the weakest-field point) with pitch $\xi_0 = v_\parallel/v$, the particle is
trapped if

$$ \xi_0^2 < 1 - \frac{B_{\min}}{B_{\max}} $$

For an isotropic distribution the trapped fraction is therefore

$$ f_t = \sqrt{\frac{2\epsilon}{1+\epsilon}} $$

**Important caveat.** In *this* field model, both components share the $1/R$
factor, so $|B| = (R_0/R)\sqrt{B_0^2 + B_{\theta,\text{ref}}^2}$ is exactly
proportional to $1/R$ on a flux surface, and the formula above is therefore exact
*by construction* rather than approximate. Comparing measurement to it validates
the integrator, the classifier and the pitch sampling — not the field model. See
`DOC_SELF_REVIEW.md` finding 4.

Trapped orbits project onto the poloidal plane as crescents — "banana" orbits —
of full radial width

$$ w_b \simeq 2\rho_\theta\sqrt{\epsilon}, \qquad
   \rho_\theta = \frac{m v}{q B_\theta(r)} $$

Textbooks differ by $O(1)$ prefactors and on whether $v$ or $v_\parallel$ at the
midplane is meant, so **the prefactor is not treated as predictive here**. What is
tested is the scaling: $w_b \propto 1/B_\theta$ and $w_b \propto \sqrt{E}$. The
fitted prefactor is reported in `RESULTS.md` rather than asserted.

## 6. The co/counter asymmetry

Passing and trapped orbits are *not* centred on their flux surface. The drift
displaces the orbit radially, in a direction set by the sign of $v_\parallel$:
co-going particles sit inboard of their flux surface, counter-going ones
outboard. Measured for $|\xi| = 0.45$ launched at $r = 0.15$ m: the co-going
orbit spans $r \in [0.071, 0.160]$, the counter-going one $r \in [0.160, 0.234]$.

This is why counter-passing fast ions are the first to be lost in a real machine,
and it appears here without being put in.

## 7. Adiabatic invariance is approximate

$\mu$ is conserved only asymptotically as $\rho/L \to 0$. Measured here,
$|\delta\mu/\mu| \approx 4.1\,(\rho/a)$ — first order, and independent of the
timestep. At 10 keV, $\rho/a = 0.034$ and $\mu$ holds to about 14% for
near-passing particles. Guiding-centre theory is used in this project for
*diagnostics and reference formulas only*; the orbits themselves are full-orbit
integrations that make no adiabatic assumption. See `DOC_SELF_REVIEW.md`
finding 6.

## 8. Collisions

Sections 1-7 are collisionless. The optional collision model is the **Lorentz
(pitch-angle) operator**

$$ C[f] = \frac{\nu_d}{2}\,\frac{\partial}{\partial\xi}
   \left[(1-\xi^2)\frac{\partial f}{\partial\xi}\right] $$

implemented in Monte Carlo form (Boozer & Kuo-Petravic 1981) as

$$ \xi' = \xi\,(1 - \nu_d\Delta t) \pm \sqrt{(1-\xi^2)\,\nu_d\Delta t} $$

with the sign drawn with equal probability. Its first two moments,
$\langle\Delta\xi\rangle = -\nu_d\Delta t\,\xi$ and
$\langle(\Delta\xi)^2\rangle = (1-\xi^2)\nu_d\Delta t$, are exactly the drift
and diffusion coefficients of the operator above, so the scheme is first-order
accurate in $\nu_d\Delta t$.

**Speed is exactly conserved**, since only the direction changes. The operator
therefore adds no energy error on top of the Boris pusher.

**The update cannot leave $[-1,1]$.** By Cauchy-Schwarz, with
$a = \nu_d\Delta t$,

$$ \xi(1-a) + \sqrt{a}\sqrt{1-\xi^2} \le \sqrt{(1-a)^2+a} = \sqrt{1-a+a^2} < 1 $$

for every $0<a<1$. The clip in the code is a round-off guard that never fires.

### Trapped or passing, without reference to the midplane

Collisions move a particle across the trapped/passing boundary, so that boundary
has to be testable at any point on an orbit. The midplane form
$|\xi| < \sqrt{2\epsilon/(1+\epsilon)}$ is **only** valid at the outboard
midplane. The general condition follows from $\mu$ and energy conservation: a
particle mirrors iff its turning-point field lies below the maximum field on its
flux surface, i.e.

$$ (1 - \xi^2)\,\frac{B_{\max}}{B} > 1 $$

with everything evaluated wherever the particle happens to be. This is what
`diagnostics.is_trapped` uses. See `DOC_SELF_REVIEW.md` finding 20.

### Detrapping

A trapped particle need only be scattered through $\Delta\xi \sim \sqrt{\epsilon}$
to leave the trapped cone, so the effective detrapping rate is
$\nu_{\rm eff} = \nu_d/\epsilon$ rather than $\nu_d$, and the banana picture
survives while $\nu_{\rm eff} < \omega_b$. The standard estimate for the
detrapping time is $\tau = \epsilon/\nu_d$.

Measured here, that estimate is **right in magnitude and wrong in scaling**:
$\langle\tau\nu_d/\epsilon\rangle = 1.04$, but
$\partial\ln\tau/\partial\ln\nu_d = -0.779$ against a predicted $-1$. The
discrepancy is not explained; see `RESULTS.md` §7.3 and finding 22.

### What is not modelled

Energy scattering, drag, slowing-down, and momentum conservation between species
are all absent — the background is a fixed Maxwellian at rest with no
back-reaction. There is therefore **no bootstrap current** and no complete
neoclassical calculation. The collision operator adds trapped-particle
detrapping physics, not transport coefficients.

## 9. Default machine

| quantity | symbol | value |
|---|---|---|
| major radius | $R_0$ | 1.0 m |
| minor radius | $a$ | 0.3 m |
| inverse aspect ratio | $\epsilon$ | 0.3 |
| toroidal field on axis | $B_0$ | 2.0 T |
| plasma-current parameter | $I_p$ | 450 kA (enclosed current at $r=a$: 471.7 kA) |
| current peaking | $\nu$ | 1 (parabolic) |
| safety factor, cylindrical label | $q_0$ / $q(a)$ | 1.0 / 2.0 |
| safety factor, true field-line | $q_0$ / $q(a)$ | 1.000 / **2.097** |
| species | — | deuterium |
| energy | $E$ | 10 keV |
| speed | $v$ | $9.790\times10^5$ m/s |
| gyro-period on axis | $T_c$ | 65.56 ns |
| gyroradius on axis | $\rho$ | 1.021 cm |
| $\rho/a$ | — | 0.034 |
