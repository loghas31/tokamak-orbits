# Tokamak Orbit Confinement — project brief

**Logan Hastie** · MPhys Astrophysics, University of Edinburgh
Python · NumPy · SciPy · pytest · GitHub Actions
`github.com/loghas31/tokamak-orbits`

---

## What it is

A single-particle orbit code for a simplified tokamak magnetic field, built to
answer one question:

> **When magnetic confinement fails as the field is weakened, does it fail
> sharply at a threshold, or gradually?**

Individual deuterium ions are integrated under the full Lorentz force — no
guiding-centre approximation in the orbits themselves — through a magnetic field
combining a 1/R toroidal component with a poloidal component from a plasma
current. The plasma current is then reduced until ions reach the wall.

## The result

**Both answers are true, and the gap between them is the finding.**

At a *fixed* pitch angle the transition is a near-step: the loss fraction falls
from 90% to 10% across a few percent of the plasma current. Across an *isotropic
population* of the same ions it takes 157% of the midpoint current — some 27
times broader, which is not a threshold in any useful sense.

The reason is that every pitch angle has its own sharp threshold, and those
thresholds are spread over a factor of 8.3 in plasma current (30.6 kA to
253.5 kA). A population contains all of them at once, so its curve is the
smeared superposition of many step functions. Nothing about the population is
gradual; it is made of sharp things that disagree about where to be sharp.

Adding collisions then destroys even the single-ion threshold. An ion that is
perfectly confined above 266 kA when it cannot change pitch angle is still being
lost at 1029 kA once it can — because collisions eventually walk it into the
wide, wall-facing counter-going orbits. **The sharp threshold turns out to be an
artefact of forbidding the ion to change direction.**

A secondary result falls out without being sought: co-going and counter-going
ions of identical energy and pitch magnitude have orbits displaced in *opposite*
radial directions, so a counter-going ion needs roughly seven times more plasma
current to stay confined. Loss in this machine is overwhelmingly a counter-going
phenomenon.

## How it was validated

Every claim is checked against a result derived independently, with nothing
fitted:

| Property | Reference | Agreement |
|---|---|---|
| ∇·**B** | must be exactly zero | 10⁻⁹ relative |
| Integrator order | Boris is second order | 1.974 → 2.000 |
| Energy conservation | — | < 10⁻¹³, bounded, non-growing |
| Grad-B + curvature drift | m(v∥² + ½v⊥²)/qBR | 0.005 – 0.698% |
| E × B drift | E×B/B² | 0.026 – 0.045%, species-independent |
| Collision operator | Legendre eigenvalues l(l+1)ν/2 | within 0.2% |
| Detrapping time | τ = ε/ν | ν^−0.962 against −1 |

148 tests run on push across Python 3.10–3.12. Every table in the results
document is generated from stored JSON rather than typed by hand, so the figures
and the prose cannot disagree.

Two numerical choices carry most of the weight. The integrator is a **Boris
pusher**, not `scipy`'s RK45: RK45's energy error accumulates without bound, and
since confinement is measured by watching an ion drift to a wall, an integrator
that changes its energy manufactures the answer. And the poloidal field carries
an extra factor of R₀/R that makes the field **divergence-free on a torus** — the
textbook cylindrical form is not, by about 5% of |**B**|, which is not a magnetic
field at all.

## What it is not

The field is divergence-free with tokamak-like topology, but it is **not an
equilibrium** — circular concentric flux surfaces with no Shafranov shift do not
solve the Grad–Shafranov equation. Nothing here is a prediction about a real
machine.

The ions are test particles: they do not carry the current that makes the field
they move in, do not see each other, and do not push back. Collisions are
pitch-angle only, with no energy scattering or drag, so there is no
slowing-down and no bootstrap current. No field ripple, no electric field, no
wall physics beyond "crossed r = a, gone". "Confinement time" here means the
time for one test ion to reach the wall — an orbit loss time, not the energy
confinement time an experimentalist would quote.

The repository states all of this in a dedicated document rather than leaving it
to be discovered.

## The part I would point at

`docs/DOC_SELF_REVIEW.md` is a running register of **23 findings against my own
work**, eleven of which changed a published number. It includes a magnetic field
that turned out not to be divergence-free, a banana-width scan that was measuring
orbits which were not bananas, a drift validation contaminated by a duplicated
copy of a formula I had already fixed elsewhere, and a numerical artefact I
documented in a docstring that provably cannot occur.

Six of the findings came from an adversarial review pass run *after* everything
already passed its tests.

The entry worth reading is finding 22. I measured a collisional detrapping
exponent of −0.779 against a predicted −1, checked three candidate explanations,
excluded all three, and published it as an unexplained failure. It was none of
them: it was my own collision timestep. I had validated the collision operator
against decay rates that are a *bulk* property of the whole distribution, which
converge quickly, and never noticed that a detrapping time is a **first-passage**
quantity, which does not — the ion was reaching the boundary in about two
discrete jumps instead of diffusing there. Resolved properly, the exponent is
−0.962.

The generalisable lesson is more useful than the number: **validating a scheme on
a bulk statistic does not validate it for a first-passage one.** There is now a
test asserting exactly that, demonstrating that the bulk rate is already
converged at the step size where the first-passage number is still wrong — which
is precisely why it went unnoticed.

## Reproducing it

```bash
git clone https://github.com/loghas31/tokamak-orbits
cd tokamak-orbits && pip install -e ".[dev]"

pytest -q                                    # 148 tests, ~4 min
python scripts/run_experiments.py --quick    # coarse reproduction, ~5 min
python scripts/make_figures.py               # regenerate every figure
```

Fixed seeds and a fixed timestep policy throughout; the same random sample is
used at every point of a scan, so a scan measures the response to the field
rather than sampling noise.

## Where it could go next

1. **Energy scattering and drag** — the collision model changes an ion's
   direction but never its speed. Adding slowing-down would make this a real
   neoclassical calculation.
2. **A Solov'ev equilibrium** — an analytic solution to Grad–Shafranov with a
   genuine Shafranov shift. This would break the exact |**B**| ∝ 1/R that
   currently makes the trapped-fraction agreement partly true by construction,
   turning a weak check into a real test.
3. **Toroidal field ripple** — real machines have discrete coils, and the
   resulting ripple is a leading fast-ion loss channel that is absent here.
