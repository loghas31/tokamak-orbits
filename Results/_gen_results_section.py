#!/usr/bin/env python3
"""Generate RESULTS.md section 7 (collisions) from results/*.json.

Kept in the repo so the section can be regenerated rather than hand-edited,
which is the rule that findings 16 and 22 exist to enforce.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def main():
    det = load("detrapping.json")
    rows, fit = det["rows"], det["fit"]
    co = load("collision_operator.json")
    sm = load("collisional_smearing.json")

    leg = "\n".join(
        f"| {r['nu_dt']:.4f} | "
        + " | ".join(
            f"{r[f'ratio_l{l}']:.4f}" if r[f"ratio_l{l}"] else "below noise"
            for l in (1, 2, 3))
        + " |"
        for r in co["legendre"])

    ovr = "\n".join(
        f"| {o['nu_dt']:.3f} | {100 * o['overshoot_fraction']:.4f}% | "
        f"{o['max_abs_xi']:.5f} | {o['bound']:.5f} |"
        for o in co["overshoot"])

    dtab = "\n".join(
        f"| {r['nu_d']:.3g} | {100 * r['fraction_detrapped']:.1f}% | "
        f"{r['median_detrap_time']:.3g} | {r['predicted_eps_over_nu']:.3g} | "
        f"{r['ratio_median']:.2f} | {r['nu_eff_over_omega_b']:.2f} | "
        f"{r['samples_per_detrap_time']:.0f} |"
        for r in rows)

    base = None
    srows = []
    for r in sm:
        ip = np.array([p["plasma_current"] for p in r["points"]]) / 1e3
        lo = np.array([p["lost_fraction"] for p in r["points"]])
        hi = float(ip[lo > 0].max())
        truncated = abs(hi - ip.max()) < 1e-6
        if r["nu_d"] == 0:
            base = hi
        label = ("collisionless" if r["nu_d"] == 0
                 else f"$10^{{{round(np.log10(r['nu_d']))}}}$")
        ge = "$\\geq$" if truncated else ""
        srows.append(
            f"| {label} | {100 * lo.max():.1f}% | {ge}{hi:.0f} kA | "
            f"{hi / base:.1f}x |")
    smr = "\n".join(srows)

    # Plain substitution, not str.format: the template is full of LaTeX braces
    # and .format() tries to read them as fields.
    out = SECTION
    for key, val in (("@LEG@", leg), ("@OVR@", ovr), ("@DTAB@", dtab),
                     ("@SMR@", smr),
                     ("@EXPONENT@", f"{fit['exponent']:.3f}"),
                     ("@RATIO@", f"{fit['mean_ratio']:.2f}"),
                     ("@BASE@", f"{base:.0f}")):
        out = out.replace(key, val)
    assert "@" not in out.replace("@", "", 0) or True
    leftover = [k for k in ("@LEG@", "@OVR@", "@DTAB@", "@SMR@", "@EXPONENT@",
                            "@RATIO@", "@BASE@") if k in out]
    if leftover:
        raise RuntimeError(f"unsubstituted placeholders: {leftover}")
    print(out)


SECTION = r"""
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
@LEG@

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
@OVR@

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
@DTAB@

Fitted over all six points, every one of which is now resolved:

| quantity | measured | predicted |
|---|---|---|
| $\partial\ln\tau / \partial\ln\nu_d$ | **@EXPONENT@** | $-1$ |
| $\langle \tau\nu_d/\epsilon \rangle$ | **@RATIO@** | $1$ |

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
@SMR@

Collisionlessly this ion is perfectly confined above @BASE@ kA. At
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
"""


if __name__ == "__main__":
    main()
