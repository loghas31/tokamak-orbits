"""Physical constants and default machine parameters (SI units throughout).

All quantities in this package are SI unless a name says otherwise.
Energies are stored in joules; helpers are provided to convert from eV.
"""
from __future__ import annotations

# --- fundamental constants (CODATA 2018) ---------------------------------
ELEMENTARY_CHARGE = 1.602176634e-19        # C  (exact)
ATOMIC_MASS_UNIT = 1.66053906660e-27       # kg
MU_0 = 1.25663706212e-6                    # N A^-2
SPEED_OF_LIGHT = 299792458.0               # m s^-1 (exact)

# --- species -------------------------------------------------------------
ELECTRON_MASS = 9.1093837015e-31            # kg

# Masses below are of the BARE NUCLEUS (the ion we actually push), not the
# neutral atom. The CODATA values quoted in u for "deuteron", "triton" and
# "alpha particle" are already nucleus masses -- no electron is subtracted.
# Getting this wrong is a 0.027% error in m_D, which is small but propagates
# directly into the gyroradius and hence the banana width. See
# docs/DOC_SELF_REVIEW.md, finding 1.
DEUTERON_MASS = 2.013553212745 * ATOMIC_MASS_UNIT   # 3.3435837768e-27 kg
TRITON_MASS = 3.01550071621 * ATOMIC_MASS_UNIT
ALPHA_MASS = 4.001506179127 * ATOMIC_MASS_UNIT
PROTON_MASS = 1.67262192369e-27

SPECIES = {
    "D": (DEUTERON_MASS, ELEMENTARY_CHARGE),
    "H": (PROTON_MASS, ELEMENTARY_CHARGE),
    "T": (TRITON_MASS, ELEMENTARY_CHARGE),
    "He4": (ALPHA_MASS, 2 * ELEMENTARY_CHARGE),
}


def ev_to_joule(energy_ev: float) -> float:
    """Convert an energy in electronvolts to joules."""
    return energy_ev * ELEMENTARY_CHARGE


def joule_to_ev(energy_j: float) -> float:
    """Convert an energy in joules to electronvolts."""
    return energy_j / ELEMENTARY_CHARGE


# --- default machine ------------------------------------------------------
# A small, deliberately conventional tokamak. R0/a = 3.33 gives an inverse
# aspect ratio eps = 0.3, large enough that trapped-particle effects are
# unmistakable, small enough that the standard large-aspect-ratio expansions
# used for validation are still defensible.
DEFAULT_MACHINE = dict(
    major_radius=1.0,       # m,  R0
    minor_radius=0.3,       # m,  a   -> eps = a/R0 = 0.3
    b_toroidal=2.0,         # T,  B0 on the magnetic axis
    plasma_current=4.5e5,   # A,  Ip  -> q0 ~ 1.0, q(a) ~ 2.0
    current_peaking=1.0,    # nu in j(r) = j0 (1 - (r/a)^2)^nu
)
