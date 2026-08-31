"""Single-particle orbit confinement in a simplified tokamak field."""
from .constants import SPECIES, ev_to_joule, joule_to_ev, DEFAULT_MACHINE
from .fields import (
    Field,
    UniformField,
    TokamakField,
    grad_b_curvature_drift,
    e_cross_b_drift,
)

__version__ = "0.1.0"
__all__ = [
    "SPECIES", "ev_to_joule", "joule_to_ev", "DEFAULT_MACHINE",
    "Field", "UniformField", "TokamakField",
    "grad_b_curvature_drift", "e_cross_b_drift",
]
