"""Geometry of a bifurcated stenosed artery.

The parent artery has axial extent ``[0, z_b]`` and the two daughter arteries
emanate from ``z = z_b`` at half-angle ``beta_half``.  A cosine stenosis is
placed at ``z_s`` in the parent artery; an optional, smaller stenosis can be
placed in the daughters.  All quantities are dimensionless, scaled by the
unobstructed parent radius ``R0 = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class StenosisSpec:
    """Cosine stenosis specification.

    R(z) = R_base * [1 - (delta / 2) * (1 + cos(2 pi (z - z_c) / L))]
    inside ``[z_c - L/2, z_c + L/2]`` and ``R_base`` outside.
    """

    z_center: float
    length: float
    delta: float          # max constriction (fraction of R_base)
    R_base: float = 1.0

    def radius(self, z: np.ndarray | float) -> np.ndarray | float:
        z = np.asarray(z, dtype=float)
        inside = np.abs(z - self.z_center) <= 0.5 * self.length
        bump = 0.5 * self.delta * (1.0 + np.cos(2.0 * np.pi * (z - self.z_center) / self.length))
        R = self.R_base * (1.0 - np.where(inside, bump, 0.0))
        return R


@dataclass
class ArteryGeometry:
    """Parent + (symmetric) daughter geometry."""

    L_parent: float = 6.0
    L_daughter: float = 6.0
    z_bifurcation: float = 6.0
    beta_half: float = np.pi / 6.0           # half bifurcation angle (rad)
    parent_stenosis: StenosisSpec = field(
        default_factory=lambda: StenosisSpec(z_center=3.0, length=2.0, delta=0.3, R_base=1.0)
    )
    daughter_stenosis: StenosisSpec | None = field(
        default_factory=lambda: StenosisSpec(z_center=8.0, length=1.5, delta=0.15, R_base=0.7)
    )
    daughter_radius_ratio: float = 0.7       # κ = R_daughter / R_parent

    # ------------------------------------------------------------------
    # radii
    # ------------------------------------------------------------------
    def parent_radius(self, z: np.ndarray | float) -> np.ndarray | float:
        return self.parent_stenosis.radius(z)

    def daughter_radius(self, s: np.ndarray | float) -> np.ndarray | float:
        """Radius of one daughter at arc-length ``s`` measured from the
        bifurcation point along the daughter axis."""
        if self.daughter_stenosis is None:
            return np.full_like(np.asarray(s, dtype=float),
                                self.daughter_radius_ratio)
        z = self.z_bifurcation + s
        spec = self.daughter_stenosis
        spec.R_base = self.daughter_radius_ratio
        return spec.radius(z)

    # ------------------------------------------------------------------
    # convenient sampling for ROM cross-sections
    # ------------------------------------------------------------------
    def parent_section(self, n_z: int = 41) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(z, R(z))`` along the parent artery."""
        z = np.linspace(0.0, self.L_parent, n_z)
        return z, self.parent_radius(z)

    def daughter_section(self, n_s: int = 41) -> tuple[np.ndarray, np.ndarray]:
        s = np.linspace(0.0, self.L_daughter, n_s)
        return s, self.daughter_radius(s)

    # ------------------------------------------------------------------
    # canonical "report" cross-sections
    # ------------------------------------------------------------------
    def report_sections(self) -> dict[str, float]:
        """Axial locations at which post-processed quantities are reported."""
        return {
            "parent_inlet":      0.05 * self.L_parent,
            "parent_throat":     self.parent_stenosis.z_center,
            "parent_outlet":     0.95 * self.L_parent,
            "daughter_inlet":    0.05 * self.L_daughter,
            "daughter_throat":   (self.daughter_stenosis.z_center
                                  - self.z_bifurcation
                                  if self.daughter_stenosis else
                                  0.5 * self.L_daughter),
            "daughter_outlet":   0.95 * self.L_daughter,
        }



# ----------------------------------------------------------------------
# Paper-faithful stenosis (Ponalagusamy & Priyadharshini, AMC 2018)
# ----------------------------------------------------------------------

# The paper places a cosine stenosis in the parent vessel, centred at
#     z_c = 1.6
# with axial extent [z_c - L/2, z_c + L/2] = [0.6, 2.6]  (so L = 2),
# and constriction depth ``delta_s`` (called δs in the paper).  Outside
# the stenotic region the parent radius is R0 = 1.

PAPER_STENOSIS_CENTER = 1.6
PAPER_STENOSIS_LENGTH = 2.0
PAPER_PARENT_BASE_R = 1.0


def paper_parent_radius(z, delta_s: float,
                        z_c: float = PAPER_STENOSIS_CENTER,
                        L: float = PAPER_STENOSIS_LENGTH,
                        R0: float = PAPER_PARENT_BASE_R) -> np.ndarray:
    """Cosine stenosis profile of the parent artery.

        R(z) = R0 * (1 - (delta_s / 2) * (1 + cos(2*pi*(z - z_c) / L)))   for |z - z_c| <= L/2
        R(z) = R0                                                          otherwise

    Notes
    -----
    With this convention:
        R(z_c)        = R0 * (1 - delta_s)        (throat)
        R(z_c +- L/2) = R0                        (boundaries)
    matching the symmetric pattern in Ponalagusamy & Priyadharshini's
    Table 1 (e.g. delta_s = 0.2 gives R_throat = 0.8).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    inside = np.abs(z - z_c) <= 0.5 * L
    bump = 0.5 * delta_s * (1.0 + np.cos(2.0 * np.pi * (z - z_c) / L))
    R = R0 * (1.0 - np.where(inside, bump, 0.0))
    return R if R.size > 1 else float(R)


def paper_axial_grid(n_z: int = 101,
                     z_min: float = 0.0,
                     z_max: float = 3.2) -> np.ndarray:
    """Default axial grid for paper-faithful WSS / Λ plots.

    The paper reports values at z ∈ {0.6, 1.0, 1.4, 1.8, 2.2, 2.6} which is
    a uniform 0.4-spacing grid covering the full stenotic region.  We use a
    finer 0.016 spacing for plotting and pick out the report z's exactly.
    """
    return np.linspace(z_min, z_max, n_z)


# Six axial report locations used in the paper's tables
PAPER_REPORT_Z = np.array([0.6, 1.0, 1.4, 1.8, 2.2, 2.6])
