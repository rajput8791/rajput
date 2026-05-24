"""Axially-distributed steady-state Casson FOM.

This module reproduces the paper of Ponalagusamy & Priyadharshini
(Applied Mathematics and Computation 333, 2018, pp. 325-343).

At each axial station ``z`` we solve the dimensionless steady-state
momentum balance for the dimensionless axial velocity ``u(r)``
across the cross-section ``r in [0, R(z)]``::

    -(1/r) d/dr( r * mu_eff(|du/dr|) * du/dr )  +  (M^2 + 1/Da) * u  =  G

with boundary conditions ``du/dr|_{r=0} = 0`` (axisymmetry) and
``u(R(z)) = 0`` (no slip).  Here

* ``M``  -- Hartmann number,
* ``Da`` -- Darcy number (``Da -> infty`` removes the porous resistance),
* ``mu_eff`` -- regularised Casson effective viscosity,
        ``mu_eff(g) = (sqrt(theta) + sqrt(g + eps))^2 / (g + eps)``,
* ``theta`` -- Casson yield-stress parameter,
* ``G``    -- (constant) axial pressure gradient ``-dp/dz``.

The wall shear stress reported in the paper's Tables 1-3 is

    tau_w(z) = mu_eff * du/dr  |_{r = R(z)}             (signed; negative
                                                         because du/dr < 0
                                                         at the wall)

We solve the cross-sectional ODE by a second-order finite-difference
discretisation on a uniform radial grid plus Picard iteration on
``mu_eff``.  The Picard iteration is robust and converges in 10-30
sweeps for the parameter ranges of interest.

Comparisons with the paper's Tables 1-3 are exposed through
``reproduce_paper_tables``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .geometry import (
    paper_parent_radius,
    paper_axial_grid,
    PAPER_REPORT_Z,
)


# ----------------------------------------------------------------------
# Cross-section solver (steady)
# ----------------------------------------------------------------------

@dataclass
class SteadyParams:
    """Steady-state Casson-MHD-Darcy parameters."""

    M: float = 1.0          # Hartmann number
    Da: float = 0.1         # Darcy number (use np.inf for no porous medium)
    theta: float = 0.05     # Casson yield-stress parameter
    G: float = 1.0          # axial pressure gradient -dp/dz (calibrated)
    eps: float = 1e-6       # regularisation for Casson law


def _mu_eff(gamma_dot: np.ndarray, theta: float, eps: float) -> np.ndarray:
    """Regularised Casson effective viscosity."""
    g = np.abs(gamma_dot) + eps
    return (np.sqrt(theta) + np.sqrt(g)) ** 2 / g


def solve_cross_section(R_wall: float,
                        params: SteadyParams,
                        Nr: int = 201,
                        max_iter: int = 80,
                        tol: float = 1e-8,
                        u_init: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the steady cross-sectional momentum equation.

    Parameters
    ----------
    R_wall : wall radius (R(z) at the cross-section of interest)
    params : SteadyParams
    Nr     : number of radial grid points (uniform on [0, R_wall])
    max_iter, tol : Picard iteration controls
    u_init : optional warm-start velocity profile sampled on the new grid

    Returns
    -------
    r        : (Nr,) radial grid
    u        : (Nr,) axial velocity profile
    mu_eff_  : (Nr,) effective viscosity at the converged solution
    """
    if R_wall <= 0:
        raise ValueError(f"R_wall must be > 0, got {R_wall}")

    r = np.linspace(0.0, R_wall, Nr)
    dr = r[1] - r[0]
    sink = params.M ** 2 + (0.0 if not np.isfinite(params.Da) else 1.0 / params.Da)

    # Initial guess: warm-start if provided, else Newtonian Hagen-Poiseuille.
    if u_init is not None and len(u_init) == Nr:
        u = u_init.copy()
    else:
        u = (params.G / 4.0) * (R_wall ** 2 - r ** 2)

    # Pre-compute geometric coefficients (depend only on r and dr,
    # not on the Picard iterate).
    n = Nr - 2
    r_int = r[1:-1]                                   # (n,)
    r_int_safe = np.where(r_int < 1e-14, 0.5 * dr, r_int)
    r_ip = r_int_safe + 0.5 * dr                     # (n,)
    r_im = np.maximum(r_int_safe - 0.5 * dr, 0.0)    # (n,)
    inv_ri_dr2 = 1.0 / (r_int_safe * dr * dr)        # (n,)

    for _ in range(max_iter):
        # Shear rate du/dr (centred, one-sided at endpoints)
        dudr = np.empty(Nr)
        dudr[1:-1] = (u[2:] - u[:-2]) / (2.0 * dr)
        dudr[0] = 0.0
        dudr[-1] = (u[-1] - u[-2]) / dr

        mu = _mu_eff(np.abs(dudr), params.theta, params.eps)

        # Vectorised tridiagonal assembly
        mu_ip = 0.5 * (mu[1:-1] + mu[2:])     # (n,) at i+1/2
        mu_im = 0.5 * (mu[1:-1] + mu[:-2])    # (n,) at i-1/2
        c_ip = r_ip * mu_ip * inv_ri_dr2
        c_im = r_im * mu_im * inv_ri_dr2

        # row 0 (i = 1) folds the symmetry BC u[0] = u[1] into the diagonal
        a_lo = np.empty(n)
        a_lo[0] = 0.0
        a_lo[1:] = -c_im[1:]
        a_d = c_ip + c_im + sink
        a_d[0] = c_ip[0] + sink                 # u[0] = u[1] -> drop c_im[0]
        a_up = np.empty(n)
        a_up[:-1] = -c_ip[:-1]
        a_up[-1] = 0.0
        b = np.full(n, params.G)
        # Right BC u[Nr-1] = 0 -> nothing to add to b[n-1]

        u_int = _thomas(a_lo, a_d, a_up, b)

        u_new = np.empty(Nr)
        u_new[1:-1] = u_int
        u_new[0] = u_new[1]     # symmetry
        u_new[-1] = 0.0         # no slip

        rel = np.linalg.norm(u_new - u) / max(np.linalg.norm(u_new), 1e-12)
        u = u_new
        if rel < tol:
            break

    # Final viscosity for diagnostics
    dudr = np.empty(Nr)
    dudr[1:-1] = (u[2:] - u[:-2]) / (2.0 * dr)
    dudr[0] = 0.0
    dudr[-1] = (u[-1] - u[-2]) / dr
    mu_final = _mu_eff(np.abs(dudr), params.theta, params.eps)

    return r, u, mu_final


def _thomas(lo: np.ndarray, d: np.ndarray, up: np.ndarray,
            b: np.ndarray) -> np.ndarray:
    """Thomas tridiagonal solver (in-place forward + back sweep)."""
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = up[0] / d[0]
    dp[0] = b[0] / d[0]
    for i in range(1, n):
        denom = d[i] - lo[i] * cp[i - 1]
        cp[i] = up[i] / denom if i < n - 1 else 0.0
        dp[i] = (b[i] - lo[i] * dp[i - 1]) / denom
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ----------------------------------------------------------------------
# Quantities of interest
# ----------------------------------------------------------------------

def wall_shear_stress(r: np.ndarray, u: np.ndarray, mu: np.ndarray) -> float:
    """Signed wall shear stress  ``tau_w = mu_eff * du/dr|_R``.

    With ``u(R) = 0`` and ``u > 0`` inside, ``du/dr|_R < 0``, so the
    returned ``tau_w`` is negative -- matching the paper's tables.
    """
    dr = r[1] - r[0]
    dudr_wall = (u[-1] - u[-2]) / dr
    return mu[-1] * dudr_wall


def volumetric_flow_rate(r: np.ndarray, u: np.ndarray) -> float:
    """Q = 2*pi * int_0^R r * u(r) dr (per cross-section)."""
    return 2.0 * np.pi * np.trapezoid(r * u, r)


def plug_core_radius(r: np.ndarray, u: np.ndarray,
                     theta: float, G: float) -> float:
    """Radial location where the local stress equals the yield stress.

    For the paper's steady model the stress at radius r is
    ``tau(r) = G r / 2`` (force balance, independent of constitutive law),
    so the plug-core radius is simply ``R_p = 2 theta / G``.
    """
    return 2.0 * theta / G


# ----------------------------------------------------------------------
# Axial sweep
# ----------------------------------------------------------------------

@dataclass
class AxialResult:
    z: np.ndarray              # (n_z,) axial grid
    R: np.ndarray              # (n_z,) wall radius profile
    tau_w: np.ndarray          # (n_z,) signed WSS
    Q: np.ndarray              # (n_z,) volumetric flow rate
    u_center: np.ndarray       # (n_z,) centerline velocity
    Rp: float                  # plug-core radius (z-independent for steady)
    params: SteadyParams = field(default_factory=SteadyParams)


def axial_sweep(delta_s: float,
                params: SteadyParams,
                z: np.ndarray | None = None,
                Nr: int = 201) -> AxialResult:
    """Sweep cross-section solver over the axial direction.

    ``R(z)`` follows the cosine stenosis profile from
    ``geometry.paper_parent_radius``.
    """
    if z is None:
        z = paper_axial_grid()
    R_z = paper_parent_radius(z, delta_s)

    tau_w = np.zeros_like(z)
    Q = np.zeros_like(z)
    u_c = np.zeros_like(z)

    for i, Ri in enumerate(R_z):
        r, u, mu = solve_cross_section(Ri, params, Nr=Nr)
        tau_w[i] = wall_shear_stress(r, u, mu)
        Q[i] = volumetric_flow_rate(r, u)
        u_c[i] = u[0]

    return AxialResult(
        z=z,
        R=R_z,
        tau_w=tau_w,
        Q=Q,
        u_center=u_c,
        Rp=plug_core_radius(np.array([0.0]), np.array([0.0]),
                            params.theta, params.G),
        params=params,
    )


# ----------------------------------------------------------------------
# Calibration of G to hit the paper's baseline tau_w = -3.8494
# ----------------------------------------------------------------------

PAPER_BASELINE_WSS = -3.8494
PAPER_BASELINE_M = 1.0
PAPER_BASELINE_Da = 0.1


def calibrate_G(theta: float = 0.05,
                target_wss: float = PAPER_BASELINE_WSS,
                M: float = PAPER_BASELINE_M,
                Da: float = PAPER_BASELINE_Da,
                R_wall: float = 1.0,
                Nr: int = 401,
                G_low: float = 1.0,
                G_high: float = 50.0,
                tol: float = 1e-6,
                max_iter: int = 60) -> float:
    """Bisection: find G such that tau_w(R_wall) = target_wss.

    Used once at startup to pin the model to the paper's baseline value
    of -3.8494 at the unobstructed cross-section R = 1.
    """
    def f(G):
        p = SteadyParams(M=M, Da=Da, theta=theta, G=G)
        r, u, mu = solve_cross_section(R_wall, p, Nr=Nr)
        return wall_shear_stress(r, u, mu) - target_wss

    f_lo = f(G_low)
    f_hi = f(G_high)
    if f_lo * f_hi > 0:
        # Expand upper bound
        for _ in range(20):
            G_high *= 2.0
            f_hi = f(G_high)
            if f_lo * f_hi <= 0:
                break

    for _ in range(max_iter):
        G_mid = 0.5 * (G_low + G_high)
        f_mid = f(G_mid)
        if abs(f_mid) < tol:
            return G_mid
        if f_lo * f_mid <= 0:
            G_high = G_mid
            f_hi = f_mid
        else:
            G_low = G_mid
            f_lo = f_mid
    return 0.5 * (G_low + G_high)


# ----------------------------------------------------------------------
# Reproduce the three paper tables
# ----------------------------------------------------------------------

def reproduce_paper_tables(theta: float = 0.05,
                           Nr: int = 401,
                           verbose: bool = True) -> dict:
    """Compute Tables 1, 2, 3 of Ponalagusamy & Priyadharshini (AMC 2018).

    Returns a dict of pandas DataFrames keyed ``'table1'``, ``'table2'``,
    ``'table3'`` plus the calibrated ``'G'``.
    """
    import pandas as pd

    G = calibrate_G(theta=theta, Nr=Nr)
    if verbose:
        print(f"  Calibrated G = {G:.6f}  (theta={theta}, Nr={Nr})")

    z = PAPER_REPORT_Z

    # ---------- Table 1: tau_w vs z for delta_s in {0.0, 0.2} -----------
    rows1 = []
    for delta_s in (0.0, 0.2):
        p = SteadyParams(M=1.0, Da=0.1, theta=theta, G=G)
        res = axial_sweep(delta_s, p, z=z, Nr=Nr)
        for zi, ti in zip(z, res.tau_w):
            rows1.append({
                "z": zi,
                "delta_s": delta_s,
                "tau_w": ti,
            })
    table1 = pd.DataFrame(rows1).pivot(index="z", columns="delta_s",
                                       values="tau_w")
    table1.columns = [f"delta_s={d}" for d in table1.columns]

    # ---------- Table 2: tau_w vs z for M in {0, 1} ---------------------
    rows2 = []
    for M_h in (0.0, 1.0):
        p = SteadyParams(M=M_h, Da=0.1, theta=theta, G=G)
        res = axial_sweep(0.2, p, z=z, Nr=Nr)
        for zi, ti in zip(z, res.tau_w):
            rows2.append({"z": zi, "M": M_h, "tau_w": ti})
    table2 = pd.DataFrame(rows2).pivot(index="z", columns="M",
                                       values="tau_w")
    table2.columns = [f"M={int(m)}" for m in table2.columns]

    # ---------- Table 3: tau_w vs z for Da in {inf, 0.1} ----------------
    rows3 = []
    for Da_h in (np.inf, 0.1):
        p = SteadyParams(M=1.0, Da=Da_h, theta=theta, G=G)
        res = axial_sweep(0.2, p, z=z, Nr=Nr)
        for zi, ti in zip(z, res.tau_w):
            rows3.append({"z": zi, "Da": Da_h, "tau_w": ti})
    table3 = pd.DataFrame(rows3).pivot(index="z", columns="Da",
                                       values="tau_w")
    # Force column order: Da=inf first, then Da=0.1 (matches paper)
    inf_cols = [c for c in table3.columns if not np.isfinite(c)]
    finite_cols = sorted([c for c in table3.columns if np.isfinite(c)],
                         reverse=True)
    table3 = table3[inf_cols + finite_cols]
    table3.columns = [
        ("Da=inf" if not np.isfinite(d) else f"Da={d}")
        for d in table3.columns
    ]

    return {"G": G, "table1": table1, "table2": table2, "table3": table3}
