"""Full-Order Model (FOM) for Pulsatile MHD Casson flow.

Solves the dimensionless momentum equation in a cylindrical artery cross-section:

    du/dt = (1/r)*d/dr(r * mu_eff * du/dr) - (M^2 + 1/Da)*u + S(t)

where S(t) = A0*(1 + e*cos(t)) + a_b*cos(omega_b*t + phi)

Casson constitutive law (regularised):
    mu_eff = (sqrt(theta) + sqrt(|du/dr| + eps))^2 / (|du/dr| + eps)

Method: Implicit Euler + Picard iteration on effective viscosity.
        (Unconditionally stable, avoids blow-up)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FlowParams:
    """Dimensionless parameters for the pulsatile MHD Casson problem."""
    M: float = 2.0                # Hartmann number
    Da: float = 0.5               # Darcy number
    theta: float = 0.1            # Casson yield stress (dimensionless)
    A0: float = 1.0               # mean pressure gradient amplitude
    e: float = 0.5                # pulsatility ratio A1/A0
    a_b: float = 0.5              # body acceleration amplitude
    omega_b: float = 1.0          # body acceleration frequency ratio
    phi: float = 0.0              # body acceleration phase
    alpha_sq: float = 1.0         # Womersley number squared (scaled to 1)
    eps: float = 1e-3             # regularisation for Casson law


@dataclass
class FOMSolver:
    """Finite-difference solver on a radial grid [0, R_wall]."""
    Nr: int = 80                  # number of radial grid points
    Nt: int = 200                 # number of time steps per cardiac cycle
    n_cycles: int = 3             # number of cardiac cycles to simulate
    R_wall: float = 1.0           # wall radius at this cross-section
    params: FlowParams = field(default_factory=FlowParams)

    def __post_init__(self):
        self.dr = self.R_wall / (self.Nr - 1)
        self.T_total = 2.0 * np.pi * self.n_cycles
        self.dt = self.T_total / self.Nt
        self.r = np.linspace(0, self.R_wall, self.Nr)
        self.t_arr = np.linspace(0, self.T_total, self.Nt + 1)

    # ------------------------------------------------------------------
    # Source term
    # ------------------------------------------------------------------
    def source(self, t: float) -> float:
        """S(t) = A0*(1 + e*cos(t)) + a_b*cos(omega_b*t + phi)."""
        p = self.params
        return p.A0 * (1.0 + p.e * np.cos(t)) + p.a_b * np.cos(p.omega_b * t + p.phi)

    def pressure_gradient(self, t: float) -> float:
        """Pulsatile pressure gradient only (for resistance calc)."""
        p = self.params
        return p.A0 * (1.0 + p.e * np.cos(t))

    # ------------------------------------------------------------------
    # Effective viscosity (Casson, regularised)
    # ------------------------------------------------------------------
    def mu_eff(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Casson effective viscosity (regularised)."""
        p = self.params
        g = np.abs(gamma_dot) + p.eps
        sq_g = np.sqrt(g)
        sq_th = np.sqrt(p.theta)
        mu = (sq_th + sq_g) ** 2 / g
        return mu

    # ------------------------------------------------------------------
    # Implicit Euler time step with Picard iteration
    # ------------------------------------------------------------------
    def step_implicit(self, u_old: np.ndarray, t_new: float,
                      picard_iters: int = 6) -> np.ndarray:
        """Advance one time step using fully implicit Euler + Picard.

        System: (I/dt + sink - L(mu)) u_new = u_old/dt + S(t_new)
        where L(mu) u = (1/r) d/dr(r * mu * du/dr)
        and sink = M^2 + 1/Da.
        """
        Nr = self.Nr
        dr = self.dr
        dt = self.dt
        r = self.r
        p = self.params
        sink = p.M ** 2 + 1.0 / p.Da
        S = self.source(t_new)

        u_new = u_old.copy()

        for _ in range(picard_iters):
            # Compute shear rate from current iterate
            gamma_dot = np.zeros(Nr)
            gamma_dot[1:-1] = np.abs(u_new[2:] - u_new[:-2]) / (2.0 * dr)
            gamma_dot[0] = 0.0
            gamma_dot[-1] = np.abs(u_new[-1] - u_new[-2]) / dr
            mu = self.mu_eff(gamma_dot)

            # Build tridiagonal system for interior points [1..Nr-2]
            # (1/dt + sink) u_i - [(r_{i+1/2} mu_{i+1/2})/(r_i dr^2)] u_{i+1}
            #                    - [(r_{i-1/2} mu_{i-1/2})/(r_i dr^2)] u_{i-1}
            #                    + [(r_{i+1/2} mu_{i+1/2} + r_{i-1/2} mu_{i-1/2})/(r_i dr^2)] u_i
            # = u_old_i / dt + S

            # We solve: A * u_interior = b
            n = Nr - 2  # interior points (indices 1..Nr-2)
            lower = np.zeros(n)  # sub-diagonal
            diag = np.zeros(n)   # main diagonal
            upper = np.zeros(n)  # super-diagonal
            b = np.zeros(n)

            for j in range(n):
                i = j + 1  # physical index
                ri = r[i]
                if ri < 1e-14:
                    ri = dr * 0.5  # avoid division by zero at r=0

                r_ip = ri + 0.5 * dr
                r_im = ri - 0.5 * dr
                mu_ip = 0.5 * (mu[i] + mu[i + 1])
                mu_im = 0.5 * (mu[i] + mu[i - 1])

                c_ip = r_ip * mu_ip / (ri * dr * dr)
                c_im = r_im * mu_im / (ri * dr * dr)

                diag[j] = 1.0 / dt + sink + c_ip + c_im
                if j > 0:
                    lower[j] = -c_im
                else:
                    # j=0 corresponds to i=1; its left neighbor is i=0
                    # Symmetry BC: u[0] = u[1], so -c_im * u[0] = -c_im * u[1]
                    # Move to diagonal
                    diag[j] += (-c_im)  # wait no: A[0,0] already has c_im,
                    # and we need -c_im on the "left" but u[0]=u[1] means
                    # that term -c_im*u[0] = -c_im*u[1], so effectively
                    # diag contribution should be reduced: diag = 1/dt+sink+c_ip+c_im - c_im = 1/dt+sink+c_ip
                    # Let me redo: the equation is
                    # (1/dt + sink + c_ip + c_im) u[1] - c_ip*u[2] - c_im*u[0] = rhs
                    # With u[0]=u[1]: (1/dt + sink + c_ip + c_im - c_im) u[1] - c_ip*u[2] = rhs
                    # So diag[0] = 1/dt + sink + c_ip
                    diag[j] = 1.0 / dt + sink + c_ip  # corrected

                if j < n - 1:
                    upper[j] = -c_ip
                # else: j=n-1 corresponds to i=Nr-2, right neighbor is i=Nr-1
                # BC: u[Nr-1] = 0, so -c_ip*u[Nr-1] = 0, nothing to add

                b[j] = u_old[i] / dt + S

            # Thomas algorithm (tridiagonal solve)
            u_interior = thomas_solve(lower, diag, upper, b)

            u_new[1:-1] = u_interior
            u_new[0] = u_new[1]    # symmetry
            u_new[-1] = 0.0        # no-slip

        return u_new

    # ------------------------------------------------------------------
    # Full simulation
    # ------------------------------------------------------------------
    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the full time-stepping simulation.

        Returns
        -------
        r : (Nr,) radial grid
        t : (Nt+1,) time array
        U : (Nr, Nt+1) velocity field snapshots
        """
        Nr = self.Nr
        Nt = self.Nt
        U = np.zeros((Nr, Nt + 1))
        # u(r, 0) = 0 (initial condition)

        for n in range(Nt):
            t_new = self.t_arr[n + 1]
            U[:, n + 1] = self.step_implicit(U[:, n], t_new, picard_iters=5)

        return self.r, self.t_arr, U

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def wall_shear_stress(self, U: np.ndarray) -> np.ndarray:
        """WSS = mu_eff * |du/dr| at r = R_wall, for each time step."""
        dr = self.dr
        # One-sided difference at wall
        gamma_wall = np.abs((U[-1, :] - U[-2, :]) / dr)
        mu_wall = self.mu_eff(gamma_wall)
        return mu_wall * gamma_wall

    def plug_core_radius(self, U: np.ndarray) -> np.ndarray:
        """Plug core radius R_p(t): largest r where |du/dr| ~ 0."""
        dr = self.dr
        r = self.r
        Nt = U.shape[1]
        Rp = np.zeros(Nt)
        threshold = 0.01  # threshold for plug detection

        for n in range(Nt):
            u = U[:, n]
            if np.max(np.abs(u)) < 1e-10:
                Rp[n] = 0.0
                continue
            gamma = np.abs(np.gradient(u, dr))
            # Normalise by max shear rate
            max_gamma = np.max(gamma)
            if max_gamma < 1e-10:
                Rp[n] = r[-1]
                continue
            gamma_norm = gamma / max_gamma
            # Find outermost point from center where gamma_norm < threshold
            idx = 0
            while idx < len(gamma_norm) and gamma_norm[idx] < threshold:
                idx += 1
            Rp[n] = r[max(0, idx - 1)]
        return Rp

    def flow_resistance(self, U: np.ndarray) -> np.ndarray:
        """Flow resistance Lambda = (-dp/dz) / Q where Q = 2*pi*int(r*u*dr)."""
        r = self.r
        Nt = U.shape[1]
        Lambda = np.zeros(Nt)
        for n in range(Nt):
            u = U[:, n]
            Q = 2.0 * np.pi * np.trapezoid(r * u, r)
            dpz = self.pressure_gradient(self.t_arr[n])
            if abs(Q) > 1e-12:
                Lambda[n] = dpz / Q
            else:
                Lambda[n] = 0.0
        return Lambda


def thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system using Thomas algorithm.

    lower[0] is unused (no sub-diagonal for first row).
    upper[-1] is unused (no super-diagonal for last row).
    """
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)

    # Forward sweep
    c[0] = upper[0] / diag[0]
    d[0] = b[0] / diag[0]
    for i in range(1, n):
        m = lower[i] / (diag[i] - lower[i] * c[i - 1])
        c[i] = upper[i] / (diag[i] - lower[i] * c[i - 1]) if i < n - 1 else 0.0
        d[i] = (b[i] - lower[i] * d[i - 1]) / (diag[i] - lower[i] * c[i - 1])

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


def generate_snapshots(param_samples: list[dict], Nr: int = 80, Nt: int = 200,
                       n_cycles: int = 2) -> dict:
    """Generate FOM snapshots for multiple parameter combinations.

    Parameters
    ----------
    param_samples : list of dicts with keys matching FlowParams fields
                   (extra keys like 'delta', 'beta_half' are ignored)
    Nr, Nt, n_cycles : FOM discretisation settings

    Returns
    -------
    dict with keys:
        'r': radial grid (Nr,)
        't': time array (Nt+1,)
        'U_all': list of (Nr, Nt+1) snapshot matrices
        'params': the param_samples list
        'wss': list of (Nt+1,) WSS arrays
        'Rp': list of (Nt+1,) plug-core arrays
        'Lambda': list of (Nt+1,) resistance arrays
    """
    results = {'U_all': [], 'wss': [], 'Rp': [], 'Lambda': [], 'params': param_samples}

    # FlowParams valid keys
    flow_keys = {'M', 'Da', 'theta', 'A0', 'e', 'a_b', 'omega_b', 'phi', 'alpha_sq', 'eps'}

    for i, ps in enumerate(param_samples):
        fp_dict = {k: v for k, v in ps.items() if k in flow_keys}
        fp = FlowParams(**fp_dict)
        solver = FOMSolver(Nr=Nr, Nt=Nt, n_cycles=n_cycles, params=fp)
        r, t, U = solver.solve()
        results['U_all'].append(U)
        results['wss'].append(solver.wall_shear_stress(U))
        results['Rp'].append(solver.plug_core_radius(U))
        results['Lambda'].append(solver.flow_resistance(U))
        if i == 0:
            results['r'] = r
            results['t'] = t
        print(f"  FOM sample {i+1}/{len(param_samples)} done "
              f"(M={ps.get('M',2)}, Da={ps.get('Da',0.5)}, theta={ps.get('theta',0.1)})")

    return results
