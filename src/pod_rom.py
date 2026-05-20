"""Proper Orthogonal Decomposition (POD) based Reduced-Order Model.

Pipeline:
1. Collect FOM snapshots into a matrix U ∈ R^{Nr × N_snapshots}
2. Apply cylindrical weighting (r*dr) for proper L2 inner product
3. SVD → POD basis Φ_k (top-k modes capturing ≥ 99.9% energy)
4. Galerkin projection of the governing PDE onto the POD basis
5. Time-integrate the reduced ODE system
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import svd


class PODBasis:
    """Compute and store POD basis from snapshot data."""

    def __init__(self, r: np.ndarray, energy_threshold: float = 0.999):
        """
        Parameters
        ----------
        r : (Nr,) radial grid points
        energy_threshold : fraction of total energy to retain (default 99.9%)
        """
        self.r = r
        self.Nr = len(r)
        self.dr = r[1] - r[0] if len(r) > 1 else 1.0
        self.energy_threshold = energy_threshold

        # Cylindrical weight matrix W = diag(r * dr) for L2 inner product
        # W[0] uses r=dr/2 to avoid zero weight at axis
        w = r.copy()
        w[0] = 0.5 * self.dr  # regularise center weight
        self.weights = w * self.dr  # (Nr,)
        self.W_sqrt = np.sqrt(self.weights)  # (Nr,)

        # POD results (populated after fit)
        self.Phi = None         # (Nr, k) POD modes
        self.sigma = None       # (k,) singular values
        self.k = 0              # number of retained modes
        self.energy_captured = 0.0

    def fit(self, snapshots: np.ndarray) -> "PODBasis":
        """Compute POD basis from snapshot matrix.

        Parameters
        ----------
        snapshots : (Nr, N_s) matrix of velocity snapshots

        Returns
        -------
        self
        """
        Nr, Ns = snapshots.shape
        assert Nr == self.Nr, f"Snapshot rows {Nr} != grid size {self.Nr}"

        # Apply weighting: X = W^{1/2} * U
        X = snapshots * self.W_sqrt[:, None]

        # Thin SVD
        U_svd, sigma, Vt = svd(X, full_matrices=False)

        # Determine truncation level
        energy_total = np.sum(sigma ** 2)
        energy_cumsum = np.cumsum(sigma ** 2) / energy_total
        k = int(np.searchsorted(energy_cumsum, self.energy_threshold)) + 1
        k = max(k, 1)
        k = min(k, len(sigma))

        # Unweight to get physical POD modes
        self.Phi = U_svd[:, :k] / self.W_sqrt[:, None]
        self.sigma = sigma[:k]
        self.k = k
        self.energy_captured = energy_cumsum[k - 1]

        print(f"  POD: retained {k} modes, energy captured = {self.energy_captured:.6f}")
        return self

    def project(self, u: np.ndarray) -> np.ndarray:
        """Project full field u onto POD basis: a = Φᵀ W u.

        Parameters
        ----------
        u : (Nr,) or (Nr, N) velocity field(s)

        Returns
        -------
        a : (k,) or (k, N) reduced coefficients
        """
        W = np.diag(self.weights)
        if u.ndim == 1:
            return self.Phi.T @ W @ u
        else:
            return self.Phi.T @ W @ u

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Reconstruct full field from reduced coefficients: u ≈ Φ a.

        Parameters
        ----------
        a : (k,) or (k, N)

        Returns
        -------
        u : (Nr,) or (Nr, N)
        """
        if a.ndim == 1:
            return self.Phi @ a
        else:
            return self.Phi @ a


class GalerkinROM:
    """Galerkin-projected reduced ODE system for the Casson MHD problem.

    The semi-discrete equation in radial direction:
        alpha^2 * du/dt = L(u) + S(t)
    where L(u) includes the diffusion, MHD sink, porous sink terms.

    For the LINEAR part (MHD + Darcy sink + Newtonian-like diffusion):
        L_lin u = (1/r) d/dr(r du/dr) - (M^2 + 1/Da) u

    Galerkin projection:
        M_red * da/dt = A_red * a + S_red(t) + N_red(a, t)

    where M_red = Φᵀ W Φ (mass), A_red = Φᵀ W L Φ (stiffness),
    S_red = Φᵀ W s (source), N_red is nonlinear Casson correction.
    """

    def __init__(self, pod: PODBasis, params: dict):
        """
        Parameters
        ----------
        pod : fitted PODBasis
        params : dict with keys M, Da, theta, A0, e, a_b, omega_b, phi, alpha_sq, eps
        """
        self.pod = pod
        self.params = params
        self.k = pod.k
        self.Nr = pod.Nr
        self.r = pod.r
        self.dr = pod.dr

        # Build reduced matrices
        self._build_reduced_system()

    def _build_reduced_system(self):
        """Pre-compute M_red, A_red (linear stiffness)."""
        Phi = self.pod.Phi  # (Nr, k)
        W = np.diag(self.pod.weights)  # (Nr, Nr)
        Nr = self.Nr
        dr = self.dr
        r = self.r
        p = self.params
        M_h = p.get('M', 2.0)
        Da = p.get('Da', 0.5)
        alpha2 = p.get('alpha_sq', 1.0)

        # Mass matrix (reduced): M_red = Phi^T W Phi
        self.M_red = Phi.T @ W @ Phi  # (k, k)

        # Linear diffusion operator (Newtonian part):
        # L u = (1/r) d/dr(r du/dr) - (M^2 + 1/Da) u
        # Discrete: second-order finite-difference
        sink = M_h ** 2 + 1.0 / Da

        L = np.zeros((Nr, Nr))
        for i in range(1, Nr - 1):
            ri = r[i]
            r_ip = ri + 0.5 * dr
            r_im = ri - 0.5 * dr
            coeff_ip = r_ip / (ri * dr ** 2)
            coeff_im = r_im / (ri * dr ** 2)
            L[i, i - 1] = coeff_im
            L[i, i] = -(coeff_ip + coeff_im) - sink
            L[i, i + 1] = coeff_ip

        # BC: symmetry at r=0 => u[0] = u[1] => row 0: L[0,:] = L[1,:]
        L[0, :] = L[1, :]
        # BC: u(R) = 0 => row Nr-1 is zero (homogeneous)
        L[-1, :] = 0.0

        # Stiffness: A_red = Phi^T W L Phi / alpha^2
        self.A_red = Phi.T @ W @ L @ Phi / alpha2  # (k, k)

        # Source projection vector: s_red = Phi^T W * ones / alpha^2
        # (the source S(t) is spatially uniform, multiplied by scalar)
        ones = np.ones(Nr)
        ones[-1] = 0.0  # BC node
        self.s_vec = (Phi.T @ W @ ones) / alpha2  # (k,)

        # For the nonlinear Casson correction we'll use hyper-reduction
        # (direct evaluation in physical space, then project)

    def source(self, t: float) -> float:
        """S(t) = A0*(1 + e*cos(t)) + a_b*cos(omega_b*t + phi)."""
        p = self.params
        A0 = p.get('A0', 1.0)
        e = p.get('e', 0.5)
        a_b = p.get('a_b', 0.5)
        omega_b = p.get('omega_b', 1.0)
        phi = p.get('phi', 0.0)
        return A0 * (1.0 + e * np.cos(t)) + a_b * np.cos(omega_b * t + phi)

    def casson_correction(self, a: np.ndarray) -> np.ndarray:
        """Nonlinear Casson viscosity correction in reduced space.

        The full Casson term introduces extra effective viscosity beyond the
        Newtonian (mu=1) part already in A_red.  We evaluate:
            N(u) = (1/r) d/dr(r * (mu_eff - 1) * du/dr)
        in physical space, then project.
        """
        Phi = self.pod.Phi
        W = np.diag(self.pod.weights)
        r = self.r
        dr = self.dr
        Nr = self.Nr
        p = self.params
        theta = p.get('theta', 0.1)
        eps = p.get('eps', 1e-3)
        alpha2 = p.get('alpha_sq', 1.0)

        # Reconstruct velocity
        u = Phi @ a  # (Nr,)

        # Shear rate
        gamma_dot = np.zeros(Nr)
        gamma_dot[1:-1] = -(u[2:] - u[:-2]) / (2.0 * dr)
        gamma_dot[0] = 0.0
        gamma_dot[-1] = -(u[-1] - u[-2]) / dr

        # Casson effective viscosity
        g = np.abs(gamma_dot) + eps
        sq_g = np.sqrt(g)
        sq_th = np.sqrt(theta)
        mu_eff = (sq_th + sq_g) ** 2 / g
        mu_extra = mu_eff - 1.0  # extra beyond Newtonian

        # Nonlinear diffusion term: (1/r) d/dr(r * mu_extra * du/dr)
        N_phys = np.zeros(Nr)
        for i in range(1, Nr - 1):
            ri = r[i]
            r_ip = ri + 0.5 * dr
            r_im = ri - 0.5 * dr
            mu_ip = 0.5 * (mu_extra[i] + mu_extra[i + 1])
            mu_im = 0.5 * (mu_extra[i] + mu_extra[i - 1])
            du_ip = (u[i + 1] - u[i]) / dr
            du_im = (u[i] - u[i - 1]) / dr
            N_phys[i] = (r_ip * mu_ip * du_ip - r_im * mu_im * du_im) / (ri * dr)

        # Project
        return (Phi.T @ W @ N_phys) / alpha2  # (k,)

    def rhs(self, t: float, a: np.ndarray) -> np.ndarray:
        """Right-hand side of the reduced ODE: da/dt = M_red^{-1} (A_red a + S(t)*s_vec + N(a))."""
        S_t = self.source(t)
        linear = self.A_red @ a + S_t * self.s_vec
        nonlinear = self.casson_correction(a)
        rhs_full = linear + nonlinear
        # Solve M_red * da/dt = rhs_full
        da_dt = np.linalg.solve(self.M_red, rhs_full)
        return da_dt

    def solve(self, t_span: tuple[float, float], t_eval: np.ndarray | None = None,
              a0: np.ndarray | None = None) -> dict:
        """Integrate the reduced ODE.

        Parameters
        ----------
        t_span : (t_start, t_end)
        t_eval : time points for output
        a0 : initial reduced coefficients (default zeros)

        Returns
        -------
        dict with 'a' (k, Nt), 't' (Nt,), 'U_rom' (Nr, Nt)
        """
        if a0 is None:
            a0 = np.zeros(self.k)

        sol = solve_ivp(self.rhs, t_span, a0, t_eval=t_eval,
                        method='RK45', rtol=1e-5, atol=1e-7,
                        max_step=0.1)

        a_sol = sol.y  # (k, Nt)
        t_sol = sol.t  # (Nt,)
        U_rom = self.pod.reconstruct(a_sol)  # (Nr, Nt)

        return {'a': a_sol, 't': t_sol, 'U_rom': U_rom}


def build_pod_rom(fom_results: dict, params: dict,
                  energy_threshold: float = 0.999) -> tuple[PODBasis, GalerkinROM]:
    """Convenience function: build POD basis from FOM results, then create Galerkin ROM.

    Parameters
    ----------
    fom_results : dict from generate_snapshots() with 'r', 'U_all'
    params : flow parameters dict
    energy_threshold : POD energy cutoff

    Returns
    -------
    (pod, rom) : PODBasis and GalerkinROM instances
    """
    r = fom_results['r']

    # Collect all snapshots into one matrix
    snapshot_list = fom_results['U_all']
    all_snapshots = np.hstack(snapshot_list)  # (Nr, total_time_steps)

    # Build POD
    pod = PODBasis(r, energy_threshold=energy_threshold)
    pod.fit(all_snapshots)

    # Build Galerkin ROM
    rom = GalerkinROM(pod, params)

    return pod, rom
