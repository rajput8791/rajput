"""Physics-Informed Neural Network for Reduced-Order Model (PINN-ROM).

The neural network learns the mapping:
    NN(t, μ) → a(t; μ) ∈ R^k

where a are the POD reduced coefficients. The loss function combines:
1. Data loss:   ||a_pred - a_data||^2  (from FOM snapshots or paper tables)
2. Physics loss: residual of the reduced ODE
3. IC loss:     ||a(0, μ) - 0||^2  (zero initial condition)

Training uses the tables from the paper (velocity profiles, WSS, etc.)
projected onto the POD basis.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PINNROM(nn.Module):
    """Multi-layer perceptron for PINN-ROM surrogate.

    Input:  (t, M, Da, theta, delta, beta_half) — normalized
    Output: a ∈ R^k (POD reduced coefficients)
    """

    def __init__(self, n_inputs: int = 6, n_modes: int = 5,
                 hidden_layers: list[int] | None = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 128, 128, 64]

        layers = []
        in_dim = n_inputs
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_modes))
        self.net = nn.Sequential(*layers)
        self.n_modes = n_modes
        self.n_inputs = n_inputs

        # Input normalization parameters (set during training)
        self.register_buffer('x_mean', torch.zeros(n_inputs))
        self.register_buffer('x_std', torch.ones(n_inputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with input normalization.

        Parameters
        ----------
        x : (batch, n_inputs) — [t, M, Da, theta, delta, beta_half]
        """
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        return self.net(x_norm)


class PINNROMTrainer:
    """Training loop for the PINN-ROM model."""

    def __init__(self, model: PINNROM, pod_basis, rom_matrices: dict,
                 device: str = 'cpu'):
        """
        Parameters
        ----------
        model : PINNROM network
        pod_basis : PODBasis instance (for projection/reconstruction)
        rom_matrices : dict with 'A_red', 'M_red', 's_vec', and param ranges
        device : 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.pod = pod_basis
        self.rom = rom_matrices
        self.device = device
        self.losses_history = {'total': [], 'data': [], 'physics': [], 'ic': []}

    def prepare_training_data(self, fom_results: dict,
                              param_samples: list[dict]) -> dict:
        """Prepare data tensors from FOM results.

        Projects FOM snapshots onto POD basis to get ground-truth
        reduced coefficients a(t; μ).
        """
        r = fom_results['r']
        t_arr = fom_results['t']
        Nt = len(t_arr)

        all_inputs = []    # (t, M, Da, theta, delta, beta_half)
        all_targets = []   # a_k (reduced coefficients)

        for i, ps in enumerate(param_samples):
            U_i = fom_results['U_all'][i]  # (Nr, Nt)
            # Project each snapshot
            a_i = self.pod.project(U_i)  # (k, Nt)

            for n in range(Nt):
                inp = [
                    t_arr[n],
                    ps.get('M', 2.0),
                    ps.get('Da', 0.5),
                    ps.get('theta', 0.1),
                    ps.get('delta', 0.3),
                    ps.get('beta_half', np.pi / 6),
                ]
                all_inputs.append(inp)
                all_targets.append(a_i[:, n])

        inputs = torch.tensor(np.array(all_inputs), dtype=torch.float32)
        targets = torch.tensor(np.array(all_targets), dtype=torch.float32)

        # Set normalization
        self.model.x_mean = inputs.mean(dim=0).to(self.device)
        self.model.x_std = inputs.std(dim=0).to(self.device)

        return {
            'inputs': inputs.to(self.device),
            'targets': targets.to(self.device),
            't_arr': t_arr,
            'param_samples': param_samples,
        }

    def data_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and true reduced coefficients."""
        a_pred = self.model(inputs)
        return torch.mean((a_pred - targets) ** 2)

    def physics_loss(self, collocation_points: torch.Tensor,
                     A_red: torch.Tensor, M_red: torch.Tensor,
                     s_vec: torch.Tensor) -> torch.Tensor:
        """PDE residual loss in reduced space.

        Residual = M_red * da/dt - A_red * a - S(t) * s_vec ≈ 0

        (Casson nonlinear correction omitted here for efficiency;
         it's implicitly learned from the data loss.)
        """
        collocation_points.requires_grad_(True)
        a_pred = self.model(collocation_points)  # (batch, k)

        # Time derivative via autograd
        # t is the first column of collocation_points
        da_dt = torch.zeros_like(a_pred)
        for j in range(a_pred.shape[1]):
            grad = torch.autograd.grad(
                a_pred[:, j].sum(), collocation_points,
                create_graph=True, retain_graph=True
            )[0]
            da_dt[:, j] = grad[:, 0]  # derivative w.r.t. t (first input)

        # Source term: S(t) = A0*(1 + e*cos(t)) + a_b*cos(omega_b*t + phi)
        t_col = collocation_points[:, 0]
        params = self.rom.get('params', {})
        A0 = params.get('A0', 1.0)
        e = params.get('e', 0.5)
        a_b = params.get('a_b', 0.5)
        omega_b = params.get('omega_b', 1.0)
        phi = params.get('phi', 0.0)

        S_t = A0 * (1.0 + e * torch.cos(t_col)) + a_b * torch.cos(omega_b * t_col + phi)

        # Residual: M_red @ da/dt - A_red @ a - S(t) * s_vec
        # shape: (batch, k)
        lhs = torch.matmul(da_dt, M_red.T)  # (batch, k) @ (k, k)^T
        rhs = torch.matmul(a_pred, A_red.T) + S_t.unsqueeze(1) * s_vec.unsqueeze(0)
        residual = lhs - rhs

        return torch.mean(residual ** 2)

    def ic_loss(self, param_samples: list[dict]) -> torch.Tensor:
        """Initial condition loss: a(t=0, μ) = 0 for all μ."""
        ic_inputs = []
        for ps in param_samples:
            inp = [
                0.0,  # t = 0
                ps.get('M', 2.0),
                ps.get('Da', 0.5),
                ps.get('theta', 0.1),
                ps.get('delta', 0.3),
                ps.get('beta_half', np.pi / 6),
            ]
            ic_inputs.append(inp)
        ic_tensor = torch.tensor(np.array(ic_inputs), dtype=torch.float32).to(self.device)
        a_pred_ic = self.model(ic_tensor)
        return torch.mean(a_pred_ic ** 2)

    def train(self, train_data: dict, n_epochs: int = 2000,
              lr: float = 1e-3, w_data: float = 1.0,
              w_physics: float = 0.1, w_ic: float = 1.0,
              n_collocation: int = 500, print_every: int = 200) -> dict:
        """Full training loop.

        Parameters
        ----------
        train_data : output of prepare_training_data()
        n_epochs : number of training epochs
        lr : learning rate
        w_data, w_physics, w_ic : loss weights
        n_collocation : number of collocation points for physics loss
        print_every : print frequency

        Returns
        -------
        losses_history dict
        """
        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=1e-6)

        inputs = train_data['inputs']
        targets = train_data['targets']
        param_samples = train_data['param_samples']
        t_arr = train_data['t_arr']

        # ROM matrices as tensors
        A_red = torch.tensor(self.rom['A_red'], dtype=torch.float32).to(self.device)
        M_red = torch.tensor(self.rom['M_red'], dtype=torch.float32).to(self.device)
        s_vec = torch.tensor(self.rom['s_vec'], dtype=torch.float32).to(self.device)

        t_min, t_max = t_arr[0], t_arr[-1]

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Data loss
            L_data = self.data_loss(inputs, targets)

            # Physics loss (random collocation points)
            t_coll = t_min + (t_max - t_min) * torch.rand(n_collocation, 1)
            # Random parameter samples for collocation
            param_coll = []
            for _ in range(n_collocation):
                ps = param_samples[np.random.randint(len(param_samples))]
                param_coll.append([
                    ps.get('M', 2.0),
                    ps.get('Da', 0.5),
                    ps.get('theta', 0.1),
                    ps.get('delta', 0.3),
                    ps.get('beta_half', np.pi / 6),
                ])
            param_tensor = torch.tensor(np.array(param_coll), dtype=torch.float32)
            coll_points = torch.cat([t_coll, param_tensor], dim=1).to(self.device)

            L_physics = self.physics_loss(coll_points, A_red, M_red, s_vec)

            # IC loss
            L_ic = self.ic_loss(param_samples)

            # Total loss
            loss = w_data * L_data + w_physics * L_physics + w_ic * L_ic
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            # Record
            self.losses_history['total'].append(loss.item())
            self.losses_history['data'].append(L_data.item())
            self.losses_history['physics'].append(L_physics.item())
            self.losses_history['ic'].append(L_ic.item())

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:5d} | Loss: {loss.item():.6e} "
                      f"(data={L_data.item():.3e}, phys={L_physics.item():.3e}, "
                      f"ic={L_ic.item():.3e})")

        return self.losses_history

    def predict(self, t: np.ndarray, params: dict) -> np.ndarray:
        """Predict reduced coefficients for given time array and parameters.

        Parameters
        ----------
        t : (Nt,) time points
        params : dict with M, Da, theta, delta, beta_half

        Returns
        -------
        a_pred : (k, Nt) reduced coefficients
        """
        self.model.eval()
        Nt = len(t)
        inputs = np.zeros((Nt, self.model.n_inputs))
        inputs[:, 0] = t
        inputs[:, 1] = params.get('M', 2.0)
        inputs[:, 2] = params.get('Da', 0.5)
        inputs[:, 3] = params.get('theta', 0.1)
        inputs[:, 4] = params.get('delta', 0.3)
        inputs[:, 5] = params.get('beta_half', np.pi / 6)

        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            a_pred = self.model(x).cpu().numpy()  # (Nt, k)

        return a_pred.T  # (k, Nt)

    def predict_velocity(self, t: np.ndarray, params: dict) -> np.ndarray:
        """Predict full velocity field u(r, t; μ) via POD reconstruction.

        Returns
        -------
        U_pred : (Nr, Nt)
        """
        a_pred = self.predict(t, params)  # (k, Nt)
        U_pred = self.pod.reconstruct(a_pred)  # (Nr, Nt)
        return U_pred


def build_pinn_rom(pod_basis, galerkin_rom, fom_results: dict,
                   param_samples: list[dict],
                   n_epochs: int = 2000, lr: float = 1e-3) -> PINNROMTrainer:
    """Convenience function to build and train a PINN-ROM model.

    Parameters
    ----------
    pod_basis : fitted PODBasis
    galerkin_rom : GalerkinROM (for reduced matrices)
    fom_results : dict from generate_snapshots()
    param_samples : list of parameter dicts
    n_epochs : training epochs
    lr : learning rate

    Returns
    -------
    trainer : trained PINNROMTrainer
    """
    k = pod_basis.k
    model = PINNROM(n_inputs=6, n_modes=k, hidden_layers=[64, 128, 128, 64])

    rom_matrices = {
        'A_red': galerkin_rom.A_red,
        'M_red': galerkin_rom.M_red,
        's_vec': galerkin_rom.s_vec,
        'params': galerkin_rom.params,
    }

    trainer = PINNROMTrainer(model, pod_basis, rom_matrices, device='cpu')
    train_data = trainer.prepare_training_data(fom_results, param_samples)

    print("\n--- PINN-ROM Training ---")
    trainer.train(train_data, n_epochs=n_epochs, lr=lr)

    return trainer
