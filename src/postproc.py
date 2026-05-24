"""Post-processing: compute derived quantities, generate tables and plots.

Outputs:
- Tables (CSV): WSS, flow resistance, plug core radius vs parameters
- Figures (PNG): velocity profiles, WSS vs M/Da, resistance vs Da/beta,
                 plug radius vs theta, POD energy, PINN loss curves
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt


# ======================================================================
# TABLE GENERATION
# ======================================================================

def make_wss_table(fom_results: dict, param_samples: list[dict]) -> pd.DataFrame:
    """Create WSS table: time-averaged WSS for each parameter combination."""
    rows = []
    t_arr = fom_results['t']
    # Use last cardiac cycle for time-averaging
    Nt = len(t_arr)
    last_cycle_start = max(0, Nt // 2)

    for i, ps in enumerate(param_samples):
        wss = fom_results['wss'][i]
        wss_avg = np.mean(wss[last_cycle_start:])
        wss_max = np.max(wss[last_cycle_start:])
        rows.append({
            'M': ps.get('M', 2.0),
            'Da': ps.get('Da', 0.5),
            'theta': ps.get('theta', 0.1),
            'delta': ps.get('delta', 0.3),
            'beta_half': ps.get('beta_half', np.pi / 6),
            'WSS_avg': wss_avg,
            'WSS_max': wss_max,
        })
    return pd.DataFrame(rows)


def make_resistance_table(fom_results: dict, param_samples: list[dict]) -> pd.DataFrame:
    """Create flow resistance table."""
    rows = []
    t_arr = fom_results['t']
    Nt = len(t_arr)
    last_cycle_start = max(0, Nt // 2)

    for i, ps in enumerate(param_samples):
        Lambda = fom_results['Lambda'][i]
        Lambda_avg = np.mean(Lambda[last_cycle_start:])
        rows.append({
            'M': ps.get('M', 2.0),
            'Da': ps.get('Da', 0.5),
            'theta': ps.get('theta', 0.1),
            'delta': ps.get('delta', 0.3),
            'beta_half': ps.get('beta_half', np.pi / 6),
            'Resistance_avg': Lambda_avg,
        })
    return pd.DataFrame(rows)


def make_plug_core_table(fom_results: dict, param_samples: list[dict]) -> pd.DataFrame:
    """Create plug core radius table."""
    rows = []
    t_arr = fom_results['t']
    Nt = len(t_arr)
    last_cycle_start = max(0, Nt // 2)

    for i, ps in enumerate(param_samples):
        Rp = fom_results['Rp'][i]
        Rp_avg = np.mean(Rp[last_cycle_start:])
        Rp_max = np.max(Rp[last_cycle_start:])
        rows.append({
            'M': ps.get('M', 2.0),
            'Da': ps.get('Da', 0.5),
            'theta': ps.get('theta', 0.1),
            'delta': ps.get('delta', 0.3),
            'Rp_avg': Rp_avg,
            'Rp_max': Rp_max,
        })
    return pd.DataFrame(rows)


def make_comparison_table(fom_results: dict, pinn_trainer, param_samples: list[dict],
                          pod_basis) -> pd.DataFrame:
    """Compare FOM vs PINN-ROM predictions at specific time instants."""
    rows = []
    t_arr = fom_results['t']
    r = fom_results['r']
    # Pick time instants in last cycle
    Nt = len(t_arr)
    t_indices = [Nt // 2, 3 * Nt // 4, Nt - 1]

    for i, ps in enumerate(param_samples):
        U_fom = fom_results['U_all'][i]
        U_pinn = pinn_trainer.predict_velocity(t_arr, ps)

        for ti in t_indices:
            u_fom = U_fom[:, ti]
            u_pinn = U_pinn[:, ti]
            # L2 relative error
            err = np.linalg.norm(u_fom - u_pinn) / (np.linalg.norm(u_fom) + 1e-12)
            # Centerline velocity
            rows.append({
                'M': ps.get('M', 2.0),
                'Da': ps.get('Da', 0.5),
                'theta': ps.get('theta', 0.1),
                't': t_arr[ti],
                'u_center_FOM': u_fom[0],
                'u_center_PINN': u_pinn[0],
                'L2_rel_error': err,
            })
    return pd.DataFrame(rows)


# ======================================================================
# FIGURE GENERATION
# ======================================================================

def plot_velocity_profiles(fom_results: dict, pinn_trainer, param_samples: list[dict],
                           save_dir: str = 'results/figs') -> None:
    """Plot velocity profiles u(r) at different times: FOM vs PINN-ROM."""
    os.makedirs(save_dir, exist_ok=True)
    r = fom_results['r']
    t_arr = fom_results['t']
    Nt = len(t_arr)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Pick first sample, several time instants
    ps = param_samples[0]
    U_fom = fom_results['U_all'][0]
    U_pinn = pinn_trainer.predict_velocity(t_arr, ps)

    t_indices = [Nt // 4, Nt // 2, 3 * Nt // 4, Nt - 1]
    titles = ['t = T/4', 't = T/2', 't = 3T/4', 't = T']

    for ax, ti, title in zip(axes, t_indices, titles):
        ax.plot(r, U_fom[:, ti], 'b-', linewidth=2, label='FOM')
        ax.plot(r, U_pinn[:, ti], 'r--', linewidth=2, label='PINN-ROM')
        ax.set_xlabel('r (radial position)')
        ax.set_ylabel('u (velocity)')
        ax.set_title(f'Velocity Profile at {title}\n'
                     f'(M={ps.get("M",2)}, Da={ps.get("Da",0.5)}, θ={ps.get("theta",0.1)})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'velocity_profiles.png'), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/velocity_profiles.png")


def plot_wss_vs_M(fom_results: dict, param_samples: list[dict],
                  save_dir: str = 'results/figs') -> None:
    """Plot WSS vs Hartmann number M."""
    os.makedirs(save_dir, exist_ok=True)

    df = make_wss_table(fom_results, param_samples)
    # Group by M
    if df['M'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        grouped = df.groupby('M')['WSS_avg'].mean()
        ax.plot(grouped.index, grouped.values, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Hartmann Number (M)', fontsize=12)
        ax.set_ylabel('Time-averaged Wall Shear Stress', fontsize=12)
        ax.set_title('WSS vs Hartmann Number', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'wss_vs_M.png'), dpi=150)
        plt.close()
        print(f"  Saved: {save_dir}/wss_vs_M.png")


def plot_resistance_vs_Da(fom_results: dict, param_samples: list[dict],
                          save_dir: str = 'results/figs') -> None:
    """Plot flow resistance vs Darcy number."""
    os.makedirs(save_dir, exist_ok=True)

    df = make_resistance_table(fom_results, param_samples)
    if df['Da'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        grouped = df.groupby('Da')['Resistance_avg'].mean()
        ax.plot(grouped.index, grouped.values, 'rs-', linewidth=2, markersize=8)
        ax.set_xlabel('Darcy Number (Da)', fontsize=12)
        ax.set_ylabel('Time-averaged Flow Resistance (Λ)', fontsize=12)
        ax.set_title('Flow Resistance vs Darcy Number', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'resistance_vs_Da.png'), dpi=150)
        plt.close()
        print(f"  Saved: {save_dir}/resistance_vs_Da.png")


def plot_plug_radius_vs_theta(fom_results: dict, param_samples: list[dict],
                              save_dir: str = 'results/figs') -> None:
    """Plot plug core radius vs yield stress theta."""
    os.makedirs(save_dir, exist_ok=True)

    df = make_plug_core_table(fom_results, param_samples)
    if df['theta'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        grouped = df.groupby('theta')['Rp_avg'].mean()
        ax.plot(grouped.index, grouped.values, 'g^-', linewidth=2, markersize=8)
        ax.set_xlabel('Yield Stress (θ)', fontsize=12)
        ax.set_ylabel('Time-averaged Plug Core Radius (Rp)', fontsize=12)
        ax.set_title('Plug Core Radius vs Casson Yield Stress', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'plug_radius_vs_theta.png'), dpi=150)
        plt.close()
        print(f"  Saved: {save_dir}/plug_radius_vs_theta.png")


def plot_pod_energy(pod_basis, save_dir: str = 'results/figs') -> None:
    """Plot POD singular value decay and cumulative energy."""
    os.makedirs(save_dir, exist_ok=True)
    sigma = pod_basis.sigma
    energy = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(range(1, len(sigma) + 1), sigma, 'ko-', markersize=6)
    ax1.set_xlabel('Mode index', fontsize=12)
    ax1.set_ylabel('Singular value σ_k', fontsize=12)
    ax1.set_title('POD Singular Value Decay', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(energy) + 1), energy * 100, 'b.-', markersize=8)
    ax2.axhline(99.9, color='r', linestyle='--', label='99.9% threshold')
    ax2.set_xlabel('Number of modes (k)', fontsize=12)
    ax2.set_ylabel('Cumulative energy (%)', fontsize=12)
    ax2.set_title('POD Cumulative Energy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pod_energy.png'), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/pod_energy.png")


def plot_pinn_loss(losses_history: dict, save_dir: str = 'results/figs') -> None:
    """Plot PINN training loss curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(losses_history['total']) + 1)

    ax.semilogy(epochs, losses_history['total'], 'k-', label='Total', linewidth=2)
    ax.semilogy(epochs, losses_history['data'], 'b--', label='Data', linewidth=1.5)
    ax.semilogy(epochs, losses_history['physics'], 'r--', label='Physics', linewidth=1.5)
    ax.semilogy(epochs, losses_history['ic'], 'g--', label='IC', linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('PINN-ROM Training Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/pinn_loss.png")


def plot_wss_vs_beta(fom_results: dict, param_samples: list[dict],
                     save_dir: str = 'results/figs') -> None:
    """Plot WSS and resistance vs bifurcation half-angle."""
    os.makedirs(save_dir, exist_ok=True)

    df_wss = make_wss_table(fom_results, param_samples)
    df_res = make_resistance_table(fom_results, param_samples)

    if df_wss['beta_half'].nunique() > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        grp_wss = df_wss.groupby('beta_half')['WSS_avg'].mean()
        ax1.plot(np.degrees(grp_wss.index), grp_wss.values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Half Bifurcation Angle β/2 (degrees)', fontsize=12)
        ax1.set_ylabel('WSS (time-averaged)', fontsize=12)
        ax1.set_title('WSS vs Bifurcation Angle', fontsize=14)
        ax1.grid(True, alpha=0.3)

        grp_res = df_res.groupby('beta_half')['Resistance_avg'].mean()
        ax2.plot(np.degrees(grp_res.index), grp_res.values, 'rs-', linewidth=2, markersize=8)
        ax2.set_xlabel('Half Bifurcation Angle β/2 (degrees)', fontsize=12)
        ax2.set_ylabel('Resistance (time-averaged)', fontsize=12)
        ax2.set_title('Resistance vs Bifurcation Angle', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'wss_resistance_vs_beta.png'), dpi=150)
        plt.close()
        print(f"  Saved: {save_dir}/wss_resistance_vs_beta.png")


# ======================================================================
# SAVE ALL RESULTS
# ======================================================================

def save_all_results(fom_results: dict, param_samples: list[dict],
                     pod_basis, pinn_trainer,
                     output_dir: str = 'results') -> None:
    """Generate and save all tables and figures."""
    tables_dir = os.path.join(output_dir, 'tables')
    figs_dir = os.path.join(output_dir, 'figs')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    print("\n=== Generating Tables ===")
    # Tables
    wss_df = make_wss_table(fom_results, param_samples)
    wss_df.to_csv(os.path.join(tables_dir, 'wss.csv'), index=False)
    print(f"  WSS Table:\n{wss_df.to_string(index=False)}\n")

    res_df = make_resistance_table(fom_results, param_samples)
    res_df.to_csv(os.path.join(tables_dir, 'resistance.csv'), index=False)
    print(f"  Resistance Table:\n{res_df.to_string(index=False)}\n")

    plug_df = make_plug_core_table(fom_results, param_samples)
    plug_df.to_csv(os.path.join(tables_dir, 'plug_core.csv'), index=False)
    print(f"  Plug Core Table:\n{plug_df.to_string(index=False)}\n")

    comp_df = make_comparison_table(fom_results, pinn_trainer, param_samples, pod_basis)
    comp_df.to_csv(os.path.join(tables_dir, 'fom_vs_pinn_comparison.csv'), index=False)
    print(f"  FOM vs PINN Comparison:\n{comp_df.to_string(index=False)}\n")

    print("\n=== Generating Figures ===")
    # Figures
    plot_velocity_profiles(fom_results, pinn_trainer, param_samples, save_dir=figs_dir)
    plot_wss_vs_M(fom_results, param_samples, save_dir=figs_dir)
    plot_resistance_vs_Da(fom_results, param_samples, save_dir=figs_dir)
    plot_plug_radius_vs_theta(fom_results, param_samples, save_dir=figs_dir)
    plot_wss_vs_beta(fom_results, param_samples, save_dir=figs_dir)
    plot_pod_energy(pod_basis, save_dir=figs_dir)
    plot_pinn_loss(pinn_trainer.losses_history, save_dir=figs_dir)

    print(f"\n✓ All results saved to '{output_dir}/'")
