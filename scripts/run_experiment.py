#!/usr/bin/env python
"""Main driver script for the Pulsatile MHD Casson Flow ROM + PINN-ROM experiment.

Usage:
    python scripts/run_experiment.py          # Full run (~5-10 min CPU)
    python scripts/run_experiment.py --quick  # Quick smoke test (~1-2 min)

This script:
1. Generates FOM (Full Order Model) snapshots for multiple parameter combinations
2. Builds a POD (Proper Orthogonal Decomposition) reduced basis
3. Constructs and integrates a Galerkin ROM
4. Trains a PINN-ROM neural network surrogate
5. Generates comparison tables and publication-quality figures
"""

import sys
import os
import time
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.casson_fom import FlowParams, FOMSolver, generate_snapshots
from src.pod_rom import PODBasis, GalerkinROM, build_pod_rom
from src.pinn_rom import PINNROM, PINNROMTrainer, build_pinn_rom
from src.postproc import save_all_results


def define_parameter_samples(quick: bool = False) -> list[dict]:
    """Define parameter combinations for snapshot generation.

    Varies M, Da, theta, delta, beta_half to capture the parametric space.
    """
    if quick:
        # Minimal set for quick testing
        samples = [
            {'M': 1.0, 'Da': 0.5, 'theta': 0.05, 'delta': 0.2, 'beta_half': np.pi/6},
            {'M': 2.0, 'Da': 0.5, 'theta': 0.1,  'delta': 0.3, 'beta_half': np.pi/6},
            {'M': 3.0, 'Da': 0.3, 'theta': 0.15, 'delta': 0.4, 'beta_half': np.pi/4},
            {'M': 2.0, 'Da': 1.0, 'theta': 0.1,  'delta': 0.3, 'beta_half': np.pi/5},
        ]
    else:
        # Full parametric sweep
        M_values = [0.5, 1.0, 2.0, 3.0, 4.0]
        Da_values = [0.1, 0.3, 0.5, 1.0, 2.0]
        theta_values = [0.0, 0.05, 0.1, 0.15, 0.2]
        delta_values = [0.1, 0.2, 0.3, 0.4]
        beta_values = [np.pi/8, np.pi/6, np.pi/5, np.pi/4]

        samples = []
        # Latin-hypercube-style sampling (not full grid — too expensive)
        np.random.seed(42)
        n_samples = 20

        for _ in range(n_samples):
            samples.append({
                'M': float(np.random.choice(M_values)),
                'Da': float(np.random.choice(Da_values)),
                'theta': float(np.random.choice(theta_values)),
                'delta': float(np.random.choice(delta_values)),
                'beta_half': float(np.random.choice(beta_values)),
            })

        # Ensure some boundary cases are included
        samples.append({'M': 0.5, 'Da': 2.0, 'theta': 0.0, 'delta': 0.1, 'beta_half': np.pi/8})
        samples.append({'M': 4.0, 'Da': 0.1, 'theta': 0.2, 'delta': 0.4, 'beta_half': np.pi/4})

    return samples


def main():
    parser = argparse.ArgumentParser(description='Pulsatile MHD Casson Flow: ROM + PINN-ROM')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with fewer samples and epochs')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for tables and figures')
    parser.add_argument('--epochs', type=int, default=None,
                        help='PINN training epochs (default: 500 quick, 2000 full)')
    parser.add_argument('--Nr', type=int, default=60,
                        help='Number of radial grid points (default: 60)')
    parser.add_argument('--Nt', type=int, default=150,
                        help='Number of time steps per cycle (default: 150)')
    args = parser.parse_args()

    quick = args.quick
    output_dir = args.output_dir
    Nr = args.Nr
    Nt = args.Nt if not quick else 100
    n_cycles = 2 if not quick else 1
    n_epochs = args.epochs or (500 if quick else 2000)

    print("=" * 70)
    print("  PULSATILE MHD CASSON FLOW — POD-ROM + PINN-ROM")
    print("  (Ponalagusamy & Priyadharshini, AMC 2018)")
    print("=" * 70)
    print(f"\n  Mode: {'QUICK' if quick else 'FULL'}")
    print(f"  Nr={Nr}, Nt={Nt}, n_cycles={n_cycles}, PINN epochs={n_epochs}")

    # ------------------------------------------------------------------
    # Step 1: Define parameter samples
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 1: Parameter sampling")
    print("=" * 70)
    param_samples = define_parameter_samples(quick=quick)
    print(f"  Number of parameter samples: {len(param_samples)}")
    for i, ps in enumerate(param_samples[:5]):
        print(f"    Sample {i+1}: M={ps['M']:.1f}, Da={ps['Da']:.1f}, "
              f"θ={ps['theta']:.2f}, δ={ps['delta']:.2f}, β/2={np.degrees(ps['beta_half']):.1f}°")
    if len(param_samples) > 5:
        print(f"    ... and {len(param_samples)-5} more")

    # ------------------------------------------------------------------
    # Step 2: Generate FOM snapshots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 2: Full-Order Model (FOM) — Generating snapshots")
    print("=" * 70)
    t0 = time.time()
    fom_results = generate_snapshots(param_samples, Nr=Nr, Nt=Nt, n_cycles=n_cycles)
    t_fom = time.time() - t0
    print(f"\n  FOM completed in {t_fom:.1f} seconds")
    print(f"  Snapshot matrix size: ({Nr}, {len(param_samples) * (Nt * n_cycles + 1)})")

    # ------------------------------------------------------------------
    # Step 3: POD-ROM
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 3: POD Basis + Galerkin ROM")
    print("=" * 70)
    t0 = time.time()

    # Use parameters from first sample as representative for Galerkin ROM
    representative_params = {
        'M': 2.0, 'Da': 0.5, 'theta': 0.1, 'A0': 1.0,
        'e': 0.5, 'a_b': 0.5, 'omega_b': 1.0, 'phi': 0.0,
        'alpha_sq': 1.0, 'eps': 1e-3,
    }
    pod_basis, galerkin_rom = build_pod_rom(fom_results, representative_params,
                                            energy_threshold=0.999)
    t_pod = time.time() - t0
    print(f"  POD + Galerkin ROM built in {t_pod:.1f} seconds")
    print(f"  Retained modes: {pod_basis.k}")

    # Quick validation: run Galerkin ROM
    print("\n  Running Galerkin ROM validation...")
    t_span = (fom_results['t'][0], fom_results['t'][-1])
    rom_sol = galerkin_rom.solve(t_span, t_eval=fom_results['t'])
    U_rom = rom_sol['U_rom']
    # Relative error vs first FOM sample
    U_fom_0 = fom_results['U_all'][0]
    err_rom = np.linalg.norm(U_fom_0 - U_rom) / (np.linalg.norm(U_fom_0) + 1e-12)
    print(f"  Galerkin ROM relative L2 error (vs sample 1): {err_rom:.4e}")

    # ------------------------------------------------------------------
    # Step 4: PINN-ROM
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 4: PINN-ROM Training")
    print("=" * 70)
    t0 = time.time()
    pinn_trainer = build_pinn_rom(pod_basis, galerkin_rom, fom_results,
                                   param_samples, n_epochs=n_epochs, lr=1e-3)
    t_pinn = time.time() - t0
    print(f"\n  PINN-ROM training completed in {t_pinn:.1f} seconds")

    # Validation
    print("\n  PINN-ROM validation (on training samples):")
    for i in range(min(3, len(param_samples))):
        ps = param_samples[i]
        U_pinn = pinn_trainer.predict_velocity(fom_results['t'], ps)
        U_fom_i = fom_results['U_all'][i]
        err_pinn = np.linalg.norm(U_fom_i - U_pinn) / (np.linalg.norm(U_fom_i) + 1e-12)
        print(f"    Sample {i+1} (M={ps['M']:.1f}, Da={ps['Da']:.1f}): "
              f"rel error = {err_pinn:.4e}")

    # ------------------------------------------------------------------
    # Step 5: Post-processing
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 5: Post-processing — Tables & Figures")
    print("=" * 70)
    save_all_results(fom_results, param_samples, pod_basis, pinn_trainer,
                     output_dir=output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  FOM solve time:       {t_fom:.1f} s")
    print(f"  POD + ROM build time: {t_pod:.1f} s")
    print(f"  PINN training time:   {t_pinn:.1f} s")
    print(f"  Total time:           {t_fom + t_pod + t_pinn:.1f} s")
    print(f"\n  Results saved to: {os.path.abspath(output_dir)}/")
    print(f"    - Tables: {output_dir}/tables/*.csv")
    print(f"    - Figures: {output_dir}/figs/*.png")
    print("\n" + "=" * 70)
    print("  DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
