# Pulsatile MHD Casson Flow — POD-ROM and PINN-ROM

Reduced-order modelling and physics-informed neural network surrogates for the
problem studied in:

> R. Ponalagusamy, S. Priyadharshini,
> *Pulsatile MHD flow of a Casson fluid through a porous bifurcated arterial
> stenosis under periodic body acceleration*,
> **Applied Mathematics and Computation** (2018).

---

## 1. Physical model (dimensionless)

Axisymmetric axial velocity `u(r, z, t)` in the parent artery (and, separately,
in each daughter artery) satisfies

```
α² ∂u/∂t = −∂p/∂z + (1/r) ∂(r τ)/∂r − (M² + 1/Da) u + B(t)
```

with

| Symbol | Meaning |
|--------|---------|
| `α² = R₀² ω / ν` | Womersley number squared (we set α² = 1 by time scaling) |
| `−∂p/∂z = A₀ (1 + e cos(t))` | pulsatile pressure gradient, `e = A₁/A₀` |
| `B(t) = a_b cos(ω̄ t + φ)` | periodic body acceleration |
| `M` | Hartmann number (magnetic field) |
| `Da` | Darcy number (porous medium) |
| `θ` | dimensionless yield stress of the Casson fluid |

**Casson constitutive law** (yielded region, `τ ≥ θ`):

```
τ = (√θ + √|γ̇|)²,   γ̇ = −∂u/∂r
```

The plug core (`τ < θ`) moves as a rigid body with `∂u/∂r = 0`.

**Boundary / initial conditions**

```
u(R(z), t) = 0     (no-slip on the wall)
∂u/∂r|_{r=0} = 0   (axisymmetry)
u(r, 0) = 0        (rest)
```

**Geometry**

* Parent artery radius (cosine stenosis):
  `R(z) = R₀ − (δ/2)(1 + cos(2π(z − z_s)/L_s))` inside the stenosis zone,
  `R(z) = R₀` elsewhere.
* Daughter artery radius `R_d = κ R₀` and orientation `β/2` (half bifurcation
  angle). Each daughter is an independent 1-D radial problem with its own
  geometry and parameters.

The full parameter vector used for ROM is

```
μ = (M, Da, θ, δ, β/2, e, a_b, ω̄, φ, κ)
```

---

## 2. Numerical pipeline

```
              ┌─────────────────────┐
              │  Full-Order Model    │
              │  (finite-difference) │
              └─────────┬────────────┘
                        │ snapshots U(r, t; μ)
                        ▼
              ┌─────────────────────┐
              │  POD basis (SVD)     │
              │  Φ_k, σ_k            │
              └─────────┬────────────┘
                        │ a(t; μ) = Φ_kᵀ u
            ┌───────────┴────────────┐
            ▼                        ▼
   ┌────────────────┐       ┌─────────────────┐
   │ POD-Galerkin   │       │  PINN-ROM       │
   │ reduced ODE    │       │  NN(t, μ) → a   │
   │ (ψ_t = F̃(ψ))   │       │  data + PDE loss│
   └────────────────┘       └─────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ post-processing      │
              │ • velocity profiles  │
              │ • wall shear stress  │
              │ • flow resistance    │
              │ • plug core radius   │
              └─────────────────────┘
```

### 2.1 Full-Order Model (FOM)

* Cylindrical `r ∈ [0, R(z)]`, uniform mesh, `N_r` points.
* Crank–Nicolson in time, Picard iteration on the Casson effective viscosity
  `μ_eff(γ̇) = θ/(|γ̇| + ε) + 2√(θ/(|γ̇| + ε)) + 1`.
* Tridiagonal solve per time step (very fast).

### 2.2 POD-ROM

* Snapshot matrix `U ∈ R^{N_r × N_s}` with columns `u(r; t_k, μ_k)`.
* Weighted SVD: `U = Φ Σ V^T` with cylindrical weights `w_i = r_i Δr`.
* Truncate to `k` modes such that captured energy ≥ 99.9 %.
* Galerkin projection of the linear part; Casson term reconstructed in the
  full space, then projected (cheap because `N_r` is small).

### 2.3 PINN-ROM

A multi-layer perceptron `NN_θ(t̂, μ̂) → a ∈ R^k`:

* **Data loss** uses the snapshot tables (paper's tables when available, FOM
  snapshots otherwise) projected onto the POD basis.
* **Physics loss** evaluates the residual of the reduced ODE
  `Φ_kᵀ M Φ_k ȧ − F_red(a, t, μ) = 0` at random collocation points
  `(t, μ)`. Time derivative `ȧ` is obtained with `torch.autograd`.
* **IC loss** enforces `a(0, μ) = 0` (rest start).

---

## 3. Outputs

After training, the following are produced in `results/`:

| File | Content |
|------|---------|
| `results/tables/wss.csv` | Wall shear stress on parent / daughter inner & outer walls vs `(M, Da, θ, δ, β/2)` |
| `results/tables/resistance.csv` | Flow resistance `Λ` vs the same set of parameters |
| `results/tables/plug_core.csv` | Plug-core radius `R_p` vs `(θ, δ, M, Da)` |
| `results/figs/velocity_profiles.png` | `u(r)` at several time instants and parameter values |
| `results/figs/wss_vs_M.png` | WSS vs Hartmann number `M` |
| `results/figs/resistance_vs_Da.png` | Resistance vs Darcy number |
| `results/figs/plug_radius_vs_theta.png` | Plug core radius vs yield stress |
| `results/figs/pod_energy.png` | POD singular value decay |
| `results/figs/pinn_loss.png` | PINN training loss curves |

---

## 4. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_experiment.py --quick    # 1–2 minutes (smoke test)
python scripts/run_experiment.py            # full run, ~5–10 minutes on CPU
```

Use the paper's tables instead of FOM snapshots:

```bash
python scripts/run_experiment.py --train-data data/paper_tables.csv
```

The expected CSV header is

```
M,Da,theta,delta,beta_half,t,r,u
```

---

## 5. File layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── geometry.py      # stenosis profile, bifurcation geometry
│   ├── casson_fom.py    # finite-difference full-order solver
│   ├── pod_rom.py       # snapshot SVD + Galerkin reduced model
│   ├── pinn_rom.py      # PyTorch PINN-ROM
│   ├── postproc.py      # WSS, resistance, plug core, plots, table I/O
│   └── __init__.py
├── scripts/
│   └── run_experiment.py
└── results/             # generated tables and figures
```

---

## 6. Notes and limitations

* The Casson regularization ε defaults to `1e-3`; it controls the
  smoothness of the plug-yielded transition.
* Body-acceleration phase, frequency ratio and pressure ratio defaults are
  taken from typical cardiovascular ranges (`e=0.5`, `ω̄=1`, `φ=0`).
* Only one daughter is computed by default; symmetric geometry is assumed,
  the second daughter is obtained by reflection.
* When the user provides paper tables in `data/paper_tables.csv` they are
  used as ground truth for the PINN data loss; otherwise FOM snapshots are
  used. Either way the PDE residual loss enforces the governing equation,
  so the surrogate generalises to unseen `(t, μ)` queries.
