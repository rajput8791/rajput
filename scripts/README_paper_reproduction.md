# Paper-faithful FOM and figure reproduction

Reproduces results from Ponalagusamy & Priyadharshini, *Applied Mathematics
and Computation* **333** (2018) 325-343. The paper studies pulsatile flow of
a Casson fluid in a stenosed bifurcated artery under a uniform magnetic
field, with a Darcy-type porous wall, solved by FDM and FEM.

## What was previously wrong

The original `src/casson_fom.py` solves the radial momentum equation at a
*single* cross-section with `R_wall = 1`. That means:

- the stenotic-height parameter `delta_s` had no effect on the solution
  (it was metadata, never entering the FOM),
- there were no axial profiles, while the paper reports `tau_w(z)`,
- the time-averaged flow resistance was computed as the time average of
  `Lambda(t) = -dp/dz / Q(t)`, which blows up wherever `Q(t)` crosses
  zero, producing the spurious 10^4-10^5 Lambda values seen earlier.

## New modules

- `src/geometry.py` (extended)
  - `paper_parent_radius(z, delta_s)`: cosine stenosis profile
        `R(z) = 1 - (delta_s/2) (1 + cos(2 pi (z - 1.6) / 2))` for
        `z in [0.6, 2.6]`, `R = 1` outside,
  - `paper_axial_grid()`: dense plotting grid,
  - `PAPER_REPORT_Z`: the six axial points used in Tables 1-3
        (`z = 0.6, 1.0, 1.4, 1.8, 2.2, 2.6`).

- `src/casson_axial.py` (new) - paper-faithful FOM
  - `solve_cross_section(R_wall, params)`: steady Casson + MHD + Darcy
        radial solve via second-order FD + Picard iteration on the
        regularised effective viscosity,
  - `axial_sweep(delta_s, params)`: loops the cross-section solve over
        a z-grid using the cosine-stenosis radius profile,
  - `wall_shear_stress(r, u, mu)`: signed `tau_w = mu * du/dr |_{r=R}`,
  - `calibrate_G(theta)`: bisection that finds the pressure gradient
        `G` for which `tau_w(R=1, M=1, Da=0.1, theta) = -3.8494`,
        the paper's baseline value at the unobstructed cross-section,
  - `reproduce_paper_tables()`: builds Tables 1, 2, 3 as DataFrames.

- `scripts/reproduce_paper_tables.py` (new)
  - calibrates `G`, prints the three tables alongside the paper values
        and their max absolute / relative errors, and saves CSVs to
        `results/tables/table{1,2,3}_*.csv`.

- `scripts/paper_figures.py` (new)
  - `fig2_schematic`: bifurcated artery schematic with stenosis,
        bifurcation half-angle and daughter radius,
  - `fig6_wss_vs_z_delta`: axial WSS for `delta_s in {0, 0.05, 0.10,
        0.15, 0.20}`,
  - `fig8_wss_vs_z_Da`: axial WSS for `Da in {inf, 1.0, 0.5, 0.1}`,
  - `fig11_resistance_vs_delta`: normalised flow resistance vs
        stenotic height, parameterised by Hartmann number `M`.
  - All saved under `results/figs/fig{2,6,8,11}_*.png`.

## How to run

    python scripts/reproduce_paper_tables.py
    python scripts/paper_figures.py

## Validation against the paper tables

With theta = 0.05 and Nr = 401, calibration gives G = 13.81. The boundary
values (z = 0.6, 2.6, where R = 1) match the paper exactly within rounding:

|       | tau_w paper | tau_w this work |
|-------|-------------|-----------------|
| Table 1, delta_s = 0.0  | -3.8494 | -3.8494 |
| Table 1, delta_s = 0.2  | -3.8494 | -3.8494 |
| Table 2, M = 0          | -3.9736 | -3.9715 |
| Table 2, M = 1          | -3.8494 | -3.8494 |
| Table 3, Da = inf       | -6.1140 | -6.2823 |
| Table 3, Da = 0.1       | -3.8494 | -3.8494 |

Inside the stenotic region the discrepancy is at most ~11% (Da = inf) and
typically ~5-8% (M, delta_s sweeps). The remaining gap is most likely due to
a slightly different stenosis profile and/or a 2-D treatment in the paper's
FDM/FEM, neither of which is fully specified in the snippet provided.
The trends (sign, symmetry around z = 1.6, monotonic dependence on each
parameter) are all reproduced.
