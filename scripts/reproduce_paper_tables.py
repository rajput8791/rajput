#!/usr/bin/env python
"""Reproduce Tables 1-3 of Ponalagusamy & Priyadharshini (AMC 2018) and
print them alongside the published values for direct comparison.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- import path bootstrap ----
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
os.chdir(_root)

from src.casson_axial import reproduce_paper_tables, PAPER_REPORT_Z


# ----------------------------------------------------------------------
# Published values (transcribed from the paper)
# ----------------------------------------------------------------------

PAPER_TABLE1 = pd.DataFrame(
    {
        "delta_s=0.0": [-3.8494, -3.8494, -3.8494, -3.8494, -3.8494, -3.8494],
        "delta_s=0.2": [-3.8494, -3.5086, -3.3355, -3.3355, -3.5086, -3.8494],
    },
    index=PAPER_REPORT_Z,
)
PAPER_TABLE2 = pd.DataFrame(
    {
        "M=0": [-3.9736, -3.6122, -3.4181, -3.4181, -3.6122, -3.9736],
        "M=1": [-3.8494, -3.5086, -3.3355, -3.3355, -3.5086, -3.8494],
    },
    index=PAPER_REPORT_Z,
)
PAPER_TABLE3 = pd.DataFrame(
    {
        "Da=inf": [-6.1140, -5.3185, -4.9181, -4.9181, -5.3185, -6.1140],
        "Da=0.1": [-3.8494, -3.5086, -3.3355, -3.3355, -3.5086, -3.8494],
    },
    index=PAPER_REPORT_Z,
)


def _print_side_by_side(title, mine: pd.DataFrame, paper: pd.DataFrame):
    print(f"\n{'='*78}\n  {title}\n{'='*78}")
    combined = pd.concat({"FOM (this work)": mine, "Paper": paper}, axis=1)
    combined.index.name = "z"
    print(combined.round(4).to_string())
    err = (mine.values - paper.values)
    rel = np.abs(err) / (np.abs(paper.values) + 1e-12)
    print(f"  -> max abs err = {np.abs(err).max():.4f},  "
          f"max rel err = {rel.max()*100:.2f}%")


def main():
    print("=" * 78)
    print("  Reproducing Ponalagusamy & Priyadharshini (AMC 2018) Tables 1-3")
    print("=" * 78)

    out = reproduce_paper_tables(theta=0.05, Nr=401, verbose=True)

    _print_side_by_side("TABLE 1: WSS vs z for stenotic height delta_s",
                        out["table1"], PAPER_TABLE1)
    _print_side_by_side("TABLE 2: WSS vs z for Hartmann number M",
                        out["table2"], PAPER_TABLE2)
    _print_side_by_side("TABLE 3: WSS vs z for Darcy number Da",
                        out["table3"], PAPER_TABLE3)

    # Save side-by-side CSVs
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mine, paper in [
        ("table1_delta_s", out["table1"], PAPER_TABLE1),
        ("table2_hartmann", out["table2"], PAPER_TABLE2),
        ("table3_darcy",   out["table3"], PAPER_TABLE3),
    ]:
        combined = pd.concat({"FOM (this work)": mine, "Paper": paper}, axis=1)
        combined.index.name = "z"
        combined.to_csv(out_dir / f"{name}.csv")
        print(f"  wrote {out_dir / f'{name}.csv'}")


if __name__ == "__main__":
    main()
