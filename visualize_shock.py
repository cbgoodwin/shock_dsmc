#!/usr/bin/env python3
"""
Visualization script for shock_simulation.py output.

Usage:
    python visualize_shock.py <plot_type> [output_folder]

plot_type : npt | rho | u | T | pdf | all
output_folder : optional; defaults to the most recently modified shock_output_* folder
"""

import sys
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def find_output_dir(folder_arg=None):
    """Return the output directory to use."""
    if folder_arg is not None:
        p = Path(folder_arg)
        if not p.is_dir():
            # Try relative to the script directory as a fallback
            p = SCRIPT_DIR / folder_arg
        if not p.is_dir():
            sys.exit(f'Error: folder "{folder_arg}" not found.')
        return str(p)
    dirs = sorted(SCRIPT_DIR.glob('shock_output_*'), key=os.path.getmtime)
    if not dirs:
        sys.exit(f'Error: no shock_output_* folders found in {SCRIPT_DIR}')
    return str(dirs[-1])


def load_dat(path):
    """Load a whitespace-delimited .dat file, skipping the first header line."""
    return np.loadtxt(path, skiprows=1)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_npt(outdir):
    path = os.path.join(outdir, 'npt.dat')
    data = load_dat(path)
    t, ratio = data[:, 0], data[:, 1]

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, ratio, linewidth=1.2)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$N_p \/ / \/ N_0$')
    ax.set_title(f'Particle count in tube\n{Path(outdir).name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_rho(outdir):
    path = os.path.join(outdir, 'rho.dat')
    data = load_dat(path)
    x, rho = data[:, 0], data[:, 1]

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, rho, linewidth=1.2)
    ax.set_xlabel(r'$x \/ / \/ \lambda_1$')
    ax.set_ylabel(r'$\rho \/ / \/ \rho_0$')
    ax.set_title(f'Density profile\n{Path(outdir).name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_u(outdir):
    path = os.path.join(outdir, 'u.dat')
    data = load_dat(path)
    x, u = data[:, 0], data[:, 1]

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, u, linewidth=1.2)
    ax.set_xlabel(r'$x \/ / \/ \lambda_1$')
    ax.set_ylabel(r'$\langle u \rangle$')
    ax.set_title(f'Mean x-velocity profile\n{Path(outdir).name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_T(outdir):
    path = os.path.join(outdir, 'T.dat')
    data = load_dat(path)
    x, T = data[:, 0], data[:, 1]

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, T, linewidth=1.2)
    ax.set_xlabel(r'$x \/ / \/ \lambda_1$')
    ax.set_ylabel('T')
    ax.set_title(f'Temperature profile\n{Path(outdir).name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_pdf(outdir):
    """
    Plot velocity PDFs for every pdf_cell_*.dat file.

    Three panels — f(u), f(v), f(w) — with one line per cell colored by
    position (upstream=blue, downstream=yellow).  Dashed lines show the
    equilibrium Maxwellians from the first file loaded.
    """
    pdf_files = sorted(glob.glob(os.path.join(outdir, 'pdf_cell_*.dat')))
    if not pdf_files:
        print(f'No pdf_cell_*.dat files found in {Path(outdir).name}')
        return

    # Parse cell numbers from filenames for the colorbar tick labels
    cell_nums = []
    datasets = []
    for fpath in pdf_files:
        try:
            data = load_dat(fpath)
        except Exception as e:
            print(f'  Skipping {fpath}: {e}')
            continue
        stem = Path(fpath).stem          # e.g. "pdf_cell_063"
        cell_nums.append(int(stem.split('_')[-1]))
        datasets.append(data)

    if not datasets:
        print('No readable PDF files found.')
        return

    n = len(datasets)
    colors = cm.viridis(np.linspace(0, 1, n))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    # dat file column layout: x  f(u)  f(v)  f(w)  f1_e(u)  f2_e(u)  f1_e(v)  f2_e(v)
    #                         0   1     2     3       4         5         6        7
    panel_cols  = [1, 2, 3]
    panel_ylabs = [r'$f(u)$', r'$f(v)$', r'$f(w)$']
    # analytical equilibrium columns: u uses 4/5, v and w use 6/7
    eq_cols = [(4, 5), (6, 7), (6, 7)]

    ref_data = datasets[0]   # analytical curves are the same in every file

    for ax, pcol, ylabel, (ec1, ec2) in zip(axes, panel_cols, panel_ylabs, eq_cols):
        vel = ref_data[:, 0]
        ax.plot(vel, ref_data[:, ec1], 'b--', linewidth=1.4,
                label='upstream Maxwell', zorder=3)
        ax.plot(vel, ref_data[:, ec2], 'r--', linewidth=1.4,
                label='downstream Maxwell', zorder=3)

        for data, color in zip(datasets, colors):
            ax.plot(data[:, 0], data[:, pcol], color=color,
                    linewidth=0.8, alpha=0.8)

        ax.set_xlabel('velocity')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Velocity PDFs across shock\n{Path(outdir).name}')
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PLOT_FUNCS = {
    'npt': plot_npt,
    'rho': plot_rho,
    'u':   plot_u,
    'T':   plot_T,
    'pdf': plot_pdf,
}

VALID_TYPES = list(PLOT_FUNCS.keys()) + ['all']


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_TYPES:
        print(f'Usage: python visualize_shock.py <{"|".join(VALID_TYPES)}> [output_folder]')
        sys.exit(1)

    plot_type = sys.argv[1]
    folder_arg = sys.argv[2] if len(sys.argv) >= 3 else None
    outdir = find_output_dir(folder_arg)
    print(f'Reading from: {Path(outdir).name}')

    to_plot = list(PLOT_FUNCS.keys()) if plot_type == 'all' else [plot_type]
    for pt in to_plot:
        PLOT_FUNCS[pt](outdir)

    plt.show()


if __name__ == '__main__':
    main()
