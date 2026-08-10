#!/usr/bin/env python3
"""
Visualization script for shock_simulation.py output.

Usage:
    python visualize_shock.py <plot_type> [output_folder]

plot_type : npt | rho | u | T | flux | pdf | shock_center | all
output_folder : optional; defaults to the most recently modified shock_output_* folder

In timeseries mode (detected automatically when per-period files are present),
rho/u/T plots overlay all periods with a viridis colormap (early=dark, late=bright).
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


OUTPUT_ROOT = SCRIPT_DIR / 'output'
PLOTS_ROOT  = SCRIPT_DIR / 'plots'


def find_output_dir(folder_arg=None):
    """Return the output directory to use."""
    if folder_arg is not None:
        p = Path(folder_arg)
        if not p.is_dir():
            p = OUTPUT_ROOT / folder_arg
        if not p.is_dir():
            p = SCRIPT_DIR / folder_arg
        if not p.is_dir():
            sys.exit(f'Error: folder "{folder_arg}" not found.')
        return str(p)
    dirs = sorted(OUTPUT_ROOT.glob('shock_output_*'), key=os.path.getmtime)
    if not dirs:
        sys.exit(f'Error: no shock_output_* folders found in {OUTPUT_ROOT}')
    return str(dirs[-1])


def load_dat(path):
    """Load a whitespace-delimited .dat file, skipping the first header line."""
    return np.loadtxt(path, skiprows=1)



def get_period_files(outdir, prefix):
    """Return sorted list of per-period files matching <prefix>_N.dat."""
    pattern = os.path.join(outdir, f'{prefix}_*.dat')
    files = glob.glob(pattern)
    # Sort by integer period number, not lexicographically
    files.sort(key=lambda f: int(Path(f).stem.split('_')[-1]))
    return files


def get_run_param(outdir, name, default=0):
    """Return an integer CLI argument from run_info.txt, or default if not found."""
    run_info = os.path.join(outdir, 'run_info.txt')
    if not os.path.isfile(run_info):
        return default
    with open(run_info) as f:
        cmd = f.readline().split()
    for i, tok in enumerate(cmd):
        if tok.startswith(f'-{name}='):
            return int(tok.split('=', 1)[1])
        if tok == f'-{name}' and i + 1 < len(cmd):
            return int(cmd[i + 1])
    return default


def get_gap(outdir):
    return get_run_param(outdir, 'gap', 0)


def get_L_tube(outdir):
    """Return L_tube from the printed parameters block in run_info.txt."""
    run_info = os.path.join(outdir, 'run_info.txt')
    if not os.path.isfile(run_info):
        return None
    with open(run_info) as f:
        for line in f:
            if 'L_tube' in line and '=' in line:
                try:
                    return float(line.split('=')[1].strip())
                except ValueError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_npt(outdir):
    path = os.path.join(outdir, 'npt.dat')
    data = load_dat(path)
    t, ratio = data[:, 0], data[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    line1, = ax.plot(t, ratio, color='tab:blue', linewidth=1.2, label=r'$N_p/N_0$')

    # Overlay right-boundary removal rate if available (5-step moving average)
    rr_path = os.path.join(outdir, 'removed_right.dat')
    if os.path.isfile(rr_path):
        rr_data = load_dat(rr_path)
        warmup  = get_run_param(outdir, 'warmup', 1000)
        n_per   = get_run_param(outdir, 'num_periods', 1)
        period  = get_run_param(outdir, 'period', 1000)
        gap     = get_run_param(outdir, 'gap', 0)
        w = max(10, int((warmup + n_per * (period + gap) - gap) / 100))
        kernel = np.ones(w) / w
        # mode='valid' gives N-(w-1) values, each a fully-windowed average;
        # rr_smooth[k] = mean of steps k..k+w-1, associated with time rr_data[k+w-1]
        rr_smooth = np.convolve(rr_data[:, 1], kernel, mode='valid')
        t_smooth = rr_data[w-1:, 0]       # time of last element in each window
        rr_smooth = rr_smooth[w:-w]        # drop first and last w averages
        t_smooth  = t_smooth[w:-w]
        # Set both y-axes so that y=1 is centred at the same vertical level
        pad = 1.15  # 15% headroom beyond the max deviation from 1
        dev1 = max(abs(ratio - 1).max(), 1e-6) * pad
        dev2 = max(abs(rr_smooth - 1).max() * pad, 0.1)
        ax.set_ylim(1 - dev1, 1 + dev1)

        ax2 = ax.twinx()
        line2, = ax2.plot(t_smooth, rr_smooth, color='tab:orange',
                          linewidth=1.0, alpha=0.8, label=rf'removed / $N_{{in}}$ ({w}-step avg)')
        ax2.set_ylim(1 - dev2, 1 + dev2)
        ax2.set_ylabel(r'removed / $N_{in}$', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        lines = [line1, line2]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    # Draw separators at period boundaries
    num_periods = get_run_param(outdir, 'num_periods', 0)
    period_steps = get_run_param(outdir, 'period', 0)
    if num_periods > 0 and period_steps > 0:
        gap = get_gap(outdir)
        stride = period_steps + gap
        warmup_steps = len(t) - num_periods * period_steps - (num_periods - 1) * gap
        for p in range(num_periods):
            start_idx = warmup_steps + p * stride
            if 0 <= start_idx < len(t):
                ax.axvline(t[start_idx], color='gray', linewidth=0.7, linestyle='--', alpha=0.7)
            if gap > 0 and p < num_periods - 1:
                end_idx = warmup_steps + p * stride + period_steps
                if 0 <= end_idx < len(t):
                    ax.axvline(t[end_idx], color='gray', linewidth=0.7, linestyle='--', alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel(r'$N_p \/ / \/ N_0$', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_title(f'Particle count in tube\n{Path(outdir).name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def _plot_spatial_timeseries(outdir, prefix, ylabel, title_label):
    """Overlay all per-period spatial profiles with a viridis colormap.

    Reads the combined <prefix>.dat file (columns: period, x, value).
    """
    path = os.path.join(outdir, f'{prefix}.dat')
    if not os.path.isfile(path):
        print(f'No {prefix}.dat found in {Path(outdir).name}')
        return
    data = load_dat(path)
    periods = np.unique(data[:, 0].astype(int))
    n = len(periods)
    colors = cm.viridis(np.linspace(0, 1, n))

    _, ax = plt.subplots(figsize=(8, 5))
    for period, color in zip(periods, colors):
        mask = data[:, 0].astype(int) == period
        ax.plot(data[mask, 1], data[mask, 2], color=color, linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(1, n))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='period')

    ax.set_xlabel(r'$x \/ / \/ \lambda_1$')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title_label} — timeseries ({n} periods)\n{Path(outdir).name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_rho(outdir):
    _plot_spatial_timeseries(outdir, 'rho', r'$\rho \/ / \/ \rho_0$', 'Density profile')


def plot_u(outdir):
    _plot_spatial_timeseries(outdir, 'u', r'$\langle u \rangle$', 'Mean x-velocity profile')


def plot_flux(outdir):
    rho_path = os.path.join(outdir, 'rho.dat')
    if not os.path.isfile(rho_path):
        print(f'No rho.dat found in {Path(outdir).name}')
        return
    rho_data = load_dat(rho_path)
    u_data   = load_dat(os.path.join(outdir, 'u.dat'))
    periods = np.unique(rho_data[:, 0].astype(int))
    n = len(periods)
    colors = cm.viridis(np.linspace(0, 1, n))
    _, ax = plt.subplots(figsize=(8, 5))
    for period, color in zip(periods, colors):
        rows = np.where(rho_data[:, 0].astype(int) == period)[0]
        rows = rows[len(rows) // 10:]
        ax.plot(rho_data[rows, 1], rho_data[rows, 2] * u_data[rows, 2], color=color, linewidth=1.0)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(1, n))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='period')
    ax.set_title(f'Mass flux profile — timeseries ({n} periods)\n{Path(outdir).name}')
    ax.set_xlabel(r'$x \/ / \/ \lambda_1$')
    ax.set_ylabel(r'$\rho u$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_T(outdir):
    _plot_spatial_timeseries(outdir, 'T', 'T', 'Temperature profile')


def plot_shock_center(outdir):
    path = os.path.join(outdir, 'shock_center.dat')
    if not os.path.isfile(path):
        print(f'No shock_center.dat found in {Path(outdir).name}')
        return
    data = load_dat(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0]
    raw = data[:, 1].copy()

    error_mask = (raw == -1)
    if error_mask.any() and not error_mask.all():
        idx = np.arange(len(raw), dtype=float)
        raw[error_mask] = np.interp(idx[error_mask], idx[~error_mask], raw[~error_mask])

    L_tube = get_L_tube(outdir)
    centers = raw / L_tube if L_tube else raw

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, centers, color='tab:blue', linewidth=0.8, alpha=0.7)
    if error_mask.any():
        ax.plot(t[error_mask], centers[error_mask], 'o', color='red', markersize=0.8, zorder=3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Shock center (tube fraction)')
    ax.set_title(f'Shock center over time\n{Path(outdir).name}')
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
    'npt':          plot_npt,
    'rho':          plot_rho,
    'u':            plot_u,
    'T':            plot_T,
    'flux':         plot_flux,
    'pdf':          plot_pdf,
    'shock_center': plot_shock_center,
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
    n = get_run_param(outdir, 'num_periods', 0)
    print(f'{n} period(s)')

    # Derive plots folder from the datetime embedded in the output folder name
    datetime_suffix = Path(outdir).name[len('shock_output_'):]
    PLOTS_ROOT.mkdir(exist_ok=True)
    plots_dir = PLOTS_ROOT / f'plots_{datetime_suffix}'
    plots_dir.mkdir(exist_ok=True)

    to_plot = list(PLOT_FUNCS.keys()) if plot_type == 'all' else [plot_type]
    any_new = False
    for pt in to_plot:
        plot_path = plots_dir / f'{pt}.pdf'
        if plot_path.exists():
            print(f'  {pt}.pdf already exists, skipping')
            continue
        figs_before = set(plt.get_fignums())
        PLOT_FUNCS[pt](outdir)
        if set(plt.get_fignums()) - figs_before:
            plt.savefig(plot_path, bbox_inches='tight')
            print(f'  Saved {plot_path}')
            any_new = True

    if any_new:
        plt.show()


if __name__ == '__main__':
    main()
