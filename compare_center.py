#!/usr/bin/env python3
"""
Compare shock center trajectories from multiple shock_center.dat-style files.

Usage:
    python compare_center.py <folder>

Reads every .dat file in <folder> (each with a one-line header followed by
columns: t  center), plots all traces on one graph. Each trace is centered by
subtracting its mean so that all lines sit at y=0 on average. The same scale
is shown on both the left and right y-axes. Saves compare_center.pdf inside
that folder.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter


def load_dat(path):
    return np.loadtxt(path, skiprows=1)


def interpolate_errors(centers):
    """Replace -1 sentinels and non-finite values with linear interpolation from neighbours."""
    bad = (centers == -1) | ~np.isfinite(centers)
    if bad.any() and not bad.all():
        idx = np.arange(len(centers), dtype=float)
        centers = centers.copy()
        centers[bad] = np.interp(idx[bad], idx[~bad], centers[~bad])
    return centers


def main():
    if len(sys.argv) < 2:
        print('Usage: python compare_center.py <folder>')
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        sys.exit(f'Error: "{folder}" is not a directory.')

    dat_files = sorted(folder.glob('*.dat'))
    if not dat_files:
        sys.exit(f'No .dat files found in {folder}')

    print(f'Reading from: {folder}')

    datasets = []
    for fpath in dat_files:
        try:
            data = load_dat(fpath)
        except Exception as e:
            print(f'  Skipping {fpath.name}: {e}')
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        datasets.append((fpath.stem, data))

    if not datasets:
        sys.exit('No readable .dat files found.')

    n = len(datasets)
    colors = cm.coolwarm(np.linspace(0, 1, n))

    traces = [(label, data[:, 0], interpolate_errors(data[:, 1])) for label, data in datasets]

    # Center each trace and find the common half-span from absolute values
    means = [np.mean(centers) for _, _, centers in traces]
    deviations = [centers - mean for (_, _, centers), mean in zip(traces, means)]
    max_dev = max((v for v in (np.max(np.abs(d)) for d in deviations) if np.isfinite(v)),
                  default=1e-6)
    max_dev = max(max_dev, 1e-6)
    half_span = max_dev * 1.15

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    for (label, t, _), dev, color in zip(traces, deviations, colors):
        ax_left.plot(t, dev, color=color, linewidth=0.8, alpha=0.8, label=label)

    ax_left.set_ylim(-half_span, half_span)
    ax_right.set_ylim(-half_span, half_span)

    # Relabel each axis with absolute values centred on that trace's mean
    mean_left, mean_right = means[0], means[-1]
    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y + mean_left:.2f}'))
    ax_right.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y + mean_right:.2f}'))

    ax_left.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.6)
    ax_left.set_xlabel('Time')
    ax_left.set_ylabel(f'Shock center — {datasets[0][0]}')
    ax_right.set_ylabel(f'Shock center — {datasets[-1][0]}')
    ax_left.legend(fontsize=7, loc='best')
    ax_left.grid(True, alpha=0.3)
    fig.suptitle(f'Shock center comparison\n{folder}')
    plt.tight_layout()

    out_path = folder / 'compare_center.pdf'
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
