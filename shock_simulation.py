#!/usr/bin/env python3
"""
Direct Simulation Monte Carlo (DSMC) shock simulation.
Translated from shock_simulation_260317.f90
"""

import argparse
import collections
import math
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
start_time = datetime.now()
# ===== Parameters =====

pi = math.pi
s = 2.0          # ratio of bulk flow to most probable speed
beta1 = 1.0      # inverse of most probable speed
U1 = s / beta1   # bulk flow velocity in terms of upstream most probable speed
n = None          # set from -n command-line argument

# Physical parameters
Ru = 8.314472      # gas constant
gamma = 5.0 / 3.0  # Laplace's coefficient (assumed for ideal gas)
AVOG = 6.022e23    # Avogadro's constant

# Upstream parameters
# Free variables
Tp1 = 1.0      # upstream temperature
mu1 = 1.0      # dynamic viscosity
rho1 = 1.0     # upstream density
area = 1.0     # cross section area of tube
# Derived variables
RR = 1.0 / (2.0 * Tp1 * beta1**2)  # specific gas constant
mm = Ru / RR                       # molar mass
dm1 = math.sqrt(5.0 * mm * math.sqrt(RR * Tp1 / pi) / (16.0 * mu1))    # hard sphere diameter
sigma_t = pi * dm1**2                                                  # hard sphere collision cross section
cm1 = 2.0 / (math.sqrt(pi) * beta1)   # upstream mean molecular speed
lamda1 = cm1 * rho1 / mu1             # upstream mean free path
a1 = math.sqrt(gamma * RR * Tp1)      # upstream speed of sound
Ma1 = U1 / a1                         # upstream Mach speed of gas

# Downstream parameters from normal shock relations
U2 = -2.0 * a1 / (gamma + 1.0) * (Ma1 - 1.0 / Ma1) + U1
rho2 = rho1 * (Ma1**2 / (1.0 + (gamma - 1.0) / (gamma + 1.0) * (Ma1**2 - 1.0)))
Tp2 = Tp1 * (1.0 + 2.0 * (gamma - 1.0) / (gamma + 1.0)**2 *
             (Ma1**2 - 1.0) * (1.0 + gamma * Ma1**2) / Ma1**2)
beta2 = (2.0 * RR * Tp2)**(-0.5)
mu2 = 5.0 * mm * math.sqrt(RR * Tp2 / pi) / (16.0 * dm1**2)
cm2 = 2.0 / math.sqrt(pi) / beta2
lamda2 = 32.0 * mu2 / (5.0 * pi * rho2 * cm2)
a2 = math.sqrt(gamma * RR * Tp2)
Ma2 = U2 / a2
deltaU = U1 - U2

# Simulation parameters
dx = 0.365961
ncell = 185
dt = 0.5 * lamda2 / cm2
Nt = 1000
L_tube = dx * ncell
Nreal = ((2.0 * rho1 + rho2) / mm) * AVOG * 20.0 * lamda1 * area
# N0, Nmax, Wt are computed in main() once -n is parsed
N0 = Nmax = Wt = None


class _Tee:
    """Mirror stdout to a file, excluding per-step np= lines.

    Buffers writes until a newline arrives so that the message and its
    trailing newline are filtered together as a complete line.
    """
    def __init__(self, filepath):
        self._file = open(filepath, 'w')
        self._stdout = sys.stdout
        self._buf = ''

    def write(self, msg):
        self._stdout.write(msg)
        self._buf += msg
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if '  np= ' not in line:
                self._file.write(line + '\n')

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        if self._buf and '  np= ' not in self._buf:
            self._file.write(self._buf)
        sys.stdout = self._stdout
        self._file.close()


# ===== Argument Parsing =====

def parse_args():
    parser = argparse.ArgumentParser(description='DSMC shock simulation')
    parser.add_argument('-initialize', type=str, default=None,
                        help='"empty" to start with an empty tube, or path to load particle state from')
    parser.add_argument('-warmup', type=int, default=None,
                        help='Number of warmup timesteps before sampling/periods '
                             '(default: 0 when loading a state, 1000 otherwise)')
    parser.add_argument('-period', type=int, default=10,
                        help='Number of time steps per sampling window (default: 10)')
    parser.add_argument('-num_periods', type=int, default=1,
                        help='Number of periods (default: 1)')
    parser.add_argument('-gap', type=int, default=0,
                        help='Steps between periods (positive integer)')
    parser.add_argument('-n', type=int, default=500,
                        help='Number of simulated particles per unit volume (default: 500)')
    parser.add_argument('-piston_speed', type=float, default=1.0,
                        help='Piston speed multiplier applied to U2 at the right boundary (default: 1.0, must be > 0)')
    parser.add_argument('-smooth_center', type=int, default=3,
                        help='Number of timesteps in the trailing average used for shock center computation (default: 3, must be > 0)')
    parser.add_argument('-seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None, i.e. random)')
    parser.add_argument('-adaptive', type=float, nargs='?', const=0.01, default=None,
                        help='Adaptively adjust piston speed each step based on midpoint density; '
                             'value sets the gain multiplier (default when flag is used: 0.01)')
    parser.add_argument('-model', type=str, default='hs', choices=['hs', 'maxwell'],
                        help='Collision model: hs = hard sphere (default), maxwell = Maxwell molecules')
    parser.add_argument('-adaptive_mode', type=str, default='density', choices=['density', 'particles'],
                        help='Adaptive piston mode: density (default) uses midpoint density; '
                             'particles uses total particle count relative to target')
    args = parser.parse_args()

    if args.warmup is None:
        args.warmup = 0 if (args.initialize not in (None, 'empty')) else 1000

    if args.period <= 0:
        parser.error('-period must be a positive integer')
    if args.num_periods <= 0:
        parser.error('-num_periods must be a positive integer')
    if args.piston_speed <= 0:
        parser.error('-piston_speed must be greater than zero')
    if args.smooth_center <= 0:
        parser.error('-smooth_center must be a positive integer')
    if args.gap < 0:
        parser.error('-gap must be a non-negative integer')

    return args


# ===== Utility Functions =====

# replace with numpy.random.normal?
def gasdev(mean, stand):
    """Gaussian random number via Box-Muller transform."""
    ran1 = max(np.random.random(), 1.0e-12)
    ran2 = np.random.random()
    g = math.sqrt(-2.0 * math.log(ran1)) * math.cos(2.0 * pi * ran2)
    return g * stand + mean


def u_ini(beta, u_bulk):
    """
    Sample the x-velocity of a particle entering through the inlet boundary.

    At an inlet surface, faster molecules arrive more frequently, so the
    distribution of entering velocities is flux-weighted:

        g(u) = norm * u * exp(-beta^2 * (u - u_bulk)^2),  u > 0

    rather than the ordinary Maxwellian.  Uses rejection sampling: propose u
    uniformly in [u_min, u_max], accept with probability g(u) / g_max.
    """
    # Proposal window: ±3 thermal widths around u_bulk, clipped to u > 0
    u_min = max(u_bulk - 3.0 / beta, 0.0)
    u_max = max(u_bulk + 3.0 / beta, 1.0 / beta)

    # Normalization constant so g integrates to 1 over u > 0
    speed_ratio = u_bulk * beta
    norm = 2.0 * beta**2 / (
        math.sqrt(pi) * speed_ratio * (1.0 + erf(speed_ratio))
        + math.exp(-speed_ratio**2)
    )

    # Peak of g(u): found analytically by setting dg/du = 0
    u_peak = (u_bulk + math.sqrt(u_bulk**2 + 2.0 / beta**2)) / 2.0
    g_max  = norm * u_peak * math.exp(-beta**2 * (u_peak - u_bulk)**2)

    # Rejection sampling
    while True:
        u_candidate = u_min + np.random.random() * (u_max - u_min)
        g_candidate = norm * u_candidate * math.exp(-beta**2 * (u_candidate - u_bulk)**2)
        if g_candidate / g_max >= np.random.random():
            return u_candidate


def _sample_u_ini_batch(beta, u_bulk, n_samples):
    """Vectorized version of u_ini for n_samples particles.

    Same target distribution and proposal window; rejection is applied with a
    numpy mask, refilling in batches until n_samples are accepted.
    """
    if n_samples == 0:
        return np.empty(0)

    u_min = max(u_bulk - 3.0 / beta, 0.0)
    u_max = max(u_bulk + 3.0 / beta, 1.0 / beta)

    speed_ratio = u_bulk * beta
    norm = 2.0 * beta**2 / (
        math.sqrt(pi) * speed_ratio * (1.0 + erf(speed_ratio))
        + math.exp(-speed_ratio**2)
    )

    u_peak = (u_bulk + math.sqrt(u_bulk**2 + 2.0 / beta**2)) / 2.0
    g_max  = norm * u_peak * math.exp(-beta**2 * (u_peak - u_bulk)**2)

    out = np.empty(n_samples)
    filled = 0
    while filled < n_samples:
        batch = max(int((n_samples - filled) * 2), 16)
        cand = u_min + np.random.random(batch) * (u_max - u_min)
        g = norm * cand * np.exp(-beta**2 * (cand - u_bulk)**2)
        accepted = cand[g >= g_max * np.random.random(batch)]
        take = min(accepted.size, n_samples - filled)
        out[filled:filled + take] = accepted[:take]
        filled += take
    return out


def print_parameters():
    print('Simulation parameters:')
    print(f'  ncell      = {ncell}')
    print(f'  Nt         = {Nt}')
    print(f'  dx         = {dx}')
    print(f'  dt         = {dt}')
    print(f'  L_tube     = {L_tube}')
    print(f'  N0         = {N0}')
    print(f'  Nmax       = {Nmax}')
    print(f'  U1, U2     = {U1}, {U2}')
    print(f'  rho1, rho2 = {rho1}, {rho2}')
    print(f'  Tp1, Tp2   = {Tp1}, {Tp2}')


def get_T(u_arr, v_arr, w_arr, RR_in):
    """Compute temperature from velocity component arrays."""
    np_local = len(u_arr)
    if np_local <= 1:
        return 0.0
    uu = np.mean(u_arr)
    vv = np.mean(v_arr)
    ww = np.mean(w_arr)
    return float(np.sum((u_arr - uu)**2 + (v_arr - vv)**2 + (w_arr - ww)**2) /
                 (3.0 * np_local * RR_in))


# ===== Simulation Functions =====

def collision(u, v, w, pc, cr_max, cell_no, vol_cell, maxwell=False):
    """DSMC collision step for a single cell.
    pc: pre-computed array of particle indices in this cell.
    All random numbers are batch-generated upfront; pair updates are applied
    sequentially so each collision sees the velocities from all prior collisions,
    preserving the correct NTC statistics.

    maxwell=True: Maxwell molecule model. sigma*c_rel is constant, so all
    candidate pairs are accepted (P_accept=1) and cr_max is not updated.
    maxwell=False (default): hard sphere model with NTC acceptance.
    """
    nc = len(pc)
    if nc < 2:
        return

    u_cell = u[pc].copy()
    v_cell = v[pc].copy()
    w_cell = w[pc].copy()

    Nt_colli = int(0.5 * nc * (nc - 1.0) * (sigma_t * cr_max[cell_no]) * dt * Wt / (vol_cell * AVOG))
    if Nt_colli == 0:
        return

    # Batch-generate all random numbers upfront to avoid per-call Python overhead
    pp1_arr  = np.random.randint(0, nc, size=Nt_colli)
    pp2_arr  = np.random.randint(0, nc, size=Nt_colli)
    r_theta  = np.random.random(size=Nt_colli)
    r_psi    = np.random.random(size=Nt_colli)
    if not maxwell:
        r_accept = np.random.random(size=Nt_colli)

    for i in range(Nt_colli):
        pp1 = int(pp1_arr[i])
        pp2 = int(pp2_arr[i])
        if pp1 == pp2:
            continue

        c_rel_x = u_cell[pp1] - u_cell[pp2]
        c_rel_y = v_cell[pp1] - v_cell[pp2]
        c_rel_z = w_cell[pp1] - w_cell[pp2]
        c_rel = math.sqrt(c_rel_x**2 + c_rel_y**2 + c_rel_z**2)

        if maxwell:
            do_collision = True
        else:
            P_accept = c_rel / max(cr_max[cell_no], 1.0e-12)
            if P_accept > 1.0:
                P_accept = 1.0
                cr_max[cell_no] = c_rel
            do_collision = r_accept[i] <= P_accept

        if do_collision:
            theta   = 2.0 * pi * r_theta[i]
            cos_psi = 1.0 - 2.0 * r_psi[i]
            sin_psi = math.sqrt(max(1.0 - cos_psi**2, 0.0))

            cprime_x = c_rel * math.cos(theta) * sin_psi
            cprime_y = c_rel * math.sin(theta) * sin_psi
            cprime_z = c_rel * cos_psi

            u_cm = 0.5 * (u_cell[pp1] + u_cell[pp2])
            v_cm = 0.5 * (v_cell[pp1] + v_cell[pp2])
            w_cm = 0.5 * (w_cell[pp1] + w_cell[pp2])

            u_cell[pp1] = u_cm + 0.5 * cprime_x
            u_cell[pp2] = u_cm - 0.5 * cprime_x
            v_cell[pp1] = v_cm + 0.5 * cprime_y
            v_cell[pp2] = v_cm - 0.5 * cprime_y
            w_cell[pp1] = w_cm + 0.5 * cprime_z
            w_cell[pp2] = w_cm - 0.5 * cprime_z

            if not maxwell:
                cr_max[cell_no] = max(c_rel, cr_max[cell_no])

    u[pc] = u_cell
    v[pc] = v_cell
    w[pc] = w_cell


def write_spatial_profiles(ua, va, wa, np_cell, outdir, n_sample_steps, period=None):
    """Write rho, u, T spatial profiles.

    Single-batch (period=None): write header + 2-column (x, value) to rho/u/T.dat.
    Timeseries (period=int): prepend a period column; open in write mode for period 1,
    append mode for subsequent periods so all periods land in one file.
    """
    rho0 = float(N0) / (float(ncell) * (2.0 / 3.0) * rho1 + float(ncell) * (1.0 / 3.0) * rho2)
    mode = 'a' if (period is not None and period > 1) else 'w'
    with open(os.path.join(outdir, 'rho.dat'), mode) as f200, \
         open(os.path.join(outdir, 'u.dat'), mode) as f201, \
         open(os.path.join(outdir, 'T.dat'), mode) as f202:
        if mode == 'w':
            if period is None:
                f200.write('variables="x/<greek>l</greek><sub>1","<greek>r</greek>"\n')
                f201.write('variables="x/<greek>l</greek><sub>1","u"\n')
                f202.write('variables="x/<greek>l</greek><sub>1","T"\n')
            else:
                f200.write('variables="period","x/<greek>l</greek><sub>1","<greek>r</greek>"\n')
                f201.write('variables="period","x/<greek>l</greek><sub>1","u"\n')
                f202.write('variables="period","x/<greek>l</greek><sub>1","T"\n')
        for i in range(ncell):
            xx = float(i) * dx / lamda1
            rho = float(np_cell[i]) / rho0 / float(n_sample_steps)
            if np_cell[i] > 0:
                uu = float(np.mean(ua[:np_cell[i], i]))
                Tp = get_T(ua[:np_cell[i], i], va[:np_cell[i], i], wa[:np_cell[i], i], RR)
            else:
                uu = 0.0
                Tp = 0.0
            if period is None:
                f200.write(f' {xx} {rho}\n')
                f201.write(f' {xx} {uu}\n')
                f202.write(f' {xx} {Tp}\n')
            else:
                f200.write(f' {period} {xx} {rho}\n')
                f201.write(f' {period} {xx} {uu}\n')
                f202.write(f' {period} {xx} {Tp}\n')


def get_pdf_cell(ua, va, wa, np_cell, ipdf, outdir, n_sample_steps):
    """Write velocity PDF for cell ipdf (0-based).

    ua, va, wa: accumulation arrays shaped (np_cell_max, ncell)
    np_cell: particle count per cell, shape (ncell,)
    ipdf: 0-based cell index for the PDF output
    n_sample_steps: number of time steps used to accumulate samples
    """
    nbin = 201
    half = (nbin - 1) // 2  # 100
    binsize = 0.1 * beta1

    # File named with 1-based cell number to match Fortran output
    cell_num = ipdf + 1
    id2 = (cell_num % 1000) // 100
    id1 = (cell_num % 100) // 10
    id0 = cell_num % 10
    fname = os.path.join(outdir, f'pdf_cell_{id2}{id1}{id0}.dat')

    with open(fname, 'w') as f101:
        f101.write('variables="x","f(u)","f(v)","f(w)","f1_e(u)","f2_e(u)","f1_e(v)","f2_e(v)"\n')

        if 0 <= ipdf < ncell:
            pdf_u = np.zeros(nbin)
            pdf_v = np.zeros(nbin)
            pdf_w = np.zeros(nbin)
            pdf_c = np.zeros(nbin)
            um = 0.0
            cm_val = 0.0
            npp = np_cell[ipdf]

            for pp in range(npp):
                u_p = ua[pp, ipdf]
                v_p = va[pp, ipdf]
                w_p = wa[pp, ipdf]

                idx_u = int(u_p / binsize + 0.5) if u_p >= 0.0 else int(u_p / binsize - 0.5)
                if -half <= idx_u <= half:
                    pdf_u[idx_u + half] += 1.0

                idx_v = int(v_p / binsize + 0.5) if v_p >= 0.0 else int(v_p / binsize - 0.5)
                if -half <= idx_v <= half:
                    pdf_v[idx_v + half] += 1.0

                idx_w = int(w_p / binsize + 0.5) if w_p >= 0.0 else int(w_p / binsize - 0.5)
                if -half <= idx_w <= half:
                    pdf_w[idx_w + half] += 1.0

                c = math.sqrt(u_p**2 + v_p**2 + w_p**2)
                idx_c = int(c / binsize + 0.5)
                if -half <= idx_c <= half:
                    pdf_c[idx_c + half] += 1.0

                cm_val += math.sqrt((u_p - U1)**2 + v_p**2 + w_p**2)
                um += u_p

            if np.sum(pdf_u) > 0.0:
                pdf_u = pdf_u / np.sum(pdf_u) / binsize
            if np.sum(pdf_v) > 0.0:
                pdf_v = pdf_v / np.sum(pdf_v) / binsize
            if np.sum(pdf_w) > 0.0:
                pdf_w = pdf_w / np.sum(pdf_w) / binsize
            if np.sum(pdf_c) > 0.0:
                pdf_c = pdf_c / np.sum(pdf_c) / binsize

            for i in range(-half, half + 1):
                x = float(i) * binsize
                pdf_ve1 = beta1 / math.sqrt(pi) * math.exp(-beta1**2 * x**2)
                pdf_ve2 = beta2 / math.sqrt(pi) * math.exp(-beta2**2 * x**2)
                c = x - U1
                pdf_ue1 = beta1 / math.sqrt(pi) * math.exp(-beta1**2 * c**2)
                c = x - U2
                pdf_ue2 = beta2 / math.sqrt(pi) * math.exp(-beta2**2 * c**2)
                f101.write(f' {x} {pdf_u[i+half]} {pdf_v[i+half]} {pdf_w[i+half]}'
                           f' {pdf_ue1} {pdf_ue2} {pdf_ve1} {pdf_ve2}\n')

sigmoid = lambda x, rho, slope, center : rho/(1+np.exp(-4*slope*(x-center)))

def get_shock_center(np_cell_avg, verbose=False):
    try:
        rho_denom = float(N0) / (float(ncell) * (2.0 / 3.0) * rho1 + float(ncell) * (1.0 / 3.0) * rho2)
        rho_cell_adj = np_cell_avg/rho_denom - 1         # normalize so that rho_adj ~ 0 upstream
        cell_centers = L_tube*np.linspace(1/ncell, 1-1/ncell, ncell)

        # initial guess to aid convergence in the next step
        rho_est = np.mean(rho_cell_adj[-5:])
        middle = np.where((rho_cell_adj > 0.33*rho_est) & (rho_cell_adj < 0.67*rho_est))[0]
        x_0, x_1 = middle[0], middle[-1]
        slope_est = (rho_cell_adj[x_1]-rho_cell_adj[x_0])/(cell_centers[x_1]-cell_centers[x_0])
        center_est = np.mean(cell_centers[middle])
        scaling_factor = np.sqrt(center_est/slope_est)
        estimates = [rho_est, slope_est, center_est]
        scaled_estimates = [rho_est, slope_est*scaling_factor, center_est/scaling_factor]

        # fit curve using gauss-newton
        scaled_fit = curve_fit(sigmoid, cell_centers[3:]/scaling_factor, rho_cell_adj[3:], p0=scaled_estimates)[0]
        fit = [scaled_fit[0], scaled_fit[1]/scaling_factor, scaled_fit[2]*scaling_factor]
        center = fit[2]
        if verbose:
            print(f'  estimates: rho={estimates[0]:.3f}  slope={estimates[1]:.3f}  center={estimates[2]:.3f}')
            print(f'  fit:       rho={fit[0]:.3f}  slope={fit[1]:.3f}  center={fit[2]:.3f}')
    except:
        center = -1
        
    return center

_timers = {'collision': 0.0, 'shock_center': 0.0, 'file_io': 0.0}

# ===== Main Program =====

def _run_step(i, u, v, w, x, idx, cell, cr_max, std1, N_in, vol_cell, Uw,
              ua, va, wa, np_cell, np_cell_max, accumulate, maxwell=False):
    """Execute one time step: inject, move, collide, optionally accumulate samples.
    Returns (np_count, n_removed_right)."""
    p_in = int(N_in + 0.5)

    # Pre-injection active mask: used for the full-step move and boundary checks.
    # Injected particles are placed directly at their end-of-step position, so
    # they skip the move.
    active_slots = (idx == 1)

    # ---- Vectorized injection: batch-sample u_ini/v/w in one call each.
    # Injected particles are placed at x = u*dt*r (r ~ U[0,1]) so they are
    # uniformly distributed in [0, u*dt] at the end of the step, matching a
    # uniformly-distributed entry time within [0, dt]. ----
    inactive_slots = np.where(~active_slots)[0]
    to_inject = min(p_in, len(inactive_slots))
    injection_slots = inactive_slots[:to_inject] if to_inject > 0 else None
    if to_inject > 0:
        u[injection_slots] = _sample_u_ini_batch(beta1, U1, to_inject)
        v[injection_slots] = np.random.normal(0.0, std1, size=to_inject)
        w[injection_slots] = np.random.normal(0.0, std1, size=to_inject)
        idx[injection_slots] = 1
        x[injection_slots] = u[injection_slots] * dt * np.random.random(size=to_inject)

    # ---- Detect right-crossers among pre-existing actives BEFORE moving
    # (injected particles cannot cross L_tube in one partial step). ----
    right = active_slots & (x + u * dt >= L_tube)

    # ---- Move pre-existing active particles by full u*dt ----
    x[active_slots] += u[active_slots] * dt

    # Fold newly-injected particles into active_slots for downstream ops
    if to_inject > 0:
        active_slots[injection_slots] = True
    np_count = int(np.sum(active_slots))

    # ---- Reflect right-crossers off the piston (Fortran-code method) ----
    if right.any():
        # Recover old positions from (new x, unchanged u) — small array, only right-crossers
        x_old_right = x[right] - u[right] * dt
        u[right] = 2.0 * Uw - u[right]
        x[right] = 2.0 * L_tube - x_old_right + u[right] * dt

        still_right = right & (x >= L_tube)
        n_removed_right = int(np.sum(still_right))
        if n_removed_right > 0:
            idx[still_right] = 0
            cell[still_right] = -1
            np_count -= n_removed_right
            active_slots &= ~still_right                # keep `active` in sync w/o recomputing (idx == 1)
    else:
        n_removed_right = 0

    # ---- Left-boundary removal (reflected particles land near L_tube, so ~right is redundant) ----
    left = active_slots & (x < 0.0)
    if left.any():
        n_left = int(np.sum(left))
        idx[left] = 0
        cell[left] = -1
        np_count -= n_left
        active_slots &= ~left

    # ---- Cell assignment (active now equals still_active) ----
    raw_cells = (x[active_slots] / L_tube * float(ncell)).astype(int)
    cell[active_slots] = np.clip(raw_cells, 0, ncell - 1)

    # ---- Build sorted cell index once — O(N log N) vs O(N * ncell) per-cell search ----
    active_indices = np.where(active_slots)[0]
    _sort_order  = np.argsort(cell[active_indices], kind='stable')
    sorted_pix   = active_indices[_sort_order]       # particle indices sorted by cell
    cells_sorted = cell[sorted_pix]                  # cell id of each sorted particle
    cell_starts  = np.searchsorted(cells_sorted, np.arange(ncell))
    cell_ends    = np.searchsorted(cells_sorted, np.arange(ncell), side='right')

    # Midpoint cell density (for adaptive piston control).
    # ncell//2 is the cell closer to the right boundary when ncell is even.
    target_shock_center = (ncell*2) // 3
    rho0_norm = float(N0) / (float(ncell) * (2.0/3.0 * rho1 + 1.0/3.0 * rho2))
    rho_halfway = (cell_ends[target_shock_center] - cell_starts[target_shock_center]) / rho0_norm

    _t0 = time.perf_counter()
    for k in range(ncell):
        collision(u, v, w, sorted_pix[cell_starts[k]:cell_ends[k]], cr_max, k, vol_cell, maxwell)
    _timers['collision'] += time.perf_counter() - _t0

    if accumulate:                        # true if period is active, i.e. not in warmup or gap
        if len(sorted_pix) > 0:
            local_pos = np.arange(len(sorted_pix)) - cell_starts[cells_sorted]
            rows = np_cell[cells_sorted] + local_pos
            valid_mask = rows < np_cell_max
            pidx = sorted_pix[valid_mask]
            rows = rows[valid_mask]
            cols = cells_sorted[valid_mask]
            ua[rows, cols] = u[pidx]
            va[rows, cols] = v[pidx]
            wa[rows, cols] = w[pidx]
            counts = cell_ends - cell_starts
            np_cell += np.minimum(counts, np.maximum(0, np_cell_max - np_cell))

    return np_count, n_removed_right, rho_halfway


def main():
    args = parse_args()

    global n, N0, Nmax, Wt
    n    = args.n
    N0   = int(n * area * L_tube)
    Nmax = int(3.0 * N0)
    Wt   = Nreal / N0

    np.random.seed(args.seed)

    n_sample_steps = args.period

    np_cell_max = int(float(Nmax) * float(n_sample_steps) / float(ncell) * 5.0)

    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / 'output'
    output_root.mkdir(exist_ok=True)
    outdir = str(output_root / ('shock_output_' + datetime.now().strftime('%m%d%y_%H%M%S')))
    os.makedirs(outdir)

    tee = _Tee(os.path.join(outdir, 'run_info.txt'))
    sys.stdout = tee
    cmd = 'python ' + ' '.join([Path(sys.argv[0]).name] + sys.argv[1:])
    print(cmd)
    print(f'Output directory: {outdir}')

    u = np.zeros(Nmax) # x velocity component array
    v = np.zeros(Nmax) # y
    w = np.zeros(Nmax) # z
    x = np.zeros(Nmax) # x position array
    ua = np.zeros((np_cell_max, ncell))
    va = np.zeros((np_cell_max, ncell))
    wa = np.zeros((np_cell_max, ncell))
    np_cell = np.zeros(ncell, dtype=int)
    idx = np.zeros(Nmax, dtype=int)     # idx[i] = 0 -> no particle with index i
    cell = np.full(Nmax, -1, dtype=int)
    cr_max = np.zeros(ncell)

    cr_max1 = 2.0 * cm1
    cr_max2 = 2.0 * cm2
    for i in range(ncell): # iterate over the length, ncell * dx = length
        cr_max[i] = cr_max1 if i < ncell * 2 // 3 else cr_max2

    vol_cell = area * dx

    tmp1 = 2.0 * beta1**2
    tmp2 = math.sqrt(pi) * s * (1.0 + erf(s)) + math.exp(-s**2)
    k1 = tmp1 / tmp2
    Nrate = (rho1 / mm) * (AVOG / Wt) * beta1 / (k1 * math.sqrt(pi))
    N_in = Nrate * dt * area

    print_parameters()
    print(f'N_in= {N_in}')
    print(f'mean collision time= {(float(N0) / float(ncell) * sigma_t * 2.0 * cm1)**(-1)}')

    # Initialize particles
    std1 = 1.0 / (math.sqrt(2.0) * beta1) # upstream
    std2 = 1.0 / (math.sqrt(2.0) * beta2) # downstream
    if args.initialize is None:
        prob = 2.0 * rho1 / (2.0 * rho1 + rho2)    # probability the particle is upstream (shock at 2/3)
        for j in range(N0):
            R1 = np.random.random()
            R2 = np.random.random()
            if R1 <= prob:
                x[j] = R2 * 2.0 * L_tube / 3.0             # place randomly on left 2/3
                u[j] = gasdev(0.0, std1) + U1              # with bulk velocity U1
                v[j] = gasdev(0.0, std1)
                w[j] = gasdev(0.0, std1)
            else:
                x[j] = R2 * L_tube / 3.0 + 2.0 * L_tube / 3.0  # place randomly on right 1/3
                u[j] = gasdev(0.0, std2) + U2              # with bulk velocity U2
                v[j] = gasdev(0.0, std2)
                w[j] = gasdev(0.0, std2)
            idx[j] = 1
            cell[j] = min(ncell - 1, int((x[j] / L_tube) * float(ncell)))
    elif args.initialize != 'empty':
        candidates = [
            Path(args.initialize) / 'state.dat',
            script_dir / args.initialize / 'state.dat',
            output_root / f'shock_output_{args.initialize}' / 'state.dat',
        ]
        state_path = next((p for p in candidates if p.is_file()), None)
        if state_path is None:
            sys.exit('Error: state.dat not found. Tried:\n' +
                     '\n'.join(f'  {p}' for p in candidates))
        state_data = np.loadtxt(str(state_path), skiprows=1)
        if state_data.ndim == 1:
            state_data = state_data.reshape(1, -1)
        for j, (xj, uj, vj, wj) in enumerate(state_data[:Nmax]):
            x[j], u[j], v[j], w[j] = xj, uj, vj, wj
            idx[j] = 1
            cell[j] = min(ncell - 1, int(xj / L_tube * float(ncell)))
        print(f'Initialized from state: {args.initialize} ({min(len(state_data), Nmax)} particles)')
    # else: args.initialize == 'empty' -> tube starts empty (arrays already zeroed)

    # Target particle count for adaptive_mode=particles
    if args.initialize not in (None, 'empty'):
        np_count_target = min(len(state_data), Nmax)
    else:
        np_count_target = N0

    Uw = U2 * args.piston_speed          # precomputed once; passed to _run_step
    maxwell = (args.model == 'maxwell')
    np_count = 0
    warmup_print_stride = max(10, args.warmup // 100)
    period_print_stride = max(10, args.period // 10) if args.period else 10

    with open(os.path.join(outdir, 'npt.dat'), 'w') as f_npt_global, \
         open(os.path.join(outdir, 'removed_right.dat'), 'w') as f_rr, \
         open(os.path.join(outdir, 'shock_center.dat'), 'w') as f_sc, \
         open(os.path.join(outdir, 'piston_speed.dat'), 'w') as f_ps:
        f_npt_global.write('variables="t","Number of particles in the tube"\n')
        f_rr.write('variables="t","removed/N_in"\n')
        f_sc.write('variables="t","center"\n')
        f_ps.write('variables="t","Uw"\n')

        # Warmup phase
        warmup_end = 0
        if args.initialize == 'empty':
            # Dynamic warmup: run until np_count falls below the count from 10 steps prior
            history = collections.deque(maxlen=10)
            while True:
                warmup_end += 1
                np_count, n_rr, rho_halfway = _run_step(warmup_end, u, v, w, x, idx, cell, cr_max, std1, N_in, vol_cell, Uw,
                                     ua, va, wa, np_cell, np_cell_max, accumulate=False, maxwell=maxwell)
                if args.adaptive is not None:
                    if args.adaptive_mode == 'particles':
                        Uw = U2 + U2 * 100 * ((np_count - np_count_target) / np_count_target) * args.adaptive
                    else:
                        Uw = U2 + U2 * (rho_halfway - (rho1 + rho2) / 2.0) * args.adaptive
                _t0 = time.perf_counter()
                f_npt_global.write(f' {warmup_end * dt} {float(np_count) / float(N0)}\n')
                f_rr.write(f' {warmup_end * dt} {float(n_rr) / N_in}\n')
                f_ps.write(f' {warmup_end * dt} {Uw}\n')
                _timers['file_io'] += time.perf_counter() - _t0
                if (warmup_end - 1) % warmup_print_stride == 0:
                    print(f'warmup step= {warmup_end}  np= {np_count}')
                history.append(np_count)
                if len(history) == 10 and np_count < history[0]:
                    break
            print(f'Warmup complete: {warmup_end} steps (dynamic)')
        else:
            for warmup_end in range(1, args.warmup + 1):
                np_count, n_rr, rho_halfway = _run_step(warmup_end, u, v, w, x, idx, cell, cr_max, std1, N_in, vol_cell, Uw,
                                     ua, va, wa, np_cell, np_cell_max, accumulate=False, maxwell=maxwell)
                if args.adaptive is not None:
                    if args.adaptive_mode == 'particles':
                        Uw = U2 + U2 * 100 * ((np_count - np_count_target) / np_count_target) * args.adaptive
                    else:
                        Uw = U2 + U2 * (rho_halfway - (rho1 + rho2) / 2.0) * args.adaptive
                _t0 = time.perf_counter()
                f_npt_global.write(f' {warmup_end * dt} {float(np_count) / float(N0)}\n')
                f_rr.write(f' {warmup_end * dt} {float(n_rr) / N_in}\n')
                f_ps.write(f' {warmup_end * dt} {Uw}\n')
                _timers['file_io'] += time.perf_counter() - _t0
                if (warmup_end - 1) % warmup_print_stride == 0:
                    print(f'warmup step= {warmup_end}  np= {np_count}')
            print(f'Warmup complete: {args.warmup} steps')

        # Timeseries phase: one output set per period, with optional gaps between periods
        gap = args.gap
        stride = args.period + gap  # steps consumed per period slot
        _init_count = np.bincount(cell[idx == 1], minlength=ncell).astype(float)
        np_cell_buf = collections.deque([_init_count] * args.smooth_center, maxlen=args.smooth_center)
        for p in range(1, args.num_periods + 1):
            ua[:] = 0.0
            va[:] = 0.0
            wa[:] = 0.0
            np_cell[:] = 0

            for local_step in range(1, args.period + 1):
                global_step = warmup_end + (p - 1) * stride + local_step
                np_count, n_rr, rho_halfway = _run_step(global_step, u, v, w, x, idx, cell, cr_max,
                                     std1, N_in, vol_cell, Uw,
                                     ua, va, wa, np_cell, np_cell_max, accumulate=True, maxwell=maxwell)
                if args.adaptive is not None:
                    if args.adaptive_mode == 'particles':
                        Uw = U2 + U2 * 100 * ((np_count - np_count_target) / np_count_target) * args.adaptive
                    else:
                        Uw = U2 + U2 * (rho_halfway - (rho1 + rho2) / 2.0) * args.adaptive
                _t0 = time.perf_counter()
                f_npt_global.write(f' {global_step * dt} {float(np_count) / float(N0)}\n')
                f_rr.write(f' {global_step * dt} {float(n_rr) / N_in}\n')
                f_ps.write(f' {global_step * dt} {Uw}\n')
                _timers['file_io'] += time.perf_counter() - _t0
                _t0 = time.perf_counter()
                np_cell_buf.append(np.bincount(cell[idx == 1], minlength=ncell).astype(float))
                center = get_shock_center(sum(np_cell_buf) / len(np_cell_buf),
                                         verbose=(local_step == 1))
                _timers['shock_center'] += time.perf_counter() - _t0
                _t0 = time.perf_counter()
                f_sc.write(f' {global_step * dt} {center}\n')
                _timers['file_io'] += time.perf_counter() - _t0
                if (local_step - 1) % period_print_stride == 0:
                    print(f'period= {p}  step= {local_step}  np= {np_count}')

            _t0 = time.perf_counter()
            write_spatial_profiles(ua, va, wa, np_cell, outdir, n_sample_steps, period=p)
            _timers['file_io'] += time.perf_counter() - _t0
            print(f'period= {p}  particle number ratio (Np/N0) {float(np_count) / float(N0)}')

            # Gap phase: run between periods (skip after the last period)
            if gap > 0 and p < args.num_periods:
                gap_print_stride = max(10, gap // 100)
                for gap_step in range(1, gap + 1):
                    global_step = warmup_end + (p - 1) * stride + args.period + gap_step
                    np_count, n_rr, rho_halfway = _run_step(global_step, u, v, w, x, idx, cell, cr_max,
                                         std1, N_in, vol_cell, Uw,
                                         ua, va, wa, np_cell, np_cell_max, accumulate=False, maxwell=maxwell)
                    if args.adaptive is not None:
                        if args.adaptive_mode == 'particles':
                            Uw = U2 + U2 * 100 * ((np_count - np_count_target) / np_count_target) * args.adaptive
                        else:
                            Uw = U2 + U2 * (rho_halfway - (rho1 + rho2) / 2.0) * args.adaptive
                    _t0 = time.perf_counter()
                    f_npt_global.write(f' {global_step * dt} {float(np_count) / float(N0)}\n')
                    f_rr.write(f' {global_step * dt} {float(n_rr) / N_in}\n')
                    f_ps.write(f' {global_step * dt} {Uw}\n')
                    _timers['file_io'] += time.perf_counter() - _t0
                    _t0 = time.perf_counter()
                    center = get_shock_center(np.bincount(cell[idx == 1], minlength=ncell).astype(float))
                    _timers['shock_center'] += time.perf_counter() - _t0
                    _t0 = time.perf_counter()
                    f_sc.write(f' {global_step * dt} {center}\n')
                    _timers['file_io'] += time.perf_counter() - _t0
                    if (gap_step - 1) % gap_print_stride == 0:
                        print(f'period= {p}  gap step= {gap_step}  np= {np_count}')

    print(f'Total particle number {N0}')

    _t0 = time.perf_counter()
    active = idx == 1
    with open(os.path.join(outdir, 'state.dat'), 'w') as f_state:
        f_state.write('variables="x","u","v","w"\n')
        np.savetxt(f_state, np.column_stack([x[active], u[active], v[active], w[active]]),
                   fmt=' %.15g')
    _timers['file_io'] += time.perf_counter() - _t0

    end_time = datetime.now()
    total_s = (end_time - start_time).total_seconds()
    print('Elapsed time:', end_time - start_time)
    print('\nTiming breakdown:')
    for cat, secs in _timers.items():
        pct = 100.0 * secs / total_s if total_s > 0 else 0.0
        print(f'  {cat:20s}: {secs:8.3f}s  ({pct:.1f}%)')
    other_s = total_s - sum(_timers.values())
    print(f'  {"other":20s}: {other_s:8.3f}s  ({100.0 * other_s / total_s:.1f}%)')
    tee.close()


if __name__ == '__main__':
    main()
