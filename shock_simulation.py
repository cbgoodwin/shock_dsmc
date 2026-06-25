#!/usr/bin/env python3
"""
Direct Simulation Monte Carlo (DSMC) shock simulation.
Translated from shock_simulation_260317.f90
"""

import math
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.special import erf
start_time = datetime.now()
# ===== Parameters =====

pi = math.pi
s = 2.0          # ratio of bulk flow to most probable speed
beta1 = 1.0      # inverse of most probable speed
U1 = s / beta1   # bulk flow velocity
n = 5000         # particles per unit volume

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
sigma_t = pi * dm1                                                     # collision cross section? 
cm1 = 2.0 / (math.sqrt(pi) * beta1)   # mean molecular speed
lamda1 = cm1 * rho1 / mu1             # mean free path
a1 = math.sqrt(gamma * RR * Tp1)      # speed of sound
Ma1 = U1 / a1                         # Mach speed of gas

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
Nt = 100
L_tube = dx * ncell
N0 = int(n * area * L_tube)
Nmax = int(1.2 * N0)
Nreal = ((rho1 + rho2) / mm) * AVOG * 60.0 * (lamda1 / 2.0) * area
Wt = Nreal / N0      # scaling factor simulated # to real #?
cell_stat = 92
Nt_start = int(Nt * 0.8)
Nt_end = Nt
np_cell_max = int(float(Nmax) * float(Nt_end - Nt_start + 1) / float(ncell) * 5.0)


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

def collision(u, v, w, cell, cell_no, cr_max, vol_cell):
    """DSMC collision step for a single cell (0-based cell_no)."""
    sigma_max = sigma_t

    pc = np.where(cell == cell_no)[0]  # get indices of all particles in the cell
    nc = len(pc)
    if nc < 2:
        return

    uc = u[pc].copy()
    vc = v[pc].copy()
    wc = w[pc].copy()

    Nt_colli = int(0.5 * nc * (nc - 1.0) * (sigma_t * cr_max[cell_no]) * dt * Wt / (vol_cell * AVOG)) # number of collisions
    Nt_colli = max(0, Nt_colli)

    for _ in range(Nt_colli):
        pp1 = int(np.random.random() * nc)
        pp2 = int(np.random.random() * nc)
        if pp1 == pp2:
            continue

        cr_x = uc[pp1] - uc[pp2]
        cr_y = vc[pp1] - vc[pp2]
        cr_z = wc[pp1] - wc[pp2]
        cr = math.sqrt(cr_x**2 + cr_y**2 + cr_z**2)

        P_accept = cr * sigma_t / max(cr_max[cell_no], 1.0e-12) / sigma_max
        P_accept = min(max(P_accept, 0.0), 1.0)

        if np.random.random() <= P_accept:
            theta = 2.0 * pi * np.random.random()             # uniformly random in theta
            psi = math.acos(1.0 - 2.0 * np.random.random())   # not uniformly random in psi

            # original
            crt_x = cr * math.cos(theta) * math.sin(psi)
            crt_y = cr * math.sin(theta) * math.sin(psi)
            crt_z = cr * math.cos(psi)

            #crt_x = cr * math.cos(psi)
            #crt_y = cr * math.cos(theta) * math.sin(psi)
            #crt_z = cr * math.sin(theta) * math.sin(psi)

            cm_x = 0.5 * (uc[pp1] + uc[pp2])
            cm_y = 0.5 * (vc[pp1] + vc[pp2])
            cm_z = 0.5 * (wc[pp1] + wc[pp2])

            uc[pp1] = cm_x + 0.5 * crt_x
            vc[pp1] = cm_y + 0.5 * crt_y
            wc[pp1] = cm_z + 0.5 * crt_z
            uc[pp2] = cm_x - 0.5 * crt_x
            vc[pp2] = cm_y - 0.5 * crt_y
            wc[pp2] = cm_z - 0.5 * crt_z

            cr_x = uc[pp1] - uc[pp2]
            cr_y = vc[pp1] - vc[pp2]
            cr_z = wc[pp1] - wc[pp2]
            cr = math.sqrt(cr_x**2 + cr_y**2 + cr_z**2)
            cr_max[cell_no] = max(cr, cr_max[cell_no])

    u[pc] = uc
    v[pc] = vc
    w[pc] = wc


def get_pdf_cell(ua, va, wa, np_cell, ipdf, outdir):
    """Write velocity PDF for cell ipdf (0-based) and spatial profiles for all cells.

    ua, va, wa: accumulation arrays shaped (np_cell_max, ncell)
    np_cell: particle count per cell, shape (ncell,)
    ipdf: 0-based cell index for the PDF output
    """
    nbin = 201
    half = (nbin - 1) // 2  # 100
    binsize = 0.1 * beta1
    rho0 = float(N0) / (float(ncell) * rho1 * 0.5 + float(ncell) * rho2 * 0.5)

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

    with open(os.path.join(outdir, 'rho.dat'), 'w') as f200, \
         open(os.path.join(outdir, 'u.dat'), 'w') as f201, \
         open(os.path.join(outdir, 'T.dat'), 'w') as f202:
        f200.write('variables="x/<greek>l</greek><sub>1","<greek>r</greek>"\n')
        f201.write('variables="x/<greek>l</greek><sub>1","u"\n')
        f202.write('variables="x/<greek>l</greek><sub>1","T"\n')

        for i in range(ncell):
            xx = float(i) * dx / lamda1
            rho = float(np_cell[i]) / rho0 / float(Nt_end - Nt_start + 1)
            if np_cell[i] > 0:
                uu = float(np.mean(ua[:np_cell[i], i]))
                Tp = get_T(ua[:np_cell[i], i], va[:np_cell[i], i], wa[:np_cell[i], i], RR)
            else:
                uu = 0.0
                Tp = 0.0
            f200.write(f' {xx} {rho}\n')
            f201.write(f' {xx} {uu}\n')
            f202.write(f' {xx} {Tp}\n')


# ===== Main Program =====

def main():
    script_dir = str(Path(__file__).resolve().parent)
    outdir = script_dir + '/shock_output_' + datetime.now().strftime('%m%d%y_%H%M%S')
    os.makedirs(outdir)
    print(f'Output directory: {outdir}')

    u = np.zeros(Nmax) # x velocity component array
    v = np.zeros(Nmax) # y
    w = np.zeros(Nmax) # z
    x = np.zeros(Nmax) # x position array
    ua = np.zeros((np_cell_max, ncell)) #
    va = np.zeros((np_cell_max, ncell)) #
    wa = np.zeros((np_cell_max, ncell)) #
    np_cell = np.zeros(ncell, dtype=int)
    idx = np.zeros(Nmax, dtype=int)     # idx[i] = 0 -> no particle with index i 
    cell = np.full(Nmax, -1, dtype=int)
    cr_max = np.zeros(ncell)

    cr_max1 = 2.0 * cm1
    cr_max2 = 2.0 * cm2
    for i in range(ncell): # iterate over the length, ncell * dx = length
        cr_max[i] = cr_max1 if i < ncell // 2 else cr_max2

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
    prob = rho1 / (rho1 + rho2)    # probability the particle is upstream, given shock is in the center?
    std1 = 1.0 / (math.sqrt(2.0) * beta1) # upstream
    std2 = 1.0 / (math.sqrt(2.0) * beta2) # downstream
    for j in range(N0):     
        R1 = np.random.random()
        R2 = np.random.random()
        if R1 <= prob:
            x[j] = R2 * L_tube / 2.0                   # place randomly on left side
            u[j] = gasdev(0.0, std1) + U1              # with bulk velocity U1
            v[j] = gasdev(0.0, std1)
            w[j] = gasdev(0.0, std1)
        else:
            x[j] = R2 * L_tube / 2.0 + L_tube / 2.0    # place randomly on right side
            u[j] = gasdev(0.0, std2) + U2              # with bulk velocity U2
            v[j] = gasdev(0.0, std2)
            w[j] = gasdev(0.0, std2)
        idx[j] = 1
        cell[j] = min(ncell - 1, int((x[j] / L_tube) * float(ncell)))

    np_count = 0

    with open(os.path.join(outdir, 'npt.dat'), 'w') as f10:
        f10.write('variables="t","Number of particles in the tube"\n')

        for i in range(1, Nt + 1):
            p_in = int(N_in + 0.5)     # N_in 
            um = 0.0

            # Inject new particles into inactive slots, then move all active particles.
            # (Mirrors Fortran's single pass: inject into idx==0, then move idx==1.)
            inactive_slots = np.where(idx == 0)[0]
            to_inject = min(p_in, len(inactive_slots))
            for ji in range(to_inject):
                j = inactive_slots[ji]
                u[j] = u_ini(beta1, U1)
                v[j] = gasdev(0.0, std1)
                w[j] = gasdev(0.0, std1)
                idx[j] = 1
                x[j] = 0.0

            # Move all active particles
            active = (idx == 1)
            x0 = x.copy()
            x[active] += u[active] * dt
            np_count = int(np.sum(active))

            # Right boundary: reflect velocity around U2, recompute position
            right = active & (x >= L_tube)
            u[right] = 2.0 * U2 - u[right]
            x[right] = 2.0 * L_tube - x0[right] + u[right] * dt

            # Still past right boundary after reflection: remove
            still_right = right & (x >= L_tube)
            idx[still_right] = 0
            cell[still_right] = -1
            np_count -= int(np.sum(still_right))

            # Left boundary: remove particles that exit
            left = active & ~right & (x < 0.0)
            idx[left] = 0
            cell[left] = -1
            np_count -= int(np.sum(left))

            # Update cell assignments for remaining active particles
            still_active = (idx == 1)
            raw_cells = (x[still_active] / L_tube * float(ncell)).astype(int)
            cell[still_active] = np.clip(raw_cells, 0, ncell - 1)

            # Collision step for each cell
            for k in range(ncell):
                collision(u, v, w, cell, k, cr_max, vol_cell)

            f10.write(f' {i * dt} {float(np_count) / float(N0)}\n')
            print(f'step= {i}  np= {np_count}')

            # Accumulate velocity samples for statistics
            if Nt_start <= i <= Nt_end:
                acc_idx = np.where(idx == 1)[0]
                for ji in range(len(acc_idx)):
                    j = acc_idx[ji]
                    c = cell[j]
                    if np_cell[c] < np_cell_max:
                        ua[np_cell[c], c] = u[j]
                        va[np_cell[c], c] = v[j]
                        wa[np_cell[c], c] = w[j]
                        np_cell[c] += 1

    print(np_cell / float(Nt_end - Nt_start + 1))
    print(f'particle number ratio (Np/N0) {float(np_count) / float(N0)}')
    print(f'um= {um}')
    print(f'Np for cell statistics= {np_cell}')
    print(f'Total particle number {N0}')

    # Write PDF output for cells near the shock (0-based: Fortran cells 63..119 => indices 62..118)
    for i in range(-30, 27, 2):
        cell_idx = 92 + i  # 0-based equivalent of Fortran's (93 + i)
        if 0 <= cell_idx < ncell:
            get_pdf_cell(ua, va, wa, np_cell, cell_idx, outdir)


if __name__ == '__main__':
    main()
    end_time = datetime.now()
    elapsed = end_time - start_time
    print('Elapsed time:' , elapsed)