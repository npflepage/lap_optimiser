"""
═══════════════════════════════════════════════════════════════════════════════
  PHYSICS-CONSTRAINED MINIMUM LAP TIME OPTIMISER  —  Fourier representation
═══════════════════════════════════════════════════════════════════════════════

Free variables (optimised):
  cd  — Fourier coefficients of the signed normal deviation d(t)   [2*N+1]
  cv  — Fourier coefficients of the physical speed       v(t)      [2*N+1]

The racing path is:   r(t) = c(t) + d(t) * N_c(t)
where c(t) is the Fourier-fitted centerline and N_c its analytical normal.

Objective:   minimise lap time  T = ∫ g_r / v  dt
Constraints:
  1. Track limits     :  -w_left(t) ≤ d(t) ≤ w_right(t)
  2. Friction ellipse :  (a_lon/a_lon_max(v))² + (a_lat/a_side(v))² ≤ 1
  3. Speed positivity :  v(t) ≥ v_min   (avoids degenerate / divide-by-zero)

Physics model (real F1, 750 kg, ~1000 hp):
  max_front_acc, max_back_acc, max_side_acc  — speed-dependent grip envelope
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from scipy.optimize import minimize
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
#  FOURIER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def build_design_matrix(t_vec, N):
    A = np.ones((len(t_vec), 2 * N + 1))
    for k in range(1, N + 1):
        A[:, 2*k-1] = np.cos(k * t_vec)
        A[:, 2*k]   = np.sin(k * t_vec)
    return A

def build_derivative_matrix(t_vec, N, order=1):
    D = np.zeros((len(t_vec), 2 * N + 1))
    for k in range(1, N + 1):
        if order == 1:
            D[:, 2*k-1] =  k    * (-np.sin(k * t_vec))
            D[:, 2*k]   =  k    * ( np.cos(k * t_vec))
        elif order == 2:
            D[:, 2*k-1] = -k**2 *   np.cos(k * t_vec)
            D[:, 2*k]   = -k**2 *   np.sin(k * t_vec)
        elif order == 3:
            D[:, 2*k-1] =  k**3 *   np.sin(k * t_vec)
            D[:, 2*k]   = -k**3 *   np.cos(k * t_vec)
    return D

def fit_fourier(t_data, y_data, N):
    A = build_design_matrix(t_data, N)
    c, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)
    return c

def compute_frenet(dx, dy, ddx, ddy):
    g      = np.sqrt(dx**2 + dy**2)
    Tx, Ty = dx / g, dy / g
    Nx, Ny = -Ty, Tx
    kappa  = (dx * ddy - dy * ddx) / (g**3)
    return g, Tx, Ty, Nx, Ny, kappa

def project_points_onto_fourier(px, py, x_f, y_f, nx_f, ny_f):
    signed_dist = np.zeros(len(px))
    for i in range(len(px)):
        dist = (px[i] - x_f)**2 + (py[i] - y_f)**2
        j    = np.argmin(dist)
        signed_dist[i] = (px[i]-x_f[j]) * nx_f[j] + (py[i]-y_f[j]) * ny_f[j]
    return signed_dist

# ═══════════════════════════════════════════════════════════════════════════════
#  F1 PHYSICS MODEL  (real car: 750 kg, ~1000 hp, speed-dependent grip)
# ═══════════════════════════════════════════════════════════════════════════════

def max_front_acc(v):
    """Forward limit: min(traction+downforce, power) minus drag."""
    return (np.clip(7978.162499470494 + 2.1259279125731587 * v**2,
                    None, 745000.0 / np.maximum(v, 1e-3))) / 750.0 \
           - 0.745 * v**2 / 750.0

def max_back_acc(v):
    """Braking limit: grip + downforce + drag-assist (no power limit)."""
    return (11254.767298703786 + 2.2808897580274747 * v**2
            + 0.745 * v**2) / 750.0

def max_side_acc(v):
    """Lateral limit: grip + downforce, ×1.2 cornering bonus."""
    return 1.2 * (11254.767298703786 + 2.2808897580274747 * v**2) / 750.0

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA & BUILD CENTERLINE  (Steps 1-3 from the working pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv('Silverstone.csv')
x  = df['x_m'].values
y  = df['y_m'].values
w_right = df['w_tr_right_m'].values
w_left  = df['w_tr_left_m'].values

dx_raw = np.gradient(x);  dy_raw = np.gradient(y)
nrm    = np.sqrt(dx_raw**2 + dy_raw**2)
nx_raw, ny_raw = -dy_raw/nrm, dx_raw/nrm
x_right_raw = x + w_right * nx_raw;  y_right_raw = y + w_right * ny_raw
x_left_raw  = x - w_left  * nx_raw;  y_left_raw  = y - w_left  * ny_raw

# Arc-length parameterization
N  = 128
ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
s  = np.concatenate([[0], np.cumsum(ds)])
t  = 2 * np.pi * s / s[-1]

cx = fit_fourier(t, x, N)
cy = fit_fourier(t, y, N)

t_fine = np.linspace(0, 2 * np.pi, 2000)
dt     = t_fine[1] - t_fine[0]

A_fine = build_design_matrix(t_fine,    N)
D1c    = build_derivative_matrix(t_fine, N, 1)
D2c    = build_derivative_matrix(t_fine, N, 2)
D3c    = build_derivative_matrix(t_fine, N, 3)

xc   = A_fine @ cx;   yc   = A_fine @ cy
dxc  = D1c    @ cx;   dyc  = D1c    @ cy
ddxc = D2c    @ cx;   ddyc = D2c    @ cy
dddxc= D3c    @ cx;   dddyc= D3c    @ cy

g_c, Tcx, Tcy, Ncx, Ncy, kappa_c = compute_frenet(dxc, dyc, ddxc, ddyc)

# Project raw bounds → corrected widths → Fourier-fit them
d_right = project_points_onto_fourier(x_right_raw, y_right_raw, xc, yc, Ncx, Ncy)
d_left  = -project_points_onto_fourier(x_left_raw, y_left_raw,  xc, yc, Ncx, Ncy)

cw_right = fit_fourier(t, d_right, N)
cw_left  = fit_fourier(t, d_left,  N)
w_right_f = A_fine @ cw_right        # max positive deviation (toward +N_c)
w_left_f  = A_fine @ cw_left         # max negative deviation magnitude

ds_c = np.sqrt(np.diff(xc)**2 + np.diff(yc)**2)
s_c  = np.concatenate([[0], np.cumsum(ds_c)])

print(f'Centerline length: {np.sum(g_c)*dt:.2f} m   (official 5891 m)')

# ── Precompute centerline-normal derivatives (constant — needed each iter) ────
dg_c_dt  = (dxc * ddxc + dyc * ddyc) / g_c
dNcx_dt  = (-ddyc * g_c + dyc * dg_c_dt) / (g_c**2)
dNcy_dt  = ( ddxc * g_c - dxc * dg_c_dt) / (g_c**2)

num              = dxc * ddyc - dyc * ddxc
dnum             = dxc * dddyc - dyc * dddxc
d_kappa_c_g_c_dt = (dnum * g_c**2 - num * 2 * g_c * dg_c_dt) / (g_c**4)
kappa_c_g_c      = kappa_c * g_c
ddNcx_dt = -d_kappa_c_g_c_dt * Tcx - kappa_c_g_c**2 * Ncx
ddNcy_dt = -d_kappa_c_g_c_dt * Tcy - kappa_c_g_c**2 * Ncy

# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMISATION SETUP  (N=48 for both deviation and speed)
# ═══════════════════════════════════════════════════════════════════════════════

N_OPT  = 128
A_o    = build_design_matrix(t_fine,    N_OPT)
D1_o   = build_derivative_matrix(t_fine, N_OPT, 1)
D2_o   = build_derivative_matrix(t_fine, N_OPT, 2)
NC     = 2 * N_OPT + 1                # coefficients per series

V_MIN  = 5.0                          # m/s — speed positivity floor

def unpack(params):
    """Split flat parameter vector into deviation and speed coefficients."""
    return params[:NC], params[NC:]

def forward(params):
    """
    Compute everything needed for objective + constraints from parameters.
    Returns dict of path/kinematic quantities.
    """
    cd_, cv_ = unpack(params)

    # Deviation d(t) and derivatives
    d_t    = A_o  @ cd_
    dd_dt  = D1_o @ cd_
    ddd_dt = D2_o @ cd_

    # Speed v(t) and derivative
    v_r   = A_o  @ cv_
    dv_dt = D1_o @ cv_

    # Racing path via product rule
    dxr  = dxc + dd_dt * Ncx + d_t * dNcx_dt
    dyr  = dyc + dd_dt * Ncy + d_t * dNcy_dt
    ddxr = ddxc + ddd_dt * Ncx + 2 * dd_dt * dNcx_dt + d_t * ddNcx_dt
    ddyr = ddyc + ddd_dt * Ncy + 2 * dd_dt * dNcy_dt + d_t * ddNcy_dt

    g_r   = np.sqrt(dxr**2 + dyr**2)
    kappa = (dxr * ddyr - dyr * ddxr) / (g_r**3)

    # Physical accelerations
    dv_dtau = (v_r / g_r) * dv_dt          # longitudinal (signed)
    a_lat   = v_r**2 * kappa               # lateral (signed)
    a_lon   = dv_dtau

    return dict(d_t=d_t, v_r=v_r, g_r=g_r, kappa=kappa,
                a_lon=a_lon, a_lat=a_lat,
                dxr=dxr, dyr=dyr)

# ── Objective: lap time ───────────────────────────────────────────────────────
_iter = [0]
def objective(params):
    st = forward(params)
    lap_time = np.sum(st['g_r'] / np.maximum(st['v_r'], 1e-3)) * dt
    return lap_time

# ── Constraint 1: track limits  (-w_left ≤ d ≤ w_right) ──────────────────────
def con_track_right(params):
    st = forward(params)
    return w_right_f - st['d_t']          # ≥ 0

def con_track_left(params):
    st = forward(params)
    return st['d_t'] + w_left_f           # ≥ 0

# ── Constraint 2: friction ellipse  (1 - (a_lon/lim)² - (a_lat/side)² ≥ 0) ───
def con_friction(params):
    st  = forward(params)
    v   = np.maximum(st['v_r'], V_MIN)
    aL  = st['a_lon']
    aT  = st['a_lat']
    # Longitudinal limit switches: front if accelerating, back if braking
    lon_lim = np.where(aL >= 0, max_front_acc(v), max_back_acc(v))
    lon_lim = np.maximum(lon_lim, 0.5)    # avoid div-by-zero at top speed
    side_lim = np.maximum(max_side_acc(v), 0.5)
    return 1.0 - (aL / lon_lim)**2 - (aT / side_lim)**2   # ≥ 0

# ── Constraint 3: speed positivity ───────────────────────────────────────────
def con_speed_min(params):
    st = forward(params)
    return st['v_r'] - V_MIN              # ≥ 0

constraints = [
    {'type': 'ineq', 'fun': con_track_right},
    {'type': 'ineq', 'fun': con_track_left},
    {'type': 'ineq', 'fun': con_friction},
    {'type': 'ineq', 'fun': con_speed_min},
]

# ── Initial guess: centerline (d=0) + constant moderate speed ────────────────
cd0 = np.zeros(NC)                         # start on centerline
cv0 = np.zeros(NC)
cv0[0] = 10.0                              # 40 m/s ≈ 144 km/h start guess
params0 = np.concatenate([cd0, cv0])

# Diagnostic callback
def cb(params):
    _iter[0] += 1
    st = forward(params)
    lap = np.sum(st['g_r'] / np.maximum(st['v_r'], 1e-3)) * dt
    fr  = con_friction(params)
    tr_r = con_track_right(params).min()
    tr_l = con_track_left(params).min()
    print(f'  iter {_iter[0]:3d} | lap {lap:8.3f}s | '
          f'min friction margin {fr.min():+.4f} | '
          f'track margin R{tr_r:+.2f} L{tr_l:+.2f} | '
          f'v[{st["v_r"].min():.1f},{st["v_r"].max():.1f}] m/s')

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

print('\n── Initial state ──────────────────────────────────────────')
st0 = forward(params0)
lap0 = np.sum(st0['g_r'] / np.maximum(st0['v_r'], 1e-3)) * dt
print(f'  Initial lap time: {lap0:.3f} s  (constant {cv0[0]:.0f} m/s on centerline)')

print('\n── Optimising (SLSQP) ─────────────────────────────────────')
result = minimize(
    objective, params0,
    method='SLSQP',
    constraints=constraints,
    callback=cb,
    options={'maxiter': 200, 'ftol': 1e-5, 'disp': True}
)

print(f'\n── Done ───────────────────────────────────────────────────')
print(f'  Success: {result.success}')
print(f'  Message: {result.message}')
print(f'  Final lap time: {result.fun:.3f} s')

# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def full_state(params):
    cd_, cv_ = unpack(params)
    d_t   = A_o @ cd_
    v_r   = A_o @ cv_
    st    = forward(params)
    xr    = xc + d_t * Ncx
    yr    = yc + d_t * Ncy
    g_r   = st['g_r']
    lap   = np.sum(g_r / np.maximum(v_r, 1e-3)) * dt
    a_lon = st['a_lon']
    a_lat = st['a_lat']
    a_mag = np.sqrt(a_lon**2 + a_lat**2)
    # Max acceleration magnitude in the SAME direction as the actual accel
    v_safe   = np.maximum(v_r, V_MIN)
    lon_lim  = np.where(a_lon >= 0, max_front_acc(v_safe), max_back_acc(v_safe))
    side_lim = max_side_acc(v_safe)
    # Project the limit ellipse along the acceleration direction
    ang      = np.arctan2(a_lat, a_lon)
    a_max_dir = 1.0 / np.sqrt((np.cos(ang)/np.maximum(lon_lim,1e-3))**2
                              + (np.sin(ang)/np.maximum(side_lim,1e-3))**2)
    return dict(xr=xr, yr=yr, d_t=d_t, v_r=v_r, g_r=g_r, lap=lap,
                a_lon=a_lon, a_lat=a_lat, a_mag=a_mag, a_max_dir=a_max_dir)

S_init = full_state(params0)
S_opt  = full_state(result.x)

print(f'\n  Initial: lap {S_init["lap"]:.2f}s  '
      f'path {np.sum(S_init["g_r"])*dt:.1f}m  '
      f'avg {np.sum(S_init["g_r"])*dt/S_init["lap"]*3.6:.1f} km/h')
print(f'  Optimal: lap {S_opt["lap"]:.2f}s  '
      f'path {np.sum(S_opt["g_r"])*dt:.1f}m  '
      f'avg {np.sum(S_opt["g_r"])*dt/S_opt["lap"]*3.6:.1f} km/h')

# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def speed_colored_line(ax, xs, ys, speeds, lw=2.5, alpha=1.0):
    """Plot a path colored by speed (jet reversed: red=slow, green=fast)."""
    pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap='RdYlGn',
                          norm=plt.Normalize(speeds.min(), speeds.max()),
                          alpha=alpha)
    lc.set_array(speeds[:-1])
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    return lc

fig = plt.figure(figsize=(24, 13))
gs  = gridspec.GridSpec(3, 3, width_ratios=[1, 1.5, 1],
                        height_ratios=[1, 1, 1],
                        hspace=0.42, wspace=0.28,
                        left=0.05, right=0.97, top=0.93, bottom=0.06)

# Left column
ax_spd = fig.add_subplot(gs[0, 0])
ax_ai  = fig.add_subplot(gs[1, 0])
ax_ao  = fig.add_subplot(gs[2, 0])
# Centre
ax_trk = fig.add_subplot(gs[:, 1])
# Right column
ax_rgt = fig.add_subplot(gs[0, 2])
ax_lft = fig.add_subplot(gs[1, 2])
ax_kap = fig.add_subplot(gs[2, 2])

# ── Centre: track + both paths, speed-colored ────────────────────────────────
ax_trk.plot(x_right_raw, y_right_raw, 'b-', lw=1.0, alpha=0.4)
ax_trk.plot(x_left_raw,  y_left_raw,  'r-', lw=1.0, alpha=0.4)
ax_trk.fill(np.concatenate([xc + w_right_f*Ncx, (xc - w_left_f*Ncx)[::-1]]),
            np.concatenate([yc + w_right_f*Ncy, (yc - w_left_f*Ncy)[::-1]]),
            color='gray', alpha=0.10)
# Initial path (low alpha)
speed_colored_line(ax_trk, S_init['xr'], S_init['yr'], S_init['v_r']*3.6,
                   lw=2.0, alpha=0.30)
# Optimised path (high alpha)
lc_main = speed_colored_line(ax_trk, S_opt['xr'], S_opt['yr'],
                             S_opt['v_r']*3.6, lw=3.0, alpha=0.95)
cbar = fig.colorbar(lc_main, ax=ax_trk, fraction=0.035, pad=0.02)
cbar.set_label('Speed (km/h)', fontsize=9)
ax_trk.set_aspect('equal')
ax_trk.grid(True, alpha=0.2)
ax_trk.set_title(
    f'Racing Line  |  Initial: {S_init["lap"]:.2f} s   →   '
    f'Optimised: {S_opt["lap"]:.2f} s   '
    f'(Δ {S_init["lap"]-S_opt["lap"]:+.2f} s)',
    fontsize=12, fontweight='bold')
ax_trk.set_xlabel('x (m)'); ax_trk.set_ylabel('y (m)')

# ── Left top: speed sequences (km/h) ─────────────────────────────────────────
ax_spd.plot(s_c, S_init['v_r']*3.6, 'gray',     lw=1.5, alpha=0.7, label='Initial')
ax_spd.plot(s_c, S_opt['v_r']*3.6,  'seagreen', lw=1.8, alpha=0.9, label='Optimised')
ax_spd.set_title('Speed profile', fontsize=10)
ax_spd.set_xlabel('Arc length (m)', fontsize=8)
ax_spd.set_ylabel('Speed (km/h)', fontsize=8)
ax_spd.legend(fontsize=8); ax_spd.grid(True, alpha=0.2)

# ── Left middle: INITIAL path accel vs max accel ─────────────────────────────
ax_ai.plot(s_c, S_init['a_mag'],     'k-',  lw=1.4, alpha=0.8, label='|a| used')
ax_ai.plot(s_c, S_init['a_max_dir'], 'r--', lw=1.2, alpha=0.7, label='|a| max (same dir)')
ax_ai.fill_between(s_c, S_init['a_mag'], S_init['a_max_dir'],
                   where=S_init['a_max_dir'] >= S_init['a_mag'],
                   color='green', alpha=0.10)
ax_ai.set_title('Initial path — acceleration vs limit', fontsize=10)
ax_ai.set_xlabel('Arc length (m)', fontsize=8)
ax_ai.set_ylabel('m/s²', fontsize=8)
ax_ai.legend(fontsize=8); ax_ai.grid(True, alpha=0.2)

# ── Left bottom: OPTIMISED path accel vs max accel ───────────────────────────
ax_ao.plot(s_c, S_opt['a_mag'],     'k-',  lw=1.4, alpha=0.8, label='|a| used')
ax_ao.plot(s_c, S_opt['a_max_dir'], 'r--', lw=1.2, alpha=0.7, label='|a| max (same dir)')
ax_ao.fill_between(s_c, S_opt['a_mag'], S_opt['a_max_dir'],
                   where=S_opt['a_max_dir'] >= S_opt['a_mag'],
                   color='green', alpha=0.10)
ax_ao.set_title('Optimised path — acceleration vs limit', fontsize=10)
ax_ao.set_xlabel('Arc length (m)', fontsize=8)
ax_ao.set_ylabel('m/s²', fontsize=8)
ax_ao.legend(fontsize=8); ax_ao.grid(True, alpha=0.2)

# ── Right top/middle: track bound vs path displacement ───────────────────────
ax_rgt.plot(s_c,  w_right_f,       'b-', lw=1.5, alpha=0.7, label='Right bound')
ax_rgt.plot(s_c,  S_opt['d_t'],    'g-', lw=1.5, alpha=0.8, label='Optimised d')
ax_rgt.plot(s_c,  S_init['d_t'],   'gray', lw=1.0, alpha=0.5, label='Initial d')
ax_rgt.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax_rgt.fill_between(s_c, S_opt['d_t'], w_right_f, alpha=0.10, color='b')
ax_rgt.set_title('Right side — bound vs path', fontsize=10)
ax_rgt.set_xlabel('Arc length (m)', fontsize=8)
ax_rgt.set_ylabel('Displacement (m)', fontsize=8)
ax_rgt.legend(fontsize=8); ax_rgt.grid(True, alpha=0.2)

ax_lft.plot(s_c, -w_left_f,        'r-', lw=1.5, alpha=0.7, label='Left bound')
ax_lft.plot(s_c,  S_opt['d_t'],    'g-', lw=1.5, alpha=0.8, label='Optimised d')
ax_lft.plot(s_c,  S_init['d_t'],   'gray', lw=1.0, alpha=0.5, label='Initial d')
ax_lft.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax_lft.fill_between(s_c, -w_left_f, S_opt['d_t'], alpha=0.10, color='r')
ax_lft.set_title('Left side — bound vs path', fontsize=10)
ax_lft.set_xlabel('Arc length (m)', fontsize=8)
ax_lft.set_ylabel('Displacement (m)', fontsize=8)
ax_lft.legend(fontsize=8); ax_lft.grid(True, alpha=0.2)

# ── Right bottom: curvature comparison ───────────────────────────────────────
st_i = forward(params0); st_o = forward(result.x)
ax_kap.plot(s_c, st_i['kappa'], 'gray',   lw=1.2, alpha=0.6, label='Initial κ')
ax_kap.plot(s_c, st_o['kappa'], 'purple', lw=1.4, alpha=0.8, label='Optimised κ')
ax_kap.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax_kap.set_title('Racing path curvature', fontsize=10)
ax_kap.set_xlabel('Arc length (m)', fontsize=8)
ax_kap.set_ylabel('κ (1/m)', fontsize=8)
ax_kap.legend(fontsize=8); ax_kap.grid(True, alpha=0.2)

plt.suptitle('Physics-Constrained Minimum Lap Time — Silverstone',
             fontsize=15, fontweight='bold')
plt.savefig('laptime_result.png', dpi=130, bbox_inches='tight')
plt.show()