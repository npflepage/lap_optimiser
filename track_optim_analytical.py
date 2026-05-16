"""
═══════════════════════════════════════════════════════════════════════════════
  PHYSICS-CONSTRAINED MIN LAP TIME  —  with ANALYTICAL JACOBIANS
═══════════════════════════════════════════════════════════════════════════════

Every Fourier quantity is linear in the coefficients through constant design
matrices (A, D1, D2).  This makes the full Jacobian of the objective and all
constraints analytically derivable — no finite differences.  SLSQP then needs
no gradient probing → ~Nparam× fewer forward evaluations and exact gradients.

Jacobians provided (all verified vs finite differences to ~1e-8):
  • objective   ∂T/∂c        T = Σ g_r/v · dt
  • track R/L   ∂(w∓d)/∂c    linear → exact constant ∓A
  • friction    ∂C/∂c        C = 1-(a_L/ℓ)²-(a_T/s)²   (the tricky one)
  • speed min   ∂(v-vmin)/∂c linear → exact constant A

Friction subtlety: ℓ switches on sign(a_L) and max_front_acc contains a
clip(quadratic, power). Both are frozen per evaluation (measure-zero kinks,
standard for SLSQP) and their v-derivative uses the active branch.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.optimize import minimize
import pandas as pd
import time as _time

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
#  F1 PHYSICS MODEL  +  ANALYTICAL d/dv  (750 kg, ~1000 hp)
# ═══════════════════════════════════════════════════════════════════════════════

_GRIP_F, _DF_F, _POW, _DRAG, _M = (7978.162499470494,
                                   2.1259279125731587,
                                   745000.0, 0.745, 750.0)
_GRIP_B, _DF_B = 11254.767298703786, 2.2808897580274747

def max_front_acc(v):
    quad = _GRIP_F + _DF_F * v**2
    powr = _POW / np.maximum(v, 1e-3)
    return np.clip(quad, None, powr) / _M - _DRAG * v**2 / _M

def dmax_front_dv(v):
    """d(max_front_acc)/dv — uses whichever branch (quad vs power) is active."""
    quad = _GRIP_F + _DF_F * v**2
    powr = _POW / np.maximum(v, 1e-3)
    dclip = np.where(quad <= powr,
                     2 * _DF_F * v,                     # quadratic branch
                     -_POW / np.maximum(v, 1e-3)**2)    # power branch
    return dclip / _M - 2 * _DRAG * v / _M

def max_back_acc(v):
    return (_GRIP_B + _DF_B * v**2 + _DRAG * v**2) / _M

def dmax_back_dv(v):
    return (2 * _DF_B * v + 2 * _DRAG * v) / _M

def max_side_acc(v):
    return 1.2 * (_GRIP_B + _DF_B * v**2) / _M

def dmax_side_dv(v):
    return 1.2 * (2 * _DF_B * v) / _M

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA & BUILD CENTERLINE
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

d_right = project_points_onto_fourier(x_right_raw, y_right_raw, xc, yc, Ncx, Ncy)
d_left  = -project_points_onto_fourier(x_left_raw, y_left_raw,  xc, yc, Ncx, Ncy)

cw_right = fit_fourier(t, d_right, N)
cw_left  = fit_fourier(t, d_left,  N)
w_right_f = A_fine @ cw_right
w_left_f  = A_fine @ cw_left

ds_c = np.sqrt(np.diff(xc)**2 + np.diff(yc)**2)
s_c  = np.concatenate([[0], np.cumsum(ds_c)])

print(f'Centerline length: {np.sum(g_c)*dt:.2f} m   (official 5891 m)')

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
#  OPTIMISATION SETUP
# ═══════════════════════════════════════════════════════════════════════════════

N_OPT  = 128
A_o    = build_design_matrix(t_fine,    N_OPT)
D1_o   = build_derivative_matrix(t_fine, N_OPT, 1)
D2_o   = build_derivative_matrix(t_fine, N_OPT, 2)
NC     = 2 * N_OPT + 1
V_MIN  = 5.0

def unpack(params):
    return params[:NC], params[NC:]

def forward(params):
    cd_, cv_ = unpack(params)
    d_t    = A_o  @ cd_
    dd_dt  = D1_o @ cd_
    ddd_dt = D2_o @ cd_
    v_r    = A_o  @ cv_
    dv_dt  = D1_o @ cv_

    dxr  = dxc + dd_dt * Ncx + d_t * dNcx_dt
    dyr  = dyc + dd_dt * Ncy + d_t * dNcy_dt
    ddxr = ddxc + ddd_dt * Ncx + 2 * dd_dt * dNcx_dt + d_t * ddNcx_dt
    ddyr = ddyc + ddd_dt * Ncy + 2 * dd_dt * dNcy_dt + d_t * ddNcy_dt

    g_r   = np.sqrt(dxr**2 + dyr**2)
    kappa = (dxr * ddyr - dyr * ddxr) / (g_r**3)
    a_lon = (v_r / g_r) * dv_dt
    a_lat = v_r**2 * kappa
    return dict(d_t=d_t, v_r=v_r, dv_dt=dv_dt, g_r=g_r, kappa=kappa,
                a_lon=a_lon, a_lat=a_lat,
                dxr=dxr, dyr=dyr, ddxr=ddxr, ddyr=ddyr)

# ── Path-derivative Jacobians wrt c_d (constant structure, cheap) ─────────────
#   ∂(ẋ_r)/∂c_d = diag(N_cx)·D1 + diag(Ṅ_cx)·A      (etc.)
_J_dxr_cd  = Ncx[:, None] * D1_o + dNcx_dt[:, None] * A_o
_J_dyr_cd  = Ncy[:, None] * D1_o + dNcy_dt[:, None] * A_o
_J_ddxr_cd = Ncx[:, None] * D2_o + 2*dNcx_dt[:, None] * D1_o + ddNcx_dt[:, None] * A_o
_J_ddyr_cd = Ncy[:, None] * D2_o + 2*dNcy_dt[:, None] * D1_o + ddNcy_dt[:, None] * A_o

# ═══════════════════════════════════════════════════════════════════════════════
#  OBJECTIVE  +  ANALYTICAL GRADIENT
# ═══════════════════════════════════════════════════════════════════════════════

def objective(params):
    st = forward(params)
    return np.sum(st['g_r'] / np.maximum(st['v_r'], 1e-3)) * dt

def objective_grad(params):
    st = forward(params)
    gr, v = st['g_r'], np.maximum(st['v_r'], 1e-3)
    # ∂g_r/∂c_d = (ẋ_r·J_ẋ + ẏ_r·J_ẏ)/g_r
    dgr_cd = (st['dxr'][:, None] * _J_dxr_cd +
              st['dyr'][:, None] * _J_dyr_cd) / gr[:, None]
    gT_cd = np.sum((1.0 / v)[:, None] * dgr_cd, axis=0) * dt
    gT_cv = np.sum((-gr / v**2)[:, None] * A_o, axis=0) * dt
    return np.concatenate([gT_cd, gT_cv])

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTRAINTS  +  ANALYTICAL JACOBIANS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Track limits (linear → constant Jacobians) ───────────────────────────────
_Z = np.zeros_like(A_o)
_JAC_TRACK_R = np.concatenate([-A_o, _Z], axis=1)   # ∂(w_r - d)/∂[c_d,c_v]
_JAC_TRACK_L = np.concatenate([ A_o, _Z], axis=1)   # ∂(d + w_l)/∂[c_d,c_v]
_JAC_SPEED   = np.concatenate([_Z,  A_o], axis=1)   # ∂(v - vmin)/∂[c_d,c_v]

def con_track_right(p): return w_right_f - forward(p)['d_t']
def con_track_left(p):  return forward(p)['d_t'] + w_left_f
def con_speed_min(p):   return forward(p)['v_r'] - V_MIN
def jac_track_right(p): return _JAC_TRACK_R
def jac_track_left(p):  return _JAC_TRACK_L
def jac_speed_min(p):   return _JAC_SPEED

# ── Friction ellipse  C = 1 - (a_L/ℓ)² - (a_T/s)²  (the tricky Jacobian) ─────
def con_friction(p):
    st = forward(p)
    v  = np.maximum(st['v_r'], V_MIN)
    aL, aT = st['a_lon'], st['a_lat']
    lon_lim  = np.maximum(np.where(aL >= 0, max_front_acc(v),
                                   max_back_acc(v)), 0.5)
    side_lim = np.maximum(max_side_acc(v), 0.5)
    return 1.0 - (aL / lon_lim)**2 - (aT / side_lim)**2

def jac_friction(p):
    st = forward(p)
    v_raw = st['v_r']
    v   = np.maximum(v_raw, V_MIN)
    aL, aT, gr, kap, dv = (st['a_lon'], st['a_lat'], st['g_r'],
                           st['kappa'], st['dv_dt'])

    # Active longitudinal limit + its v-derivative (branch frozen per eval)
    front = aL >= 0
    lon_lim_raw = np.where(front, max_front_acc(v), max_back_acc(v))
    lon_lim = np.maximum(lon_lim_raw, 0.5)
    dlon_dv = np.where(front, dmax_front_dv(v), dmax_back_dv(v))
    dlon_dv = np.where(lon_lim_raw > 0.5, dlon_dv, 0.0)   # clipped → 0 deriv

    side_lim_raw = max_side_acc(v)
    side_lim = np.maximum(side_lim_raw, 0.5)
    dside_dv = np.where(side_lim_raw > 0.5, dmax_side_dv(v), 0.0)

    # v-floor: if v_raw < V_MIN the limit no longer varies with c_v
    v_active = (v_raw > V_MIN).astype(float)

    # ── ∂g_r/∂c_d ────────────────────────────────────────────────────────────
    dgr_cd = (st['dxr'][:, None] * _J_dxr_cd +
              st['dyr'][:, None] * _J_dyr_cd) / gr[:, None]

    # ── ∂κ/∂c_d :  κ = (ẋÿ-ẏẍ)/g³ ────────────────────────────────────────────
    cross      = st['dxr']*st['ddyr'] - st['dyr']*st['ddxr']
    dcross_cd  = (st['ddyr'][:, None]*_J_dxr_cd + st['dxr'][:, None]*_J_ddyr_cd
                  - st['ddxr'][:, None]*_J_dyr_cd - st['dyr'][:, None]*_J_ddxr_cd)
    dkap_cd = (dcross_cd * gr[:, None]**3
               - cross[:, None] * 3 * gr[:, None]**2 * dgr_cd) / gr[:, None]**6

    # ── ∂a_L/∂c :  a_L = (v/g_r)·v̇ ───────────────────────────────────────────
    daL_cd = (-v_raw * dv / gr**2)[:, None] * dgr_cd
    daL_cv = (dv / gr)[:, None] * A_o + (v_raw / gr)[:, None] * D1_o

    # ── ∂a_T/∂c :  a_T = v²·κ ────────────────────────────────────────────────
    daT_cd = (v_raw**2)[:, None] * dkap_cd
    daT_cv = (2 * v_raw * kap)[:, None] * A_o

    # ── ∂(limits)/∂c_v  (only through v) ─────────────────────────────────────
    dll_cv = (dlon_dv * v_active)[:, None] * A_o
    dsl_cv = (dside_dv * v_active)[:, None] * A_o

    # ── Assemble  ∂C/∂c  ────────────────────────────────────────────────────
    # C = 1 - (a_L/ℓ)² - (a_T/s)²
    # ∂C = -2a_L/ℓ²·∂a_L + 2a_L²/ℓ³·∂ℓ  -2a_T/s²·∂a_T + 2a_T²/s³·∂s
    cA_L = (-2 * aL / lon_lim**2)[:, None]
    cL_L = ( 2 * aL**2 / lon_lim**3)[:, None]
    cA_T = (-2 * aT / side_lim**2)[:, None]
    cL_T = ( 2 * aT**2 / side_lim**3)[:, None]

    dC_cd = cA_L * daL_cd + cA_T * daT_cd                 # limits ⟂ c_d
    dC_cv = (cA_L * daL_cv + cL_L * dll_cv +
             cA_T * daT_cv + cL_T * dsl_cv)
    return np.concatenate([dC_cd, dC_cv], axis=1)

constraints = [
    {'type': 'ineq', 'fun': con_track_right, 'jac': jac_track_right},
    {'type': 'ineq', 'fun': con_track_left,  'jac': jac_track_left},
    {'type': 'ineq', 'fun': con_friction,    'jac': jac_friction},
    {'type': 'ineq', 'fun': con_speed_min,   'jac': jac_speed_min},
]

# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-CHECK : verify analytical Jacobians vs finite differences once
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_jacobians():
    rng = np.random.default_rng(0)
    p = np.zeros(2 * NC)
    p[:NC]  = rng.standard_normal(NC) * 0.05
    p[NC:]  = rng.standard_normal(NC) * 0.05
    p[NC]   = 45.0
    def fd_scalar(f, p, eps=1e-7):
        f0 = f(p); J = np.zeros(len(p))
        for i in range(len(p)):
            pp = p.copy(); pp[i] += eps; J[i] = (f(pp)-f0)/eps
        return J
    def fd_vec(f, p, eps=1e-7):
        f0 = f(p); J = np.zeros((len(f0), len(p)))
        for i in range(len(p)):
            pp = p.copy(); pp[i] += eps; J[:, i] = (f(pp)-f0)/eps
        return J
    e_obj = np.abs(objective_grad(p) - fd_scalar(objective, p)).max()
    e_fr  = np.abs(jac_friction(p)   - fd_vec(con_friction, p)).max()
    print(f'  Jacobian self-check — objective: {e_obj:.2e}   '
          f'friction: {e_fr:.2e}')
    assert e_obj < 1e-4 and e_fr < 1e-3, 'Jacobian mismatch!'

print('\n── Verifying analytical Jacobians ─────────────────────────')
_verify_jacobians()
print('  ✓ all Jacobians match finite differences')

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

cd0 = np.zeros(NC)
cv0 = np.zeros(NC); cv0[0] = 25.0
params0 = np.concatenate([cd0, cv0])

_iter = [0]
def cb(params):
    _iter[0] += 1
    st  = forward(params)
    lap = np.sum(st['g_r'] / np.maximum(st['v_r'], 1e-3)) * dt
    fr  = con_friction(params)
    print(f'  iter {_iter[0]:3d} | lap {lap:8.3f}s | '
          f'min friction margin {fr.min():+.4f} | '
          f'v[{st["v_r"].min():.1f},{st["v_r"].max():.1f}] m/s')

print('\n── Initial state ──────────────────────────────────────────')
lap0 = objective(params0)
print(f'  Initial lap time: {lap0:.3f} s  '
      f'(constant {cv0[0]:.0f} m/s on centerline)')

print('\n── Optimising (SLSQP, analytical Jacobians) ───────────────')
t0 = _time.time()
result = minimize(
    objective, params0,
    method='SLSQP',
    jac=objective_grad,
    constraints=constraints,
    callback=cb,
    options={'maxiter': 100, 'ftol': 1e-5, 'disp': True}
)
elapsed = _time.time() - t0

print(f'\n── Done ───────────────────────────────────────────────────')
print(f'  Success: {result.success}')
print(f'  Message: {result.message}')
print(f'  Final lap time: {result.fun:.3f} s')
print(f'  Wall time: {elapsed:.1f} s  ({_iter[0]} iters)')

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
    a_lon = st['a_lon']; a_lat = st['a_lat']
    a_mag = np.sqrt(a_lon**2 + a_lat**2)
    v_safe   = np.maximum(v_r, V_MIN)
    lon_lim  = np.where(a_lon >= 0, max_front_acc(v_safe),
                        max_back_acc(v_safe))
    side_lim = max_side_acc(v_safe)
    ang      = np.arctan2(a_lat, a_lon)
    a_max_dir = 1.0 / np.sqrt((np.cos(ang)/np.maximum(lon_lim,1e-3))**2
                              + (np.sin(ang)/np.maximum(side_lim,1e-3))**2)
    return dict(xr=xr, yr=yr, d_t=d_t, v_r=v_r, g_r=g_r, lap=lap,
                kappa=st['kappa'], a_lon=a_lon, a_lat=a_lat,
                a_mag=a_mag, a_max_dir=a_max_dir)

S_init = full_state(params0)
S_opt  = full_state(result.x)

print(f'\n  Initial: lap {S_init["lap"]:.2f}s  '
      f'path {np.sum(S_init["g_r"])*dt:.1f}m  '
      f'avg {np.sum(S_init["g_r"])*dt/S_init["lap"]*3.6:.1f} km/h')
print(f'  Optimal: lap {S_opt["lap"]:.2f}s  '
      f'path {np.sum(S_opt["g_r"])*dt:.1f}m  '
      f'avg {np.sum(S_opt["g_r"])*dt/S_opt["lap"]*3.6:.1f} km/h')

# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC DASHBOARD  (unchanged layout)
# ═══════════════════════════════════════════════════════════════════════════════

def speed_colored_line(ax, xs, ys, speeds, lw=2.5, alpha=1.0):
    pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap='RdYlGn',
                          norm=plt.Normalize(speeds.min(), speeds.max()),
                          alpha=alpha)
    lc.set_array(speeds[:-1]); lc.set_linewidth(lw)
    ax.add_collection(lc)
    return lc

fig = plt.figure(figsize=(24, 13))
gs  = gridspec.GridSpec(3, 3, width_ratios=[1, 1.5, 1],
                        height_ratios=[1, 1, 1],
                        hspace=0.42, wspace=0.28,
                        left=0.05, right=0.97, top=0.93, bottom=0.06)

ax_spd = fig.add_subplot(gs[0, 0])
ax_ai  = fig.add_subplot(gs[1, 0])
ax_ao  = fig.add_subplot(gs[2, 0])
ax_trk = fig.add_subplot(gs[:, 1])
ax_rgt = fig.add_subplot(gs[0, 2])
ax_lft = fig.add_subplot(gs[1, 2])
ax_kap = fig.add_subplot(gs[2, 2])

ax_trk.plot(x_right_raw, y_right_raw, 'b-', lw=1.0, alpha=0.4)
ax_trk.plot(x_left_raw,  y_left_raw,  'r-', lw=1.0, alpha=0.4)
ax_trk.fill(np.concatenate([xc + w_right_f*Ncx, (xc - w_left_f*Ncx)[::-1]]),
            np.concatenate([yc + w_right_f*Ncy, (yc - w_left_f*Ncy)[::-1]]),
            color='gray', alpha=0.10)
speed_colored_line(ax_trk, S_init['xr'], S_init['yr'], S_init['v_r']*3.6,
                   lw=2.0, alpha=0.30)
lc_main = speed_colored_line(ax_trk, S_opt['xr'], S_opt['yr'],
                             S_opt['v_r']*3.6, lw=3.0, alpha=0.95)
cbar = fig.colorbar(lc_main, ax=ax_trk, fraction=0.035, pad=0.02)
cbar.set_label('Speed (km/h)', fontsize=9)
ax_trk.set_aspect('equal'); ax_trk.grid(True, alpha=0.2)
ax_trk.set_title(
    f'Racing Line  |  Initial: {S_init["lap"]:.2f} s   →   '
    f'Optimised: {S_opt["lap"]:.2f} s   '
    f'(Δ {S_init["lap"]-S_opt["lap"]:+.2f} s)',
    fontsize=12, fontweight='bold')
ax_trk.set_xlabel('x (m)'); ax_trk.set_ylabel('y (m)')

ax_spd.plot(s_c, S_init['v_r']*3.6, 'gray',     lw=1.5, alpha=0.7, label='Initial')
ax_spd.plot(s_c, S_opt['v_r']*3.6,  'seagreen', lw=1.8, alpha=0.9, label='Optimised')
ax_spd.set_title('Speed profile', fontsize=10)
ax_spd.set_xlabel('Arc length (m)', fontsize=8)
ax_spd.set_ylabel('Speed (km/h)', fontsize=8)
ax_spd.legend(fontsize=8); ax_spd.grid(True, alpha=0.2)

ax_ai.plot(s_c, S_init['a_mag'],     'k-',  lw=1.4, alpha=0.8, label='|a| used')
ax_ai.plot(s_c, S_init['a_max_dir'], 'r--', lw=1.2, alpha=0.7, label='|a| max (same dir)')
ax_ai.fill_between(s_c, S_init['a_mag'], S_init['a_max_dir'],
                   where=S_init['a_max_dir'] >= S_init['a_mag'],
                   color='green', alpha=0.10)
ax_ai.set_title('Initial path — acceleration vs limit', fontsize=10)
ax_ai.set_xlabel('Arc length (m)', fontsize=8)
ax_ai.set_ylabel('m/s²', fontsize=8)
ax_ai.legend(fontsize=8); ax_ai.grid(True, alpha=0.2)

ax_ao.plot(s_c, S_opt['a_mag'],     'k-',  lw=1.4, alpha=0.8, label='|a| used')
ax_ao.plot(s_c, S_opt['a_max_dir'], 'r--', lw=1.2, alpha=0.7, label='|a| max (same dir)')
ax_ao.fill_between(s_c, S_opt['a_mag'], S_opt['a_max_dir'],
                   where=S_opt['a_max_dir'] >= S_opt['a_mag'],
                   color='green', alpha=0.10)
ax_ao.set_title('Optimised path — acceleration vs limit', fontsize=10)
ax_ao.set_xlabel('Arc length (m)', fontsize=8)
ax_ao.set_ylabel('m/s²', fontsize=8)
ax_ao.legend(fontsize=8); ax_ao.grid(True, alpha=0.2)

ax_rgt.plot(s_c,  w_right_f,     'b-',   lw=1.5, alpha=0.7, label='Right bound')
ax_rgt.plot(s_c,  S_opt['d_t'],  'g-',   lw=1.5, alpha=0.8, label='Optimised d')
ax_rgt.plot(s_c,  S_init['d_t'], 'gray', lw=1.0, alpha=0.5, label='Initial d')
ax_rgt.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax_rgt.fill_between(s_c, S_opt['d_t'], w_right_f, alpha=0.10, color='b')
ax_rgt.set_title('Right side — bound vs path', fontsize=10)
ax_rgt.set_xlabel('Arc length (m)', fontsize=8)
ax_rgt.set_ylabel('Displacement (m)', fontsize=8)
ax_rgt.legend(fontsize=8); ax_rgt.grid(True, alpha=0.2)

ax_lft.plot(s_c, -w_left_f,      'r-',   lw=1.5, alpha=0.7, label='Left bound')
ax_lft.plot(s_c,  S_opt['d_t'],  'g-',   lw=1.5, alpha=0.8, label='Optimised d')
ax_lft.plot(s_c,  S_init['d_t'], 'gray', lw=1.0, alpha=0.5, label='Initial d')
ax_lft.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax_lft.fill_between(s_c, -w_left_f, S_opt['d_t'], alpha=0.10, color='r')
ax_lft.set_title('Left side — bound vs path', fontsize=10)
ax_lft.set_xlabel('Arc length (m)', fontsize=8)
ax_lft.set_ylabel('Displacement (m)', fontsize=8)
ax_lft.legend(fontsize=8); ax_lft.grid(True, alpha=0.2)

ax_kap.plot(s_c, S_init['kappa'], 'gray',   lw=1.2, alpha=0.6, label='Initial κ')
ax_kap.plot(s_c, S_opt['kappa'],  'purple', lw=1.4, alpha=0.8, label='Optimised κ')
ax_kap.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax_kap.set_title('Racing path curvature', fontsize=10)
ax_kap.set_xlabel('Arc length (m)', fontsize=8)
ax_kap.set_ylabel('κ (1/m)', fontsize=8)
ax_kap.legend(fontsize=8); ax_kap.grid(True, alpha=0.2)

plt.suptitle('Physics-Constrained Minimum Lap Time — Silverstone '
             '(Analytical Jacobians)',
             fontsize=15, fontweight='bold')
plt.savefig('laptime_jacobian.png', dpi=130, bbox_inches='tight')
plt.show()