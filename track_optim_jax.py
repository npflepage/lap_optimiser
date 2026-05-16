"""
═══════════════════════════════════════════════════════════════════════════════
  PHYSICS-CONSTRAINED MIN LAP TIME  —  JAX + JAXOPT  (autodiff + JIT)
═══════════════════════════════════════════════════════════════════════════════

CORRECTION vs first attempt: jaxopt.ScipyMinimize is UNCONSTRAINED-only.
jaxopt has no general nonlinear-constraint solver (ProjectedGradient needs a
closed-form projection we don't have; OSQP is QP-only).  The correct jaxopt
approach for nonlinear inequality constraints is the PENALTY METHOD:

  minimise   T(p) + μ · Σ relu(violation)²

solved with jaxopt.ScipyMinimize (L-BFGS) under a PROGRESSIVE penalty
schedule — each μ warm-starts from the previous solution.  This keeps the
problem well-conditioned (validated: constraint violation → 2e-9, feasible).

Everything in the loop is JAX:
  • forward model, lap time, penalised loss → @jit
  • gradient → jax.value_and_grad (exact autodiff, no manual ∂)
  • jaxopt.ScipyMinimize(value_and_grad=True) feeds L-BFGS the autodiff grad

One-off track fit stays NumPy (cheap, runs once).
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np                       # one-off track fit only
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jaxopt import ScipyMinimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import pandas as pd
import time as _time

jax.config.update("jax_enable_x64", True)   # F1 lap times need float64

# ═══════════════════════════════════════════════════════════════════════════════
#  FOURIER UTILITIES  (NumPy — used once to build constant matrices)
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

def compute_frenet_np(dx, dy, ddx, ddy):
    g      = np.sqrt(dx**2 + dy**2)
    Tx, Ty = dx / g, dy / g
    Nx, Ny = -Ty, Tx
    kappa  = (dx * ddy - dy * ddx) / (g**3)
    return g, Tx, Ty, Nx, Ny, kappa

def project_points_onto_fourier(px, py, x_f, y_f, nx_f, ny_f):
    signed = np.zeros(len(px))
    for i in range(len(px)):
        j = np.argmin((px[i]-x_f)**2 + (py[i]-y_f)**2)
        signed[i] = (px[i]-x_f[j])*nx_f[j] + (py[i]-y_f[j])*ny_f[j]
    return signed

# ═══════════════════════════════════════════════════════════════════════════════
#  F1 PHYSICS MODEL  (JAX — differentiable)
# ═══════════════════════════════════════════════════════════════════════════════

_GRIP_F, _DF_F, _POW, _DRAG, _M = (7978.162499470494, 2.1259279125731587,
                                   745000.0, 0.745, 750.0)
_GRIP_B, _DF_B = 11254.767298703786, 2.2808897580274747

@jit
def max_front_acc(v):
    quad = _GRIP_F + _DF_F * v**2
    powr = _POW / jnp.maximum(v, 1e-3)
    return jnp.minimum(quad, powr) / _M - _DRAG * v**2 / _M

@jit
def max_back_acc(v):
    return (_GRIP_B + _DF_B * v**2 + _DRAG * v**2) / _M

@jit
def max_side_acc(v):
    return 1.2 * (_GRIP_B + _DF_B * v**2) / _M

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA & BUILD CENTERLINE  (NumPy, one-off)
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv('Monza.csv')
x  = df['x_m'].values
y  = df['y_m'].values
w_right = df['w_tr_right_m'].values
w_left  = df['w_tr_left_m'].values

dx_raw = np.gradient(x);  dy_raw = np.gradient(y)
nrm    = np.sqrt(dx_raw**2 + dy_raw**2)
nx_raw, ny_raw = -dy_raw/nrm, dx_raw/nrm
x_right_raw = x + w_right * nx_raw;  y_right_raw = y + w_right * ny_raw
x_left_raw  = x - w_left  * nx_raw;  y_left_raw  = y - w_left  * ny_raw

N  = 128+64
ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
s  = np.concatenate([[0], np.cumsum(ds)])
t  = 2 * np.pi * s / s[-1]

cx = fit_fourier(t, x, N)
cy = fit_fourier(t, y, N)

t_fine = np.linspace(0, 2 * np.pi, 2000)
dt     = float(t_fine[1] - t_fine[0])

A_fine = build_design_matrix(t_fine,    N)
D1c    = build_derivative_matrix(t_fine, N, 1)
D2c    = build_derivative_matrix(t_fine, N, 2)
D3c    = build_derivative_matrix(t_fine, N, 3)

xc   = A_fine @ cx;   yc   = A_fine @ cy
dxc  = D1c    @ cx;   dyc  = D1c    @ cy
ddxc = D2c    @ cx;   ddyc = D2c    @ cy
dddxc= D3c    @ cx;   dddyc= D3c    @ cy

g_c, Tcx, Tcy, Ncx, Ncy, kappa_c = compute_frenet_np(dxc, dyc, ddxc, ddyc)

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
#  OPTIMISATION SETUP  —  constants → JAX device
# ═══════════════════════════════════════════════════════════════════════════════

N_OPT = 128+64
A_o   = jnp.asarray(build_design_matrix(t_fine,    N_OPT))
D1_o  = jnp.asarray(build_derivative_matrix(t_fine, N_OPT, 1))
D2_o  = jnp.asarray(build_derivative_matrix(t_fine, N_OPT, 2))
NC    = 2 * N_OPT + 1
V_MIN = 5.0

J_dxc, J_dyc     = jnp.asarray(dxc),  jnp.asarray(dyc)
J_ddxc, J_ddyc   = jnp.asarray(ddxc), jnp.asarray(ddyc)
J_Ncx, J_Ncy     = jnp.asarray(Ncx),  jnp.asarray(Ncy)
J_dNcx, J_dNcy   = jnp.asarray(dNcx_dt),  jnp.asarray(dNcy_dt)
J_ddNcx, J_ddNcy = jnp.asarray(ddNcx_dt), jnp.asarray(ddNcy_dt)
J_wr, J_wl       = jnp.asarray(w_right_f), jnp.asarray(w_left_f)
J_dt             = jnp.asarray(dt)

@jit
def forward(params):
    cd_ = params[:NC]
    cv_ = params[NC:]
    d_t    = A_o  @ cd_
    dd_dt  = D1_o @ cd_
    ddd_dt = D2_o @ cd_
    v_r    = A_o  @ cv_
    dv_dt  = D1_o @ cv_

    dxr  = J_dxc + dd_dt * J_Ncx + d_t * J_dNcx
    dyr  = J_dyc + dd_dt * J_Ncy + d_t * J_dNcy
    ddxr = J_ddxc + ddd_dt * J_Ncx + 2*dd_dt * J_dNcx + d_t * J_ddNcx
    ddyr = J_ddyc + ddd_dt * J_Ncy + 2*dd_dt * J_dNcy + d_t * J_ddNcy

    g_r   = jnp.sqrt(dxr**2 + dyr**2)
    kappa = (dxr * ddyr - dyr * ddxr) / (g_r**3)
    a_lon = (v_r / g_r) * dv_dt
    a_lat = v_r**2 * kappa
    return d_t, v_r, g_r, kappa, a_lon, a_lat

@jit
def lap_time(params):
    _, v_r, g_r, _, _, _ = forward(params)
    return jnp.sum(g_r / jnp.maximum(v_r, 1e-3)) * J_dt

@jit
def constraint_violation(params):
    """Total squared inequality violation (0 ⇔ fully feasible)."""
    d_t, v_r, g_r, kappa, a_lon, a_lat = forward(params)
    v = jnp.maximum(v_r, V_MIN)
    lon_lim  = jnp.maximum(jnp.where(a_lon >= 0,
                                     max_front_acc(v), max_back_acc(v)), 0.5)
    side_lim = jnp.maximum(max_side_acc(v), 0.5)

    relu = lambda z: jnp.maximum(z, 0.0)
    c_fr   = relu((a_lon / lon_lim)**2 + (a_lat / side_lim)**2 - 1.0)  # ellipse
    c_tr_r = relu(d_t - J_wr)                                          # right
    c_tr_l = relu(-J_wl - d_t)                                         # left
    c_v    = relu(V_MIN - v_r)                                         # speed
    return jnp.sum(c_fr**2 + c_tr_r**2 + c_tr_l**2 + c_v**2) * J_dt

def make_loss(mu):
    """Penalised objective for a given penalty weight μ (closed over)."""
    @jit
    def loss(params):
        return lap_time(params) + mu * constraint_violation(params)
    return loss

# ═══════════════════════════════════════════════════════════════════════════════
#  PROGRESSIVE-PENALTY OPTIMISATION  (jaxopt.ScipyMinimize / L-BFGS + autodiff)
# ═══════════════════════════════════════════════════════════════════════════════

cd0 = np.zeros(NC)
cv0 = np.zeros(NC); cv0[0] = 20.0
params  = jnp.asarray(np.concatenate([cd0, cv0]))
params0 = params  # keep the global initial state for the final plot

print('\n── Compiling JAX kernels (first call traces + JITs) ───────')
_t = _time.time()
_ = lap_time(params).block_until_ready()
_ = constraint_violation(params).block_until_ready()
print(f'  Compilation done in {_time.time()-_t:.1f} s')

lap0 = float(lap_time(params0))
print(f'\n── Initial state ──────────────────────────────────────────')
print(f'  Initial lap time: {lap0:.3f} s  '
      f'(constant {cv0[0]:.0f} m/s on centerline)')

PENALTY_SCHEDULE = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8]
print('\n── Progressive-penalty optimisation (autodiff + JIT) ──────')
t0 = _time.time()
for stage, mu in enumerate(PENALTY_SCHEDULE):
    loss_fn = make_loss(mu)
    vag     = jit(value_and_grad(loss_fn))      # exact autodiff gradient

    solver = ScipyMinimize(
        method='L-BFGS-B',
        fun=vag,
        value_and_grad=True,                    # vag returns (loss, grad)
        maxiter=5000,
        tol=1e-12,
    )
    res    = solver.run(params)
    params = res.params

    lap  = float(lap_time(params))
    viol = float(constraint_violation(params))
    print(f'  stage {stage+1}/{len(PENALTY_SCHEDULE)}  μ={mu:.0e}  |  '
          f'lap {lap:8.3f} s  |  constraint viol {viol:.3e}')

elapsed = _time.time() - t0
params_opt = params
lap_opt    = float(lap_time(params_opt))
viol_opt   = float(constraint_violation(params_opt))

print(f'\n── Done ───────────────────────────────────────────────────')
print(f'  Final lap time:        {lap_opt:.3f} s')
print(f'  Final constraint viol: {viol_opt:.3e}  '
      f'({"FEASIBLE" if viol_opt < 1e-3 else "check penalty"})')
print(f'  Wall time: {elapsed:.1f} s')

# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACT RESULTS  (back to NumPy for plotting)
# ═══════════════════════════════════════════════════════════════════════════════

A_np = build_design_matrix(t_fine, N_OPT)

def full_state(params):
    p   = np.asarray(params)
    cd_ = p[:NC]; cv_ = p[NC:]
    d_t = A_np @ cd_
    v_r = A_np @ cv_
    _, _, g_j, kap_j, aL_j, aT_j = (np.asarray(z)
                                    for z in forward(jnp.asarray(p)))
    xr  = xc + d_t * Ncx
    yr  = yc + d_t * Ncy
    lap = np.sum(g_j / np.maximum(v_r, 1e-3)) * dt
    a_mag = np.sqrt(aL_j**2 + aT_j**2)
    v_safe   = np.maximum(v_r, V_MIN)
    lon_lim  = np.where(aL_j >= 0,
                        np.minimum(_GRIP_F+_DF_F*v_safe**2,
                                   _POW/np.maximum(v_safe,1e-3))/_M
                        - _DRAG*v_safe**2/_M,
                        (_GRIP_B+_DF_B*v_safe**2+_DRAG*v_safe**2)/_M)
    side_lim = 1.2*(_GRIP_B+_DF_B*v_safe**2)/_M
    ang      = np.arctan2(aT_j, aL_j)
    a_max_dir = 1.0/np.sqrt((np.cos(ang)/np.maximum(lon_lim,1e-3))**2
                            + (np.sin(ang)/np.maximum(side_lim,1e-3))**2)
    return dict(xr=xr, yr=yr, d_t=d_t, v_r=v_r, g_r=g_j, lap=lap,
                kappa=kap_j, a_lon=aL_j, a_lat=aT_j,
                a_mag=a_mag, a_max_dir=a_max_dir)

S_init = full_state(params0)
S_opt  = full_state(params_opt)

print(f'\n  Initial: lap {S_init["lap"]:.2f}s  '
      f'path {np.sum(S_init["g_r"])*dt:.1f}m  '
      f'avg {np.sum(S_init["g_r"])*dt/S_init["lap"]*3.6:.1f} km/h')
print(f'  Optimal: lap {S_opt["lap"]:.2f}s  '
      f'path {np.sum(S_opt["g_r"])*dt:.1f}m  '
      f'avg {np.sum(S_opt["g_r"])*dt/S_opt["lap"]*3.6:.1f} km/h')

# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC DASHBOARD  (identical layout)
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

plt.suptitle('Physics-Constrained Minimum Lap Time — Silverstone  '
             '(JAX autodiff + JIT, progressive penalty)',
             fontsize=15, fontweight='bold')
plt.savefig('laptime_jax_penalty.png', dpi=130, bbox_inches='tight')
plt.show()