"""
═══════════════════════════════════════════════════════════════════════════════
  MULTI-TRACK PROGRESSIVE LAP-TIME OPTIMISER  →  JSON for the web viewer
═══════════════════════════════════════════════════════════════════════════════

For EVERY <Track>.csv in this folder:

  Two nested loops + innermost solver:
    OUTER  : resolution schedule  N ∈ [1,2,4,8,16,32,48,64,96,128,192]
             each N warm-starts from the previous N (zero-padded coeffs)
    INNER  : progressive penalty   μ ∈ schedule
             each μ warm-starts from the previous μ
    SOLVER : jaxopt.ScipyMinimize (L-BFGS) + JAX autodiff + JIT

  For each resolution level we store the racing line: path (x,y),
  physical speed, longitudinal/lateral/|a| accelerations, the
  speed-direction max-accel envelope, arc length, and lap time.

Output (written to ./site/data/):
    index.json          — track list + F1 records
    <Track>.json        — bounds + centerline + 11 resolution solutions
═══════════════════════════════════════════════════════════════════════════════
"""

import os, json, glob, time as _time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jaxopt import ScipyMinimize

jax.config.update("jax_enable_x64", True)

# ── Resolution schedule (warm-started progressively) ─────────────────────────
RES_SCHEDULE     = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192]
PENALTY_SCHEDULE = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8]
# N_FIT is set adaptively per-track inside solve_track() based on point count
N_PLOT_PTS       = 600
V_MIN            = 5.0
INNER_MAXITER    = 4000
INNER_TOL        = 1e-11

# Set FAST_TEST=1 env var for a quick pipeline check (1 track, 3 resolutions)
FAST = os.environ.get('FAST_TEST', '0') == '1'
if FAST:
    RES_SCHEDULE     = [1, 4, 16]
    PENALTY_SCHEDULE = [1e1, 1e3, 1e5]
    INNER_MAXITER    = 300

# ═══════════════════════════════════════════════════════════════════════════════
#  F1 RACE LAP RECORDS (seconds). Edit freely — only used for the delta display.
# ═══════════════════════════════════════════════════════════════════════════════
F1_RECORDS = {
    'Silverstone': 87.097,
    'Monza':       80.329,
    'Spa':         103.003,
    'Catalunya':   78.149,
    'Budapest':    78.739,
    'Spielberg':   65.619,
    'Suzuka':      90.983,
    'Sakhir':      91.447,
    'Shanghai':    92.238,
    'Sepang':      94.080,
    'Sochi':       95.761,
    'YasMarina':   86.103,
    'Melbourne':   80.235,
    'Montreal':    73.078,
    'Austin':      96.169,
    'MexicoCity':  77.774,
    'SaoPaulo':    71.011,
    'Hockenheim':  73.780,
    'Nuerburgring':89.212,
    'Zandvoort':   71.097,
    # Non-F1 layouts:
    'BrandsHatch': None, 'IMS': None, 'MoscowRaceway': None,
    'Norisring':   None, 'Oschersleben': None,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  FOURIER / GEOMETRY UTILITIES  (NumPy — one-off per track)
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

def pad_coeffs(c, N_from, N_to):
    out = np.zeros(2 * N_to + 1)
    out[:2 * N_from + 1] = c
    return out

# ═══════════════════════════════════════════════════════════════════════════════
#  F1 PHYSICS MODEL  (JAX, differentiable; 750 kg, ~1000 hp)
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

# NumPy mirrors for final diagnostics / serialisation
def mfa_np(v): return np.minimum(_GRIP_F+_DF_F*v**2, _POW/np.maximum(v,1e-3))/_M - _DRAG*v**2/_M
def mba_np(v): return (_GRIP_B+_DF_B*v**2+_DRAG*v**2)/_M
def msa_np(v): return 1.2*(_GRIP_B+_DF_B*v**2)/_M

# ═══════════════════════════════════════════════════════════════════════════════
#  PER-TRACK SOLVE
# ═══════════════════════════════════════════════════════════════════════════════

def solve_track(csv_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]

    # ── CSV parsing: skip the header line (plain text, no '#') ───────────────
    df = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if df.ndim != 2 or df.shape[1] < 4:
        raise ValueError(f"Unexpected CSV shape {df.shape}, expected (N, 4)")

    x, y            = df[:, 0], df[:, 1]
    w_right, w_left = df[:, 2], df[:, 3]

    # ── Adaptive Fourier order: need ~3x data points for numerical stability ──
    n_pts = len(x)
    N_FIT = min(192, max(32, n_pts // 3))
    print(f'    {name}: {n_pts} pts → N_FIT={N_FIT}')

    # ── Raw track bounds ──────────────────────────────────────────────────────
    dx_raw = np.gradient(x); dy_raw = np.gradient(y)
    nrm    = np.sqrt(dx_raw**2 + dy_raw**2)
    nx_raw, ny_raw  = -dy_raw/nrm, dx_raw/nrm
    x_right_raw = x + w_right*nx_raw; y_right_raw = y + w_right*ny_raw
    x_left_raw  = x - w_left *nx_raw; y_left_raw  = y - w_left *ny_raw

    # ── Arc-length parameterisation & centerline Fourier fit ─────────────────
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s  = np.concatenate([[0], np.cumsum(ds)])
    t  = 2*np.pi*s / s[-1]

    cx = fit_fourier(t, x, N_FIT)
    cy = fit_fourier(t, y, N_FIT)

    t_fine = np.linspace(0, 2*np.pi, 2000)
    dt     = float(t_fine[1] - t_fine[0])

    A_fine = build_design_matrix(t_fine,     N_FIT)
    D1c    = build_derivative_matrix(t_fine, N_FIT, 1)
    D2c    = build_derivative_matrix(t_fine, N_FIT, 2)
    D3c    = build_derivative_matrix(t_fine, N_FIT, 3)

    xc    = A_fine@cx;  yc    = A_fine@cy
    dxc   = D1c@cx;     dyc   = D1c@cy
    ddxc  = D2c@cx;     ddyc  = D2c@cy
    dddxc = D3c@cx;     dddyc = D3c@cy

    g_c, Tcx, Tcy, Ncx, Ncy, kappa_c = compute_frenet_np(dxc, dyc, ddxc, ddyc)

    # ── Project raw bounds onto centerline normals, fit as Fourier series ─────
    d_right =  project_points_onto_fourier(x_right_raw, y_right_raw, xc, yc, Ncx, Ncy)
    d_left  = -project_points_onto_fourier(x_left_raw,  y_left_raw,  xc, yc, Ncx, Ncy)
    cw_r = fit_fourier(t, d_right, N_FIT)
    cw_l = fit_fourier(t, d_left,  N_FIT)
    w_right_f = A_fine@cw_r
    w_left_f  = A_fine@cw_l

    ds_c = np.sqrt(np.diff(xc)**2 + np.diff(yc)**2)
    s_c  = np.concatenate([[0], np.cumsum(ds_c)])
    track_len = float(np.sum(g_c)*dt)

    # ── Centerline-normal derivatives (constant, used inside JAX forward) ─────
    dg_c_dt  = (dxc*ddxc + dyc*ddyc) / g_c
    dNcx_dt  = (-ddyc*g_c + dyc*dg_c_dt) / (g_c**2)
    dNcy_dt  = ( ddxc*g_c - dxc*dg_c_dt) / (g_c**2)
    numv     = dxc*ddyc - dyc*ddxc
    dnumv    = dxc*dddyc - dyc*dddxc
    dkgg     = (dnumv*g_c**2 - numv*2*g_c*dg_c_dt) / (g_c**4)
    kgg      = kappa_c*g_c
    ddNcx_dt = -dkgg*Tcx - kgg**2*Ncx
    ddNcy_dt = -dkgg*Tcy - kgg**2*Ncy

    # ── Push track constants to JAX device ───────────────────────────────────
    Jc = {k: jnp.asarray(v) for k, v in dict(
        dxc=dxc, dyc=dyc, ddxc=ddxc, ddyc=ddyc,
        Ncx=Ncx, Ncy=Ncy,
        dNcx=dNcx_dt, dNcy=dNcy_dt,
        ddNcx=ddNcx_dt, ddNcy=ddNcy_dt,
        wr=w_right_f, wl=w_left_f).items()}
    J_dt = jnp.asarray(dt)

    # ── JAX forward model factory (one per resolution level) ─────────────────
    def make_level(N_OPT):
        A_o  = jnp.asarray(build_design_matrix(t_fine,     N_OPT))
        D1_o = jnp.asarray(build_derivative_matrix(t_fine, N_OPT, 1))
        D2_o = jnp.asarray(build_derivative_matrix(t_fine, N_OPT, 2))
        NC   = 2*N_OPT + 1

        @jit
        def forward(p):
            cd_, cv_ = p[:NC], p[NC:]
            d_t = A_o@cd_; dd = D1_o@cd_; ddd = D2_o@cd_
            v_r = A_o@cv_; dv = D1_o@cv_
            dxr  = Jc['dxc']  + dd*Jc['Ncx']  + d_t*Jc['dNcx']
            dyr  = Jc['dyc']  + dd*Jc['Ncy']  + d_t*Jc['dNcy']
            ddxr = Jc['ddxc'] + ddd*Jc['Ncx'] + 2*dd*Jc['dNcx'] + d_t*Jc['ddNcx']
            ddyr = Jc['ddyc'] + ddd*Jc['Ncy'] + 2*dd*Jc['dNcy'] + d_t*Jc['ddNcy']
            g_r  = jnp.sqrt(dxr**2 + dyr**2)
            kap  = (dxr*ddyr - dyr*ddxr) / (g_r**3)
            aL   = (v_r/g_r) * dv
            aT   = v_r**2 * kap
            return d_t, v_r, g_r, kap, aL, aT

        @jit
        def lap_time_fn(p):
            _, v_r, g_r, _, _, _ = forward(p)
            return jnp.sum(g_r / jnp.maximum(v_r, 1e-3)) * J_dt

        @jit
        def cviol_fn(p):
            d_t, v_r, g_r, kap, aL, aT = forward(p)
            v  = jnp.maximum(v_r, V_MIN)
            ll = jnp.maximum(jnp.where(aL>=0, max_front_acc(v), max_back_acc(v)), 0.5)
            sl = jnp.maximum(max_side_acc(v), 0.5)
            r  = lambda z: jnp.maximum(z, 0.0)
            return jnp.sum(
                r((aL/ll)**2 + (aT/sl)**2 - 1.0)**2
                + r(d_t - Jc['wr'])**2
                + r(-Jc['wl'] - d_t)**2
                + r(V_MIN - v_r)**2
            ) * J_dt

        return forward, lap_time_fn, cviol_fn, NC

    # ── Two nested loops: resolution (outer) × penalty (inner) ───────────────
    solutions = []
    params    = None
    N_prev    = None

    for N_OPT in RES_SCHEDULE:
        fwd, lap_time_fn, cviol_fn, NC = make_level(N_OPT)

        # Warm-start from previous resolution (zero-pad coefficients)
        if params is None:
            cd0 = np.zeros(NC); cv0 = np.zeros(NC); cv0[0] = 20.0
            p   = jnp.asarray(np.concatenate([cd0, cv0]))
        else:
            cd_p = np.asarray(params[:2*N_prev+1])
            cv_p = np.asarray(params[2*N_prev+1:])
            p    = jnp.asarray(np.concatenate([
                pad_coeffs(cd_p, N_prev, N_OPT),
                pad_coeffs(cv_p, N_prev, N_OPT),
            ]))

        # Inner penalty loop
        for mu in PENALTY_SCHEDULE:
            def loss(q, _mu=mu):
                return lap_time_fn(q) + _mu * cviol_fn(q)
            vag    = jit(value_and_grad(loss))
            solver = ScipyMinimize(
                method='L-BFGS-B', fun=vag, value_and_grad=True,
                maxiter=INNER_MAXITER, tol=INNER_TOL,
            )
            p = solver.run(p).params

        params = p
        N_prev = N_OPT

        # ── Serialise this resolution level ───────────────────────────────────
        pn    = np.asarray(p)
        A_np  = build_design_matrix(t_fine, N_OPT)
        d_t   = A_np @ pn[:NC]
        v_r   = A_np @ pn[NC:]
        _, _, g_j, kap_j, aL_j, aT_j = (np.asarray(z)
                                          for z in fwd(jnp.asarray(pn)))
        xr  = xc + d_t*Ncx
        yr  = yc + d_t*Ncy
        lap = float(np.sum(g_j / np.maximum(v_r, 1e-3)) * dt)
        viol= float(cviol_fn(jnp.asarray(pn)))

        a_mag  = np.sqrt(aL_j**2 + aT_j**2)
        v_sf   = np.maximum(v_r, V_MIN)
        lon_l  = np.where(aL_j>=0, mfa_np(v_sf), mba_np(v_sf))
        side_l = msa_np(v_sf)
        ang    = np.arctan2(aT_j, aL_j)
        a_max  = 1.0 / np.sqrt(
            (np.cos(ang) / np.maximum(lon_l,  1e-3))**2 +
            (np.sin(ang) / np.maximum(side_l, 1e-3))**2
        )
        heading = np.arctan2(np.gradient(yr), np.gradient(xr))

        # Decimate to N_PLOT_PTS for the web
        idx   = np.linspace(0, len(xr)-1, N_PLOT_PTS).astype(int)
        dec   = lambda a: [round(float(v), 4) for v in np.asarray(a)[idx]]
        s_dec = [round(float(v), 4) for v in s_c[idx]]

        solutions.append(dict(
            N=int(N_OPT), lap_time=round(lap, 4),
            constraint_viol=viol,
            x=dec(xr), y=dec(yr), s=s_dec,
            speed=dec(v_r*3.6),       # km/h for display
            speed_ms=dec(v_r),
            a_lon=dec(aL_j), a_lat=dec(aT_j),
            a_mag=dec(a_mag), a_max=dec(a_max),
            heading=dec(heading),
        ))
        print(f'    [{name}] N={N_OPT:4d}  lap {lap:8.3f}s  viol {viol:.1e}')

    # ── Bounds at full N_PLOT_PTS resolution ─────────────────────────────────
    idx2 = np.linspace(0, len(xc)-1, N_PLOT_PTS).astype(int)
    decc = lambda a: [round(float(v), 4) for v in np.asarray(a)[idx2]]

    track_json = dict(
        name=name,
        f1_record=F1_RECORDS.get(name, None),
        track_length=round(track_len, 1),
        bounds=dict(
            xr=decc(xc + w_right_f*Ncx), yr=decc(yc + w_right_f*Ncy),
            xl=decc(xc - w_left_f *Ncx), yl=decc(yc - w_left_f *Ncy),
            xc=decc(xc), yc=decc(yc),
        ),
        resolutions=RES_SCHEDULE,
        solutions=solutions,
    )
    return name, track_json

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — loop every CSV, write JSON
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    here    = os.path.dirname(os.path.abspath(__file__))
    csvs    = sorted(glob.glob(os.path.join(here, '*.csv')))
    out_dir = os.path.join(here, 'site', 'data')
    os.makedirs(out_dir, exist_ok=True)

    if FAST:
        csvs = csvs[:1]
        print('FAST_TEST mode: 1 track, reduced schedule')

    index = []
    for ci, csv_path in enumerate(csvs):
        nm = os.path.splitext(os.path.basename(csv_path))[0]
        print(f'\n══ [{ci+1}/{len(csvs)}] {nm} ══════════════════════════')
        t0 = _time.time()
        try:
            name, tj = solve_track(csv_path)
        except Exception as e:
            print(f'  !! skipped {nm}: {e}')
            continue
        with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
            json.dump(tj, f, separators=(',', ':'))
        best = tj['solutions'][-1]
        index.append(dict(
            name=name, f1_record=tj['f1_record'],
            best_lap=best['lap_time'],
            track_length=tj['track_length'],
        ))
        print(f'  ✓ {name}: best {best["lap_time"]:.2f}s  '
              f'({_time.time()-t0:.0f}s elapsed)')

    with open(os.path.join(out_dir, 'index.json'), 'w') as f:
        json.dump(dict(tracks=index), f, indent=2)

    print(f'\n✓ Wrote {len(index)} tracks → {out_dir}')