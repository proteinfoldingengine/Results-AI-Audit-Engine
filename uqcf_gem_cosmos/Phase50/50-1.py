# =========================================================
# UQCF–GEM Phase 50.1 (Targeted Noise Model Update)
# =========================================================
# Goal: Re-run Phase 50.0 pipeline with modified noise
# (beta=0.5, sigma=5e-5) to test if IntegratorAccepted passes.
# All other parameters and prereg thresholds remain identical.
# =========================================================

# =========================================================
# 0. ENV + IMPORTS
# =========================================================

import os, sys, time, math, json, hashlib, platform
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.linalg import expm
from scipy.stats import linregress
from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy.fft as fft
import matplotlib.pyplot as plt

# Deterministic BLAS/threading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_CORETYPE"] = "generic"

np.set_printoptions(precision=12, floatmode="maxprec_equal", suppress=True)
np.seterr(all="raise")

try:
    import tensornetwork as tn
    print(f"TensorNetwork loaded (v{tn.__version__})")
except ImportError:
    print("TensorNetwork not available; MPO path disabled.")
    tn = None


# =========================================================
# 1. CONFIG / PREREG
# =========================================================

@dataclass
class RunConfig:
    # --- Stochastic Parameters ---
    stochastic: bool   = True      # keep stochastic mode ON
    noise_sigma: float = 5e-5      # PHASE 50.1 CHANGE (was 1e-4)
    noise_beta: float  = 0.5       # PHASE 50.1 CHANGE (was 0.0)
    seed: int          = 42        # DO NOT CHANGE

    # --- Temporal Kahan Parameters (unchanged) ---
    enable_kahan_coupling: bool = True
    TAU_C: float               = 1.0
    ETA_SCALE_C: float         = 0.1
    COUPLING_LAMBDA_C: float   = 0.01

    # --- Audit Toggles (unchanged) ---
    do_A: bool = True  # BCH commutator budget
    do_BC: bool = True # Integrator drift / scaling
    do_D: bool = True  # PrecisionGate (c64 vs c128 divergence)
    do_F: bool = True  # Precision sweep (eps -> drift slope κ)
    do_E: bool = True  # Physics grid check (only if gates pass)
    do_TN: bool = False # Legacy tensor network demo (off by default)

args = RunConfig()


def detect_platform_id() -> str:
    """Get arch + BLAS ID for audit stamp."""
    try:
        old_stdout, buf = sys.stdout, StringIO()
        sys.stdout = buf
        np.show_config()
        sys.stdout = old_stdout
        cfg = buf.getvalue()
        if "openblas" in cfg.lower():
            blas = "OpenBLAS"
        elif "mkl" in cfg.lower():
            blas = "MKL"
        else:
            blas = "BLAS_Unknown"
    except Exception:
        blas = "BLAS_Unknown"
    return f"cpu_{platform.machine()}_{blas}"

PLATFORM_ID = detect_platform_id()

# Physical constants (used in Audit E, cosmology path)
C_M_PER_S = 2.99792458e8
MPC_M = 3.085677581e22
k0_ref_calc = (2.0 * np.pi * 1e-7) / (C_M_PER_S / MPC_M) # ~pivot k in 1/Mpc units

# ---------- PREREG: frozen thresholds / knobs (Identical to 50.0) ----------
PREREG = {
    "title": "UQCF-GEM Phase 50.1 — Integrator Test (Pink Noise)", # <-- Title updated
    "integrator_accept_det": {
        "k_E_window":   (1.8, 2.2),
        "k_S_window":   (2.8, 3.2),
        "alpha_window": (1.45, 1.55),
        "R2_E_min": 0.95,
        "R2_S_min": 0.95,
        "R2_a_min": 0.90,
        "monotone_eps": 0.05
    },
    "integrator_accept_stoch": {
        "k_window":     (-1.6, -0.4), # both k_E and k_S
        "alpha_window": (0.7, 1.3),
        "R2_min": 0.90,
        "monotone_eps": 0.20
    },
    "precision_gate": {
        "kappa_prime_thresh_det": 1e-5,
        "kappa_prime_thresh_stoch": 5e-4,
        "R2_min": 0.90,
        "dt": 0.04,
        "steps": 1500
    },
    "dt_ref_emergent": 0.01,     # where we read gamma_bar for E_unit
    "kappa_universality": {
        "tol": 0.2,             # |κ_strang - κ_trotter| must be <= tol
        "R2_min": 0.90
    },
    "geom": {
        "Q_audit": 8,
        "T_total": 24.0,
        "dt_grid": [0.03,0.025,0.02,0.015,0.0125,0.01,0.0075,0.00625,0.005,0.004]
    },
    # Cosmology / grid sanity prereg (unchanged)
    "numerics": {
        "integrator": {"sigma": 0.5, "logx_sq_clamp": 100.0},
        "k0_ref_mpc_inv": k0_ref_calc,
        # Placeholders for Xi0, Hc_star - filled if normalize_xi0 runs
        "Xi0_dict": {},
        "Hc_star_dict": {},
        "k_grid_forced": {
            "min_mpc_inv": 2.2e+02,
            "max_mpc_inv": 1.9e+10,
            "n_log": 400
        },
        "cosmo": {"h": 0.674},
        "bands": {
            "PTA":  {"f_min_Hz":1e-9, "f_max_Hz":1e-6, "n_log":60},
            "LISA": {"f_min_Hz":1e-4, "f_max_Hz":1e0, "n_log":80}
        }
    },
    "pump_grid": {
        "a_star":[1e-24,1e-22,1e-20],
        "deltaN":[0.5,1.0],
        "p":[-1,0,1],
        "Xi_factors":[0.5,1.0,1.5]
    },
    "predictions": {
        "V_over_I_peak":{"mean":0.040,"sigma":0.005},
        "n_t":{"mean":-0.42,"sigma":0.10},
        "Omega_GW_max":1e-8
    },
    "pass_fail": {
        "robustness_amp_drift_pct_max": 30.0,
        "robustness_loc_drift_dex_max": 0.30
    },
    # runtime-filled after audits:
    "results": {
        "IntegratorAccepted": None,
        "PrecisionGate": None,
        "GridSanityPassed": None,
        "UniversalityPassed": None,
        "E_unit": None,
        "alpha_fit": None,
        "k_E": None,
        "k_S": None,
        "kappa": {},
    }
}


# =========================================================
# 2. BASIC NUMERICS & HELPERS (Unchanged from 50.0)
# =========================================================

def crn_seed_seq(*parts) -> np.random.SeedSequence:
    """Deterministic "common random numbers"."""
    ints = []
    for p in parts:
        if isinstance(p, (int, np.integer)):
            ints.append(int(p) & 0xFFFFFFFF)
        else:
            h = hashlib.blake2s(str(p).encode("utf-8"), digest_size=8)
            ints.append(int.from_bytes(h.digest(), "little") & 0xFFFFFFFF)
    return np.random.SeedSequence(ints)

def kahan_sum(arr) -> float:
    s = 0.0
    c = 0.0
    for x in np.asarray(arr, dtype=np.float64).ravel():
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return float(s)

def energy_variance(psi, H) -> float:
    """σ_H^2 = <H^2> - <H>^2"""
    psi_c = psi.astype(np.complex128)
    H_c = H.astype(np.complex128)
    e = psi_c.conj() @ (H_c @ psi_c)
    e2 = psi_c.conj() @ (H_c @ (H_c @ psi_c))
    var = (e2.real - (e.real**2))
    return float(max(0.0, var))

def tail_stat_aligned(t_vec, y_vec, window=2.0, nsamp=200, method="median", envelope=False):
    """Sample tail into common window, return scalar stat."""
    t_vec = np.asarray(t_vec, float)
    y_vec = np.asarray(y_vec, float)
    if len(t_vec) < 2: return float("nan")
    T_end = t_vec[-1]
    T0 = max(0.0, T_end - window)
    t_common = np.linspace(T0, T_end, nsamp) if T_end > T0 else np.array([T_end])
    y_interp = np.interp(t_common, t_vec, np.abs(y_vec))
    y_use = np.maximum.accumulate(y_interp) if envelope else y_interp
    if method == "median": return float(np.median(y_use))
    elif method == "p75": return float(np.percentile(y_use, 75))
    else: raise ValueError("method must be 'median' or 'p75'")

def linear_fit(x, y, label=""):
    """Simple linear regression y = a*x + b -> slope a, intercept b, R²."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y); x, y = x[valid], y[valid]
    if len(x) < 2: return np.nan, np.nan, 0.0
    slope, intercept, r_val, *_ = linregress(x, y)
    return float(slope), float(intercept), float(r_val**2)

def near_monotone(vals, eps_frac):
    """Check if tail error is mostly monotone decreasing."""
    v = np.asarray(vals, float);
    if len(v) < 2: return True
    diffs = np.diff(v); wiggle = np.abs(v[:-1]) * eps_frac
    return bool(np.all(diffs <= wiggle))


# =========================================================
# 3. RYDBERG GEOMETRY + HAMILTONIAN (Unchanged from 50.0)
# =========================================================

_op_cache: Dict[tuple, np.ndarray] = {}

def pauli_operator(op: str, site: int, Q: int, dtype=np.complex128) -> np.ndarray:
    """Build single-site operator embedded into Q-qubit space."""
    cache_key = (op, site, Q, dtype)
    if cache_key in _op_cache: return _op_cache[cache_key]
    I_f64 = np.eye(2, dtype=np.float64); X_f64 = np.array([[0,1],[1,0]], dtype=np.float64)
    Z_f64 = np.array([[1,0],[0,-1]], dtype=np.float64); Y_c128 = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
    n_f64 = 0.5 * (I_f64 - Z_f64) # projector on |1>
    base = {"X": X_f64, "Y": Y_c128, "Z": Z_f64, "n_std": n_f64}[op]
    out = np.array([[1.0]], dtype=dtype); I_d = np.eye(2, dtype=dtype)
    for q in range(Q):
        block = base.astype(dtype) if q == site else I_d
        out = np.kron(out, block)
    _op_cache[cache_key] = out
    return out

def sierpinski_points(level=1, spacing=4.0):
    """2D Sierpinski carpet points."""
    import itertools
    base = [(x,y) for x,y in itertools.product(range(3), repeat=2) if not (x==1 and y==1)]
    pts = np.array(base, float)
    for _ in range(1, level):
        nxt=[];
        for gx,gy in pts:
            for x,y in base: nxt.append((3*gx + x, 3*gy + y))
        pts = np.array(nxt, float)
    if level > 0:
        pts = pts / (3**level - 1); pts *= spacing * (3**(level-1))
    return pts

def mean_nn_dist(P):
    dmins=[];
    for i in range(len(P)):
        d = np.inf
        for j in range(len(P)):
            if i==j: continue
            d = min(d, norm(P[i]-P[j]))
        dmins.append(d)
    return float(np.mean(dmins)) if dmins else 0.0

def matched_density_points(Q, base_spacing=1.0):
    """Pick Q points, rescale for stable avg nn distance."""
    _op_cache.clear()
    Pref = sierpinski_points(level=1, spacing=base_spacing); d_ref = mean_nn_dist(Pref)
    level=1
    while True:
        _op_cache.clear()
        Pfull = sierpinski_points(level=level, spacing=base_spacing)
        if len(Pfull) >= Q: break
        level += 1
    P = Pfull[:Q].copy(); d_now = mean_nn_dist(P); scale = d_ref / max(d_now, 1e-12)
    return P * scale

def rydberg_V(P, C6=1.0, blockade_R=None):
    """Pairwise Rydberg interaction ~ C6 / r^6."""
    Q = len(P); V = np.zeros((Q,Q), float)
    for i in range(Q):
        for j in range(i+1, Q):
            r = norm(P[i]-P[j]) + 1e-12; vij = C6 / (r**6)
            if blockade_R is not None and r < blockade_R: vij = 1e6
            V[i,j] = V[j,i] = vij
    return V

def build_blocks(Q:int, Omega:float, Delta:float, V_mat:np.ndarray, dtype=np.complex128):
    """Build HX, HZ, HZZ for Rydberg Hamiltonian."""
    dim = 2**Q; HX = np.zeros((dim,dim), dtype=dtype); HZ = np.zeros((dim,dim), dtype=dtype); HZZ = np.zeros((dim,dim), dtype=dtype)
    n_ops = [pauli_operator("n_std", i, Q, dtype) for i in range(Q)]
    for i in range(Q):
        HX += 0.5*Omega * pauli_operator("X", i, Q, dtype)
        HZ += -Delta   * n_ops[i]
    for i in range(Q):
        for j in range(i+1, Q):
            if V_mat[i,j] != 0: HZZ += V_mat[i,j] * (n_ops[i] @ n_ops[j])
    norms = {"HX_norm_F": float(np.linalg.norm(HX,"fro")), "HZ_norm_F": float(np.linalg.norm(HZ,"fro")), "HZZ_norm_F": float(np.linalg.norm(HZZ,"fro"))}
    return HX, HZ, HZZ, norms


# =========================================================
# 4. UNITARY FACTORS + NOISE + TEMPORAL KAHAN COUPLING (Unchanged from 50.0)
# =========================================================

def build_factors(HX, HZ, HZZ, dt, dtype):
    """Precompute exp(-i H_piece dt) etc."""
    HX = HX.astype(dtype); HZ = HZ.astype(dtype); HZZ= HZZ.astype(dtype)
    start = time.time()
    UZ = expm(-1j*HZ * dt).astype(dtype); UX = expm(-1j*HX * dt).astype(dtype); UZZ = expm(-1j*HZZ * dt).astype(dtype)
    UZ_half = expm(-1j*HZ * (dt/2)).astype(dtype); UZZ_half= expm(-1j*HZZ * (dt/2)).astype(dtype)
    print(f"   build_factors(dt={dt}): {time.time()-start:.2f}s")
    return {"UZ": UZ, "UX": UX, "UZZ": UZZ, "UZ_half": UZ_half, "UZZ_half": UZZ_half}

def apply_noise(U, sigma, beta, seed_tag):
    """Apply tiny skew-Hermitian random kick ~exp(sigma*K)."""
    if sigma <= 0: return U
    rng = np.random.default_rng(crn_seed_seq(*seed_tag))
    R = (rng.normal(0,1,U.shape) + 1j*rng.normal(0,1,U.shape)).astype(np.complex128)
    if beta > 0:
        flat = R.reshape(-1); fft_eps = fft.fft(flat); freqs = fft.fftfreq(flat.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(freqs!=0, np.abs(freqs)**(-beta/2.0), 0.0); scale[~np.isfinite(scale)] = 0.0
        R = fft.ifft(fft_eps * scale).reshape(R.shape)
    K = R - R.conj().T; KF = np.linalg.norm(K,"fro")
    if KF > 0: K /= KF
    U_noise = expm((sigma*K).astype(U.dtype))
    return (U @ U_noise).astype(U.dtype)

def kahan_update(c_prev, dt, tau, eta_scale, run_tag, step_idx, global_seed):
    """OU process for the temporal Kahan compensator c(t)."""
    beta = math.exp(-dt/tau) if tau>0 else 0.0
    rng = np.random.default_rng(crn_seed_seq("c_state_eta", run_tag, step_idx, global_seed))
    eta = rng.normal(0.0,1.0)
    c_new = beta*c_prev + math.sqrt(max(0.0,1.0-beta**2))*eta*eta_scale
    return c_new

def apply_coupled_compensator(U, c_state, coupling_lambda):
    """Feed c_state back into U via skew-Hermitian correction."""
    if not args.enable_kahan_coupling or abs(coupling_lambda) < 1e-15: return U
    Uc = U.astype(np.complex128); K = Uc - Uc.conj().T; K_norm = np.linalg.norm(K,"fro")
    if K_norm < 1e-12: return U
    K_scaled = (coupling_lambda * c_state / K_norm) * K
    try:
        U_corr = expm(K_scaled.astype(U.dtype)); return (U_corr @ U).astype(U.dtype)
    except Exception: return U # fallback


def step_trotter1(psi, fac, step_idx, *, dt, run_tag, c_state,
                  tau, eta_scale, coupling_lambda,
                  stochastic, sigma, beta, gseed, renorm=False):
    """First-order Trotter: UZ -> UX -> UZZ."""
    if args.enable_kahan_coupling: c_state = kahan_update(c_state, dt, tau, eta_scale, run_tag, step_idx, gseed)
    ops = [("UZ",fac["UZ"]), ("UX",fac["UX"]), ("UZZ",fac["UZZ"])]
    for opname, Uop in ops:
        if stochastic: Uop = apply_noise(Uop, sigma, beta, (run_tag, opname, step_idx, gseed))
        Uop = apply_coupled_compensator(Uop, c_state, coupling_lambda); psi = Uop @ psi
    if renorm: nrm = kahan_sum(np.abs(psi)**2); psi = psi/np.sqrt(nrm) if nrm>1e-300 else np.eye(len(psi))[0]
    return psi, c_state

def step_strang2(psi, fac, step_idx, *, dt, run_tag, c_state,
                 tau, eta_scale, coupling_lambda,
                 stochastic, sigma, beta, gseed, renorm=False):
    """Second-order Strang: UZ/2 -> UZZ/2 -> UX -> UZZ/2 -> UZ/2."""
    if args.enable_kahan_coupling: c_state = kahan_update(c_state, dt, tau, eta_scale, run_tag, step_idx, gseed)
    seq = [("UZ_half_pre", fac["UZ_half"]), ("UZZ_half_pre", fac["UZZ_half"]), ("UX", fac["UX"]),
           ("UZZ_half_post",fac["UZZ_half"]), ("UZ_half_post", fac["UZ_half"])]
    for opname, Uop in seq:
        if stochastic: Uop = apply_noise(Uop, sigma, beta, (run_tag, opname, step_idx, gseed))
        Uop = apply_coupled_compensator(Uop, c_state, coupling_lambda); psi = Uop @ psi
    if renorm: nrm = kahan_sum(np.abs(psi)**2); psi = psi/np.sqrt(nrm) if nrm>1e-300 else np.eye(len(psi))[0]
    return psi, c_state

def step_rts1(psi, fac, step_idx, *, dt, run_tag, c_state,
              tau, eta_scale, coupling_lambda,
              stochastic, sigma, beta, gseed, renorm=False):
    """Randomized Trotter sequence (RTS1)."""
    if args.enable_kahan_coupling: c_state = kahan_update(c_state, dt, tau, eta_scale, run_tag, step_idx, gseed)
    perm_rng = np.random.default_rng(crn_seed_seq(run_tag, "perm", step_idx, gseed)); order = perm_rng.permutation(["UZ","UX","UZZ"])
    for opname in order:
        Uop = fac[opname]
        if stochastic: Uop = apply_noise(Uop, sigma, beta, (run_tag, opname, step_idx, gseed))
        Uop = apply_coupled_compensator(Uop, c_state, coupling_lambda); psi = Uop @ psi
    if renorm: nrm = kahan_sum(np.abs(psi)**2); psi = psi/np.sqrt(nrm) if nrm>1e-300 else np.eye(len(psi))[0]
    return psi, c_state

INTEGRATORS = {"trotter1": step_trotter1, "strang2": step_strang2, "rts1": step_rts1}


# =========================================================
# 5. AUDIT A (COMMUTATOR BUDGET) (Unchanged from 50.0)
# =========================================================

def commutator_budget(HX, HZ, HZZ):
    """BCH commutator norm ||C||_F."""
    HXc, HZc, HZZc = HX.astype(np.complex128), HZ.astype(np.complex128), HZZ.astype(np.complex128)
    C1 = HXc@HZc - HZc@HXc; C2 = HXc@HZZc - HZZc@HXc; C3 = HZc@HZZc - HZZc@HZc; C = C1 + C2 + C3
    fn = lambda A: float(np.linalg.norm(A, "fro"))
    return {"HXHZ": fn(C1), "HXHZZ": fn(C2), "HZHZZ": fn(C3), "C_tot": fn(C)}


# =========================================================
# 6. AUDITS B & C (DRIFT / SCALING / ALPHA / E_unit) (Unchanged from 50.0)
# =========================================================

def run_integrator_tracks(HX, HZ, HZZ, H_full) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Evolve psi(t), log drifts, compute tail stats."""
    Q = int(round(np.log2(H_full.shape[0]))); T_tot = PREREG["geom"]["T_total"]; dt_list = PREREG["geom"]["dt_grid"]
    results_rows = []; traces_by_int = {}
    for int_name, step_fn in INTEGRATORS.items():
        for dt in dt_list:
            fac = build_factors(HX, HZ, HZZ, dt, dtype=np.complex128)
            psi = np.zeros(2**Q, np.complex128); psi[0] = 1.0; c_state = 0.0
            E0 = (psi.conj() @ (H_full @ psi)).real; S0 = energy_variance(psi, H_full)
            t_axis = []; dE_trace = []; dS_trace = []; c_trace = []
            steps_total = int(round(T_tot / dt)); run_tag = f"BC:{int_name}:dt={dt}:seed={args.seed}"
            sample_stride = max(1, steps_total // 100)
            for step_idx in range(steps_total):
                psi, c_state = step_fn(psi, fac, step_idx, dt=dt, run_tag=run_tag, c_state=c_state,
                                       tau=args.TAU_C, eta_scale=args.ETA_SCALE_C, coupling_lambda=args.COUPLING_LAMBDA_C,
                                       stochastic=args.stochastic, sigma=args.noise_sigma, beta=args.noise_beta, gseed=args.seed,
                                       renorm=False)
                if (step_idx % sample_stride == 0) or (step_idx == steps_total-1):
                    Ek = (psi.conj() @ (H_full @ psi)).real; Sk = energy_variance(psi, H_full)
                    dE_trace.append(abs(Ek - E0)); dS_trace.append(abs(Sk - S0)); c_trace.append(c_state); t_axis.append(step_idx * dt)
            if args.stochastic and args.noise_sigma > 0:
                tail_E = tail_stat_aligned(t_axis, dE_trace, window=2.0, method="median", envelope=False)
                tail_S2= tail_stat_aligned(t_axis, dS_trace, window=2.0, method="p75", envelope=False)
            else:
                tail_E = tail_stat_aligned(t_axis, dE_trace, window=2.0, method="median", envelope=False)
                tail_S2= tail_stat_aligned(t_axis, dS_trace, window=2.0, method="p75", envelope=True)
            traces_by_int.setdefault(int_name, {})[dt] = {"t": t_axis, "dE": dE_trace, "dS": dS_trace, "c": c_trace}
            results_rows.append({"integrator": int_name, "dt": dt, "DeltaE_tail": tail_E, "DeltaS2_tail": tail_S2})
    df_results = pd.DataFrame(results_rows)
    return df_results, traces_by_int

def fit_scaling_and_alpha(df_results: pd.DataFrame) -> Dict[str, Dict]:
    """Fit |ΔE| vs dt, |Δσ²| vs dt, and alpha = slope(log Δσ² vs log ΔE)."""
    out = {}
    for int_name in df_results["integrator"].unique():
        sub = df_results[df_results["integrator"] == int_name].copy()
        dt_all = np.asarray(sub["dt"], float); dE_tail = np.asarray(sub["DeltaE_tail"], float); dS_tail = np.asarray(sub["DeltaS2_tail"], float)
        mask_fit = dt_all <= 0.015; x_fit = dt_all[mask_fit]; yE = dE_tail[mask_fit]; yS = dS_tail[mask_fit]
        with np.errstate(divide="ignore"): log_dt = np.log10(x_fit); log_E = np.log10(yE); log_S = np.log10(yS)
        k_E, bE, r2_E = linear_fit(log_dt, log_E, label=f"{int_name} k_E")
        k_S, bS, r2_S = linear_fit(log_dt, log_S, label=f"{int_name} k_S")
        a_fit, ba, r2_a = linear_fit(log_E, log_S, label=f"{int_name} alpha")
        out[int_name] = {"k_E": k_E, "R2_E": r2_E, "k_S": k_S, "R2_S": r2_S, "alpha": a_fit, "R2_alpha": r2_a}
    return out

def check_integrator_accept(scaling_summary: Dict, df_results: pd.DataFrame):
    """Check IntegratorAccepted gate and compute E_unit."""
    is_stoch = args.stochastic and args.noise_sigma > 0
    gate = PREREG["integrator_accept_stoch"] if is_stoch else PREREG["integrator_accept_det"]
    strang_stats = scaling_summary.get("strang2", {}); k_E = strang_stats.get("k_E", np.nan); k_S = strang_stats.get("k_S", np.nan); a_fit = strang_stats.get("alpha", np.nan)
    R2_E = strang_stats.get("R2_E", 0.0); R2_S = strang_stats.get("R2_S", 0.0); R2_a = strang_stats.get("R2_alpha", 0.0)
    sub_strang = df_results[df_results["integrator"]=="strang2"].sort_values("dt", ascending=False); tail_list = sub_strang["DeltaE_tail"].to_numpy()
    mono_ok = near_monotone(tail_list, eps_frac=gate["monotone_eps"])
    if is_stoch:
        k_ok = gate["k_window"][0] <= k_E <= gate["k_window"][1]; kS_ok = gate["k_window"][0] <= k_S <= gate["k_window"][1]
        a_ok = gate["alpha_window"][0] <= a_fit <= gate["alpha_window"][1]; r2_ok = (R2_E >= gate["R2_min"] and R2_S >= gate["R2_min"] and R2_a >= gate["R2_min"])
    else:
        k_ok = (gate["k_E_window"][0] <= k_E <= gate["k_E_window"][1]); kS_ok = (gate["k_S_window"][0] <= k_S <= gate["k_S_window"][1])
        a_ok = (gate["alpha_window"][0]<= a_fit<= gate["alpha_window"][1]); r2_ok = (R2_E >= gate["R2_E_min"] and R2_S >= gate["R2_S_min"] and R2_a >= gate["R2_a_min"])
    accepted = (k_ok and kS_ok and a_ok and r2_ok and mono_ok)
    E_unit = None
    if accepted:
        dt_ref = PREREG["dt_ref_emergent"]; strang_sorted = df_results[df_results["integrator"]=="strang2"].sort_values("dt")
        dts = strang_sorted["dt"].to_numpy(); vals= strang_sorted["DeltaE_tail"].to_numpy()
        if np.any(np.isclose(dts, dt_ref)): gamma_bar_ref = vals[np.argmin(abs(dts-dt_ref))]
        else:
            valid = (vals>0);
            if valid.sum() >= 2:
                log_dt = np.log10(dts[valid]); log_val = np.log10(vals[valid]); m, b, _ = linear_fit(log_dt, log_val, "gamma_bar_ref fit")
                gamma_bar_ref = 10**(b + m*np.log10(dt_ref))
            else: gamma_bar_ref = np.nan
        if np.isfinite(gamma_bar_ref) and np.isfinite(a_fit) and abs(a_fit) > 1e-12: E_unit = gamma_bar_ref / a_fit
    return accepted, {"k_E":k_E, "k_S":k_S, "alpha":a_fit, "R2_E":R2_E, "R2_S":R2_S, "R2_alpha":R2_a, "E_unit":E_unit}


# =========================================================
# 7. AUDIT D (PRECISION GATE) (Unchanged from 50.0)
# =========================================================

def precision_gate(H128_parts, H64_parts, Q):
    """Evolve c128 vs c64, fit divergence div ~ a * sqrt(t) + b."""
    dt = PREREG["precision_gate"]["dt"]; steps = PREREG["precision_gate"]["steps"]
    HX128,HZ128,HZZ128,H128 = H128_parts; HX64,HZ64,HZZ64,H64 = H64_parts
    fac128 = build_factors(HX128,HZ128,HZZ128,dt,np.complex128); fac64 = build_factors(HX64, HZ64, HZZ64, dt,np.complex64)
    psi128 = np.zeros(2**Q, np.complex128); psi128[0]=1.0; psi64 = np.zeros(2**Q, np.complex64); psi64[0]=1.0; c128 = 0.0; c64 = 0.0
    div_trace = []; run_tag = f"D:dt={dt}:seed={args.seed}"
    for step_idx in range(steps):
        psi128, c128 = step_strang2(psi128, fac128, step_idx, dt=dt, run_tag=run_tag, c_state=c128, tau=args.TAU_C, eta_scale=args.ETA_SCALE_C, coupling_lambda=args.COUPLING_LAMBDA_C,
                                    stochastic=args.stochastic, sigma=args.noise_sigma, beta=args.noise_beta, gseed=args.seed, renorm=False)
        psi64, c64 = step_strang2(psi64, fac64, step_idx, dt=dt, run_tag=run_tag, c_state=c64, tau=args.TAU_C, eta_scale=args.ETA_SCALE_C, coupling_lambda=args.COUPLING_LAMBDA_C,
                                  stochastic=args.stochastic, sigma=args.noise_sigma, beta=args.noise_beta, gseed=args.seed, renorm=False)
        diff = np.linalg.norm(psi128 - psi64.astype(np.complex128)); div_trace.append(float(diff))
    t_arr = np.arange(len(div_trace))*dt; sqrt_t = np.sqrt(np.maximum(t_arr,1e-300))
    kappa_prime, intercept, R2 = linear_fit(sqrt_t, div_trace, "precision κ'")
    gate = PREREG["precision_gate"]
    if args.stochastic and args.noise_sigma>0: pass_gate = (kappa_prime < gate["kappa_prime_thresh_stoch"]) and (R2 > gate["R2_min"])
    else: pass_gate = (kappa_prime < gate["kappa_prime_thresh_det"]) and (R2 > gate["R2_min"])
    return pass_gate, {"kappa_prime": kappa_prime, "R2": R2, "div_final": div_trace[-1]}


# =========================================================
# 8. AUDIT F (PRECISION SWEEP κ-LAW) (Unchanged from 50.0)
# =========================================================

def machine_eps(dtype_name):
    if dtype_name == "float32": return float(np.finfo(np.float32).eps)
    if dtype_name == "float64": return float(np.finfo(np.float64).eps)
    return np.nan

def cast_complex_dtype(float_dtype):
    return np.complex128 if float_dtype == np.float64 else np.complex64

def cast_hamiltonians_to_dtype(HX, HZ, HZZ, float_dtype):
    cdtype = cast_complex_dtype(float_dtype)
    return (HX.astype(cdtype), HZ.astype(cdtype), HZZ.astype(cdtype), (HX+HZ+HZZ).astype(cdtype))

def run_precision_sweep(HX, HZ, HZZ, Q):
    """Run strang2/trotter1 at diff precisions, fit log(drift) vs log(eps) -> slope κ."""
    dt_ref = PREREG["dt_ref_emergent"]; T_tot = PREREG["geom"]["T_total"]; steps = int(round(T_tot / dt_ref))
    integrators_test = {"strang2": step_strang2, "trotter1": step_trotter1}
    out_by_integrator = {}
    for int_name, step_fn in integrators_test.items():
        records = []
        for dtype_name, ftype in [("float32", np.float32), ("float64", np.float64)]:
            HX_d,HZ_d,HZZ_d,H_d = cast_hamiltonians_to_dtype(HX,HZ,HZZ,ftype)
            fac = build_factors(HX_d,HZ_d,HZZ_d,dt_ref,HX_d.dtype)
            psi = np.zeros(2**Q, HX_d.dtype); psi[0]=1.0; c_state = 0.0
            E0 = (psi.conj().astype(np.complex128) @ (H_d.astype(np.complex128) @ psi.astype(np.complex128))).real
            t_axis = []; drift = []; stride = max(1, steps//100); run_tag = f"F:{int_name}:{dtype_name}:seed={args.seed}"
            good = True
            for step_idx in range(steps):
                try:
                    psi, c_state = step_fn(psi, fac, step_idx, dt=dt_ref, run_tag=run_tag, c_state=c_state, tau=args.TAU_C, eta_scale=args.ETA_SCALE_C, coupling_lambda=args.COUPLING_LAMBDA_C,
                                           stochastic=args.stochastic, sigma=args.noise_sigma, beta=args.noise_beta, gseed=args.seed, renorm=False)
                except FloatingPointError: good=False; break
                if (step_idx % stride == 0) or (step_idx == steps-1):
                    Ek = (psi.conj().astype(np.complex128) @ (H_d.astype(np.complex128) @ psi.astype(np.complex128))).real; dE = abs(Ek - E0)
                    if not np.isfinite(dE): good=False; break
                    t_axis.append(step_idx*dt_ref); drift.append(dE)
            if not good or len(drift) < 2: continue
            gamma_tail = tail_stat_aligned(t_axis, drift, window=max(2.0, 0.1*T_tot), method="median", envelope=False)
            eps_val = machine_eps(dtype_name)
            if np.isfinite(eps_val) and np.isfinite(gamma_tail) and gamma_tail>0:
                records.append({"dtype": dtype_name, "eps": eps_val, "gamma_tail": gamma_tail})
        if len(records) >= 2:
            eps_arr = np.array([r["eps"] for r in records]); gamma_arr = np.array([r["gamma_tail"] for r in records])
            log_eps = np.log10(eps_arr); log_gamma = np.log10(gamma_arr)
            kappa, b0, R2 = linear_fit(log_eps, log_gamma, f"kappa({int_name})")
        else: kappa, R2 = np.nan, np.nan
        out_by_integrator[int_name] = {"records": records, "kappa": kappa, "R2": R2}
    kappa_s = out_by_integrator["strang2"]["kappa"]; kappa_t = out_by_integrator["trotter1"]["kappa"]
    R2_s = out_by_integrator["strang2"]["R2"]; R2_t = out_by_integrator["trotter1"]["R2"]
    gate = PREREG["kappa_universality"]
    universality = bool((np.isfinite(kappa_s) and np.isfinite(kappa_t) and abs(kappa_s - kappa_t) <= gate["tol"] and (R2_s >= gate["R2_min"]) and (R2_t >= gate["R2_min"])))
    return universality, out_by_integrator


# =========================================================
# 9. AUDIT E (GRID / COSMO CHECK) (Unchanged from 50.0)
# =========================================================

def integrand_func(u, k_mpc, Hc_star, DeltaN, p, Xi0, sigma, logx_sq_clamp):
    k_mpc=float(k_mpc); Hc_star=max(float(Hc_star),1e-300); DeltaN=float(DeltaN); p=float(p); Xi0=float(Xi0); sigma=float(sigma); logx_sq_clamp=float(logx_sq_clamp)
    log_k_over_Hc = math.log(k_mpc) - math.log(Hc_star); log_x = log_k_over_Hc + (u + DeltaN/2.0); x = math.exp(log_x)
    exponent_term = min((log_x**2)/(2.0*sigma**2), logx_sq_clamp); W = math.exp(-exponent_term)
    Theta_p = Xi0 * math.exp(p*u); return x * Theta_p * W

def pv_delta_k(k_mpc, Hc_star, DeltaN, p, Xi0, sigma, logx_sq_clamp):
    umin = -DeltaN/2.0; umax = DeltaN/2.0
    val, err = quad(integrand_func, umin, umax, args=(k_mpc, Hc_star, DeltaN, p, Xi0, sigma, logx_sq_clamp), epsabs=1.49e-12, epsrel=1.49e-12, limit=200)
    return 2.0 * val

def normalize_xi0(PR):
    """Calibrate Xi0_dict for each DeltaN."""
    numerics = PR["numerics"]; pump_grid= PR["pump_grid"]; sigma = numerics["integrator"]["sigma"]; clamp = numerics["integrator"]["logx_sq_clamp"]
    k0_ref = numerics["k0_ref_mpc_inv"]; target_VI = PR["predictions"]["V_over_I_peak"]["mean"]
    Xi0_dict = {}; Hc_dict = {}
    for DeltaN in sorted(set(pump_grid["deltaN"])):
        Hc_star_val = k0_ref * math.exp(DeltaN/2.0)
        I_base = 0.5 * pv_delta_k(k0_ref, Hc_star_val, DeltaN, 0.0, 1.0, sigma=sigma, logx_sq_clamp=clamp)
        if abs(I_base) < 1e-60: Xi0_val = 1e-5
        else: delta_pv_target = np.arctanh(np.clip(target_VI, -0.9999, 0.9999)); Xi0_val = delta_pv_target / (2.0*I_base)
        Xi0_dict[float(DeltaN)] = float(Xi0_val); Hc_dict[float(DeltaN)] = float(Hc_star_val)
    PR["numerics"]["Xi0_dict"] = Xi0_dict; PR["numerics"]["Hc_star_dict"] = Hc_dict
    return PR

def solve_chiral_modes_onek(k_mpc, DeltaN, p, Xi_factor, PR):
    """Return P_L(k), P_R(k) ~ chiral tensor spectrum."""
    Xi0_dict=PR["numerics"]["Xi0_dict"]; Hc_dict=PR["numerics"]["Hc_star_dict"]; sigma=PR["numerics"]["integrator"]["sigma"]; clamp=PR["numerics"]["integrator"]["logx_sq_clamp"]
    n_t_mean = PR["predictions"]["n_t"]["mean"]; dN_key = float(DeltaN)
    if dN_key not in Xi0_dict or dN_key not in Hc_dict: raise KeyError("DeltaN not normalized")
    Xi0_local = Xi0_dict[dN_key] * Xi_factor; Hc_star = Hc_dict[dN_key]
    delta_pv = pv_delta_k(k_mpc, Hc_star, DeltaN, p, Xi0_local, sigma, clamp); v_over_i = math.tanh(max(-20.0, min(20.0, delta_pv)))
    A_s = 2.1e-9; r = 0.01; k_pivot = 0.05
    P_T = (r*A_s)*((k_mpc/max(k_pivot,1e-60))**(n_t_mean))
    P_L = 0.5 * P_T * (1.0 + v_over_i); P_R = 0.5 * P_T * (1.0 - v_over_i)
    return max(P_L,0.0), max(P_R,0.0)

def apply_transfer_function(k_grid, P_L_arr, P_R_arr, band, cosmo):
    """Map primordial PL,PR(k) -> today's Ω_GW(f), V/I(f), n_t_local(f)."""
    h = cosmo["h"]; f_axis = np.logspace(np.log10(band["f_min_Hz"]), np.log10(band["f_max_Hz"]), band["n_log"], dtype=float)
    k_from_f = (2.0*np.pi * f_axis) / (C_M_PER_S/MPC_M)
    log_k = np.log10(np.maximum(k_grid,1e-60)); log_PL = np.log10(np.maximum(P_L_arr,1e-60)); log_PR = np.log10(np.maximum(P_R_arr,1e-60))
    uniq_k, idx = np.unique(log_k, return_index=True); log_PL = log_PL[idx]; log_PR = log_PR[idx]
    interp_PL = interp1d(uniq_k, log_PL, kind="linear", fill_value="extrapolate"); interp_PR = interp1d(uniq_k, log_PR, kind="linear", fill_value="extrapolate")
    log_PL_t = interp_PL(np.log10(np.maximum(k_from_f,1e-60))); log_PR_t = interp_PR(np.log10(np.maximum(k_from_f,1e-60)))
    P_L_t = 10**log_PL_t; P_R_t = 10**log_PR_t; P_T_t = P_L_t + P_R_t
    T2 = (0.803**2); Omega_GW = (9.2e-5/12.0) * P_T_t * T2
    V_over_I = (P_L_t - P_R_t) / np.maximum(P_T_t, 1e-60)
    log_Om = np.log(np.maximum(Omega_GW,1e-300)); log_f = np.log(np.maximum(f_axis,1e-300))
    n_t_local = np.gradient(log_Om, log_f, edge_order=2) if np.all(np.diff(log_f)>0) else np.full_like(f_axis, np.nan)
    return pd.DataFrame({"f_Hz": f_axis, "Omega_GW": Omega_GW, "V_over_I": V_over_I, "n_t_local": n_t_local})

def audit_grid(PR) -> Tuple[bool, Dict]:
    """Run cosmology grid checks if numeric gates passed."""
    try: PR = normalize_xi0(PR) # Ensure Xi0 is calculated
    except Exception as e: print(f"ERROR during Xi0 normalization: {e}"); return False, {"error": "Xi0 norm failed"}

    num = PR["numerics"]; pump = PR["pump_grid"]; bands= num["bands"]; cosmo= num["cosmo"]
    kmin = num["k_grid_forced"]["min_mpc_inv"]; kmax = num["k_grid_forced"]["max_mpc_inv"]; nk = num["k_grid_forced"]["n_log"]
    k_modes = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    summary_rows = []
    for DeltaN in pump["deltaN"]:
        for p in pump["p"]:
            for Xi_factor in pump["Xi_factors"]:
                P_L_list=[]; P_R_list=[]
                for k_mpc in k_modes:
                    try: PL,PRc = solve_chiral_modes_onek(k_mpc, DeltaN, p, Xi_factor, PR); P_L_list.append(PL); P_R_list.append(PRc)
                    except Exception as e: print(f"WARN: solve_chiral failed k={k_mpc}, dN={DeltaN}, p={p}, Xi={Xi_factor}: {e}"); P_L_list.append(np.nan); P_R_list.append(np.nan)
                P_L_arr = np.array(P_L_list); P_R_arr = np.array(P_R_list)
                if np.any(np.isnan(P_L_arr)): continue # Skip if solve failed
                for band_name, band in bands.items():
                    df_today = apply_transfer_function(k_modes, P_L_arr, P_R_arr, band, cosmo)
                    if df_today.empty or df_today["Omega_GW"].isnull().all(): continue
                    f_mid = np.sqrt(band["f_min_Hz"]*band["f_max_Hz"]); idx_mid = (df_today["f_Hz"]-f_mid).abs().idxmin(); row_mid = df_today.loc[idx_mid]
                    Vpk = row_mid["V_over_I"]; nt = row_mid["n_t_local"]; Om_max = df_today["Omega_GW"].max(skipna=True)
                    ok_primary = (not np.isnan(Vpk)) and (abs(Vpk - PR["predictions"]["V_over_I_peak"]["mean"]) <= 2.0*PR["predictions"]["V_over_I_peak"]["sigma"]) and (Vpk > 0) and (Om_max <= PR["predictions"]["Omega_GW_max"])
                    ok_secondary = (not np.isnan(nt)) and (abs(nt - PR["predictions"]["n_t"]["mean"]) <= 2.0*PR["predictions"]["n_t"]["sigma"])
                    summary_rows.append({"DeltaN": DeltaN, "p": p, "Xi_factor": Xi_factor, "band": band_name, "V_over_I_peak": float(Vpk), "n_t_local": float(nt), "Omega_GW_max": float(Om_max), "ok_primary": bool(ok_primary), "ok_secondary": bool(ok_secondary), "f_mid": float(f_mid)})
    summary_df = pd.DataFrame(summary_rows)
    GridSanityPassed = False
    if not summary_df.empty:
        core = summary_df[(summary_df["Xi_factor"]==1.0) & (summary_df["p"]==0)]
        if not core.empty:
            vmax = core["V_over_I_peak"].max(); vmin = core["V_over_I_peak"].min(); vmean= core["V_over_I_peak"].mean()
            amp_drift_pct = 100.0*(vmax-vmin)/max(abs(vmean),1e-12) if abs(vmean)>1e-12 else 0.0
            fmax = core["f_mid"].max(); fmin = core["f_mid"].min(); loc_drift_dex = math.log10(max(fmax,1e-30)/max(fmin,1e-30)) if fmin>1e-30 else 0.0
            amp_ok = amp_drift_pct <= PR["pass_fail"]["robustness_amp_drift_pct_max"]
            loc_ok = loc_drift_dex <= PR["pass_fail"]["robustness_loc_drift_dex_max"]
            phys_ok = core["ok_primary"].all() and core["ok_secondary"].all()
            GridSanityPassed = bool(amp_ok and loc_ok and phys_ok)
    return GridSanityPassed, {"summary_df": summary_df, "core_passed": GridSanityPassed}


# =========================================================
# 10. MAIN (Phase 50.1 Execution)
# =========================================================

timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
print("="*60)
print(f"Phase 50.1 starting on {PLATFORM_ID}") # <-- Version updated
print(f"stochastic={args.stochastic}, sigma={args.noise_sigma}, beta={args.noise_beta}, seed={args.seed}")
print(f"kahan: enabled={args.enable_kahan_coupling}, tau={args.TAU_C}, eta={args.ETA_SCALE_C}, λ={args.COUPLING_LAMBDA_C}")
print("="*60)

# --- Build geometry / Hamiltonians for audits (Unchanged) ---
Q = PREREG["geom"]["Q_audit"]; P = matched_density_points(Q, base_spacing=1.0); V_mat = rydberg_V(P, C6=1.0, blockade_R=None)
HX128, HZ128, HZZ128, norms128 = build_blocks(Q, Omega=1.0, Delta=0.0, V_mat=V_mat, dtype=np.complex128); H128 = HX128 + HZ128 + HZZ128
HX64, HZ64, HZZ64, _ = build_blocks(Q, Omega=1.0, Delta=0.0, V_mat=V_mat, dtype=np.complex64); H64 = HX64 + HZ64 + HZZ64

# --- Audit A (Unchanged logic) ---
if args.do_A: A_budget = commutator_budget(HX128, HZ128, HZZ128); C_budget = A_budget["C_tot"]; print(f"[Audit A] ||C||_F = {C_budget:.3e}")
else: print("[Audit A] skipped."); C_budget = 1.0

# --- Audits B & C (Run with NEW noise parameters) ---
if args.do_BC:
    df_results, traces = run_integrator_tracks(HX128, HZ128, HZZ128, H128)
    scaling_summary = fit_scaling_and_alpha(df_results)
    IntegratorAccepted, integ_info = check_integrator_accept(scaling_summary, df_results)
    print("[Audit B/C] IntegratorAccepted =", IntegratorAccepted)
else:
    df_results = pd.DataFrame(); scaling_summary = {}; IntegratorAccepted = False; integ_info = {"E_unit": None}
PREREG["results"]["IntegratorAccepted"] = IntegratorAccepted; PREREG["results"]["k_E"] = integ_info.get("k_E"); PREREG["results"]["k_S"] = integ_info.get("k_S"); PREREG["results"]["alpha_fit"] = integ_info.get("alpha"); PREREG["results"]["E_unit"] = integ_info.get("E_unit")

# --- Audit D (Precision Gate - run with NEW noise parameters) ---
if args.do_D:
    PrecisionGate, prec_info = precision_gate((HX128,HZ128,HZZ128,H128), (HX64, HZ64, HZZ64, H64), Q)
    print("[Audit D] PrecisionGate =", PrecisionGate, "κ'=", prec_info["kappa_prime"], "R²=", prec_info["R2"])
else: PrecisionGate = False
PREREG["results"]["PrecisionGate"] = PrecisionGate

# --- Audit F (Precision sweep - run with NEW noise parameters) ---
if args.do_F:
    UniversalityPassed, kappa_data = run_precision_sweep(HX128, HZ128, HZZ128, Q)
    print("[Audit F] UniversalityPassed =", UniversalityPassed)
    for name, info in kappa_data.items(): PREREG["results"]["kappa"][name] = {"kappa": info["kappa"], "R2": info["R2"]}
else: UniversalityPassed = None
PREREG["results"]["UniversalityPassed"] = UniversalityPassed

# --- Audit E (Physics Grid - now depends on potentially NEW IntegratorAccepted state) ---
GridAllowed = (IntegratorAccepted and PrecisionGate) # Logic unchanged
if args.do_E and GridAllowed:
    GridSanityPassed, grid_info = audit_grid(PREREG)
    print("[Audit E] GridSanityPassed =", GridSanityPassed)
    # Optionally save grid summary if needed: grid_info['summary_df'].to_csv(...)
else:
    GridSanityPassed = False
    print("[Audit E] skipped or blocked: GridAllowed =", GridAllowed)
PREREG["results"]["GridSanityPassed"] = GridSanityPassed

# --- FINAL VERDICT (Based on NEW results) ---
FINAL_VERDICT = (IntegratorAccepted and PrecisionGate and GridSanityPassed) # Logic unchanged

print("="*60)
print("PHASE 50.1 VERDICT") # <-- Version updated
print("  IntegratorAccepted :", IntegratorAccepted)
print("  PrecisionGate      :", PrecisionGate)
print("  GridSanityPassed   :", GridSanityPassed)
print("  κ Universality     :", UniversalityPassed)
print("  E_unit (emergent)  :", PREREG["results"]["E_unit"])
print("  OVERALL            :", "PASS" if FINAL_VERDICT else "FAIL")
print("="*60)

# --- SAVE PREREG SNAPSHOT ---
PREREG_SNAPSHOT = {
    "config": {
        "stochastic": args.stochastic,
        "noise_sigma": args.noise_sigma, # Captures the 50.1 value
        "noise_beta": args.noise_beta,   # Captures the 50.1 value
        "seed": args.seed,
        "temporal_kahan": {"enabled": args.enable_kahan_coupling, "tau": args.TAU_C, "eta_scale": args.ETA_SCALE_C, "lambda": args.COUPLING_LAMBDA_C},
        "platform_id": PLATFORM_ID
    },
    "PREREG": PREREG # Contains results specific to this 50.1 run
}

fname = f"phase50_1_prereg_{timestamp}.json" # <-- Filename updated
with open(fname, "w") as f: json.dump(PREREG_SNAPSHOT, f, indent=2, default=str)
print("Saved prereg snapshot ->", fname)

# --- Optional: Save Audit Plots (if matplotlib is used and plots generated) ---
# Example: if 'plots' dictionary exists:
# for name, fig in plots.items():
#     fig_fname = f"phase50_1_plot_{name}_{timestamp}.png"
#     try:
#         fig.savefig(fig_fname, dpi=150, bbox_inches='tight')
#         print(f"Saved plot -> {fig_fname}")
#     except Exception as e:
#         print(f"Error saving plot {name}: {e}")
