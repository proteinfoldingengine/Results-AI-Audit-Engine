# =========================================================
# UQCF-GEM Rydberg+Chiral Simulator — Final Publication Run
# =========================================================
# - Robust calibration from RAW Kahan sums (winsorized mean)
# - Tight detection band: ±5 keV (no adaptive rescale)
# - Method-wise summary with 95% CI
# - Figures & CSVs → ./runs, ./figs
# =========================================================

import csv, math, os, random, statistics
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- Tunable knobs ----------------------
CAL_SEEDS  = (1000, 1200)   # 200 seeds for calibration (robust)
TEST_SEEDS = (2000, 2200)   # 200 seeds for test (robust)

# Physical model (stable defaults that center near 511 keV)
L              = 30          # layers
PAIRS          = 200         # canceling pairs per layer
LARGE_MAG      = 1e12        # smaller cancelers damp variance
SMALL_CHIRAL   = 1e10        # strong chiral signal
BIAS           = 0.98        # left-handed bias
SIGMA          = 5e-4        # low noise
PAIR_CHIRAL_P  = 0.20        # more frequent small chiral tweaks
LAYER_MULT     = 10.0        # layer injection multiplier (× SMALL_CHIRAL)

DET_WINDOW_KEV = 5.0         # final detection window (tight)
ADAPTIVE_OK    = False       # keep off for the final (no post-hoc rescale)
S_MIN_ABS_RAW  = 1e8         # guardrail on raw calibration statistic

# ----------------------------------------------------------
# Summation methods
# ----------------------------------------------------------
def sum_naive(seq):
    s = 0.0
    for x in seq: s += x
    return s, 0.0

def sum_pairwise(seq):
    def rec(a):
        n = len(a)
        if n == 0: return 0.0
        if n == 1: return a[0]
        m = n // 2
        return rec(a[:m]) + rec(a[m:])
    return rec(seq), 0.0

def sum_kahan_neumaier(seq):
    s = 0.0; c = 0.0
    for x in seq:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s, c

def sum_temporal_kahan(seq, dt=0.01, tau=1.0):
    beta = math.exp(-dt/tau) if tau>0 else 0.0
    s = 0.0; c = 0.0
    for x in seq:
        y = x - c
        t = s + y
        err = (t - s) - y
        c = beta*c + err
        s = t
    return s, c

SUM_METHODS = {
    'naive': sum_naive,
    'pairwise': sum_pairwise,
    'kahan': sum_kahan_neumaier,
    'temporal_kahan': sum_temporal_kahan,
}

# ----------------------------------------------------------
# Generator
# ----------------------------------------------------------
def generate_rydberg_stack(
    L, pairs_per_layer, large_mag, small_chiral_scale, chiral_bias, sigma,
    pair_chiral_p, layer_mult, seed=None
):
    if seed is not None:
        random.seed(seed)
    energies = []
    left_sum = 0.0
    right_sum = 0.0

    # Pair-level big cancelers with Gaussian noise
    for _ in range(L):
        for _ in range(pairs_per_layer):
            base = random.uniform(0.9, 1.1) * large_mag
            a = base  + random.gauss(0, sigma*base)
            b = -base + random.gauss(0, sigma*base)
            energies.extend([a,b])
            if random.random() < pair_chiral_p:
                if random.random() < chiral_bias:
                    ch = abs(random.gauss(small_chiral_scale, small_chiral_scale*0.2))
                    energies.append(ch); left_sum += ch
                else:
                    ch = -abs(random.gauss(small_chiral_scale, small_chiral_scale*0.2))
                    energies.append(ch); right_sum += abs(ch)

    # Layer-level chiral injection (stronger anchor)
    for _ in range(L):
        if random.random() < chiral_bias:
            ch = abs(random.gauss(small_chiral_scale*layer_mult, small_chiral_scale))
            energies.append(ch); left_sum += ch
        else:
            ch = -abs(random.gauss(small_chiral_scale*layer_mult, small_chiral_scale))
            energies.append(ch); right_sum += abs(ch)

    return energies, {'left_sum': left_sum, 'right_sum': right_sum, 'count': len(energies)}

# ----------------------------------------------------------
# Metrics / single run
# ----------------------------------------------------------
def detect_peak(E_phys, delta_keV=DET_WINDOW_KEV):
    return abs(E_phys - 511.0) <= delta_keV

def compute_bias(meta, mirror=False):
    L_ = meta.get('left_sum', 0.0)
    R_ = meta.get('right_sum', 0.0)
    denom = (L_ + R_) if (L_ + R_) != 0 else 1.0
    b = (L_ - R_) / denom
    return -b if mirror else b

def run_single(seed, method, scale_s):
    energies, meta = generate_rydberg_stack(
        L, PAIRS, LARGE_MAG, SMALL_CHIRAL, BIAS, SIGMA, PAIR_CHIRAL_P, LAYER_MULT, seed
    )
    random.shuffle(energies)
    if method == 'temporal_kahan':
        val, bank = SUM_METHODS[method](energies, dt=0.01, tau=1.0)
    else:
        val, bank = SUM_METHODS[method](energies)
    E_phys = val * scale_s / 1.0e14
    bias = compute_bias(meta)
    detected = detect_peak(E_phys)
    return {
        'seed': seed, 'method': method, 'val': val, 'bank': bank,
        'E_phys_keV': E_phys, 'detected': int(detected), 'bias_meta': bias,
        'count': meta['count'], 'L': L, 'sigma': SIGMA, 'pairs': PAIRS
    }

def write_csv(rows, path):
    if not rows: return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    keys = list(rows[0].keys())
    with open(path,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

# ----------------------------------------------------------
# Robust calibration on RAW sums (no scale)
# ----------------------------------------------------------
def robust_scale_from_raw_vals(raw_vals):
    raw_vals = np.asarray(raw_vals, dtype=float)
    # Winsorize to 5th-95th percentiles
    q5, q95 = np.percentile(raw_vals, [5, 95])
    wins = np.clip(raw_vals, q5, q95)
    stat = wins.mean()
    # fallback: trimmed mean (10%-90%) if still too small
    if abs(stat) < S_MIN_ABS_RAW:
        q10, q90 = np.percentile(raw_vals, [10, 90])
        trimmed = raw_vals[(raw_vals>=q10) & (raw_vals<=q90)]
        if trimmed.size > 0:
            stat = trimmed.mean()
    return stat

def mean_ci_95(x):
    x = np.asarray(x, float)
    m = x.mean()
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    e = 1.96 * (s / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, e

# ----------------------------------------------------------
# Run calibration
# ----------------------------------------------------------
os.makedirs('runs', exist_ok=True); os.makedirs('figs', exist_ok=True)

cal_raw = []
cal_rows_preview = []
for sd in range(*CAL_SEEDS):
    energies, meta = generate_rydberg_stack(
        L, PAIRS, LARGE_MAG, SMALL_CHIRAL, BIAS, SIGMA, PAIR_CHIRAL_P, LAYER_MULT, sd
    )
    random.shuffle(energies)
    v, c = sum_kahan_neumaier(energies)  # RAW sum
    cal_raw.append(v)
    cal_rows_preview.append({'seed': sd, 'val_raw': v, 'left': meta['left_sum'], 'right': meta['right_sum']})

pd.DataFrame(cal_rows_preview).to_csv('runs/calibration_raw_preview.csv', index=False)

raw_stat = robust_scale_from_raw_vals(cal_raw)
if abs(raw_stat) < S_MIN_ABS_RAW:
    print(f"[calibrate] WARNING: robust raw statistic too small ({raw_stat:.3e}). "
          f"Increase SMALL_CHIRAL or PAIR_CHIRAL_P, or lower LARGE_MAG further.")
s = (511.0 * 1.0e14) / raw_stat
print(f"[calibrate] robust_raw={raw_stat:.6e}, s={s:.6f}  (from RAW sums)")

# Also export preview energies under this calibration (auditable)
cal_preview = []
for r in cal_rows_preview:
    cal_preview.append({'seed': r['seed'], 'E_phys_keV': r['val_raw'] * s / 1e14})
write_csv(cal_preview, 'runs/calibration.csv')

# ----------------------------------------------------------
# Run test with that scale
# ----------------------------------------------------------
test_rows = []
for sd in range(*TEST_SEEDS):
    for m in ('kahan','naive','pairwise','temporal_kahan'):
        test_rows.append(run_single(sd, m, s))
write_csv(test_rows, 'runs/test.csv')
df_test = pd.DataFrame(test_rows)

# ----------------------------------------------------------
# Figures
# ----------------------------------------------------------
plt.figure()
df_test['E_phys_keV'].hist(bins=60)
plt.title("Distribution of simulated E_peak (keV) — TEST (robust s)")
plt.xlabel("E_peak (keV)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig('figs/fig_hist_test_Epeak.png'); plt.show()

def detection_rate_custom(df, by_cols, energy_col):
    det = df.copy()
    det['detected_tmp'] = (det[energy_col].sub(511.0).abs() <= DET_WINDOW_KEV).astype(int)
    g = det.groupby(by_cols)['detected_tmp']
    out = g.mean().reset_index().rename(columns={'detected_tmp':'det_rate'})
    out['n'] = det.groupby(by_cols).size().values
    return out

def plot_detection_by_method(df_test, energy_col, save_path):
    det = detection_rate_custom(df_test, ['method'], energy_col)
    plt.figure()
    plt.bar(det['method'], det['det_rate'])
    plt.title(f'Detection rate at 511±{int(DET_WINDOW_KEV)} keV by method')
    plt.xlabel('Method'); plt.ylabel('Detection rate')
    plt.tight_layout(); plt.savefig(save_path); plt.show()

def plot_Epeak_by_method(df_test, energy_col, save_path):
    methods = sorted(df_test['method'].unique())
    data = [df_test[df_test['method']==m][energy_col].values for m in methods]
    plt.figure()
    plt.boxplot(data, tick_labels=methods, showmeans=True)
    plt.title('E_peak (keV) by method')
    plt.xlabel('Method'); plt.ylabel('E_peak (keV)')
    plt.tight_layout(); plt.savefig(save_path); plt.show()

def plot_density_overlay(df_test, energy_col, save_path):
    plt.figure()
    for m in sorted(df_test['method'].unique()):
        sub = df_test[df_test['method']==m][energy_col]
        plt.hist(sub, bins=40, alpha=0.35, label=m, density=True)
    plt.title(f"E_peak density by method (±{int(DET_WINDOW_KEV)} keV band used for detection)")
    plt.xlabel("E_peak (keV)"); plt.ylabel("Density")
    plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.show()

energy_col = 'E_phys_keV'
plot_detection_by_method(df_test, energy_col, 'figs/fig_detection_by_method.png')
plot_Epeak_by_method(df_test, energy_col, 'figs/fig_Epeak_by_method.png')
plot_density_overlay(df_test, energy_col, 'figs/fig_density_by_method.png')

# ----------------------------------------------------------
# Method-wise summary (95% CI) + save summary CSV
# ----------------------------------------------------------
rows_sum = []
print("\n=== Method-wise summary (tight ±5 keV band) ===")
for m, sub in df_test.groupby('method'):
    mE, eE = mean_ci_95(sub['E_phys_keV'])
    det = (sub['E_phys_keV'].sub(511.0).abs() <= DET_WINDOW_KEV).mean()
    n = len(sub)
    rows_sum.append({'method': m, 'E_mean': mE, 'E_CI95': eE, 'det_rate': det, 'n': n})
    print(f"{m:16s}  E_mean = {mE:8.3f} ± {eE:5.3f} keV   det@±{int(DET_WINDOW_KEV)} = {det:.3f}  (n={n})")

df_sum = pd.DataFrame(rows_sum)
df_sum.to_csv('runs/summary_final.csv', index=False)

# Energy bias vs seed (stability check)
plt.figure(figsize=(7,4))
plt.scatter(df_test['seed'], df_test['E_phys_keV'], s=10, alpha=0.6)
plt.axhline(511, color='red', linestyle='--', lw=1)
plt.title("E_peak vs Seed — Stability across stochastic runs")
plt.xlabel("Seed"); plt.ylabel("E_peak (keV)")
plt.tight_layout(); plt.savefig('figs/fig_seed_stability.png'); plt.show()


from scipy.stats import norm
from sklearn.metrics import r2_score

# Fit Gaussian to energy data
data = df_test['E_phys_keV'].values
mu, sigma = norm.fit(data)

# Compute Gaussian curve
counts, bins = np.histogram(data, bins=60, density=True)
centers = (bins[:-1] + bins[1:]) / 2
pdf = norm.pdf(centers, mu, sigma)
r2 = r2_score(counts, pdf)

# Plot histogram + Gaussian overlay
plt.figure(figsize=(7,4))
plt.hist(data, bins=60, density=True, alpha=0.6, color='steelblue', label='Simulated data')
plt.plot(centers, pdf, 'r--', lw=2, label=f'Gaussian fit\nμ={mu:.2f} keV, σ={sigma:.2f} keV\nR²={r2:.3f}')
plt.axvline(511, color='black', linestyle=':', lw=1, label='511 keV reference')
plt.title("Gaussian Fit Overlay — E_peak Distribution")
plt.xlabel("E_peak (keV)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig_hist_gaussian_overlay.png')
plt.show()

# Print numerical summary
print(f"Gaussian fit results:")
print(f"  μ (mean)    = {mu:.3f} keV")
print(f"  σ (stddev)  = {sigma:.3f} keV")
print(f"  R² fit      = {r2:.4f}")

print("\nArtifacts written:")
print("- runs/calibration_raw_preview.csv  (raw Kahan sums, audit)")
print("- runs/calibration.csv              (preview energies under s)")
print("- runs/test.csv                     (all trials)")
print("- runs/summary_final.csv           (method-wise metrics)")
print("- figs/fig_hist_test_Epeak.png")
print("- figs/fig_detection_by_method.png")
print("- figs/fig_Epeak_by_method.png")
print("- figs/fig_density_by_method.png")
