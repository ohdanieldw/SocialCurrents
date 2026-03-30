#!/usr/bin/env python3
"""
Dyad-level interpersonal synchrony analysis toolkit.

Computes synchrony between two participants' multimodal timeseries using
time-domain, frequency-domain, nonlinear, and directional methods.
Includes permutation-based surrogate testing.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.utils import (
    load_features,
    bin_timeseries,
    filter_timevarying_columns,
    align_bins,
    compute_grouped_dimensions,
    run_reduction,
    write_csv_with_header,
)

# Optional dependencies
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

ALL_METHODS = [
    "pearson", "crosscorr", "concordance",
    "rqa", "dfa",
    "coherence", "wavelet",
    "granger", "transfer-entropy",
    "coupled-oscillator",
]

DIRECTIONAL_METHODS = {"crosscorr", "granger", "transfer-entropy", "coupled-oscillator"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Interpersonal synchrony analysis for dyadic interactions"
    )
    p.add_argument("--person-a", required=True, help="Path to Person A timeseries CSV")
    p.add_argument("--person-b", required=True, help="Path to Person B timeseries CSV")
    p.add_argument("-o", "--output-dir", default="./analysis_output",
                   help="Output directory (default: ./analysis_output)")
    p.add_argument("--methods", default="all",
                   help=f"Comma-separated methods or 'all' (default: all). "
                        f"Options: {', '.join(ALL_METHODS)}")
    p.add_argument("--reduce-features",
                   choices=["pca", "fa", "ica", "grouped", "every", "all"],
                   default="grouped", help="Feature reduction (default: grouped)")
    p.add_argument("--n-components", type=int, default=5, help="Components for PCA/FA/ICA (default: 5)")
    p.add_argument("--window-size", type=float, default=30, help="Window size in seconds (default: 30)")
    p.add_argument("--step-size", type=float, default=5, help="Window step in seconds (default: 5)")
    p.add_argument("--time-resolution", type=float, default=0.5, help="Bin size in seconds (default: 0.5)")
    p.add_argument("--permutations", type=int, default=1000, help="Surrogate permutations (default: 1000)")
    p.add_argument("--lag-range", default="-10:10",
                   help="Lag range in seconds as min:max (default: -10:10)")
    p.add_argument("--no-zscore", action="store_true", help="Skip z-scoring")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return p.parse_args(argv)


def _parse_lag_range(lag_str, bin_size):
    """Parse lag range string like '-10:10' into sample counts."""
    parts = lag_str.split(":")
    lag_min_s, lag_max_s = float(parts[0]), float(parts[1])
    return int(round(lag_min_s / bin_size)), int(round(lag_max_s / bin_size))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_pair(args):
    """Load, bin, filter, reduce, align both persons. Returns aligned data."""
    print("Loading Person A...")
    df_a = load_features(args.person_a)
    print("Loading Person B...")
    df_b = load_features(args.person_b)

    print("\nBinning...")
    bin_a = bin_timeseries(df_a, "time_seconds", args.time_resolution)
    bin_b = bin_timeseries(df_b, "time_seconds", args.time_resolution)

    # Align on common bins
    common = bin_a.index.intersection(bin_b.index)
    if len(common) == 0:
        sys.exit("Error: no overlapping time bins between Person A and B")
    bin_a = bin_a.loc[common]
    bin_b = bin_b.loc[common]
    print(f"  Aligned: {len(common)} common bins")

    print("\nFiltering Person A...")
    filt_a = filter_timevarying_columns(bin_a)
    print("Filtering Person B...")
    filt_b = filter_timevarying_columns(bin_b)

    # Use common columns only
    shared_cols = sorted(set(filt_a.columns) & set(filt_b.columns))
    if not shared_cols:
        sys.exit("Error: no shared feature columns between Person A and B")
    filt_a = filt_a[shared_cols]
    filt_b = filt_b[shared_cols]
    print(f"  Shared columns: {len(shared_cols)}")

    method = args.reduce_features if args.reduce_features != "all" else "grouped"

    if method in ("pca", "fa", "ica"):
        # Fit reduction on concatenated data for shared component space
        combined = pd.concat([filt_a, filt_b], axis=0)
        print(f"\nReducing features ({method.upper()}) on combined data...")
        comp_df, labels, _ = run_reduction(combined, method, args.n_components)
        n = len(filt_a)
        data_a = comp_df.iloc[:n].values
        data_b = comp_df.iloc[n:].values
        dim_names = list(comp_df.columns)
    elif method == "grouped":
        print("\nComputing grouped dimensions...")
        grp_a, _ = compute_grouped_dimensions(df_a, filt_a, args.time_resolution)
        grp_b, _ = compute_grouped_dimensions(df_b, filt_b, args.time_resolution)
        shared_dims = sorted(set(grp_a.keys()) & set(grp_b.keys()))
        data_a = np.column_stack([grp_a[d] for d in shared_dims])
        data_b = np.column_stack([grp_b[d] for d in shared_dims])
        dim_names = shared_dims
    elif method == "every":
        data_a = filt_a.values
        data_b = filt_b.values
        dim_names = list(filt_a.columns)
    else:
        data_a = filt_a.values
        data_b = filt_b.values
        dim_names = list(filt_a.columns)

    # Z-score
    if not args.no_zscore and method in ("grouped", "every"):
        for j in range(data_a.shape[1]):
            for arr in (data_a, data_b):
                s = np.nanstd(arr[:, j])
                if s > 0:
                    arr[:, j] = (arr[:, j] - np.nanmean(arr[:, j])) / s

    data_a = np.nan_to_num(data_a, nan=0.0)
    data_b = np.nan_to_num(data_b, nan=0.0)
    bin_times = common.values * args.time_resolution

    print(f"  Final: {data_a.shape[0]} bins x {data_a.shape[1]} dimensions")
    return data_a, data_b, dim_names, bin_times


# ---------------------------------------------------------------------------
# Windowing helper
# ---------------------------------------------------------------------------

def _windows(n_bins, window_bins, step_bins):
    """Yield (start, end) index pairs for sliding windows."""
    start = 0
    while start + window_bins <= n_bins:
        yield start, start + window_bins
        start += step_bins
    # Final partial window if any data remains
    if start < n_bins and start + window_bins > n_bins:
        yield start, n_bins


# ---------------------------------------------------------------------------
# TIME-DOMAIN METHODS
# ---------------------------------------------------------------------------

def _pearsonr_at_lag(a, b, lag):
    """Pearson r between a and b shifted by lag samples."""
    n = len(a)
    if lag >= 0:
        x, y = a[:n - lag] if lag > 0 else a, b[lag:] if lag > 0 else b
    else:
        x, y = a[-lag:], b[:n + lag]
    if len(x) < 3:
        return np.nan, np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() == 0 or y[mask].std() == 0:
        return np.nan, np.nan
    return stats.pearsonr(x[mask], y[mask])


def method_pearson(data_a, data_b, dim_names, bin_times, args):
    """Windowed Pearson correlation with lag sweep."""
    bs = args.time_resolution
    w_bins = int(args.window_size / bs)
    s_bins = int(args.step_size / bs)
    lag_min, lag_max = _parse_lag_range(args.lag_range, bs)

    rows = []
    for ws, we in _windows(len(data_a), w_bins, s_bins):
        t = bin_times[ws]
        for j, dim in enumerate(dim_names):
            a_w, b_w = data_a[ws:we, j], data_b[ws:we, j]
            best_r, best_lag = np.nan, 0
            for lag in range(lag_min, lag_max + 1):
                r, _ = _pearsonr_at_lag(a_w, b_w, lag)
                if np.isfinite(r) and (np.isnan(best_r) or abs(r) > abs(best_r)):
                    best_r, best_lag = r, lag
            rows.append({
                "window_time": t, "dimension": dim,
                "pearson_r": best_r, "pearson_peak_lag": best_lag * bs,
            })
    return pd.DataFrame(rows)


def method_crosscorr(data_a, data_b, dim_names, bin_times, args):
    """Windowed cross-correlation with peak lag."""
    bs = args.time_resolution
    w_bins = int(args.window_size / bs)
    s_bins = int(args.step_size / bs)
    lag_min, lag_max = _parse_lag_range(args.lag_range, bs)

    rows = []
    for ws, we in _windows(len(data_a), w_bins, s_bins):
        t = bin_times[ws]
        for j, dim in enumerate(dim_names):
            a_w, b_w = data_a[ws:we, j], data_b[ws:we, j]
            lags = np.arange(lag_min, lag_max + 1)
            best_r, best_lag = np.nan, 0
            for lag in lags:
                r, _ = _pearsonr_at_lag(a_w, b_w, lag)
                if np.isfinite(r) and (np.isnan(best_r) or abs(r) > abs(best_r)):
                    best_r, best_lag = r, lag
            leader = "A" if best_lag > 0 else ("B" if best_lag < 0 else "sync")
            rows.append({
                "window_time": t, "dimension": dim,
                "crosscorr_r": best_r, "crosscorr_peak_lag": best_lag * bs,
                "crosscorr_leader": leader,
            })
    return pd.DataFrame(rows)


def _lins_ccc(a, b):
    """Lin's concordance correlation coefficient."""
    if len(a) < 3:
        return np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    return (2 * r * a.std() * b.std()) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)


def method_concordance(data_a, data_b, dim_names, bin_times, args):
    """Windowed Lin's CCC with lag sweep."""
    bs = args.time_resolution
    w_bins = int(args.window_size / bs)
    s_bins = int(args.step_size / bs)
    lag_min, lag_max = _parse_lag_range(args.lag_range, bs)

    rows = []
    for ws, we in _windows(len(data_a), w_bins, s_bins):
        t = bin_times[ws]
        for j, dim in enumerate(dim_names):
            a_w, b_w = data_a[ws:we, j], data_b[ws:we, j]
            best_ccc, best_lag = np.nan, 0
            for lag in range(lag_min, lag_max + 1):
                n = len(a_w)
                if lag >= 0:
                    a_s, b_s = (a_w[:n-lag] if lag > 0 else a_w), (b_w[lag:] if lag > 0 else b_w)
                else:
                    a_s, b_s = a_w[-lag:], b_w[:n+lag]
                ccc = _lins_ccc(a_s, b_s)
                if np.isfinite(ccc) and (np.isnan(best_ccc) or abs(ccc) > abs(best_ccc)):
                    best_ccc, best_lag = ccc, lag
            rows.append({
                "window_time": t, "dimension": dim,
                "concordance_ccc": best_ccc, "concordance_peak_lag": best_lag * bs,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NONLINEAR METHODS
# ---------------------------------------------------------------------------

def _cross_recurrence_matrix(a, b, threshold_pct=10):
    """Build binary cross-recurrence matrix."""
    dist = np.abs(a[:, None] - b[None, :])
    threshold = np.percentile(dist, threshold_pct)
    return dist <= threshold


def _diagonal_lines(rec_matrix):
    """Extract diagonal line lengths from recurrence matrix."""
    n, m = rec_matrix.shape
    lines = []
    for offset in range(-n + 1, m):
        diag = np.diag(rec_matrix, k=offset)
        length = 0
        for val in diag:
            if val:
                length += 1
            elif length > 0:
                lines.append(length)
                length = 0
        if length > 0:
            lines.append(length)
    return lines


def _vertical_lines(rec_matrix):
    """Extract vertical line lengths."""
    lines = []
    for col in range(rec_matrix.shape[1]):
        length = 0
        for row in range(rec_matrix.shape[0]):
            if rec_matrix[row, col]:
                length += 1
            elif length > 0:
                lines.append(length)
                length = 0
        if length > 0:
            lines.append(length)
    return lines


def method_rqa(data_a, data_b, dim_names, bin_times, args):
    """Cross-Recurrence Quantification Analysis."""
    bs = args.time_resolution
    lag_min, lag_max = _parse_lag_range(args.lag_range, bs)
    n_bins = len(data_a)

    summary_rows = []
    recurrence_plot_data = None  # store first dimension for plotting

    for j, dim in enumerate(dim_names):
        a, b = data_a[:, j], data_b[:, j]
        rec = _cross_recurrence_matrix(a, b)

        if j == 0:
            recurrence_plot_data = rec

        total_points = rec.size
        rr = rec.sum() / total_points if total_points > 0 else 0

        diag_lines = _diagonal_lines(rec)
        det_lines = [l for l in diag_lines if l >= 2]
        det = sum(det_lines) / sum(diag_lines) if diag_lines else 0
        mean_l = np.mean(det_lines) if det_lines else 0
        max_l = max(det_lines) if det_lines else 0
        if det_lines:
            hist, _ = np.histogram(det_lines, bins=max(1, max(det_lines)))
            hist = hist[hist > 0]
            probs = hist / hist.sum()
            entr = -np.sum(probs * np.log(probs + 1e-12))
        else:
            entr = 0

        vert_lines = _vertical_lines(rec)
        lam_lines = [l for l in vert_lines if l >= 2]
        lam = sum(lam_lines) / sum(vert_lines) if vert_lines else 0
        tt = np.mean(lam_lines) if lam_lines else 0

        # Diagonal profile (RR as function of lag offset)
        diag_profile = {}
        for offset in range(lag_min, lag_max + 1):
            diag = np.diag(rec, k=offset)
            diag_profile[offset * bs] = diag.mean() if len(diag) > 0 else 0

        summary_rows.append({
            "dimension": dim, "rqa_RR": rr, "rqa_DET": det,
            "rqa_L": mean_l, "rqa_Lmax": max_l, "rqa_ENTR": entr,
            "rqa_LAM": lam, "rqa_TT": tt,
        })

    return pd.DataFrame(summary_rows), recurrence_plot_data


def method_dfa(data_a, data_b, dim_names, args):
    """Detrended cross-correlation analysis (DCCA)."""
    scales = np.unique(np.logspace(1, np.log10(len(data_a) // 4), 15).astype(int))
    scales = scales[scales >= 4]

    rows = []
    for j, dim in enumerate(dim_names):
        a, b = data_a[:, j], data_b[:, j]
        # Integrated profiles
        ya = np.cumsum(a - a.mean())
        yb = np.cumsum(b - b.mean())

        f_xy = []
        f_xx = []
        f_yy = []
        valid_scales = []

        for s in scales:
            n_windows = len(ya) // s
            if n_windows < 1:
                continue
            cov_xy, cov_xx, cov_yy = [], [], []
            for w in range(n_windows):
                seg_a = ya[w * s:(w + 1) * s]
                seg_b = yb[w * s:(w + 1) * s]
                t = np.arange(s)
                # Detrend (linear)
                pa = np.polyfit(t, seg_a, 1)
                pb = np.polyfit(t, seg_b, 1)
                da = seg_a - np.polyval(pa, t)
                db = seg_b - np.polyval(pb, t)
                cov_xy.append(np.mean(da * db))
                cov_xx.append(np.mean(da ** 2))
                cov_yy.append(np.mean(db ** 2))

            fxy = np.sqrt(abs(np.mean(cov_xy)))
            fxx = np.sqrt(abs(np.mean(cov_xx)))
            fyy = np.sqrt(abs(np.mean(cov_yy)))
            f_xy.append(fxy)
            f_xx.append(fxx)
            f_yy.append(fyy)
            valid_scales.append(s)

        if len(valid_scales) >= 3:
            log_s = np.log(valid_scales)
            log_fxy = np.log(np.array(f_xy) + 1e-12)
            slope, _, _, _, _ = stats.linregress(log_s, log_fxy)
            # rho_DCCA at each scale
            rho = [f_xy[i] ** 2 / (f_xx[i] * f_yy[i] + 1e-12) for i in range(len(valid_scales))]
            mean_rho = np.mean(rho)
        else:
            slope = np.nan
            mean_rho = np.nan

        rows.append({"dimension": dim, "dfa_exponent": slope, "dfa_mean_rho": mean_rho})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FREQUENCY-DOMAIN METHODS
# ---------------------------------------------------------------------------

def method_coherence(data_a, data_b, dim_names, args):
    """Spectral coherence with phase at peak frequency."""
    fs = 1.0 / args.time_resolution
    rows = []
    coherence_data = {}

    for j, dim in enumerate(dim_names):
        a, b = data_a[:, j], data_b[:, j]
        freqs, coh = signal.coherence(a, b, fs=fs, nperseg=min(256, len(a) // 2))
        f_csd, psd_ab = signal.csd(a, b, fs=fs, nperseg=min(256, len(a) // 2))

        peak_idx = np.argmax(coh[1:]) + 1  # skip DC
        peak_freq = freqs[peak_idx]
        peak_coh = coh[peak_idx]
        phase_at_peak = np.angle(psd_ab[peak_idx])
        leader = "A" if phase_at_peak > 0 else ("B" if phase_at_peak < 0 else "sync")

        coherence_data[dim] = (freqs, coh)

        rows.append({
            "dimension": dim, "coherence_peak": peak_coh,
            "coherence_peak_freq": peak_freq,
            "coherence_phase": phase_at_peak,
            "coherence_leader": leader,
        })

    return pd.DataFrame(rows), coherence_data


def method_wavelet(data_a, data_b, dim_names, bin_times, args):
    """Continuous wavelet transform coherence."""
    if not HAS_PYWT:
        print("  Warning: pywt not installed, skipping wavelet method")
        return None, None, None

    fs = 1.0 / args.time_resolution
    # Use first dimension for the heatmap
    a, b = data_a[:, 0], data_b[:, 0]

    widths = np.arange(2, min(64, len(a) // 4))
    freqs = pywt.scale2frequency('morl', widths) * fs

    cwtA, _ = pywt.cwt(a, widths, 'morl', sampling_period=args.time_resolution)
    cwtB, _ = pywt.cwt(b, widths, 'morl', sampling_period=args.time_resolution)

    # Cross-wavelet
    Wxy = cwtA * np.conj(cwtB)

    # Smoothed coherence (simple gaussian smooth)
    from scipy.ndimage import gaussian_filter
    smooth_s = 3
    power_a = gaussian_filter(np.abs(cwtA) ** 2, smooth_s)
    power_b = gaussian_filter(np.abs(cwtB) ** 2, smooth_s)
    cross_smooth = gaussian_filter(np.abs(Wxy), smooth_s)

    coherence = cross_smooth ** 2 / (power_a * power_b + 1e-12)
    phase_diff = np.angle(Wxy)

    # Phase dynamics: mean phase diff per time bin across frequencies
    phase_rows = []
    for t_idx in range(len(bin_times)):
        mean_phase = np.mean(phase_diff[:, t_idx])
        mean_coh = np.mean(coherence[:, t_idx])
        phase_rows.append({
            "time_seconds": bin_times[t_idx],
            "mean_phase_diff": mean_phase,
            "mean_coherence": mean_coh,
            "leader": "A" if mean_phase > 0 else "B",
        })

    return coherence, freqs, pd.DataFrame(phase_rows)


# ---------------------------------------------------------------------------
# DIRECTIONAL METHODS
# ---------------------------------------------------------------------------

def method_granger(data_a, data_b, dim_names, bin_times, args):
    """Granger causality (overall + windowed)."""
    if not HAS_STATSMODELS:
        print("  Warning: statsmodels not installed, skipping Granger")
        return None, None

    max_lag = max(1, int(5 / args.time_resolution))  # 5 second max lag

    summary_rows = []
    for j, dim in enumerate(dim_names):
        a, b = data_a[:, j], data_b[:, j]
        pair = np.column_stack([b, a])  # test if A Granger-causes B
        pair_rev = np.column_stack([a, b])  # test if B Granger-causes A

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_ab = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
                res_ba = grangercausalitytests(pair_rev, maxlag=max_lag, verbose=False)

            # Best lag by min p-value
            best_p_ab = min(res_ab[l][0]['ssr_ftest'][1] for l in res_ab)
            best_f_ab = max(res_ab[l][0]['ssr_ftest'][0] for l in res_ab)
            best_p_ba = min(res_ba[l][0]['ssr_ftest'][1] for l in res_ba)
            best_f_ba = max(res_ba[l][0]['ssr_ftest'][0] for l in res_ba)
        except Exception:
            best_p_ab = best_f_ab = best_p_ba = best_f_ba = np.nan

        leader = "A" if best_f_ab > best_f_ba else "B"
        summary_rows.append({
            "dimension": dim,
            "granger_F_AtoB": best_f_ab, "granger_p_AtoB": best_p_ab,
            "granger_F_BtoA": best_f_ba, "granger_p_BtoA": best_p_ba,
            "granger_leader": leader,
        })

    # Windowed Granger for transition dynamics
    bs = args.time_resolution
    w_bins = int(args.window_size / bs)
    s_bins = int(args.step_size / bs)

    dyn_rows = []
    for ws, we in _windows(len(data_a), w_bins, s_bins):
        t = bin_times[ws]
        for j, dim in enumerate(dim_names):
            a_w, b_w = data_a[ws:we, j], data_b[ws:we, j]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = grangercausalitytests(
                        np.column_stack([b_w, a_w]), maxlag=min(max_lag, len(a_w) // 3), verbose=False
                    )
                    res_r = grangercausalitytests(
                        np.column_stack([a_w, b_w]), maxlag=min(max_lag, len(a_w) // 3), verbose=False
                    )
                f_ab = max(res[l][0]['ssr_ftest'][0] for l in res)
                f_ba = max(res_r[l][0]['ssr_ftest'][0] for l in res_r)
            except Exception:
                f_ab = f_ba = np.nan

            dyn_rows.append({
                "window_time": t, "dimension": dim,
                "granger_F_AtoB": f_ab, "granger_F_BtoA": f_ba,
                "granger_leader": "A" if f_ab > f_ba else "B",
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(dyn_rows)


def method_transfer_entropy(data_a, data_b, dim_names, args):
    """Transfer entropy via binning-based estimation."""
    n_bins_disc = 6

    def _te(source, target, n_bins=6):
        """TE(source -> target) using histogram estimation."""
        # Discretize
        s = np.digitize(source, np.linspace(source.min() - 1e-10, source.max() + 1e-10, n_bins + 1)) - 1
        t = np.digitize(target, np.linspace(target.min() - 1e-10, target.max() + 1e-10, n_bins + 1)) - 1
        t_past = t[:-1]
        t_curr = t[1:]
        s_past = s[:-1]
        n = len(t_curr)

        # Joint probabilities via histograms
        # H(t_curr | t_past) - H(t_curr | t_past, s_past)
        # = H(t_curr, t_past) - H(t_past) - H(t_curr, t_past, s_past) + H(t_past, s_past)
        def _entropy(*arrays):
            combined = np.column_stack(arrays)
            _, counts = np.unique(combined, axis=0, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-12))

        h_tc_tp = _entropy(t_curr, t_past)
        h_tp = _entropy(t_past)
        h_tc_tp_sp = _entropy(t_curr, t_past, s_past)
        h_tp_sp = _entropy(t_past, s_past)

        return (h_tc_tp - h_tp) - (h_tc_tp_sp - h_tp_sp)

    rows = []
    for j, dim in enumerate(dim_names):
        a, b = data_a[:, j], data_b[:, j]
        te_ab = _te(a, b, n_bins_disc)
        te_ba = _te(b, a, n_bins_disc)
        net = te_ab - te_ba
        leader = "A" if net > 0 else ("B" if net < 0 else "sync")
        rows.append({
            "dimension": dim,
            "te_AtoB": te_ab, "te_BtoA": te_ba,
            "te_net": net, "te_leader": leader,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# COUPLED OSCILLATOR
# ---------------------------------------------------------------------------

def method_coupled_oscillator(data_a, data_b, dim_names, args):
    """Coupled linear differential equation model (Ferrer & Helm, 2013)."""
    rows = []
    for j, dim in enumerate(dim_names):
        a, b = data_a[:, j], data_b[:, j]
        # Δa = damping_a * a + coupling_ab * b + intercept_a
        da = np.diff(a)
        db = np.diff(b)
        a_t = a[:-1]
        b_t = b[:-1]

        # OLS for A: da = alpha_a * a_t + beta_ab * b_t + c_a
        X = np.column_stack([a_t, b_t, np.ones(len(a_t))])
        try:
            coef_a, _, _, _ = np.linalg.lstsq(X, da, rcond=None)
            damping_a, coupling_ab, intercept_a = coef_a
        except Exception:
            damping_a = coupling_ab = intercept_a = np.nan

        # OLS for B: db = alpha_b * b_t + beta_ba * a_t + c_b
        X_b = np.column_stack([b_t, a_t, np.ones(len(b_t))])
        try:
            coef_b, _, _, _ = np.linalg.lstsq(X_b, db, rcond=None)
            damping_b, coupling_ba, intercept_b = coef_b
        except Exception:
            damping_b = coupling_ba = intercept_b = np.nan

        rows.append({
            "dimension": dim,
            "damping_A": damping_a, "coupling_AtoB": coupling_ab,
            "damping_B": damping_b, "coupling_BtoA": coupling_ba,
            "coupling_leader": "A" if abs(coupling_ab) > abs(coupling_ba) else "B",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SURROGATE TESTING
# ---------------------------------------------------------------------------

def _block_shuffle(signal, block_size, rng):
    """Block-shuffle a 1-D signal to preserve autocorrelation."""
    n = len(signal)
    n_blocks = max(1, n // block_size)
    indices = np.arange(n_blocks)
    rng.shuffle(indices)
    shuffled = np.concatenate([signal[i * block_size:(i + 1) * block_size] for i in indices])
    # Handle remainder
    remainder = n - n_blocks * block_size
    if remainder > 0:
        shuffled = np.concatenate([shuffled, signal[-remainder:]])
    return shuffled[:n]


def surrogate_test(data_a, data_b, dim_names, observed_summary, methods_to_test,
                   args):
    """Run permutation-based surrogate testing."""
    rng = np.random.default_rng(args.seed)
    block_size = max(1, int(5.0 / args.time_resolution))  # ~5s blocks
    n_perm = args.permutations
    n_dims = len(dim_names)

    # Collect observed values per (method, dimension)
    obs = {}
    for _, row in observed_summary.iterrows():
        dim = row["dimension"]
        for col in observed_summary.columns:
            if col != "dimension" and np.isfinite(row.get(col, np.nan) if isinstance(row.get(col, np.nan), (int, float)) else np.nan):
                try:
                    obs[(col, dim)] = float(row[col])
                except (ValueError, TypeError):
                    pass

    # Count exceedances
    exceed = {k: 0 for k in obs}

    print(f"  Running {n_perm} surrogate permutations...")
    for perm in range(n_perm):
        # Block-shuffle person B
        data_b_shuf = np.column_stack([
            _block_shuffle(data_b[:, j], block_size, rng) for j in range(n_dims)
        ])

        # Recompute summary metrics for each method
        for method_name in methods_to_test:
            try:
                if method_name == "pearson":
                    # Overall mean pearson
                    for j, dim in enumerate(dim_names):
                        r, _ = stats.pearsonr(data_a[:, j], data_b_shuf[:, j])
                        key = ("pearson_r_overall", dim)
                        if key in exceed and abs(r) >= abs(obs.get(key, 0)):
                            exceed[key] += 1
                elif method_name == "concordance":
                    for j, dim in enumerate(dim_names):
                        ccc = _lins_ccc(data_a[:, j], data_b_shuf[:, j])
                        key = ("concordance_ccc_overall", dim)
                        if key in exceed and abs(ccc) >= abs(obs.get(key, 0)):
                            exceed[key] += 1
            except Exception:
                pass

        if (perm + 1) % 100 == 0:
            print(f"    Permutation {perm + 1}/{n_perm}")

    # Compute p-values
    p_surr = {k: (v + 1) / (n_perm + 1) for k, v in exceed.items()}
    return p_surr


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_synchrony_timeseries(ts_df, dim_names, output_path):
    """Plot windowed synchrony over time."""
    # Find columns that contain synchrony values
    sync_cols = [c for c in ts_df.columns if c.endswith("_r") or c.endswith("_ccc")]
    if not sync_cols:
        sync_cols = [c for c in ts_df.columns if "pearson" in c or "crosscorr" in c or "concordance" in c]

    if not sync_cols or "window_time" not in ts_df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(dim_names)))

    for j, dim in enumerate(dim_names):
        sub = ts_df[ts_df["dimension"] == dim]
        if sub.empty:
            continue
        for col in sync_cols:
            if col in sub.columns:
                ax.plot(sub["window_time"], sub[col], label=f"{dim} ({col})",
                        color=cmap[j % len(cmap)], alpha=0.8)
                break

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Synchrony", fontsize=11)
    ax.set_title("Windowed Synchrony Over Time", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_coherence_spectrum(coherence_data, output_path):
    """Plot spectral coherence per dimension."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(coherence_data)))

    for i, (dim, (freqs, coh)) in enumerate(coherence_data.items()):
        ax.plot(freqs, coh, label=dim, color=cmap[i % len(cmap)])

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Coherence", fontsize=11)
    ax.set_title("Spectral Coherence", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_wavelet_heatmap(coherence_matrix, freqs, bin_times, output_path):
    """Plot time-frequency coherence heatmap."""
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.pcolormesh(bin_times, freqs, coherence_matrix, cmap="magma", shading="auto")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title("Wavelet Coherence (time-frequency)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Coherence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_recurrence(rec_matrix, output_path):
    """Plot cross-recurrence matrix."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rec_matrix, cmap="Greys", origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Person B (time bins)", fontsize=11)
    ax.set_ylabel("Person A (time bins)", fontsize=11)
    ax.set_title("Cross-Recurrence Plot", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# SUMMARY REPORT
# ---------------------------------------------------------------------------

def write_summary_report(path, args, selected_methods, summary_df, leader_df):
    """Write human-readable summary report."""
    lines = [
        "=" * 60,
        "  SocialCurrents — Synchrony Analysis Report",
        "=" * 60,
        "",
        f"Person A: {args.person_a}",
        f"Person B: {args.person_b}",
        f"Methods:  {', '.join(selected_methods)}",
        f"Reduce:   {args.reduce_features}",
        f"Window:   {args.window_size}s (step {args.step_size}s)",
        f"Lag range: {args.lag_range}s",
        f"Permutations: {args.permutations}",
        "",
    ]

    if summary_df is not None and not summary_df.empty:
        lines.append("Summary:")
        lines.append("-" * 50)
        lines.append(summary_df.to_string(index=False, float_format="%.4f"))
        lines.append("")

    if leader_df is not None and not leader_df.empty:
        lines.append("Leadership / Directionality:")
        lines.append("-" * 50)
        lines.append(leader_df.to_string(index=False, float_format="%.4f"))
        lines.append("")

    Path(path).write_text("\n".join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and (out_dir / "synchrony_summary.csv").exists():
        print(f"  Already exists: {out_dir / 'synchrony_summary.csv'}")
        print("  Skipping (use --overwrite to replace)")
        sys.exit(0)

    # Parse methods
    if args.methods == "all":
        selected = list(ALL_METHODS)
    else:
        selected = [m.strip() for m in args.methods.split(",")]

    # Check dependencies
    if "granger" in selected and not HAS_STATSMODELS:
        print("  Warning: statsmodels not installed, skipping granger")
        selected.remove("granger")
    if "wavelet" in selected and not HAS_PYWT:
        print("  Warning: pywt not installed, skipping wavelet")
        selected.remove("wavelet")

    # Preprocess
    data_a, data_b, dim_names, bin_times = preprocess_pair(args)

    # Run methods
    all_ts_frames = []
    all_summary_frames = {}  # {method_name: [DataFrames...]}
    leader_rows = []
    coherence_data = None
    wavelet_coh = None
    wavelet_freqs = None
    phase_dynamics = None
    recurrence_plot = None
    granger_dynamics = None

    for method in selected:
        print(f"\n{'='*60}")
        print(f"  Running: {method}")
        print(f"{'='*60}")

        try:
            if method == "pearson":
                ts = method_pearson(data_a, data_b, dim_names, bin_times, args)
                all_ts_frames.append(ts)
                # Overall summary
                for j, dim in enumerate(dim_names):
                    r, p = stats.pearsonr(data_a[:, j], data_b[:, j])
                    all_summary_frames.setdefault("pearson", []).append(pd.DataFrame([{
                        "dimension": dim, "pearson_r_overall": r, "pearson_p": p,
                    }]))

            elif method == "crosscorr":
                ts = method_crosscorr(data_a, data_b, dim_names, bin_times, args)
                all_ts_frames.append(ts)
                # Summary-level leader: median lag direction across windows
                for dim in dim_names:
                    sub = ts[ts["dimension"] == dim].dropna(subset=["crosscorr_r"])
                    if sub.empty:
                        continue
                    median_lag = sub["crosscorr_peak_lag"].median()
                    mean_r = sub["crosscorr_r"].mean()
                    leader_rows.append({
                        "dimension": dim, "method": "crosscorr",
                        "leader": "A" if median_lag > 0 else ("B" if median_lag < 0 else "sync"),
                        "peak_lag": median_lag,
                        "effect_size": mean_r,
                    })

            elif method == "concordance":
                ts = method_concordance(data_a, data_b, dim_names, bin_times, args)
                all_ts_frames.append(ts)
                for j, dim in enumerate(dim_names):
                    ccc = _lins_ccc(data_a[:, j], data_b[:, j])
                    all_summary_frames.setdefault("concordance", []).append(pd.DataFrame([{
                        "dimension": dim, "concordance_ccc_overall": ccc,
                    }]))

            elif method == "rqa":
                rqa_summary, recurrence_plot = method_rqa(data_a, data_b, dim_names, bin_times, args)
                all_summary_frames.setdefault("rqa", []).append(rqa_summary)

            elif method == "dfa":
                dfa_summary = method_dfa(data_a, data_b, dim_names, args)
                all_summary_frames.setdefault("dfa", []).append(dfa_summary)

            elif method == "coherence":
                coh_summary, coherence_data = method_coherence(data_a, data_b, dim_names, args)
                all_summary_frames.setdefault("coherence", []).append(coh_summary)
                for _, row in coh_summary.iterrows():
                    leader_rows.append({
                        "dimension": row["dimension"], "method": "coherence",
                        "leader": row["coherence_leader"],
                        "peak_lag": row["coherence_phase"],
                        "effect_size": row["coherence_peak"],
                    })

            elif method == "wavelet":
                wavelet_coh, wavelet_freqs, phase_dynamics = method_wavelet(
                    data_a, data_b, dim_names, bin_times, args
                )

            elif method == "granger":
                granger_summary, granger_dynamics = method_granger(
                    data_a, data_b, dim_names, bin_times, args
                )
                if granger_summary is not None:
                    all_summary_frames.setdefault("granger", []).append(granger_summary)
                    for _, row in granger_summary.iterrows():
                        leader_rows.append({
                            "dimension": row["dimension"], "method": "granger",
                            "leader": row["granger_leader"],
                            "peak_lag": np.nan,
                            "effect_size": max(row["granger_F_AtoB"], row["granger_F_BtoA"]),
                        })

            elif method == "transfer-entropy":
                te_summary = method_transfer_entropy(data_a, data_b, dim_names, args)
                all_summary_frames.setdefault("transfer_entropy", []).append(te_summary)
                for _, row in te_summary.iterrows():
                    leader_rows.append({
                        "dimension": row["dimension"], "method": "transfer-entropy",
                        "leader": row["te_leader"],
                        "peak_lag": np.nan,
                        "effect_size": abs(row["te_net"]),
                    })

            elif method == "coupled-oscillator":
                co_summary = method_coupled_oscillator(data_a, data_b, dim_names, args)
                all_summary_frames.setdefault("coupled_oscillator", []).append(co_summary)
                for _, row in co_summary.iterrows():
                    leader_rows.append({
                        "dimension": row["dimension"], "method": "coupled-oscillator",
                        "leader": row["coupling_leader"],
                        "peak_lag": np.nan,
                        "effect_size": max(abs(row["coupling_AtoB"]), abs(row["coupling_BtoA"])),
                    })

            print(f"  {method}: done")

        except Exception as e:
            print(f"  {method}: FAILED ({e})")

    # Merge outputs
    print("\n\nSaving outputs...")

    # Synchrony timeseries
    if all_ts_frames:
        ts_merged = all_ts_frames[0]
        for df in all_ts_frames[1:]:
            if "window_time" in df.columns and "dimension" in df.columns:
                ts_merged = ts_merged.merge(df, on=["window_time", "dimension"], how="outer")
        ts_merged.to_csv(out_dir / "synchrony_timeseries.csv", index=False, float_format="%.6f")
        print(f"  Saved: {out_dir / 'synchrony_timeseries.csv'} ({len(ts_merged)} rows)")

    # Summary — prefix columns per method to avoid duplicates, then merge on dimension
    if all_summary_frames:
        summary_frames = []
        for method_name, frames in all_summary_frames.items():
            method_df = pd.concat(frames, ignore_index=True)
            # Group by dimension in case multiple rows per dimension from the same method
            if "dimension" in method_df.columns:
                method_df = method_df.groupby("dimension", as_index=False).first()
            # Prefix non-dimension columns with method name
            method_df.columns = [
                c if c == "dimension" else f"{method_name}_{c}"
                for c in method_df.columns
            ]
            summary_frames.append(method_df)

        summary = summary_frames[0]
        for df in summary_frames[1:]:
            summary = summary.merge(df, on="dimension", how="outer")
        summary.to_csv(out_dir / "synchrony_summary.csv", index=False, float_format="%.6f")
        print(f"  Saved: {out_dir / 'synchrony_summary.csv'}")
    else:
        summary = pd.DataFrame()

    # Leader/follower
    leader_df = pd.DataFrame(leader_rows) if leader_rows else pd.DataFrame()
    if not leader_df.empty:
        leader_df.to_csv(out_dir / "leader_follower.csv", index=False, float_format="%.6f")
        print(f"  Saved: {out_dir / 'leader_follower.csv'}")

    # Transition dynamics
    if granger_dynamics is not None and not granger_dynamics.empty:
        granger_dynamics.to_csv(out_dir / "transition_dynamics.csv", index=False, float_format="%.6f")
        print(f"  Saved: {out_dir / 'transition_dynamics.csv'}")

    # Phase dynamics (wavelet)
    if phase_dynamics is not None:
        phase_dynamics.to_csv(out_dir / "phase_dynamics.csv", index=False, float_format="%.6f")
        print(f"  Saved: {out_dir / 'phase_dynamics.csv'}")

    # Plots
    print("\nGenerating plots...")
    if all_ts_frames:
        ts_all = pd.concat(all_ts_frames, ignore_index=True)
        plot_synchrony_timeseries(ts_all, dim_names, out_dir / "synchrony_timeseries_plot.png")

    if coherence_data:
        plot_coherence_spectrum(coherence_data, out_dir / "coherence_spectrum.png")

    if wavelet_coh is not None and wavelet_freqs is not None:
        plot_wavelet_heatmap(wavelet_coh, wavelet_freqs, bin_times, out_dir / "wavelet_heatmap.png")

    if recurrence_plot is not None:
        plot_recurrence(recurrence_plot, out_dir / "rqa_recurrence_plot.png")

    # Summary report
    write_summary_report(out_dir / "summary_report.txt", args, selected, summary, leader_df)

    print(f"\nDone. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
