"""
Shared utilities for cross-correlation analysis.

Provides loading, binning, filtering, dimensionality reduction,
lagged cross-correlation, FDR correction, and plotting functions.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# Loading & alignment
# ---------------------------------------------------------------------------

_TIME_COL_CANDIDATES = ["time_seconds", "time", "window_time", "time_second"]
_TIME_COL_MS = ["Timestamp", "VideoTime"]  # millisecond columns


def _detect_time_column(df):
    """Find and normalise the time column to ``time_seconds``.

    Checks columns in priority order.  Millisecond columns (Timestamp,
    VideoTime) are converted to seconds when values suggest ms (max > 1000).
    Returns the (possibly mutated) DataFrame.
    """
    for col in _TIME_COL_CANDIDATES:
        if col in df.columns:
            if col != "time_seconds":
                df = df.rename(columns={col: "time_seconds"})
            return df

    for col in _TIME_COL_MS:
        if col in df.columns:
            vals = df[col]
            if vals.max() > 1000:
                df["time_seconds"] = vals / 1000.0
            else:
                df["time_seconds"] = vals
            return df

    accepted = _TIME_COL_CANDIDATES + _TIME_COL_MS
    sys.exit(f"Error: CSV must contain a time column. "
             f"Accepted names: {', '.join(accepted)}")


_ORIENTATION_FLIP_PREFIXES = ("GMP_world_x_", "GMP_land_x_", "pf_facerectx")

_SUBJECTS_CACHE = {}


def normalize_orientation(df, facing_direction):
    """Flip x-axis pose columns so all subjects face the canonical direction.

    Right-facing is canonical (no change).  For left-facing participants,
    x-axis columns are negated so spatial coordinates are comparable.
    """
    if facing_direction == "right":
        print(f"  Normalized orientation: no change (facing right, canonical)")
        return df
    if facing_direction != "left":
        print(f"  Warning: unknown facing_direction '{facing_direction}', skipping normalization")
        return df

    flip_cols = [c for c in df.columns
                 if any(c.startswith(p) or c == p for p in _ORIENTATION_FLIP_PREFIXES)]
    for c in flip_cols:
        df[c] = -df[c]
    print(f"  Normalized orientation: flipped {len(flip_cols)} x-axis columns (facing left)")
    return df


def load_subjects_csv(path):
    """Load a subjects CSV and return {subject_id: facing_direction} dict."""
    path = str(path)
    if path in _SUBJECTS_CACHE:
        return _SUBJECTS_CACHE[path]
    df = pd.read_csv(path)
    if "subject" not in df.columns or "facing_direction" not in df.columns:
        sys.exit("Error: subjects CSV must contain 'subject' and 'facing_direction' columns")
    mapping = dict(zip(df["subject"].astype(str), df["facing_direction"].astype(str)))
    _SUBJECTS_CACHE[path] = mapping
    return mapping


def extract_subject_id(features_path):
    """Extract subject ID from a features file path (dyadNNN_subNNN pattern)."""
    import re
    path_str = str(features_path)
    m = re.search(r"(sub\d+)", Path(path_str).stem)
    if m:
        return m.group(1)
    for part in Path(path_str).parts:
        m = re.search(r"^(sub\d+)$", part)
        if m:
            return m.group(1)
    return None


def resolve_facing(features_path, subjects_path):
    """Look up facing_direction for a features file from a subjects CSV."""
    if subjects_path is None:
        return None
    mapping = load_subjects_csv(subjects_path)
    sub_id = extract_subject_id(features_path)
    if sub_id is None:
        print(f"  Warning: could not extract subject ID from {features_path}")
        return None
    facing = mapping.get(sub_id)
    if facing is None:
        print(f"  Warning: subject {sub_id} not found in subjects CSV")
        return None
    print(f"  Subject {sub_id}: facing {facing}")
    return facing


def load_features(path, facing_direction=None):
    """Load pipeline timeseries features CSV.

    If *facing_direction* is provided, x-axis pose columns are flipped
    so that all subjects face the canonical (right) direction.
    """
    df = pd.read_csv(path)
    df = _detect_time_column(df)
    print(f"  Loaded features: {df.shape[0]} rows x {df.shape[1]} columns")
    if facing_direction is not None:
        df = normalize_orientation(df, facing_direction)
    return df


def load_target(path, value_col=None):
    """Load a target CSV, auto-detecting the time column.

    Accepts either ``time_seconds`` (seconds) or ``VideoTime`` (ms).
    For single-channel targets, *value_col* selects the rating column
    and renames it to ``Value`` internally.  For multi-channel targets
    (no *value_col*), all numeric columns are kept.
    """
    df = pd.read_csv(path)

    if "time_seconds" in df.columns:
        pass
    elif "VideoTime" in df.columns:
        df["time_seconds"] = df["VideoTime"] / 1000.0
    else:
        sys.exit("Error: target CSV must contain 'time_seconds' or 'VideoTime' column")

    if value_col is not None:
        if value_col not in df.columns:
            sys.exit(f"Error: target CSV must contain a '{value_col}' column")
        if value_col != "Value":
            df["Value"] = df[value_col]

    print(f"  Loaded target: {df.shape[0]} samples, "
          f"range {df['time_seconds'].min():.1f}s – {df['time_seconds'].max():.1f}s")
    return df


def bin_timeseries(df, time_col, bin_size):
    """Bin a DataFrame into equal time bins, aggregating by mean."""
    bin_idx = np.floor(df[time_col] / bin_size).astype(int)
    grouped = df.groupby(bin_idx, sort=True).mean(numeric_only=True)
    grouped.index.name = "bin_idx"
    grouped["bin_time"] = grouped.index * bin_size
    return grouped


def filter_timevarying_columns(df, nan_threshold=0.5):
    """Keep only numeric, time-varying columns (drop constants, high-NaN)."""
    num_df = df.select_dtypes(include=[np.number])
    drop_cols = {"frame_idx", "time_seconds", "bin_time"}
    num_df = num_df.drop(columns=[c for c in drop_cols if c in num_df.columns])

    variances = num_df.var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        print(f"  Dropped {len(zero_var)} constant columns")
    num_df = num_df.drop(columns=zero_var)

    nan_frac = num_df.isna().mean()
    high_nan = nan_frac[nan_frac > nan_threshold].index.tolist()
    if high_nan:
        print(f"  Dropped {len(high_nan)} columns with >{nan_threshold*100:.0f}% NaN")
    num_df = num_df.drop(columns=high_nan)

    num_df = num_df.ffill().bfill()
    print(f"  Retained {num_df.shape[1]} time-varying feature columns")
    return num_df


def align_bins(binned_feat, binned_target, bin_size):
    """Align two binned DataFrames on common bin indices."""
    common = binned_feat.index.intersection(binned_target.index)
    if len(common) == 0:
        sys.exit("Error: no overlapping time bins between features and target")
    print(f"  Aligned: {len(common)} common bins "
          f"({common.min() * bin_size:.1f}s – {common.max() * bin_size:.1f}s)")
    return binned_feat.loc[common], binned_target.loc[common], common


# ---------------------------------------------------------------------------
# Conceptual (grouped) dimensions
# ---------------------------------------------------------------------------

def compute_movement_energy(features_df):
    """Compute per-frame movement energy from GMP world coordinates.

    Must be called BEFORE binning. Returns (Series|None, status_str).
    """
    x_cols = [f"GMP_world_x_{i}" for i in range(1, 34)]
    y_cols = [f"GMP_world_y_{i}" for i in range(1, 34)]
    z_cols = [f"GMP_world_z_{i}" for i in range(1, 34)]

    avail_x = [c for c in x_cols if c in features_df.columns]
    avail_y = [c for c in y_cols if c in features_df.columns]
    avail_z = [c for c in z_cols if c in features_df.columns]
    n = min(len(avail_x), len(avail_y), len(avail_z))

    if n == 0:
        print("  Warning: No GMP_world columns found, skipping movement_energy")
        return None, "no_columns"

    all_world = (avail_x + avail_y + avail_z)[:n * 3]
    if (features_df[all_world].abs() == 0).all().all():
        print("  Warning: GMP_world coordinates are all zero — "
              "pose detection produced no landmarks for this subject")
        return None, "all_zero"

    coords = np.stack([
        features_df[avail_x[:n]].values,
        features_df[avail_y[:n]].values,
        features_df[avail_z[:n]].values,
    ], axis=-1)
    diff = np.diff(coords, axis=0)
    magnitudes = np.linalg.norm(diff, axis=2)
    avg_mag = np.nanmean(magnitudes, axis=1)
    energy = np.concatenate([[0.0], avg_mag])
    series = pd.Series(energy, index=features_df.index, name="movement_energy")
    print(f"  Computed movement_energy from {n} landmarks")
    return series, "ok"


def _zscore_and_mean(df, cols, name):
    """Z-score selected columns and return their row-wise mean."""
    available = [c for c in cols if c in df.columns]
    if not available:
        print(f"  Warning: No columns found for {name}, skipping")
        return None
    sub = df[available].copy()
    for c in sub.columns:
        s = sub[c].std()
        sub[c] = (sub[c] - sub[c].mean()) / s if s > 0 else 0.0
    result = sub.mean(axis=1)
    result.name = name
    print(f"  Computed {name} from {len(available)} columns")
    return result


def compute_vocal_energy(binned_df):
    cols = [c for c in binned_df.columns
            if c.startswith("oc_audvol") or c.startswith("oc_audpit")]
    return _zscore_and_mean(binned_df, cols, "vocal_energy")


def compute_spectral_complexity(binned_df):
    cols = [c for c in binned_df.columns
            if c.startswith("lbrs_") and not c.endswith("_singlevalue") and c != "lbrs_tempo"]
    return _zscore_and_mean(binned_df, cols, "spectral_complexity")


def compute_opensmile_summary(binned_df):
    cols = [c for c in binned_df.columns if c.startswith("osm_")]
    return _zscore_and_mean(binned_df, cols, "opensmile_summary")


def compute_grouped_dimensions(raw_features_df, binned_filtered_df, bin_size):
    """Compute movement + 3 audio/spectral grouped dimensions.

    Returns (dict of name->array, movement_status).
    """
    movement, movement_status = compute_movement_energy(raw_features_df)
    dims = {}

    if movement is not None:
        tmp = raw_features_df[["time_seconds"]].copy()
        tmp["_me"] = movement.values
        binned_me = bin_timeseries(tmp, "time_seconds", bin_size)
        common = binned_me.index.intersection(binned_filtered_df.index)
        if len(common) > 0:
            dims["movement_energy"] = binned_me.loc[common, "_me"].values

    for fn in (compute_vocal_energy, compute_spectral_complexity, compute_opensmile_summary):
        s = fn(binned_filtered_df)
        if s is not None:
            dims[s.name] = s.values

    return dims, movement_status


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _truncate(name, maxlen=12):
    return name[:maxlen] if len(name) > maxlen else name


def run_reduction(df, method, n_components):
    """Run PCA, FA, or ICA on filtered features.

    Returns (components_df, labels_dict, loadings_df).
    """
    from sklearn.decomposition import PCA, FactorAnalysis, FastICA

    n_components = min(n_components, df.shape[0], df.shape[1])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    method_upper = method.upper()
    if method == "pca":
        model = PCA(n_components=n_components)
    elif method == "fa":
        model = FactorAnalysis(n_components=n_components, random_state=42)
    elif method == "ica":
        model = FastICA(n_components=n_components, random_state=42, max_iter=500)
    else:
        sys.exit(f"Unknown reduction method: {method}")

    components = model.fit_transform(scaled)
    feature_names = list(df.columns)

    if method == "pca":
        loadings_matrix = model.components_
    elif method == "fa":
        loadings_matrix = model.components_
    elif method == "ica":
        loadings_matrix = model.mixing_.T if model.mixing_ is not None else np.zeros((n_components, len(feature_names)))

    if method == "pca":
        var_expl = model.explained_variance_ratio_
        print(f"  {method_upper}: {n_components} components, "
              f"cumulative variance = {var_expl.sum():.1%}")
    else:
        var_expl = None
        print(f"  {method_upper}: {n_components} components")

    labels = {}
    loading_rows = []
    for i in range(n_components):
        loadings = loadings_matrix[i]
        top3_idx = np.argsort(np.abs(loadings))[::-1][:3]
        top3_short = [_truncate(feature_names[j]) for j in top3_idx]
        comp_name = f"{method_upper}{i+1}"
        labels[comp_name] = f"{comp_name} ({', '.join(top3_short)})"

        var_str = f"{var_expl[i]:.1%}  " if var_expl is not None else ""
        top3_detail = ", ".join(
            f"{feature_names[j]} ({loadings[j]:+.3f})" for j in top3_idx
        )
        print(f"    {comp_name}: {var_str}top: {top3_detail}")

        for j, fname in enumerate(feature_names):
            loading_rows.append({
                "component": comp_name,
                "feature": fname,
                "loading": float(loadings[j]),
            })

    col_names = [f"{method_upper}{i+1}" for i in range(n_components)]
    components_df = pd.DataFrame(components, index=df.index, columns=col_names)
    loadings_df = pd.DataFrame(loading_rows)

    return components_df, labels, loadings_df


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def lagged_crosscorrelation(x, y, min_lag_samples, max_lag_samples):
    """Compute Pearson r between x and y at integer lags.

    Positive lag: x precedes y by ``lag`` steps.
    """
    lags = np.arange(min_lag_samples, max_lag_samples + 1)
    rs = np.full_like(lags, np.nan, dtype=float)
    ps = np.full_like(lags, np.nan, dtype=float)

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)

    for i, lag in enumerate(lags):
        if lag >= 0:
            x_seg = x_arr[:n - lag] if lag > 0 else x_arr
            y_seg = y_arr[lag:] if lag > 0 else y_arr
        else:
            x_seg = x_arr[-lag:]
            y_seg = y_arr[:n + lag]

        mask = np.isfinite(x_seg) & np.isfinite(y_seg)
        if mask.sum() < 3:
            continue
        x_valid, y_valid = x_seg[mask], y_seg[mask]
        if x_valid.std() == 0 or y_valid.std() == 0:
            continue
        r, p = stats.pearsonr(x_valid, y_valid)
        rs[i] = r
        ps[i] = p

    return lags, rs, ps


def apply_fdr(results_df):
    """Add p_fdr column via Benjamini-Hochberg correction."""
    if HAS_STATSMODELS:
        valid = results_df["p_value"].notna()
        if valid.any():
            _, fdr_vals, _, _ = multipletests(
                results_df.loc[valid, "p_value"], method="fdr_bh"
            )
            results_df["p_fdr"] = np.nan
            results_df.loc[valid, "p_fdr"] = fdr_vals
        else:
            results_df["p_fdr"] = np.nan
    else:
        print("  Warning: statsmodels not installed, skipping FDR correction")
        results_df["p_fdr"] = np.nan
    return results_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

GROUPED_DIMS = ["movement_energy", "vocal_energy", "spectral_complexity", "opensmile_summary"]
GROUPED_COLORS = {
    "movement_energy": "#e63946",
    "vocal_energy": "#457b9d",
    "spectral_complexity": "#2a9d8f",
    "opensmile_summary": "#e9c46a",
}


def plot_single_crosscorr(results_df, output_path, label="Trustworthiness",
                          rater=None, target_id=None, dim_labels=None):
    """Plot cross-correlation curves for single-target mode."""
    fig, ax = plt.subplots(figsize=(12, 7))

    n_reduced = sum(1 for d in results_df["dimension"].unique() if d not in GROUPED_DIMS)
    cmap = plt.cm.cool(np.linspace(0.0, 1.0, max(n_reduced, 1)))

    reduced_idx = 0
    for dim_name, group in results_df.groupby("dimension", sort=False):
        lag = group["lag_seconds"].values
        corr = group["correlation"].values

        if dim_name in GROUPED_DIMS:
            ax.plot(lag, corr, label=dim_name, linewidth=2.5,
                    color=GROUPED_COLORS.get(dim_name, "black"), zorder=10)
        else:
            legend_label = (dim_labels or {}).get(dim_name, dim_name)
            ax.plot(lag, corr, label=legend_label, linewidth=1.2, alpha=0.8,
                    color=cmap[min(reduced_idx, len(cmap) - 1)], zorder=5)
            reduced_idx += 1

    ax.axhline(0, color="black", linewidth=2, zorder=0)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (seconds) \u2014 negative: target precedes feature / "
                  "positive: feature precedes target", fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=12)
    title = f"Lagged Cross-Correlation with {label}"
    if rater and target_id:
        title += f"\n{rater} rates {target_id}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_path}")


def plot_multi_heatmap(results_df, feat_dims, target_dims, output_path,
                       label="Trustworthiness"):
    """Plot heatmap of peak-lag correlations (feature dim x target component)."""
    peak_r = np.full((len(feat_dims), len(target_dims)), np.nan)
    peak_lag = np.full((len(feat_dims), len(target_dims)), np.nan)

    for i, fd in enumerate(feat_dims):
        for j, td in enumerate(target_dims):
            sub = results_df[
                (results_df["dimension"] == fd) & (results_df["target_component"] == td)
            ].dropna(subset=["correlation"])
            if sub.empty:
                continue
            idx = sub["correlation"].abs().idxmax()
            peak_r[i, j] = sub.loc[idx, "correlation"]
            peak_lag[i, j] = sub.loc[idx, "lag_seconds"]

    fig, ax = plt.subplots(figsize=(max(8, len(target_dims) * 1.2),
                                     max(6, len(feat_dims) * 0.5)))
    vmax = np.nanmax(np.abs(peak_r)) if np.any(np.isfinite(peak_r)) else 0.5
    im = ax.imshow(peak_r, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(len(feat_dims)):
        for j in range(len(target_dims)):
            if np.isfinite(peak_r[i, j]):
                ax.text(j, i, f"{peak_r[i,j]:+.2f}\n({peak_lag[i,j]:+.1f}s)",
                        ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(target_dims)))
    ax.set_xticklabels(target_dims, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(feat_dims)))
    ax.set_yticklabels(feat_dims, fontsize=8)
    ax.set_xlabel("Target components", fontsize=11)
    ax.set_ylabel("Feature dimensions", fontsize=11)
    ax.set_title(f"Peak-Lag Cross-Correlation with {label}", fontsize=13)
    fig.colorbar(im, ax=ax, label="Pearson r at peak lag")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {output_path}")


# ---------------------------------------------------------------------------
# Group-level helpers
# ---------------------------------------------------------------------------

def fisher_z(r):
    """Fisher z-transform for correlation coefficients."""
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def inverse_fisher_z(z):
    """Inverse Fisher z-transform back to r."""
    return np.tanh(z)


def discover_outputs(input_dir, pattern):
    """Recursively find files matching *pattern* under *input_dir*."""
    return sorted(Path(input_dir).rglob(pattern))


def parse_csv_header(path):
    """Parse ``# key: value`` comment header from a CSV file."""
    meta = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                key, _, val = line[2:].partition(":")
                meta[key.strip()] = val.strip()
            else:
                break
    return meta


def extract_peak_correlation(corr_df, dimension_col="dimension"):
    """Extract the peak absolute correlation per dimension."""
    rows = []
    for dim, sub in corr_df.groupby(dimension_col):
        sub = sub.dropna(subset=["correlation"])
        if sub.empty:
            continue
        idx = sub["correlation"].abs().idxmax()
        row = sub.loc[idx]
        rows.append({
            "dimension": dim,
            "peak_r": row["correlation"],
            "peak_lag": row["lag_seconds"],
            "peak_p": row["p_value"],
            "peak_p_fdr": row.get("p_fdr", np.nan),
        })
    return pd.DataFrame(rows)


MODALITY_MAP = {
    "GMP_world_": "movement",
    "GMP_norm_": "movement",
    "oc_audvol": "vocal_energy",
    "oc_audpit": "vocal_pitch",
    "lbrs_": "spectral",
    "osm_": "opensmile",
    "pf_AU": "facial_AU",
    "pf_emotion": "facial_emotion",
    "pf_facepose": "head_pose",
    "ee_": "emotieffnet",
}


def feature_to_modality(feature_name):
    """Map a feature column name to its modality via prefix matching."""
    for prefix, modality in MODALITY_MAP.items():
        if feature_name.startswith(prefix):
            return modality
    return "other"


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv_with_header(df, path, header_dict, float_fmt="%.6f"):
    """Write a CSV with a comment header block."""
    lines = [f"# {k}: {v}" for k, v in header_dict.items()]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
        df.to_csv(fh, index=False, float_format=float_fmt)
    print(f"  Saved CSV: {path} ({len(df)} rows)")
