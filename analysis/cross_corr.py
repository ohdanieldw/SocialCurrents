#!/usr/bin/env python3
"""
Lagged cross-correlation analysis between pipeline timeseries features
and continuous trustworthiness ratings.

Aligns both signals to a common time resolution (default 0.5s bins),
extracts PCA components and conceptual summary dimensions, then computes
lagged cross-correlation with the rating signal.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Lagged cross-correlation between pipeline features and ratings"
    )
    p.add_argument("-f", "--features", required=True,
                   help="Path to timeseries_features.csv")
    p.add_argument("-r", "--ratings", required=True,
                   help="Path to rating CSV (must have VideoTime and Value columns)")
    p.add_argument("-o", "--output-dir", default="./analysis_output",
                   help="Output directory (default: ./analysis_output)")
    p.add_argument("--rater", default=None,
                   help="Rater subject ID (e.g. sub001)")
    p.add_argument("--target", default=None,
                   help="Target subject ID (e.g. sub007)")
    p.add_argument("--rating-col", default="Value",
                   help="Column name for rating values (default: Value)")
    p.add_argument("--bin-size", type=float, default=0.5,
                   help="Time bin size in seconds (default: 0.5)")
    p.add_argument("--lag-min", type=float, default=-5.0,
                   help="Minimum lag in seconds (default: -5)")
    p.add_argument("--lag-max", type=float, default=15.0,
                   help="Maximum lag in seconds (default: 15)")
    p.add_argument("--n-components", type=int, default=10,
                   help="Number of PCA components (default: 10)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files (default: skip)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_features(path):
    """Load timeseries features CSV."""
    df = pd.read_csv(path)
    if "time_seconds" not in df.columns:
        sys.exit("Error: features CSV must contain a 'time_seconds' column")
    print(f"  Loaded features: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def load_ratings(path, rating_col="Value"):
    """Load rating CSV, convert VideoTime from ms to seconds."""
    df = pd.read_csv(path)
    if "VideoTime" not in df.columns:
        sys.exit("Error: rating CSV must contain a 'VideoTime' column")
    if rating_col not in df.columns:
        sys.exit(f"Error: rating CSV must contain a '{rating_col}' column")
    df["time_seconds"] = df["VideoTime"] / 1000.0
    # Rename rating column to 'Value' internally for consistency
    if rating_col != "Value":
        df["Value"] = df[rating_col]
    print(f"  Loaded ratings: {df.shape[0]} samples, "
          f"column='{rating_col}', "
          f"range {df['time_seconds'].min():.1f}s – {df['time_seconds'].max():.1f}s")
    return df


# ---------------------------------------------------------------------------
# Binning & filtering
# ---------------------------------------------------------------------------

def bin_timeseries(df, time_col, bin_size):
    """Bin a DataFrame into equal time bins, aggregating by mean."""
    bin_idx = np.floor(df[time_col] / bin_size).astype(int)
    grouped = df.groupby(bin_idx, sort=True).mean(numeric_only=True)
    grouped.index.name = "bin_idx"
    grouped["bin_time"] = grouped.index * bin_size
    return grouped


def filter_timevarying_columns(df, nan_threshold=0.5):
    """Keep only numeric, time-varying columns (drop constants, high-NaN)."""
    # Numeric only
    num_df = df.select_dtypes(include=[np.number])

    # Drop index/time columns
    drop_cols = {"frame_idx", "time_seconds", "bin_time"}
    num_df = num_df.drop(columns=[c for c in drop_cols if c in num_df.columns])

    # Drop zero-variance (constant/scalar broadcast) columns
    variances = num_df.var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        print(f"  Dropped {len(zero_var)} constant columns")
    num_df = num_df.drop(columns=zero_var)

    # Drop columns with >threshold NaN
    nan_frac = num_df.isna().mean()
    high_nan = nan_frac[nan_frac > nan_threshold].index.tolist()
    if high_nan:
        print(f"  Dropped {len(high_nan)} columns with >{nan_threshold*100:.0f}% NaN")
    num_df = num_df.drop(columns=high_nan)

    # Fill remaining NaN
    num_df = num_df.ffill().bfill()

    print(f"  Retained {num_df.shape[1]} time-varying feature columns")
    return num_df


# ---------------------------------------------------------------------------
# Conceptual dimensions
# ---------------------------------------------------------------------------

def compute_movement_energy(features_df):
    """Compute per-frame movement energy from GMP world coordinates.

    Must be called BEFORE binning — velocity requires native-FPS differencing.
    Returns a Series aligned to the features DataFrame index.
    """
    # Collect GMP_world_x/y/z columns for 33 landmarks
    x_cols = [f"GMP_world_x_{i}" for i in range(1, 34)]
    y_cols = [f"GMP_world_y_{i}" for i in range(1, 34)]
    z_cols = [f"GMP_world_z_{i}" for i in range(1, 34)]

    available_x = [c for c in x_cols if c in features_df.columns]
    available_y = [c for c in y_cols if c in features_df.columns]
    available_z = [c for c in z_cols if c in features_df.columns]

    n_landmarks = min(len(available_x), len(available_y), len(available_z))
    if n_landmarks == 0:
        print("  Warning: No GMP_world coordinate columns found, skipping movement_energy")
        return None, "no_columns"

    # Check if all coordinates are zero (pose detection failed)
    all_world = (available_x + available_y + available_z)[:n_landmarks * 3]
    if (features_df[all_world].abs() == 0).all().all():
        print("  Warning: GMP_world coordinates are all zero — "
              "pose detection produced no landmarks for this subject")
        return None, "all_zero"

    # Build (n_frames, n_landmarks, 3) array
    coords = np.stack([
        features_df[available_x[:n_landmarks]].values,
        features_df[available_y[:n_landmarks]].values,
        features_df[available_z[:n_landmarks]].values,
    ], axis=-1)  # shape: (n_frames, n_landmarks, 3)

    # Frame-to-frame displacement
    diff = np.diff(coords, axis=0)  # (n_frames-1, n_landmarks, 3)
    magnitudes = np.linalg.norm(diff, axis=2)  # (n_frames-1, n_landmarks)
    avg_mag = np.nanmean(magnitudes, axis=1)  # (n_frames-1,)

    # Prepend 0 for first frame
    energy = np.concatenate([[0.0], avg_mag])
    series = pd.Series(energy, index=features_df.index, name="movement_energy")
    print(f"  Computed movement_energy from {n_landmarks} landmarks")
    return series, "ok"


def _zscore_and_mean(df, cols, name):
    """Z-score selected columns and return their row-wise mean."""
    available = [c for c in cols if c in df.columns]
    if not available:
        print(f"  Warning: No columns found for {name}, skipping")
        return None
    sub = df[available].copy()
    # Z-score each column
    for c in sub.columns:
        s = sub[c].std()
        if s > 0:
            sub[c] = (sub[c] - sub[c].mean()) / s
        else:
            sub[c] = 0.0
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


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def _truncate(name, maxlen=12):
    """Truncate a feature name for legend readability."""
    return name[:maxlen] if len(name) > maxlen else name


def run_pca(df, n_components):
    """Run PCA on the filtered features. Returns (components_df, pc_labels)."""
    n_components = min(n_components, df.shape[0], df.shape[1])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    # Replace any residual NaN/inf from scaling
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)

    var_explained = pca.explained_variance_ratio_
    feature_names = list(df.columns)

    # Build top-3 loadings per PC
    pc_labels = {}
    print(f"  PCA: {n_components} components, "
          f"cumulative variance = {var_explained.sum():.1%}")
    for i in range(n_components):
        loadings = pca.components_[i]
        top3_idx = np.argsort(np.abs(loadings))[::-1][:3]
        top3_names = [feature_names[j] for j in top3_idx]
        top3_short = [_truncate(n) for n in top3_names]
        pc_name = f"PC{i+1}"
        pc_labels[pc_name] = f"{pc_name} ({', '.join(top3_short)})"

        top3_detail = ", ".join(
            f"{feature_names[j]} ({loadings[j]:+.3f})" for j in top3_idx
        )
        print(f"    {pc_name}: {var_explained[i]:.1%}  top: {top3_detail}")

    col_names = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(components, index=df.index, columns=col_names), pc_labels


# ---------------------------------------------------------------------------
# Lagged cross-correlation
# ---------------------------------------------------------------------------

def lagged_crosscorrelation(x, y, min_lag_samples, max_lag_samples):
    """Compute Pearson r between x and y at integer lags.

    Positive lag: x precedes y by `lag` steps (x at t correlates with y at t+lag).
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

        # Drop pairs where either is NaN
        mask = np.isfinite(x_seg) & np.isfinite(y_seg)
        if mask.sum() < 3:
            continue
        # Skip if either segment is constant (std == 0)
        x_valid, y_valid = x_seg[mask], y_seg[mask]
        if x_valid.std() == 0 or y_valid.std() == 0:
            continue
        r, p = stats.pearsonr(x_valid, y_valid)
        rs[i] = r
        ps[i] = p

    return lags, rs, ps


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_all_dimensions(features_df, ratings_df, bin_size, n_components):
    """Build all dimensions and compute cross-correlations. Returns results DataFrame."""

    # --- Movement energy at native resolution, then bin ---
    movement, movement_status = compute_movement_energy(features_df)
    if movement is not None:
        # Add to features for binning
        features_df = features_df.copy()
        features_df["_movement_energy"] = movement.values

    # --- Bin both signals ---
    print("\nBinning features...")
    binned_feat = bin_timeseries(features_df, "time_seconds", bin_size)
    print(f"  Binned features: {binned_feat.shape[0]} bins")

    print("Binning ratings...")
    binned_rat = bin_timeseries(ratings_df, "time_seconds", bin_size)
    print(f"  Binned ratings: {binned_rat.shape[0]} bins")

    # --- Align on common bins ---
    common_bins = binned_feat.index.intersection(binned_rat.index)
    if len(common_bins) == 0:
        sys.exit("Error: no overlapping time bins between features and ratings")
    binned_feat = binned_feat.loc[common_bins]
    binned_rat = binned_rat.loc[common_bins]
    rating_signal = binned_rat["Value"].values
    print(f"  Aligned: {len(common_bins)} common bins "
          f"({common_bins.min() * bin_size:.1f}s – {common_bins.max() * bin_size:.1f}s)")

    # --- Extract binned movement energy before filtering ---
    binned_movement = None
    if "_movement_energy" in binned_feat.columns:
        binned_movement = binned_feat["_movement_energy"].copy()
        binned_feat = binned_feat.drop(columns=["_movement_energy"])

    # --- Filter time-varying columns ---
    print("\nFiltering features...")
    filtered = filter_timevarying_columns(binned_feat)

    # --- PCA ---
    print("\nRunning PCA...")
    pca_df, pc_labels = run_pca(filtered, n_components)

    # --- Conceptual dimensions ---
    print("\nComputing conceptual dimensions...")
    dimensions = {}

    # PCA components
    for col in pca_df.columns:
        dimensions[col] = pca_df[col].values

    # Conceptual dimensions
    if binned_movement is not None:
        dimensions["movement_energy"] = binned_movement.values

    vocal = compute_vocal_energy(filtered)
    if vocal is not None:
        dimensions["vocal_energy"] = vocal.values

    spectral = compute_spectral_complexity(filtered)
    if spectral is not None:
        dimensions["spectral_complexity"] = spectral.values

    osm = compute_opensmile_summary(filtered)
    if osm is not None:
        dimensions["opensmile_summary"] = osm.values

    return dimensions, rating_signal, bin_size, movement_status, pc_labels


def compute_all_crosscorrelations(dimensions, rating_signal, bin_size, lag_min, lag_max):
    """Compute lagged cross-correlations for all dimensions."""
    min_lag_samples = int(round(lag_min / bin_size))
    max_lag_samples = int(round(lag_max / bin_size))

    rows = []
    for name, signal in dimensions.items():
        lags, rs, ps = lagged_crosscorrelation(signal, rating_signal,
                                                min_lag_samples, max_lag_samples)
        for lag, r, p in zip(lags, rs, ps):
            rows.append({
                "dimension": name,
                "lag_seconds": lag * bin_size,
                "correlation": r,
                "p_value": p,
            })

    results = pd.DataFrame(rows)

    # FDR correction (Benjamini-Hochberg) across all valid p-values
    if HAS_STATSMODELS:
        valid_mask = results["p_value"].notna()
        if valid_mask.any():
            _, fdr_vals, _, _ = multipletests(
                results.loc[valid_mask, "p_value"], method="fdr_bh"
            )
            results["p_fdr"] = np.nan
            results.loc[valid_mask, "p_fdr"] = fdr_vals
        else:
            results["p_fdr"] = np.nan
    else:
        print("  Warning: statsmodels not installed, skipping FDR correction "
              "(pip install statsmodels)")
        results["p_fdr"] = np.nan

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cross_correlations(results_df, output_path, rater=None, target=None,
                            pc_labels=None):
    """Plot cross-correlation curves: one line per dimension."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Separate PCA and conceptual dimensions for styling
    conceptual = ["movement_energy", "vocal_energy", "spectral_complexity", "opensmile_summary"]
    conceptual_colors = {"movement_energy": "#e63946", "vocal_energy": "#457b9d",
                         "spectral_complexity": "#2a9d8f", "opensmile_summary": "#e9c46a"}

    pca_cmap = plt.cm.cool(np.linspace(0.0, 1.0, 10))

    for dim_name, group in results_df.groupby("dimension", sort=False):
        lag = group["lag_seconds"].values
        corr = group["correlation"].values

        if dim_name in conceptual:
            ax.plot(lag, corr, label=dim_name, linewidth=2.5,
                    color=conceptual_colors.get(dim_name, "black"), zorder=10)
        else:
            # PCA component — use descriptive label if available
            legend_label = (pc_labels or {}).get(dim_name, dim_name)
            idx = int(dim_name.replace("PC", "")) - 1 if dim_name.startswith("PC") else 0
            ax.plot(lag, corr, label=legend_label, linewidth=1.2, alpha=0.8,
                    color=pca_cmap[min(idx, len(pca_cmap) - 1)], zorder=5)

    ax.axhline(0, color="black", linewidth=2, zorder=0)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (seconds) \u2014 negative: rating precedes feature / "
                  "positive: feature precedes rating", fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=12)
    title = "Lagged Cross-Correlation with Trustworthiness Rating"
    if rater and target:
        title += f"\n{rater} rates {target}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _output_prefix(args):
    """Build output filename prefix from --rater and --target, or fall back."""
    if args.rater and args.target:
        return f"{args.rater}_rates_{args.target}_"
    return ""


def _csv_header_comment(args):
    """Build a header comment block for the output CSV."""
    lines = [
        f"# rater: {args.rater or '(not specified)'}",
        f"# target: {args.target or '(not specified)'}",
        f"# rating_col: {args.rating_col}",
        f"# bin_size: {args.bin_size}s",
        f"# lag_range: {args.lag_min}s to {args.lag_max}s",
        f"# n_components: {args.n_components}",
        f"# features_file: {args.features}",
        f"# ratings_file: {args.ratings}",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = _output_prefix(args)

    # Check for existing outputs unless --overwrite
    csv_path = out_dir / f"{prefix}cross_correlation.csv"
    plot_path = out_dir / f"{prefix}cross_correlation_plot.png"
    if not args.overwrite:
        existing = [p for p in (csv_path, plot_path) if p.exists()]
        if existing:
            for p in existing:
                print(f"  Already exists: {p}")
            print("  Skipping (use --overwrite to replace)")
            sys.exit(0)

    print("Loading data...")
    features_df = load_features(args.features)
    ratings_df = load_ratings(args.ratings, rating_col=args.rating_col)

    dimensions, rating_signal, bin_size, movement_status, pc_labels = build_all_dimensions(
        features_df, ratings_df, args.bin_size, args.n_components
    )

    print(f"\nComputing lagged cross-correlations "
          f"(lags {args.lag_min}s to {args.lag_max}s, step {args.bin_size}s)...")
    results = compute_all_crosscorrelations(
        dimensions, rating_signal, bin_size, args.lag_min, args.lag_max
    )

    # Save CSV with header comment
    with open(csv_path, "w") as fh:
        fh.write(_csv_header_comment(args))
        results.to_csv(fh, index=False, float_format="%.6f")
    print(f"  Saved CSV: {csv_path} ({len(results)} rows)")

    # Save plot
    plot_cross_correlations(results, plot_path, rater=args.rater, target=args.target,
                            pc_labels=pc_labels)

    # Summary
    print("\n--- Summary ---")
    print(f"  Dimensions: {len(dimensions)}")
    print(f"  Lag range: {args.lag_min}s to {args.lag_max}s ({int(round((args.lag_max - args.lag_min) / args.bin_size)) + 1} steps)")
    print(f"  Total correlations: {len(results)}")

    # Diagnostic reasons for missing dimensions
    _missing_reasons = {
        "no_columns": "GMP_world columns not found in features CSV",
        "all_zero": "pose detection produced no landmarks for this subject",
    }

    # Peak correlations per dimension
    print("\n  Peak |r| per dimension:")
    for dim_name in dimensions:
        sub = results[results["dimension"] == dim_name].dropna(subset=["correlation"])
        if sub.empty:
            reason = ""
            if dim_name == "movement_energy" and movement_status != "ok":
                reason = f" ({_missing_reasons.get(movement_status, movement_status)})"
            print(f"    {dim_name:25s}  (no valid correlations){reason}")
            continue
        idx = sub["correlation"].abs().idxmax()
        row = sub.loc[idx]
        fdr_str = f"  p_fdr={row['p_fdr']:.4f}" if HAS_STATSMODELS and np.isfinite(row['p_fdr']) else ""
        print(f"    {dim_name:25s}  r={row['correlation']:+.4f}  "
              f"lag={row['lag_seconds']:+.1f}s  p={row['p_value']:.4f}{fdr_str}")

    # Check for missing dimensions that were skipped entirely
    if "movement_energy" not in dimensions and movement_status != "ok":
        reason = _missing_reasons.get(movement_status, movement_status)
        print(f"    {'movement_energy':25s}  (skipped: {reason})")

    n_valid = results["p_value"].notna().sum()
    fdr_note = " p_fdr = Benjamini-Hochberg FDR-corrected." if HAS_STATSMODELS else ""
    print(f"\nNote: {n_valid} valid tests out of {len(results)} total.{fdr_note}")


if __name__ == "__main__":
    main()
