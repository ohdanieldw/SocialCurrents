#!/usr/bin/env python3
"""
Descriptive summary of extracted features.

Generates per-feature statistics, group summaries, diagnostic plots,
and stationarity tests.  Supports single-subject and batch modes.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.utils import (
    load_features,
    bin_timeseries,
    filter_timevarying_columns,
    compute_grouped_dimensions,
    run_reduction,
    resolve_facing,
)

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_ADF = True
except ImportError:
    HAS_ADF = False

# Feature group prefixes
FEATURE_GROUPS = {
    "Audio (oc_)": "oc_",
    "Spectral (lbrs_)": "lbrs_",
    "OpenSMILE (osm_)": "osm_",
    "Pose (GMP_)": "GMP_",
    "Py-Feat (pf_)": "pf_",
    "EmotiEffNet (eln_)": "eln_",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Descriptive summary of extracted features")
    p.add_argument("-f", "--features", required=True,
                   help="Path to timeseries_features.csv or directory (batch mode)")
    p.add_argument("-o", "--output-dir", default="./analysis_output",
                   help="Output directory (default: ./analysis_output)")
    p.add_argument("--reduce-features",
                   choices=["pca", "fa", "ica", "grouped", "every"],
                   default="grouped", help="Feature reduction (default: grouped)")
    p.add_argument("--n-components", type=int, default=5,
                   help="Components for PCA/FA/ICA (default: 5)")
    p.add_argument("--time-resolution", type=float, default=0.5,
                   help="Bin size in seconds (default: 0.5)")
    p.add_argument("--no-zscore", action="store_true", help="Skip z-scoring")
    p.add_argument("--subjects", default=None,
                   help="Path to subjects.csv for orientation normalization "
                        "(columns: dyad, subject, seat_position, facing_direction)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-subject descriptives
# ---------------------------------------------------------------------------

def compute_descriptives(df):
    """Compute per-column descriptive statistics. Returns DataFrame."""
    num = df.select_dtypes(include=[np.number])
    drop = {"frame_idx", "time_seconds", "bin_time"}
    num = num.drop(columns=[c for c in drop if c in num.columns])

    rows = []
    for col in num.columns:
        vals = num[col].values
        finite = vals[np.isfinite(vals)]
        n = len(finite)
        rows.append({
            "feature": col,
            "mean": float(np.mean(finite)) if n > 0 else np.nan,
            "sd": float(np.std(finite)) if n > 0 else np.nan,
            "min": float(np.min(finite)) if n > 0 else np.nan,
            "max": float(np.max(finite)) if n > 0 else np.nan,
            "median": float(np.median(finite)) if n > 0 else np.nan,
            "skewness": float(sp_stats.skew(finite)) if n > 2 and np.var(finite) > 0 else np.nan,
            "kurtosis": float(sp_stats.kurtosis(finite)) if n > 2 and np.var(finite) > 0 else np.nan,
            "variance": float(np.var(finite)) if n > 0 else 0.0,
            "percent_zero": float((finite == 0).sum() / n * 100) if n > 0 else 100.0,
            "percent_nan": float(np.isnan(vals).sum() / len(vals) * 100),
            "is_constant": bool(np.var(finite) == 0) if n > 0 else True,
        })

    result = pd.DataFrame(rows).sort_values("variance", ascending=False).reset_index(drop=True)
    return result


def compute_feature_groups(descriptives_df):
    """Group features by prefix and compute group-level stats."""
    rows = []
    for group_name, prefix in FEATURE_GROUPS.items():
        sub = descriptives_df[descriptives_df["feature"].str.startswith(prefix)]
        if sub.empty:
            continue
        rows.append({
            "group": group_name,
            "prefix": prefix,
            "n_features": len(sub),
            "mean_variance": float(sub["variance"].mean()),
            "mean_range": float((sub["max"] - sub["min"]).mean()),
            "n_constant": int(sub["is_constant"].sum()),
        })
    return pd.DataFrame(rows).sort_values("mean_variance", ascending=False).reset_index(drop=True)


def compute_stationarity(data_df, dim_names):
    """Run ADF test on each dimension. Returns DataFrame."""
    if not HAS_ADF:
        return None

    rows = []
    for col in dim_names:
        vals = data_df[col].dropna().values if col in data_df.columns else np.array([])
        if len(vals) < 20:
            rows.append({"dimension": col, "adf_statistic": np.nan, "adf_pvalue": np.nan, "stationary": None})
            continue
        try:
            stat, pval, _, _, _, _ = adfuller(vals, maxlag=min(20, len(vals) // 4))
            rows.append({
                "dimension": col, "adf_statistic": float(stat),
                "adf_pvalue": float(pval), "stationary": pval < 0.05,
            })
        except Exception:
            rows.append({"dimension": col, "adf_statistic": np.nan, "adf_pvalue": np.nan, "stationary": None})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_correlation_matrix(data_df, dim_names, output_path):
    """Heatmap of inter-dimension correlations."""
    corr = data_df[dim_names].corr()
    fig, ax = plt.subplots(figsize=(max(6, len(dim_names) * 0.8), max(5, len(dim_names) * 0.7)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(len(dim_names)):
        for j in range(len(dim_names)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(dim_names)))
    ax.set_xticklabels(dim_names, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(dim_names)))
    ax.set_yticklabels(dim_names, fontsize=8)
    ax.set_title("Inter-Dimension Correlation Matrix", fontsize=13)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(data_df, dim_names, bin_times, output_path):
    """Small multiples of each dimension over time."""
    n = len(dim_names)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3, n * 2)), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n))
    for i, dim in enumerate(dim_names):
        ax = axes[i]
        vals = data_df[dim].values if dim in data_df.columns else np.zeros(len(bin_times))
        ax.plot(bin_times, vals, color=colors[i], linewidth=0.8)
        ax.set_ylabel(dim, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle("Feature Dimensions Over Time", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scree(filtered_df, output_path, max_components=20):
    """Scree plot of cumulative variance explained."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filtered_df)
    scaled = np.nan_to_num(scaled, nan=0.0)

    n = min(max_components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n)
    pca.fit(scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n + 1), pca.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.plot(range(1, n + 1), cumvar, "o-", color="red", label="Cumulative")
    ax.set_xlabel("Component", fontsize=11)
    ax.set_ylabel("Variance Explained", fontsize=11)
    ax.set_title("PCA Scree Plot", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pca


def plot_pca_loadings_heatmap(pca_model, feature_names, output_path, max_features=30):
    """Heatmap of PCA loadings."""
    n_comp = pca_model.n_components_
    loadings = pca_model.components_

    # Show top features by max absolute loading across any component
    max_abs = np.max(np.abs(loadings), axis=0)
    top_idx = np.argsort(max_abs)[::-1][:max_features]
    top_names = [feature_names[i] for i in top_idx]
    top_loadings = loadings[:, top_idx]

    fig, ax = plt.subplots(figsize=(max(8, n_comp * 1.5), max(6, len(top_names) * 0.3)))
    im = ax.imshow(top_loadings.T, cmap="RdBu_r", aspect="auto",
                   vmin=-np.max(np.abs(top_loadings)), vmax=np.max(np.abs(top_loadings)))
    ax.set_xticks(range(n_comp))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_comp)], fontsize=9)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=7)
    ax.set_title("PCA Loadings (top features by importance)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Loading")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(data_df, dim_names, output_path):
    """Histogram/density for each dimension."""
    n = len(dim_names)
    cols = min(n, 4)
    rows_n = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.5, rows_n * 3))
    axes = np.atleast_2d(axes)

    for i, dim in enumerate(dim_names):
        ax = axes[i // cols, i % cols]
        vals = data_df[dim].dropna().values if dim in data_df.columns else []
        if len(vals) > 0:
            ax.hist(vals, bins=50, density=True, alpha=0.7, color=plt.cm.Set2(i / max(n, 1)))
        ax.set_title(dim, fontsize=9)
        ax.set_xlabel("")

    # Hide unused axes
    for i in range(n, rows_n * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_batch_comparison(batch_df, output_path):
    """Box plots comparing grouped dimensions across subjects."""
    dim_cols = [c for c in batch_df.columns if c.startswith("mean_") or c.startswith("var_")]
    if not dim_cols:
        return

    n = len(dim_cols)
    fig, axes = plt.subplots(1, n, figsize=(max(8, n * 2.5), 5))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(dim_cols):
        ax = axes[i]
        vals = batch_df[col].dropna().values
        ax.boxplot(vals, widths=0.6)
        ax.set_title(col.replace("mean_", "").replace("var_", "var:"), fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feature Comparison Across Subjects", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single subject
# ---------------------------------------------------------------------------

def describe_subject(csv_path, out_dir, args):
    """Run full descriptive analysis on one subject. Returns summary dict for batch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive subject/dyad from filename
    stem = Path(csv_path).stem.replace("_timeseries_features", "")
    parts = stem.split("_", 1)
    dyad_id = parts[0] if len(parts) > 1 else ""
    subject_id = parts[1] if len(parts) > 1 else stem

    facing = resolve_facing(csv_path, getattr(args, "subjects", None))
    features_df = load_features(csv_path, facing_direction=facing)
    n_frames = len(features_df)
    fps = n_frames / features_df["time_seconds"].max() if features_df["time_seconds"].max() > 0 else 25
    duration = features_df["time_seconds"].max()
    mins = int(duration // 60)
    secs = int(duration % 60)

    # Descriptives on raw features
    print("\nComputing per-feature descriptives...")
    desc = compute_descriptives(features_df)
    desc.to_csv(out_dir / "descriptives.csv", index=False, float_format="%.4f")
    print(f"  Saved: {out_dir / 'descriptives.csv'} ({len(desc)} features)")

    groups = compute_feature_groups(desc)
    groups.to_csv(out_dir / "feature_groups.csv", index=False, float_format="%.4f")
    print(f"  Saved: {out_dir / 'feature_groups.csv'}")

    # Bin & filter
    print("\nBinning...")
    binned = bin_timeseries(features_df, "time_seconds", args.time_resolution)
    print("Filtering...")
    filtered = filter_timevarying_columns(binned)

    # PCA scree + loadings (always, regardless of --reduce-features)
    print("\nPCA scree analysis...")
    pca_model = plot_pca_scree(filtered, out_dir / "pca_scree.png")
    print(f"  Saved: {out_dir / 'pca_scree.png'}")
    plot_pca_loadings_heatmap(pca_model, list(filtered.columns), out_dir / "pca_loadings_heatmap.png")
    print(f"  Saved: {out_dir / 'pca_loadings_heatmap.png'}")

    # Reduce for remaining plots
    method = args.reduce_features
    if method in ("pca", "fa", "ica"):
        print(f"\nReducing features ({method.upper()})...")
        comp_df, labels, _ = run_reduction(filtered, method, args.n_components)
        dim_names = list(comp_df.columns)
        data_df = comp_df
    elif method == "grouped":
        print("\nComputing grouped dimensions...")
        grouped, _ = compute_grouped_dimensions(features_df, filtered, args.time_resolution)
        data_df = pd.DataFrame(grouped, index=filtered.index)
        dim_names = list(data_df.columns)
    else:
        data_df = filtered
        dim_names = list(filtered.columns)[:20]  # cap for readability

    if not args.no_zscore and method in ("grouped", "every"):
        scaler = StandardScaler()
        data_df = pd.DataFrame(scaler.fit_transform(data_df), index=data_df.index, columns=data_df.columns)

    bin_times = data_df.index.values * args.time_resolution

    # Plots
    print("\nGenerating plots...")
    plot_correlation_matrix(data_df, dim_names, out_dir / "correlation_matrix.png")
    print(f"  Saved: correlation_matrix.png")
    plot_timeseries(data_df, dim_names, bin_times, out_dir / "timeseries_plot.png")
    print(f"  Saved: timeseries_plot.png")
    plot_distributions(data_df, dim_names, out_dir / "distribution_plot.png")
    print(f"  Saved: distribution_plot.png")

    # Stationarity
    stat_df = compute_stationarity(data_df, dim_names)
    if stat_df is not None:
        stat_df.to_csv(out_dir / "stationarity.csv", index=False, float_format="%.6f")
        print(f"  Saved: stationarity.csv")
        n_stationary = stat_df["stationary"].sum()
        n_total = len(stat_df)
    else:
        n_stationary = n_total = 0

    # Terminal summary
    print(f"\n\u2500\u2500 {stem} ({mins}m{secs:02d}s, {n_frames} frames) " + "\u2500" * 30)
    print(f"\n  Feature groups:")
    for _, row in groups.iterrows():
        print(f"    {row['group']:25s} {int(row['n_features']):4d} features   variance: {row['mean_variance']:.2f}")

    top5 = desc.head(5)
    print(f"\n  Top 5 most variable features:")
    for _, row in top5.iterrows():
        print(f"    {row['feature']:30s} var={row['variance']:.2f}  range=[{row['min']:.2f}, {row['max']:.2f}]")

    if pca_model:
        cumvar = np.sum(pca_model.explained_variance_ratio_[:args.n_components])
        print(f"\n  PCA: {args.n_components} components explain {cumvar:.1%} of variance")

    if stat_df is not None and n_total > 0:
        print(f"  Stationarity: {n_stationary}/{n_total} dimensions are stationary (ADF p<0.05)")
    print(f"  {'─' * 55}")

    # Return batch summary row
    summary = {
        "subject_id": subject_id,
        "dyad_id": dyad_id,
        "duration_seconds": float(duration),
        "n_frames": int(n_frames),
        "n_features": int(len(desc)),
    }
    for dim in dim_names:
        if dim in data_df.columns:
            summary[f"mean_{dim}"] = float(data_df[dim].mean())
            summary[f"var_{dim}"] = float(data_df[dim].var())
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_path = Path(args.features)

    if input_path.is_file():
        # Single subject
        describe_subject(str(input_path), args.output_dir, args)
    elif input_path.is_dir():
        # Batch mode
        csvs = sorted(input_path.rglob("*_timeseries_features.csv"))
        if not csvs:
            sys.exit(f"No timeseries CSV files found in {input_path}")

        print(f"Batch mode: found {len(csvs)} subjects\n")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        batch_rows = []
        for csv_path in csvs:
            stem = csv_path.stem.replace("_timeseries_features", "")
            sub_dir = out_dir / stem
            print(f"\n{'=' * 60}")
            print(f"  Processing: {stem}")
            print(f"{'=' * 60}")
            try:
                row = describe_subject(str(csv_path), str(sub_dir), args)
                batch_rows.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")

        if batch_rows:
            batch_df = pd.DataFrame(batch_rows)
            batch_df.to_csv(out_dir / "batch_summary.csv", index=False, float_format="%.4f")
            print(f"\n  Saved: {out_dir / 'batch_summary.csv'} ({len(batch_df)} subjects)")
            plot_batch_comparison(batch_df, out_dir / "batch_comparison.png")
            print(f"  Saved: {out_dir / 'batch_comparison.png'}")

            print(f"\n\nBatch complete: {len(batch_rows)} subjects processed.")
    else:
        sys.exit(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()
