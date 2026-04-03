#!/usr/bin/env python3
"""
Group-level analysis of which behavioral features predict outcome ratings
across subjects.

Scans the output directory for all subject-level correlate results, stacks
them, and runs group-level statistics (one-sample t-tests on Fisher-z
transformed peak correlations, feature vote counting across reduction
methods, and optional individual-difference moderator analyses).

The --label flag controls what the outcome is called in plots, CSVs, and
report text (default: Trustworthiness).  Output filenames use a slugified
version of the label.
"""

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.utils import (
    discover_outputs,
    extract_peak_correlation,
    feature_to_modality,
    fisher_z,
    inverse_fisher_z,
    parse_csv_header,
    write_csv_with_header,
    GROUPED_DIMS,
    GROUPED_COLORS,
    HAS_STATSMODELS,
)

if HAS_STATSMODELS:
    from statsmodels.stats.multitest import multipletests


def _slugify(label):
    return label.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Group-level analysis: which behavioral features predict "
                    "outcome ratings across subjects"
    )
    p.add_argument("--input-dir", required=True,
                   help="Study output root containing dyad*/sub*/correlate/")
    p.add_argument("--reduce-method",
                   choices=["grouped", "pca", "fa", "ica", "all"],
                   default="all",
                   help="Which feature reduction to analyze (default: all)")
    p.add_argument("--label", default="Trustworthiness",
                   help="What the outcome measures, used in plot titles and filenames "
                        "(default: Trustworthiness)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: {input-dir}/group/group_correlate/)")
    p.add_argument("--subjects", default=None,
                   help="Path to subjects.csv with orientation, demographics, and "
                        "questionnaire data (used for orientation note and moderator analysis)")
    p.add_argument("--covariates", default=None,
                   help="Comma-separated column names from subjects.csv for moderator analysis "
                        "(e.g. ASQ_anxiety,IRI_EC). Requires --subjects.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_CORR_FILENAME_RE = re.compile(
    r"^(?P<rater>sub\d+)_rates_(?P<target>sub\d+)_correlation"
    r"(?:_(?P<method>pca|fa|ica|grouped))?\.csv$"
)


def _parse_correlation_file(path):
    """Parse a single correlation CSV, returning metadata + peak correlations."""
    meta = parse_csv_header(path)
    match = _CORR_FILENAME_RE.match(path.name)
    if match:
        rater = match.group("rater")
        target_id = match.group("target")
        method_from_name = match.group("method")
    else:
        rater = meta.get("rater", "unknown")
        target_id = meta.get("target_id", "unknown")
        method_from_name = None

    method = method_from_name or meta.get("reduce_features", "unknown")

    dyad = None
    for part in path.parts:
        if part.startswith("dyad"):
            dyad = part
            break

    return {
        "path": path,
        "rater": rater,
        "target_id": target_id,
        "dyad": dyad or "unknown",
        "method": method,
        "meta": meta,
    }


def discover_correlation_files(input_dir, reduce_method):
    """Find all correlation CSVs and group by (target_id, method)."""
    all_csvs = discover_outputs(input_dir, "*correlation*.csv")
    corr_files = []
    for p in all_csvs:
        if "group_" in p.name or "group" in str(p):
            continue
        if not _CORR_FILENAME_RE.match(p.name):
            continue
        info = _parse_correlation_file(p)
        if reduce_method != "all" and info["method"] != reduce_method:
            continue
        corr_files.append(info)

    print(f"  Discovered {len(corr_files)} correlation files")
    methods_found = sorted(set(f["method"] for f in corr_files))
    print(f"  Methods: {', '.join(methods_found)}")
    subjects = sorted(set(f["target_id"] for f in corr_files))
    print(f"  Subjects rated: {len(subjects)} ({', '.join(subjects[:10])}{'...' if len(subjects) > 10 else ''})")
    return corr_files


def discover_loadings_files(input_dir):
    """Find all loadings CSVs and return dict keyed by (rater, target_id, method)."""
    all_csvs = discover_outputs(input_dir, "*loadings_*.csv")
    loadings = {}
    pat = re.compile(
        r"^(?P<rater>sub\d+)_rates_(?P<target>sub\d+)_loadings_(?P<method>pca|fa|ica)\.csv$"
    )
    for p in all_csvs:
        if "group" in str(p):
            continue
        match = pat.match(p.name)
        if match:
            key = (match.group("rater"), match.group("target"), match.group("method"))
            loadings[key] = p
    print(f"  Discovered {len(loadings)} loadings files")
    return loadings


# ---------------------------------------------------------------------------
# Level 1: Grouped dimensions
# ---------------------------------------------------------------------------

def run_level1_grouped(corr_files, out_dir, label, slug):
    """One-sample t-test on Fisher-z peak r per grouped dimension."""
    print(f"\n{'='*60}")
    print("  Level 1: Grouped dimensions")
    print(f"{'='*60}")

    rows = []
    for finfo in corr_files:
        try:
            df = pd.read_csv(finfo["path"], comment="#")
        except Exception as e:
            print(f"  Warning: skipping {finfo['path'].name} ({e})")
            continue
        grouped_dims = [d for d in df["dimension"].unique() if d in GROUPED_DIMS]
        if not grouped_dims:
            continue
        df_grouped = df[df["dimension"].isin(grouped_dims)]
        peaks = extract_peak_correlation(df_grouped)
        for _, peak in peaks.iterrows():
            rows.append({
                "subject": finfo["target_id"],
                "dyad": finfo["dyad"],
                "rater": finfo["rater"],
                "target_id": finfo["target_id"],
                "dimension": peak["dimension"],
                "peak_r": peak["peak_r"],
                "peak_lag": peak["peak_lag"],
                "peak_p": peak["peak_p"],
                "method": finfo["method"],
            })

    if not rows:
        print("  No grouped dimension data found, skipping Level 1")
        return pd.DataFrame()

    stacked = pd.DataFrame(rows)
    unique_per_dim = stacked.groupby(["target_id", "dimension"]).first().reset_index()

    group_rows = []
    for dim, sub in unique_per_dim.groupby("dimension"):
        z_vals = fisher_z(sub["peak_r"].values)
        n = len(z_vals)
        if n < 2:
            group_rows.append({
                "dimension": dim, "n_subjects": n,
                "mean_r": sub["peak_r"].mean(), "sd_r": sub["peak_r"].std(),
                "mean_lag": sub["peak_lag"].mean(),
                "t_stat": np.nan, "p_value": np.nan, "p_fdr": np.nan,
                "cohens_d": np.nan,
            })
            continue
        t_stat, p_val = stats.ttest_1samp(z_vals, 0)
        cohens_d = np.mean(z_vals) / np.std(z_vals, ddof=1)
        group_rows.append({
            "dimension": dim,
            "n_subjects": n,
            "mean_r": inverse_fisher_z(np.mean(z_vals)),
            "sd_r": sub["peak_r"].std(),
            "mean_lag": sub["peak_lag"].mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "p_fdr": np.nan,
            "cohens_d": cohens_d,
        })

    group_df = pd.DataFrame(group_rows)
    if HAS_STATSMODELS and group_df["p_value"].notna().any():
        valid = group_df["p_value"].notna()
        _, fdr_vals, _, _ = multipletests(group_df.loc[valid, "p_value"], method="fdr_bh")
        group_df.loc[valid, "p_fdr"] = fdr_vals

    header = {
        "analysis": "group_correlate Level 1 -- grouped dimensions",
        "test": "one-sample t-test on Fisher-z transformed peak |r|",
        "n_files_scanned": len(corr_files),
    }
    csv_path = out_dir / f"group_{slug}_from_features_grouped.csv"
    write_csv_with_header(group_df, csv_path, header)

    print("\n  Group-level results (grouped dimensions):")
    for _, row in group_df.iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        fdr_str = f"  p_fdr={row['p_fdr']:.4f}" if np.isfinite(row["p_fdr"]) else ""
        print(f"    {row['dimension']:25s}  mean_r={row['mean_r']:+.4f}  "
              f"t={row['t_stat']:+.3f}  p={row['p_value']:.4f}{fdr_str}  "
              f"d={row['cohens_d']:+.3f}  n={row['n_subjects']}{sig}")

    return unique_per_dim


# ---------------------------------------------------------------------------
# Level 2: Feature vote counting
# ---------------------------------------------------------------------------

def run_level2_votes(corr_files, loadings_map, out_dir, reduce_method, label, slug):
    """Count which original features are outcome-predictive across subjects."""
    print(f"\n{'='*60}")
    print("  Level 2: Feature vote counting")
    print(f"{'='*60}")

    vote_methods = ["pca", "fa", "ica"]
    if reduce_method != "all":
        vote_methods = [m for m in vote_methods if m == reduce_method]

    all_votes = []
    for finfo in corr_files:
        method = finfo["method"]
        if method not in vote_methods:
            continue

        try:
            corr_df = pd.read_csv(finfo["path"], comment="#")
        except Exception as e:
            print(f"  Warning: skipping {finfo['path'].name} ({e})")
            continue
        peaks = extract_peak_correlation(corr_df)

        sig_dims = peaks[(peaks["peak_r"].abs() > 0.15) & (peaks["peak_p"] < 0.05)]
        if sig_dims.empty:
            continue

        key = (finfo["rater"], finfo["target_id"], method)
        if key not in loadings_map:
            continue

        loadings_df = pd.read_csv(loadings_map[key])

        for _, sig_row in sig_dims.iterrows():
            dim = sig_row["dimension"]
            dim_loadings = loadings_df[loadings_df["component"] == dim].copy()
            if dim_loadings.empty:
                continue
            dim_loadings["abs_loading"] = dim_loadings["loading"].abs()
            top5 = dim_loadings.nlargest(5, "abs_loading")

            for _, lr in top5.iterrows():
                all_votes.append({
                    "feature": lr["feature"],
                    "abs_loading": lr["abs_loading"],
                    "subject": finfo["target_id"],
                    "rater": finfo["rater"],
                    "method": method,
                    "dimension": dim,
                    "peak_r": sig_row["peak_r"],
                })

    if not all_votes:
        print("  No significant dimensions found for vote counting")
        return pd.DataFrame()

    votes_df = pd.DataFrame(all_votes)

    n_subjects = len(set(f["target_id"] for f in corr_files
                         if f["method"] in vote_methods))

    feature_summary = (
        votes_df.groupby("feature")
        .agg(
            n_votes=("subject", "count"),
            n_unique_subjects=("subject", "nunique"),
            mean_abs_loading=("abs_loading", "mean"),
            methods_where_significant=("method", lambda x: ",".join(sorted(set(x)))),
        )
        .reset_index()
    )
    feature_summary["n_subjects_available"] = n_subjects
    feature_summary["vote_proportion"] = feature_summary["n_votes"] / max(n_subjects, 1)
    feature_summary["modality"] = feature_summary["feature"].apply(feature_to_modality)
    feature_summary = feature_summary.sort_values("n_votes", ascending=False).reset_index(drop=True)

    header = {
        "analysis": "group_correlate Level 2 -- feature vote counting",
        "criteria": "peak |r| > 0.15 AND peak_p < 0.05, top 5 features per sig dimension",
        "methods": ",".join(vote_methods),
        "n_subjects": n_subjects,
    }
    csv_path = out_dir / f"group_{slug}_feature_votes.csv"
    cols = ["feature", "n_votes", "n_subjects_available", "vote_proportion",
            "mean_abs_loading", "methods_where_significant", "modality"]
    write_csv_with_header(feature_summary[cols], csv_path, header)

    by_method = (
        votes_df.groupby(["feature", "method"])
        .agg(
            n_votes=("subject", "count"),
            mean_abs_loading=("abs_loading", "mean"),
        )
        .reset_index()
    )
    by_method["modality"] = by_method["feature"].apply(feature_to_modality)
    by_method = by_method.sort_values(["method", "n_votes"], ascending=[True, False])
    csv_path2 = out_dir / f"group_{slug}_feature_votes_by_method.csv"
    write_csv_with_header(by_method, csv_path2, header)

    print(f"\n  Top 10 features by vote count:")
    for _, row in feature_summary.head(10).iterrows():
        print(f"    {row['feature']:40s}  votes={row['n_votes']:3d}  "
              f"proportion={row['vote_proportion']:.2f}  "
              f"modality={row['modality']}")

    return feature_summary


# ---------------------------------------------------------------------------
# Level 3: Everything stacked
# ---------------------------------------------------------------------------

def run_level3_stacked(corr_files, out_dir, slug):
    """Stack all subject-level peak correlations into one CSV."""
    print(f"\n{'='*60}")
    print("  Level 3: All methods stacked")
    print(f"{'='*60}")

    rows = []
    for finfo in corr_files:
        try:
            df = pd.read_csv(finfo["path"], comment="#")
        except Exception as e:
            print(f"  Warning: skipping {finfo['path'].name} ({e})")
            continue
        peaks = extract_peak_correlation(df)
        for _, peak in peaks.iterrows():
            rows.append({
                "subject": finfo["target_id"],
                "dyad": finfo["dyad"],
                "rater": finfo["rater"],
                "target_id": finfo["target_id"],
                "method": finfo["method"],
                "dimension": peak["dimension"],
                "peak_r": peak["peak_r"],
                "peak_lag": peak["peak_lag"],
                "peak_p": peak["peak_p"],
            })

    if not rows:
        print("  No data to stack")
        return pd.DataFrame()

    stacked = pd.DataFrame(rows)
    header = {
        "analysis": "group_correlate Level 3 -- all methods stacked",
        "n_files": len(corr_files),
        "n_subjects": stacked["target_id"].nunique(),
        "methods": ",".join(sorted(stacked["method"].unique())),
    }
    csv_path = out_dir / f"group_{slug}_from_features_all.csv"
    write_csv_with_header(stacked, csv_path, header)

    print(f"  Stacked {len(stacked)} rows across "
          f"{stacked['target_id'].nunique()} subjects, "
          f"{stacked['method'].nunique()} methods")
    return stacked


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

MODALITY_COLORS = {
    "movement": "#e63946",
    "vocal_energy": "#457b9d",
    "vocal_pitch": "#1d3557",
    "spectral": "#2a9d8f",
    "opensmile": "#e9c46a",
    "facial_AU": "#f4a261",
    "facial_emotion": "#d4a373",
    "head_pose": "#a8dadc",
    "emotieffnet": "#6d6875",
    "other": "#adb5bd",
}


def plot_forest(grouped_peaks, out_dir, label):
    """Forest plot of per-subject peak r by grouped dimension."""
    if grouped_peaks.empty:
        return

    dims = [d for d in GROUPED_DIMS if d in grouped_peaks["dimension"].unique()]
    if not dims:
        return

    n_dims = len(dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2.5 * n_dims), squeeze=False)

    for i, dim in enumerate(dims):
        ax = axes[i, 0]
        sub = grouped_peaks[grouped_peaks["dimension"] == dim].sort_values("peak_r")
        n = len(sub)
        y_pos = np.arange(n)

        se = 1.0 / np.sqrt(np.maximum(3 - 3, 1))
        ci_lo = sub["peak_r"] - 1.96 * se
        ci_hi = sub["peak_r"] + 1.96 * se

        color = GROUPED_COLORS.get(dim, "#333333")
        ax.hlines(y_pos, ci_lo, ci_hi, color=color, alpha=0.4, linewidth=1.5)
        ax.scatter(sub["peak_r"], y_pos, color=color, s=40, zorder=5)

        mean_r = sub["peak_r"].mean()
        ax.scatter(mean_r, n + 0.5, marker="D", color=color, s=80, zorder=10,
                   edgecolors="black", linewidths=0.8)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(list(y_pos) + [n + 0.5])
        ax.set_yticklabels(list(sub["target_id"]) + ["Group mean"], fontsize=8)
        ax.set_xlabel("Peak r", fontsize=10)
        ax.set_title(dim, fontsize=12, fontweight="bold", color=color)
        ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle(f"Forest Plot: Peak Feature-{label} Correlations (Grouped Dimensions)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "forest_plot_grouped.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_feature_votes(vote_summary, out_dir, label):
    """Horizontal bar chart of top 20 features by vote count."""
    if vote_summary.empty:
        return

    top20 = vote_summary.head(20).iloc[::-1]
    colors = [MODALITY_COLORS.get(m, "#adb5bd") for m in top20["modality"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top20) * 0.35)))
    bars = ax.barh(range(len(top20)), top20["n_votes"], color=colors, edgecolor="white")

    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"], fontsize=8)
    ax.set_xlabel("Vote Count", fontsize=11)
    ax.set_title(f"Top 20 {label}-Predictive Features (Vote Count)", fontsize=13)
    ax.grid(True, alpha=0.2, axis="x")

    modalities_shown = top20["modality"].unique()
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLORS.get(m, "#adb5bd"))
               for m in modalities_shown]
    ax.legend(handles, modalities_shown, loc="lower right", fontsize=8,
              title="Modality", title_fontsize=9)

    fig.tight_layout()
    path = out_dir / "feature_vote_barplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_lag_distribution(stacked, out_dir):
    """Histogram/KDE of peak lags across all subjects and dimensions."""
    if stacked.empty:
        return

    lags = stacked["peak_lag"].dropna()
    if lags.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lags, bins=30, density=True, alpha=0.5, color="#457b9d", edgecolor="white")

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(lags)
        x_grid = np.linspace(lags.min(), lags.max(), 200)
        ax.plot(x_grid, kde(x_grid), color="#e63946", linewidth=2)
    except Exception:
        pass

    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="lag = 0")
    ax.axvline(lags.median(), color="#2a9d8f", linewidth=1.5, linestyle=":",
               label=f"median = {lags.median():.1f}s")
    ax.set_xlabel("Peak Lag (seconds)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution of Peak Lags Across Subjects", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = out_dir / "lag_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_method_comparison(stacked, out_dir, label):
    """Box plot of mean |peak_r| per subject, one box per reduction method."""
    if stacked.empty:
        return

    per_subject = (
        stacked.groupby(["target_id", "method"])["peak_r"]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="mean_abs_peak_r")
    )

    methods = sorted(per_subject["method"].unique())
    if len(methods) < 2:
        print("  Skipping method comparison plot (fewer than 2 methods)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_method = [per_subject[per_subject["method"] == m]["mean_abs_peak_r"].values
                      for m in methods]
    bp = ax.boxplot(data_by_method, labels=methods, patch_artist=True, widths=0.5)
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Mean |Peak r| per Subject", fontsize=11)
    ax.set_xlabel("Reduction Method", fontsize=11)
    ax.set_title(f"Method Comparison: Feature-{label} Correlation Strength", fontsize=13)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    path = out_dir / "method_comparison_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# Individual differences (covariates from subjects.csv)
# ---------------------------------------------------------------------------

def run_moderator_analysis(grouped_peaks, subjects_path, scale_cols, out_dir):
    """Correlate grouped peak_r with covariate scales (Spearman)."""
    print(f"\n{'='*60}")
    print("  Moderator analysis (individual differences)")
    print(f"{'='*60}")
    print("  NOTE: N is small; moderator analyses are exploratory.")

    from analysis.utils import load_subjects_df
    q_df = load_subjects_df(subjects_path)

    if not scale_cols:
        print("  No covariate columns specified")
        return

    print(f"  Covariates: {', '.join(scale_cols)}")

    pivot = grouped_peaks.groupby(["target_id", "dimension"])["peak_r"].first().reset_index()

    mod_rows = []
    for dim in pivot["dimension"].unique():
        dim_data = pivot[pivot["dimension"] == dim][["target_id", "peak_r"]].copy()
        dim_data = dim_data.rename(columns={"target_id": "subject"})
        merged = dim_data.merge(q_df[["subject"] + scale_cols], on="subject", how="inner")
        for scale in scale_cols:
            valid = merged[["peak_r", scale]].dropna()
            if len(valid) < 4:
                continue
            rho, p_val = stats.spearmanr(valid["peak_r"], valid[scale])
            mod_rows.append({
                "dimension": dim,
                "scale": scale,
                "spearman_r": rho,
                "p_value": p_val,
                "n": len(valid),
            })

    if not mod_rows:
        print("  No valid moderator correlations computed (too few overlapping subjects)")
        return

    mod_df = pd.DataFrame(mod_rows)
    header = {
        "analysis": "group_correlate -- moderator correlations",
        "test": "Spearman rank correlation between peak_r and questionnaire scale",
        "caveat": "N is small; moderator analyses are exploratory",
    }
    csv_path = out_dir / "moderator_correlations.csv"
    write_csv_with_header(mod_df, csv_path, header)

    print(f"\n  Moderator correlations ({len(mod_df)} pairs tested):")
    top = mod_df.reindex(mod_df["spearman_r"].abs().sort_values(ascending=False).index).head(6)
    for _, row in top.iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        print(f"    {row['dimension']:25s} x {row['scale']:20s}  "
              f"rho={row['spearman_r']:+.3f}  p={row['p_value']:.4f}  n={row['n']}{sig}")

    _plot_moderator_scatter(top, pivot, q_df, scale_cols, out_dir)


def _plot_moderator_scatter(top_mods, pivot, q_df, scale_cols, out_dir):
    """Scatter matrix for top 6 moderator correlations."""
    n_panels = min(len(top_mods), 6)
    if n_panels == 0:
        return

    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, (_, row) in enumerate(top_mods.head(n_panels).iterrows()):
        ax = axes[idx // ncols, idx % ncols]
        dim = row["dimension"]
        scale = row["scale"]

        dim_data = pivot[pivot["dimension"] == dim][["target_id", "peak_r"]].copy()
        dim_data = dim_data.rename(columns={"target_id": "subject"})
        merged = dim_data.merge(q_df[["subject", scale]], on="subject", how="inner")
        valid = merged.dropna(subset=["peak_r", scale])

        ax.scatter(valid[scale], valid["peak_r"], alpha=0.7, s=40, color="#457b9d")
        if len(valid) >= 2:
            z = np.polyfit(valid[scale], valid["peak_r"], 1)
            x_line = np.linspace(valid[scale].min(), valid[scale].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), color="#e63946", linewidth=1.5,
                    linestyle="--")
        ax.set_xlabel(scale, fontsize=9)
        ax.set_ylabel(f"Peak r ({dim})", fontsize=9)
        ax.set_title(f"rho={row['spearman_r']:+.3f}, p={row['p_value']:.3f}",
                     fontsize=9)
        ax.grid(True, alpha=0.2)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Moderator Analyses (Exploratory)", fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "moderator_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary_report(grouped_group_df, stacked, out_dir, label):
    """Print terminal summary with reference benchmarks."""
    print(f"\n{'='*60}")
    print("  Summary Report")
    print(f"{'='*60}")

    if not stacked.empty:
        all_abs_r = stacked["peak_r"].abs()
        print(f"\n  Overall peak |r| across all subjects and methods:")
        print(f"    Mean = {all_abs_r.mean():.4f}, SD = {all_abs_r.std():.4f}, "
              f"Median = {all_abs_r.median():.4f}")
        print(f"    Range = [{all_abs_r.min():.4f}, {all_abs_r.max():.4f}]")

        all_lags = stacked["peak_lag"].dropna()
        print(f"\n  Peak lag distribution:")
        print(f"    Mean = {all_lags.mean():.2f}s, SD = {all_lags.std():.2f}s, "
              f"Median = {all_lags.median():.2f}s")

    print(f"\n  Reference benchmarks:")
    print(f"    Feature-outcome correlations in continuous rating paradigms:")
    print(f"      Typical peak |r| = 0.10-0.35 (Hirschberg et al., 2020; Scherer et al., 2012)")
    print(f"    Expected peak lag: 2-5 seconds")
    print(f"      (perceptual delay in slider tasks; Schoeller & Eskenazi, 2019)")

    report_lines = [
        f"Group Correlate Summary Report ({label})",
        "=" * 40,
        "",
    ]
    if not stacked.empty:
        report_lines.extend([
            f"N subjects: {stacked['target_id'].nunique()}",
            f"Methods: {', '.join(sorted(stacked['method'].unique()))}",
            f"Total subject-dimension pairs: {len(stacked)}",
            "",
            f"Overall peak |r|: M = {all_abs_r.mean():.4f}, SD = {all_abs_r.std():.4f}",
            f"Peak lag: M = {all_lags.mean():.2f}s, SD = {all_lags.std():.2f}s",
            "",
        ])

    report_lines.extend([
        "Reference benchmarks:",
        "  Typical peak |r| = 0.10-0.35 (Hirschberg et al., 2020; Scherer et al., 2012)",
        "  Expected peak lag: 2-5s (Schoeller & Eskenazi, 2019)",
    ])

    report_path = out_dir / "summary_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\n  Saved report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    label = args.label
    slug = _slugify(label)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Error: --input-dir does not exist: {input_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "group" / "group_correlate"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = list(out_dir.glob(f"group_{slug}_*.csv"))
        if existing:
            print(f"  Output files already exist in {out_dir}")
            print(f"  Use --overwrite to replace them")
            sys.exit(0)

    print(f"Group-level feature-{label} analysis")
    print(f"  Input: {input_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Label: {label}")
    print(f"  Method filter: {args.reduce_method}")

    print(f"\nDiscovering correlation files...")
    corr_files = discover_correlation_files(input_dir, args.reduce_method)
    if not corr_files:
        sys.exit("Error: no correlation files found. Run correlate.py first.")

    loadings_map = discover_loadings_files(input_dir)

    grouped_peaks = run_level1_grouped(corr_files, out_dir, label, slug)

    vote_summary = pd.DataFrame()
    if args.reduce_method == "all" or args.reduce_method in ("pca", "fa", "ica"):
        vote_summary = run_level2_votes(corr_files, loadings_map, out_dir, args.reduce_method, label, slug)

    stacked = run_level3_stacked(corr_files, out_dir, slug)

    print(f"\nGenerating plots...")
    plot_forest(grouped_peaks, out_dir, label)
    plot_feature_votes(vote_summary, out_dir, label)
    plot_lag_distribution(stacked, out_dir)
    plot_method_comparison(stacked, out_dir, label)

    if args.subjects and args.covariates:
        from analysis.utils import load_subjects_df, get_covariate_columns
        s_df = load_subjects_df(args.subjects)
        cov_cols = get_covariate_columns(s_df, args.covariates)
        if cov_cols:
            run_moderator_analysis(grouped_peaks, args.subjects, cov_cols, out_dir)

    print_summary_report(
        out_dir / f"group_{slug}_from_features_grouped.csv",
        stacked,
        out_dir,
        label,
    )

    orientation_note = "applied (subjects.csv provided)" if args.subjects else "not applied"
    print(f"  Orientation normalization: {orientation_note}")
    with open(out_dir / "summary_report.txt", "a") as f:
        f.write(f"\nOrientation normalization: {orientation_note}\n")

    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
