#!/usr/bin/env python3
"""
Group-level analysis of interpersonal synchrony patterns across dyads.

Scans the output directory for all dyad-level synchrony results
(synchrony_summary.csv, leader_follower.csv), stacks them, and runs
group-level statistics: one-sample tests on synchrony magnitudes,
leader-follower consistency (binomial tests), method comparisons
(Friedman test), and RQA summaries.
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
    fisher_z,
    inverse_fisher_z,
    write_csv_with_header,
    GROUPED_DIMS,
    GROUPED_COLORS,
    HAS_STATSMODELS,
)

if HAS_STATSMODELS:
    from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Group-level analysis of interpersonal synchrony across dyads"
    )
    p.add_argument("--input-dir", required=True,
                   help="Study output root containing dyad*/sub*_sub*/synchrony/")
    p.add_argument("--reduce-method",
                   choices=["grouped", "pca", "fa", "ica", "cca",
                            "grouped-pca", "cluster", "all"],
                   default="all",
                   help="Which feature reduction to analyze (default: all)")
    p.add_argument("--label", default="Synchrony",
                   help="Label for plot titles and report text (default: Synchrony)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: {input-dir}/_batch/group_synchrony/)")
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

_DYAD_PAIR_RE = re.compile(r"(sub\d+)_(sub\d+)")

# Reduction method folder names as they appear on disk
_METHOD_FOLDERS = {"grouped", "pca", "fa", "ica", "cca", "grouped-pca", "cluster", "every"}


def _parse_dyad_from_path(path):
    """Extract dyad, sub_a, sub_b from a synchrony output path."""
    parts = path.parts
    dyad = None
    sub_a = sub_b = None
    for part in parts:
        if part.startswith("dyad"):
            dyad = part
        m = _DYAD_PAIR_RE.match(part)
        if m:
            sub_a, sub_b = m.group(1), m.group(2)
    return dyad, sub_a, sub_b


def _detect_method_from_path(path):
    """Determine the reduction method from the directory structure."""
    for part in reversed(path.parts):
        if part in _METHOD_FOLDERS:
            return part
        if part == "synchrony":
            return "unknown"
    return "unknown"


def discover_synchrony_files(input_dir, reduce_method):
    """Find all synchrony_summary.csv files and group metadata."""
    all_summaries = discover_outputs(input_dir, "synchrony_summary.csv")
    results = []
    for p in all_summaries:
        if "_batch" in str(p):
            continue
        dyad, sub_a, sub_b = _parse_dyad_from_path(p)
        method = _detect_method_from_path(p)
        if reduce_method != "all" and method != reduce_method:
            continue
        results.append({
            "path": p,
            "dir": p.parent,
            "dyad": dyad or "unknown",
            "sub_a": sub_a or "unknown",
            "sub_b": sub_b or "unknown",
            "method": method,
        })

    print(f"  Discovered {len(results)} synchrony_summary.csv files")
    methods_found = sorted(set(r["method"] for r in results))
    print(f"  Methods: {', '.join(methods_found)}")
    dyads = sorted(set(r["dyad"] for r in results))
    print(f"  Dyads: {len(dyads)} ({', '.join(dyads[:10])}{'...' if len(dyads) > 10 else ''})")
    return results


def discover_leader_files(input_dir, reduce_method):
    """Find all leader_follower.csv files."""
    all_leaders = discover_outputs(input_dir, "leader_follower.csv")
    results = []
    for p in all_leaders:
        if "_batch" in str(p):
            continue
        dyad, sub_a, sub_b = _parse_dyad_from_path(p)
        method = _detect_method_from_path(p)
        if reduce_method != "all" and method != reduce_method:
            continue
        results.append({
            "path": p,
            "dyad": dyad or "unknown",
            "sub_a": sub_a or "unknown",
            "sub_b": sub_b or "unknown",
            "method": method,
        })
    print(f"  Discovered {len(results)} leader_follower.csv files")
    return results


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

def _find_col(df, candidates):
    """Return the first column name from *candidates* that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


_PEARSON_R_COLS = ["pearson_pearson_r_overall", "pearson_r_overall"]
_CONCORDANCE_COLS = ["concordance_concordance_ccc_overall", "concordance_ccc_overall"]
_RQA_PREFIX_CANDIDATES = ["rqa_rqa_", "rqa_"]
_COHERENCE_PEAK_COLS = ["coherence_coherence_peak", "coherence_peak"]
_GRANGER_F_ATOB_COLS = ["granger_granger_F_AtoB", "granger_F_AtoB"]
_GRANGER_F_BTOA_COLS = ["granger_granger_F_BtoA", "granger_F_BtoA"]

_RQA_METRICS = ["RR", "DET", "L", "Lmax", "ENTR", "LAM", "TT"]


def _get_rqa_col(df, metric):
    """Find the RQA column for a given metric (handles prefix variations)."""
    for prefix in _RQA_PREFIX_CANDIDATES:
        col = prefix + metric
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# A. Synchrony > 0?
# ---------------------------------------------------------------------------

def run_analysis_a(synch_files, out_dir):
    """Test whether synchrony is reliably greater than zero across dyads."""
    print(f"\n{'='*60}")
    print("  Analysis A: Is synchrony reliably > 0?")
    print(f"{'='*60}")

    rows = []
    for finfo in synch_files:
        try:
            df = pd.read_csv(finfo["path"], comment="#")
        except Exception as e:
            print(f"  Warning: skipping {finfo['path'].name} in {finfo['dyad']} ({e})")
            continue
        if "dimension" not in df.columns:
            print(f"  Warning: no 'dimension' column in {finfo['path'].name}, skipping")
            continue

        pearson_col = _find_col(df, _PEARSON_R_COLS)
        ccc_col = _find_col(df, _CONCORDANCE_COLS)
        coh_col = _find_col(df, _COHERENCE_PEAK_COLS)
        granger_ab_col = _find_col(df, _GRANGER_F_ATOB_COLS)
        granger_ba_col = _find_col(df, _GRANGER_F_BTOA_COLS)

        for _, row in df.iterrows():
            entry = {
                "dyad": finfo["dyad"],
                "sub_a": finfo["sub_a"],
                "sub_b": finfo["sub_b"],
                "method": finfo["method"],
                "dimension": row["dimension"],
                "pearson_r": row.get(pearson_col) if pearson_col else np.nan,
                "concordance_ccc": row.get(ccc_col) if ccc_col else np.nan,
                "coherence_peak": row.get(coh_col) if coh_col else np.nan,
                "granger_F_AtoB": row.get(granger_ab_col) if granger_ab_col else np.nan,
                "granger_F_BtoA": row.get(granger_ba_col) if granger_ba_col else np.nan,
            }
            for metric in _RQA_METRICS:
                rqa_col = _get_rqa_col(df, metric)
                entry[f"rqa_{metric}"] = row.get(rqa_col) if rqa_col else np.nan
            rows.append(entry)

    if not rows:
        print("  No synchrony data found")
        return pd.DataFrame()

    stacked = pd.DataFrame(rows)

    group_rows = []
    for (dim, method), sub in stacked.groupby(["dimension", "method"]):
        r_vals = sub["pearson_r"].dropna()
        n = len(r_vals)
        entry = {
            "dimension": dim,
            "method": method,
            "n_dyads": n,
            "mean_pearson_r": r_vals.mean() if n > 0 else np.nan,
            "sd_pearson_r": r_vals.std() if n > 1 else np.nan,
        }

        if n >= 2:
            z_vals = fisher_z(r_vals.values)
            t_stat, p_val = stats.ttest_1samp(z_vals, 0)
            cohens_d = np.mean(z_vals) / np.std(z_vals, ddof=1) if np.std(z_vals, ddof=1) > 0 else np.nan
            entry.update({
                "t_stat": t_stat,
                "p_ttest": p_val,
                "cohens_d": cohens_d,
            })
            try:
                w_stat, p_wilcox = stats.wilcoxon(r_vals)
                entry["wilcoxon_W"] = w_stat
                entry["p_wilcoxon"] = p_wilcox
            except ValueError:
                entry["wilcoxon_W"] = np.nan
                entry["p_wilcoxon"] = np.nan
        else:
            entry.update({
                "t_stat": np.nan, "p_ttest": np.nan, "cohens_d": np.nan,
                "wilcoxon_W": np.nan, "p_wilcoxon": np.nan,
            })

        ccc_vals = sub["concordance_ccc"].dropna()
        entry["mean_concordance_ccc"] = ccc_vals.mean() if len(ccc_vals) > 0 else np.nan
        coh_vals = sub["coherence_peak"].dropna()
        entry["mean_coherence_peak"] = coh_vals.mean() if len(coh_vals) > 0 else np.nan

        group_rows.append(entry)

    group_df = pd.DataFrame(group_rows)

    if HAS_STATSMODELS and group_df["p_ttest"].notna().any():
        valid = group_df["p_ttest"].notna()
        _, fdr_vals, _, _ = multipletests(group_df.loc[valid, "p_ttest"], method="fdr_bh")
        group_df["p_fdr"] = np.nan
        group_df.loc[valid, "p_fdr"] = fdr_vals
    else:
        group_df["p_fdr"] = np.nan

    header = {
        "analysis": "group_synchrony A -- synchrony > 0 test",
        "test": "one-sample t-test on Fisher-z(pearson_r) + Wilcoxon signed-rank",
        "n_files": len(synch_files),
    }
    csv_path = out_dir / "synchrony_by_dimension.csv"
    write_csv_with_header(group_df, csv_path, header)

    print("\n  Group-level synchrony by dimension:")
    for _, row in group_df.iterrows():
        sig = "*" if row.get("p_ttest", 1) < 0.05 else ""
        fdr_str = f"  p_fdr={row['p_fdr']:.4f}" if np.isfinite(row.get("p_fdr", np.nan)) else ""
        print(f"    [{row['method']:12s}] {row['dimension']:25s}  "
              f"mean_r={row['mean_pearson_r']:+.4f}  "
              f"t={row['t_stat']:+.3f}  p={row['p_ttest']:.4f}{fdr_str}  "
              f"n={row['n_dyads']}{sig}")

    return stacked


# ---------------------------------------------------------------------------
# B. Leader-follower consistency
# ---------------------------------------------------------------------------

def run_analysis_b(leader_files, out_dir):
    """Test whether leader-follower patterns are consistent across dyads."""
    print(f"\n{'='*60}")
    print("  Analysis B: Leader-follower consistency")
    print(f"{'='*60}")

    all_rows = []
    for finfo in leader_files:
        try:
            df = pd.read_csv(finfo["path"], comment="#")
        except Exception as e:
            print(f"  Warning: skipping leader_follower.csv in {finfo['dyad']} ({e})")
            continue
        df["dyad"] = finfo["dyad"]
        df["reduce_method"] = finfo["method"]
        all_rows.append(df)

    if not all_rows:
        print("  No leader-follower data found")
        return pd.DataFrame()

    stacked = pd.concat(all_rows, ignore_index=True)

    if "method" not in stacked.columns:
        print("  No 'method' column in leader_follower.csv, skipping")
        return pd.DataFrame()

    summary_rows = []
    for (dim, method), sub in stacked.groupby(["dimension", "method"]):
        n_a = (sub["leader"] == "A").sum()
        n_b = (sub["leader"] == "B").sum()
        n_sync = (sub["leader"] == "sync").sum()
        n_total = n_a + n_b + n_sync

        if n_a + n_b >= 1:
            result = stats.binomtest(max(n_a, n_b), n_a + n_b, 0.5)
            binom_p = result.pvalue
        else:
            binom_p = np.nan

        summary_rows.append({
            "dimension": dim,
            "method": method,
            "n_A_leads": n_a,
            "n_B_leads": n_b,
            "n_sync": n_sync,
            "n_total": n_total,
            "binom_p": binom_p,
        })

    summary_df = pd.DataFrame(summary_rows)
    header = {
        "analysis": "group_synchrony B -- leader-follower consistency",
        "test": "binomial test (H0: equal probability of A-lead vs B-lead)",
        "n_files": len(leader_files),
    }
    csv_path = out_dir / "leader_follower_summary.csv"
    write_csv_with_header(summary_df, csv_path, header)

    print("\n  Leader-follower summary:")
    for _, row in summary_df.iterrows():
        sig = "*" if row["binom_p"] < 0.05 else "" if np.isfinite(row["binom_p"]) else ""
        print(f"    [{row['method']:18s}] {row['dimension']:25s}  "
              f"A={row['n_A_leads']}  B={row['n_B_leads']}  "
              f"sync={row['n_sync']}  binom_p={row['binom_p']:.4f}{sig}")

    return summary_df


# ---------------------------------------------------------------------------
# C. Compare reduction methods
# ---------------------------------------------------------------------------

def run_analysis_c(stacked_a, out_dir):
    """Compare mean |pearson_r| across reduction methods using Friedman test."""
    print(f"\n{'='*60}")
    print("  Analysis C: Method comparison")
    print(f"{'='*60}")

    if stacked_a.empty or "pearson_r" not in stacked_a.columns:
        print("  No pearson_r data available for method comparison")
        return pd.DataFrame()

    per_dyad = (
        stacked_a.groupby(["dyad", "method"])["pearson_r"]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="mean_abs_pearson_r")
    )

    methods = sorted(per_dyad["method"].unique())
    if len(methods) < 2:
        print("  Fewer than 2 methods, skipping comparison")
        header = {
            "analysis": "group_synchrony C -- method comparison",
            "note": "fewer than 2 methods available",
        }
        csv_path = out_dir / "method_comparison.csv"
        write_csv_with_header(per_dyad, csv_path, header)
        return per_dyad

    pivot = per_dyad.pivot(index="dyad", columns="method", values="mean_abs_pearson_r")
    pivot_clean = pivot.dropna()

    friedman_stat = friedman_p = np.nan
    if len(pivot_clean) >= 2 and len(methods) >= 3:
        try:
            method_arrays = [pivot_clean[m].values for m in pivot_clean.columns]
            friedman_stat, friedman_p = stats.friedmanchisquare(*method_arrays)
        except Exception:
            pass
    elif len(pivot_clean) >= 2 and len(methods) == 2:
        try:
            w_stat, friedman_p = stats.wilcoxon(
                pivot_clean.iloc[:, 0], pivot_clean.iloc[:, 1]
            )
            friedman_stat = w_stat
        except ValueError:
            pass

    header = {
        "analysis": "group_synchrony C -- method comparison",
        "test": "Friedman test" if len(methods) >= 3 else "Wilcoxon signed-rank",
        "friedman_chi2": f"{friedman_stat:.4f}" if np.isfinite(friedman_stat) else "N/A",
        "friedman_p": f"{friedman_p:.4f}" if np.isfinite(friedman_p) else "N/A",
        "n_dyads_complete": len(pivot_clean),
    }
    csv_path = out_dir / "method_comparison.csv"
    write_csv_with_header(per_dyad, csv_path, header)

    print(f"\n  Mean |Pearson r| per method:")
    for method in methods:
        vals = per_dyad[per_dyad["method"] == method]["mean_abs_pearson_r"]
        print(f"    {method:15s}  M={vals.mean():.4f}  SD={vals.std():.4f}  n={len(vals)}")
    if np.isfinite(friedman_p):
        test_name = "Friedman" if len(methods) >= 3 else "Wilcoxon"
        print(f"  {test_name} test: stat={friedman_stat:.3f}, p={friedman_p:.4f}")

    return per_dyad


# ---------------------------------------------------------------------------
# D. RQA summary
# ---------------------------------------------------------------------------

def run_analysis_d(stacked_a, out_dir):
    """Descriptive statistics of RQA metrics across dyads."""
    print(f"\n{'='*60}")
    print("  Analysis D: RQA summary")
    print(f"{'='*60}")

    rqa_cols = [f"rqa_{m}" for m in _RQA_METRICS]
    available = [c for c in rqa_cols if c in stacked_a.columns
                 and stacked_a[c].notna().any()]

    if not available:
        print("  No RQA data available")
        return pd.DataFrame()

    keep_cols = ["dyad", "dimension", "method"] + available
    rqa_data = stacked_a[keep_cols].dropna(subset=available, how="all")

    summary_rows = []
    for dim, sub in rqa_data.groupby("dimension"):
        entry = {"dimension": dim, "n_dyads": sub["dyad"].nunique()}
        for col in available:
            vals = sub[col].dropna()
            if len(vals) > 0:
                entry[f"mean_{col}"] = vals.mean()
                entry[f"sd_{col}"] = vals.std()
        summary_rows.append(entry)

    summary_df = pd.DataFrame(summary_rows)
    header = {
        "analysis": "group_synchrony D -- RQA descriptive statistics",
        "metrics": ", ".join(available),
    }
    csv_path = out_dir / "rqa_summary.csv"
    write_csv_with_header(summary_df, csv_path, header)

    full_csv_path = out_dir / "rqa_all_dyads.csv"
    write_csv_with_header(rqa_data, full_csv_path, {
        "analysis": "group_synchrony D -- RQA per dyad",
    })

    print(f"\n  RQA descriptives per dimension:")
    for _, row in summary_df.iterrows():
        rr_str = f"RR={row.get('mean_rqa_RR', np.nan):.4f}" if "mean_rqa_RR" in row else ""
        det_str = f"DET={row.get('mean_rqa_DET', np.nan):.4f}" if "mean_rqa_DET" in row else ""
        lam_str = f"LAM={row.get('mean_rqa_LAM', np.nan):.4f}" if "mean_rqa_LAM" in row else ""
        print(f"    {row['dimension']:25s}  {rr_str}  {det_str}  {lam_str}  n={row['n_dyads']}")

    return summary_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

METRIC_COLORS = {
    "pearson_r": "#e63946",
    "concordance_ccc": "#457b9d",
    "coherence_peak": "#2a9d8f",
    "rqa_DET": "#e9c46a",
    "rqa_LAM": "#f4a261",
    "rqa_ENTR": "#264653",
    "granger_F_AtoB": "#6d6875",
    "granger_F_BtoA": "#b5838d",
}


def plot_synchrony_heatmap(stacked, out_dir, label):
    """Dimensions x metrics heatmap, cell = mean across dyads."""
    if stacked.empty:
        return

    metric_cols = ["pearson_r", "concordance_ccc", "coherence_peak",
                   "rqa_DET", "rqa_LAM", "rqa_ENTR"]
    available_metrics = [c for c in metric_cols if c in stacked.columns
                         and stacked[c].notna().any()]
    if not available_metrics:
        return

    methods = sorted(stacked["method"].unique())

    for method in methods:
        sub = stacked[stacked["method"] == method]
        dims = sorted(sub["dimension"].unique())
        if not dims:
            continue

        matrix = np.full((len(dims), len(available_metrics)), np.nan)
        for i, dim in enumerate(dims):
            dsub = sub[sub["dimension"] == dim]
            for j, metric in enumerate(available_metrics):
                vals = dsub[metric].dropna()
                if len(vals) > 0:
                    matrix[i, j] = vals.mean()

        fig, ax = plt.subplots(figsize=(max(8, len(available_metrics) * 1.5),
                                         max(4, len(dims) * 0.5)))
        vmax = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 1.0
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")

        for i in range(len(dims)):
            for j in range(len(available_metrics)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3f}",
                            ha="center", va="center", fontsize=8)

        ax.set_xticks(range(len(available_metrics)))
        ax.set_xticklabels(available_metrics, fontsize=9, rotation=45, ha="right")
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels(dims, fontsize=9)
        ax.set_title(f"Mean {label} Across Dyads ({method})", fontsize=13)
        fig.colorbar(im, ax=ax, label="Mean value")
        fig.tight_layout()

        suffix = f"_{method}" if len(methods) > 1 else ""
        path = out_dir / f"synchrony_heatmap{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {path}")


def plot_leader_follower(leader_summary, out_dir):
    """Stacked bar chart of A-leads / B-leads / sync per dimension."""
    if leader_summary.empty:
        return

    methods = sorted(leader_summary["method"].unique())
    dims = sorted(leader_summary["dimension"].unique())

    for lf_method in methods:
        sub = leader_summary[leader_summary["method"] == lf_method]
        if sub.empty:
            continue

        sub_dims = sub.set_index("dimension").reindex(dims).fillna(0)

        fig, ax = plt.subplots(figsize=(10, max(4, len(dims) * 0.4)))
        y_pos = np.arange(len(dims))
        width = sub_dims["n_A_leads"].values.astype(float)
        ax.barh(y_pos, width, color="#e63946", label="A leads", edgecolor="white")
        left = width.copy()
        width_b = sub_dims["n_B_leads"].values.astype(float)
        ax.barh(y_pos, width_b, left=left, color="#457b9d", label="B leads", edgecolor="white")
        left += width_b
        width_s = sub_dims["n_sync"].values.astype(float)
        ax.barh(y_pos, width_s, left=left, color="#adb5bd", label="Sync", edgecolor="white")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(dims, fontsize=9)
        ax.set_xlabel("Number of Dyads", fontsize=11)
        ax.set_title(f"Leader-Follower Patterns ({lf_method})", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="x")
        fig.tight_layout()

        suffix = f"_{lf_method}" if len(methods) > 1 else ""
        path = out_dir / f"leader_follower_barplot{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {path}")


def plot_method_comparison(per_dyad_method, out_dir, label):
    """Box plots of mean |pearson_r| per dyad, one box per reduction method."""
    if per_dyad_method.empty:
        return

    methods = sorted(per_dyad_method["method"].unique())
    if len(methods) < 2:
        print("  Skipping method comparison plot (fewer than 2 methods)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [per_dyad_method[per_dyad_method["method"] == m]["mean_abs_pearson_r"].values
            for m in methods]
    bp = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.5)
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
              "#264653", "#6d6875", "#b5838d"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Mean |Pearson r| per Dyad", fontsize=11)
    ax.set_xlabel("Reduction Method", fontsize=11)
    ax.set_title(f"Reduction Method Comparison: {label} Strength", fontsize=13)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    path = out_dir / "method_comparison_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_cross_dyad_pearson(stacked, out_dir, label):
    """Dot plot of each dyad's Pearson r per dimension, lines connect same dyad."""
    if stacked.empty or "pearson_r" not in stacked.columns:
        return

    methods = sorted(stacked["method"].unique())

    for method in methods:
        sub = stacked[stacked["method"] == method]
        dims = sorted(sub["dimension"].unique())
        dyads = sorted(sub["dyad"].unique())

        if not dims or not dyads:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(dims) * 1.2), 6))
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(dyads), 1)))

        for d_idx, dyad in enumerate(dyads):
            dsub = sub[sub["dyad"] == dyad]
            x_vals = []
            y_vals = []
            for dim in dims:
                dim_row = dsub[dsub["dimension"] == dim]
                if not dim_row.empty:
                    x_vals.append(dims.index(dim))
                    y_vals.append(dim_row["pearson_r"].iloc[0])
            if x_vals:
                ax.plot(x_vals, y_vals, marker="o", markersize=5, alpha=0.6,
                        color=cmap[d_idx % len(cmap)], linewidth=1.2, label=dyad)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dims, fontsize=9, rotation=45, ha="right")
        ax.set_ylabel("Pearson r", fontsize=11)
        ax.set_title(f"Cross-Dyad {label} ({method})", fontsize=13)
        if len(dyads) <= 15:
            ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        suffix = f"_{method}" if len(methods) > 1 else ""
        path = out_dir / f"cross_dyad_pearson_r{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# Individual differences (questionnaire)
# ---------------------------------------------------------------------------

def run_moderator_analysis(stacked, subjects_path, scale_cols, out_dir):
    """Correlate dyad-level synchrony with covariate scales (Spearman)."""
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

    mod_rows = []
    for (dim, method), sub in stacked.groupby(["dimension", "method"]):
        for _, row in sub.iterrows():
            dyad = row["dyad"]
            r_val = row.get("pearson_r", np.nan)
            if not np.isfinite(r_val):
                continue

            q_a = q_df[q_df["subject"] == row["sub_a"]]
            q_b = q_df[q_df["subject"] == row["sub_b"]]
            if q_a.empty and q_b.empty:
                continue
            dyad_scores = {}
            for scale in scale_cols:
                vals = []
                if not q_a.empty and np.isfinite(q_a[scale].iloc[0]):
                    vals.append(q_a[scale].iloc[0])
                if not q_b.empty and np.isfinite(q_b[scale].iloc[0]):
                    vals.append(q_b[scale].iloc[0])
                if vals:
                    dyad_scores[scale] = np.mean(vals)

            for scale, score in dyad_scores.items():
                mod_rows.append({
                    "dimension": dim,
                    "method": method,
                    "dyad": dyad,
                    "pearson_r": r_val,
                    "scale": scale,
                    "dyad_mean_score": score,
                })

    if not mod_rows:
        print("  No valid moderator data (no questionnaire matches)")
        return

    mod_df = pd.DataFrame(mod_rows)

    corr_rows = []
    for (dim, method, scale), sub in mod_df.groupby(["dimension", "method", "scale"]):
        if len(sub) < 4:
            continue
        rho, p_val = stats.spearmanr(sub["pearson_r"], sub["dyad_mean_score"])
        corr_rows.append({
            "dimension": dim,
            "method": method,
            "scale": scale,
            "spearman_r": rho,
            "p_value": p_val,
            "n": len(sub),
        })

    if not corr_rows:
        print("  Too few overlapping dyads for moderator analysis")
        return

    corr_df = pd.DataFrame(corr_rows)
    header = {
        "analysis": "group_synchrony -- moderator correlations",
        "test": "Spearman rank correlation (dyad mean questionnaire score x pearson_r)",
        "caveat": "N is small; moderator analyses are exploratory",
    }
    csv_path = out_dir / "moderator_correlations.csv"
    write_csv_with_header(corr_df, csv_path, header)

    print(f"\n  Moderator correlations ({len(corr_df)} pairs tested):")
    top = corr_df.reindex(corr_df["spearman_r"].abs().sort_values(ascending=False).index).head(6)
    for _, row in top.iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        print(f"    {row['dimension']:20s} x {row['scale']:20s}  "
              f"rho={row['spearman_r']:+.3f}  p={row['p_value']:.4f}  n={row['n']}{sig}")

    _plot_moderator_scatter(top, mod_df, out_dir)


def _plot_moderator_scatter(top_mods, mod_df, out_dir):
    """Scatter matrix for top 6 moderator correlations."""
    n_panels = min(len(top_mods), 6)
    if n_panels == 0:
        return

    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, (_, row) in enumerate(top_mods.head(n_panels).iterrows()):
        ax = axes[idx // ncols, idx % ncols]
        dim, method, scale = row["dimension"], row["method"], row["scale"]

        sub = mod_df[(mod_df["dimension"] == dim) & (mod_df["method"] == method)
                     & (mod_df["scale"] == scale)].dropna(subset=["pearson_r", "dyad_mean_score"])

        ax.scatter(sub["dyad_mean_score"], sub["pearson_r"], alpha=0.7, s=40, color="#457b9d")
        if len(sub) >= 2:
            z = np.polyfit(sub["dyad_mean_score"], sub["pearson_r"], 1)
            x_line = np.linspace(sub["dyad_mean_score"].min(), sub["dyad_mean_score"].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), color="#e63946", linewidth=1.5,
                    linestyle="--")
        ax.set_xlabel(f"{scale} (dyad mean)", fontsize=9)
        ax.set_ylabel(f"Pearson r ({dim})", fontsize=9)
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

def print_summary_report(stacked, out_dir, label):
    """Print terminal summary with reference benchmarks."""
    print(f"\n{'='*60}")
    print("  Summary Report")
    print(f"{'='*60}")

    if not stacked.empty and "pearson_r" in stacked.columns:
        r_vals = stacked["pearson_r"].dropna()
        print(f"\n  Overall Pearson r across all dyads and dimensions:")
        print(f"    Mean = {r_vals.mean():.4f}, SD = {r_vals.std():.4f}, "
              f"Median = {r_vals.median():.4f}")
        print(f"    Range = [{r_vals.min():.4f}, {r_vals.max():.4f}]")

        ccc_vals = stacked["concordance_ccc"].dropna()
        if len(ccc_vals) > 0:
            print(f"\n  Overall Concordance (CCC):")
            print(f"    Mean = {ccc_vals.mean():.4f}, SD = {ccc_vals.std():.4f}")

        rqa_det = stacked.get("rqa_DET")
        if rqa_det is not None:
            det_vals = rqa_det.dropna()
            if len(det_vals) > 0:
                print(f"\n  Overall RQA Determinism (DET):")
                print(f"    Mean = {det_vals.mean():.4f}, SD = {det_vals.std():.4f}")

            rr_vals = stacked.get("rqa_RR")
            if rr_vals is not None:
                rr_clean = rr_vals.dropna()
                if len(rr_clean) > 0:
                    print(f"\n  Overall RQA Recurrence Rate (RR):")
                    print(f"    Mean = {rr_clean.mean():.4f} ({rr_clean.mean()*100:.1f}%)")

    print(f"\n  Reference benchmarks:")
    print(f"    Interpersonal behavioral synchrony:")
    print(f"      r = 0.10-0.40 (Bernieri & Rosenthal, 1991)")
    print(f"    RQA %recurrence:")
    print(f"      5-15% typical (Dale & Spivey, 2006)")
    print(f"    Concordance (CCC):")
    print(f"      0.2-0.5 (Boker & Rotondo, 2002)")

    report_lines = [
        f"Group {label} Summary Report",
        "=" * 40,
        "",
    ]
    if not stacked.empty:
        report_lines.extend([
            f"N dyads: {stacked['dyad'].nunique()}",
            f"Methods: {', '.join(sorted(stacked['method'].unique()))}",
            f"Dimensions: {', '.join(sorted(stacked['dimension'].unique()))}",
            "",
        ])
        r_vals = stacked["pearson_r"].dropna()
        if len(r_vals) > 0:
            report_lines.append(
                f"Overall Pearson r: M = {r_vals.mean():.4f}, SD = {r_vals.std():.4f}"
            )

    report_lines.extend([
        "",
        "Reference benchmarks:",
        "  Interpersonal synchrony: r = 0.10-0.40 (Bernieri & Rosenthal, 1991)",
        "  RQA %recurrence: 5-15% typical (Dale & Spivey, 2006)",
        "  Concordance (CCC): 0.2-0.5 (Boker & Rotondo, 2002)",
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

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Error: --input-dir does not exist: {input_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "_batch" / "group_synchrony"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = list(out_dir.glob("synchrony_by_dimension.csv"))
        if existing:
            print(f"  Output files already exist in {out_dir}")
            print(f"  Use --overwrite to replace them")
            sys.exit(0)

    print(f"Group-level {label} analysis")
    print(f"  Input: {input_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Label: {label}")
    print(f"  Method filter: {args.reduce_method}")

    print(f"\nDiscovering synchrony files...")
    synch_files = discover_synchrony_files(input_dir, args.reduce_method)
    if not synch_files:
        sys.exit("Error: no synchrony_summary.csv files found. Run synchronize.py first.")

    leader_files = discover_leader_files(input_dir, args.reduce_method)

    stacked_a = run_analysis_a(synch_files, out_dir)
    leader_summary = run_analysis_b(leader_files, out_dir)
    per_dyad_method = run_analysis_c(stacked_a, out_dir)
    run_analysis_d(stacked_a, out_dir)

    print(f"\nGenerating plots...")
    plot_synchrony_heatmap(stacked_a, out_dir, label)
    plot_leader_follower(leader_summary, out_dir)
    plot_method_comparison(per_dyad_method, out_dir, label)
    plot_cross_dyad_pearson(stacked_a, out_dir, label)

    if args.subjects and args.covariates:
        from analysis.utils import load_subjects_df, get_covariate_columns
        s_df = load_subjects_df(args.subjects)
        cov_cols = get_covariate_columns(s_df, args.covariates)
        if cov_cols:
            run_moderator_analysis(stacked_a, args.subjects, cov_cols, out_dir)

    print_summary_report(stacked_a, out_dir, label)

    orientation_note = "applied (subjects.csv provided)" if args.subjects else "not applied"
    print(f"  Orientation normalization: {orientation_note}")
    with open(out_dir / "summary_report.txt", "a") as f:
        f.write(f"\nOrientation normalization: {orientation_note}\n")

    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
