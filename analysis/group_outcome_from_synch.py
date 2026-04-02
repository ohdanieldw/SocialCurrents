#!/usr/bin/env python3
"""
Group-level analysis: does interpersonal synchrony predict outcome ratings?

Scans the output directory for all dyad-level trust_from_synch correlation
results (produced by correlate.py with synchrony_timeseries.csv as input),
stacks them, and runs group-level statistics.

Each observation is one rater-direction (not averaged across raters).
A dyad with two raters contributes two observations.

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
    fisher_z,
    inverse_fisher_z,
    parse_csv_header,
    write_csv_with_header,
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
        description="Group-level analysis: does synchrony predict outcome ratings?"
    )
    p.add_argument("--input-dir", required=True,
                   help="Study output root containing dyad*/sub*_sub*/trust_from_synch/")
    p.add_argument("--reduce-method",
                   choices=["grouped", "pca", "fa", "ica", "cca",
                            "grouped-pca", "cluster", "all"],
                   default="all",
                   help="Which synchrony reduction method to analyze (default: all)")
    p.add_argument("--label", default="Trustworthiness",
                   help="What the outcome measures, used in plot titles and filenames "
                        "(default: Trustworthiness)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: {input-dir}/_batch/group_outcome_from_synch/)")
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

_METHOD_FOLDERS = {"grouped", "pca", "fa", "ica", "cca", "grouped-pca", "cluster"}

_RATER_SUFFIX_RE = re.compile(
    r"^(?P<method>grouped|pca|fa|ica|cca|grouped-pca|cluster)_(?P<rater>sub\d+)$"
)

_TARGET_FILE_RE = re.compile(r"(?:dyad\d+)_(?P<target>sub\d+)_trustworthiness")


def _parse_trust_synch_entry(path):
    """Parse a trust_from_synch correlation.csv into metadata."""
    meta = parse_csv_header(path)

    parts = path.parts
    dyad = sub_a = sub_b = None
    for part in parts:
        if part.startswith("dyad"):
            dyad = part
        m = _DYAD_PAIR_RE.match(part)
        if m:
            sub_a, sub_b = m.group(1), m.group(2)

    method = "unknown"
    rater = meta.get("rater") if meta.get("rater") not in (None, "", "(not specified)") else None
    target_id = meta.get("target_id") if meta.get("target_id") not in (None, "", "(not specified)") else None

    parent_name = path.parent.name
    rater_match = _RATER_SUFFIX_RE.match(parent_name)
    if rater_match:
        method = rater_match.group("method")
        if not rater:
            rater = rater_match.group("rater")
    elif parent_name in _METHOD_FOLDERS:
        method = parent_name

    if not target_id:
        target_file = meta.get("target_file", "")
        tf_match = _TARGET_FILE_RE.search(target_file)
        if tf_match:
            target_id = tf_match.group("target")

    if not rater and target_id and sub_a and sub_b:
        rater = sub_b if target_id == sub_a else sub_a

    return {
        "path": path,
        "dyad": dyad or "unknown",
        "sub_a": sub_a or "unknown",
        "sub_b": sub_b or "unknown",
        "rater": rater or "unknown",
        "target_id": target_id or "unknown",
        "method": method,
        "meta": meta,
    }


def discover_trust_synch_files(input_dir, reduce_method):
    """Find all trust_from_synch correlation.csv files."""
    all_csvs = discover_outputs(input_dir, "correlation.csv")
    results = []
    for p in all_csvs:
        if "trust_from_synch" not in str(p):
            continue
        if "_batch" in str(p):
            continue
        info = _parse_trust_synch_entry(p)
        if reduce_method != "all" and info["method"] != reduce_method:
            continue
        results.append(info)

    print(f"  Discovered {len(results)} trust_from_synch correlation files")
    methods_found = sorted(set(r["method"] for r in results))
    print(f"  Methods: {', '.join(methods_found)}")
    dyads = sorted(set(r["dyad"] for r in results))
    print(f"  Dyads: {len(dyads)} ({', '.join(dyads[:10])}{'...' if len(dyads) > 10 else ''})")
    n_obs = len(results)
    n_unique_raters = len(set((r["dyad"], r["rater"]) for r in results))
    print(f"  Observations (dyad x rater): {n_obs}")
    return results


# ---------------------------------------------------------------------------
# Core stacking
# ---------------------------------------------------------------------------

def stack_all_peaks(corr_files):
    """Load all correlation CSVs and extract peak |r| per dimension."""
    rows = []
    for finfo in corr_files:
        try:
            df = pd.read_csv(finfo["path"], comment="#")
        except Exception as e:
            print(f"  Warning: skipping {finfo['path']} ({e})")
            continue

        if "dimension" not in df.columns or "correlation" not in df.columns:
            print(f"  Warning: unexpected columns in {finfo['path'].name}, skipping")
            continue

        peaks = extract_peak_correlation(df)
        for _, peak in peaks.iterrows():
            rows.append({
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
        return pd.DataFrame()

    stacked = pd.DataFrame(rows)
    print(f"  Stacked {len(stacked)} peak correlations across "
          f"{stacked['dyad'].nunique()} dyads, "
          f"{stacked['method'].nunique()} methods, "
          f"{stacked['dimension'].nunique()} dimensions")
    return stacked


# ---------------------------------------------------------------------------
# Analysis: synchrony-outcome group tests
# ---------------------------------------------------------------------------

def run_group_tests(stacked, out_dir, label, slug):
    """Per synchrony metric: one-sample t-test on Fisher-z(peak_r)."""
    print(f"\n{'='*60}")
    print(f"  Synchrony-{label} group-level tests")
    print(f"{'='*60}")

    group_rows = []
    for (dim, method), sub in stacked.groupby(["dimension", "method"]):
        r_vals = sub["peak_r"].dropna()
        n = len(r_vals)
        entry = {
            "dimension": dim,
            "method": method,
            "n_obs": n,
            "mean_r": r_vals.mean() if n > 0 else np.nan,
            "sd_r": r_vals.std() if n > 1 else np.nan,
            "mean_lag": sub["peak_lag"].mean(),
        }

        if n >= 2:
            z_vals = fisher_z(r_vals.values)
            t_stat, p_val = stats.ttest_1samp(z_vals, 0)
            sd_z = np.std(z_vals, ddof=1)
            cohens_d = np.mean(z_vals) / sd_z if sd_z > 0 else np.nan
            entry.update({
                "t_stat": t_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
            })
        else:
            entry.update({"t_stat": np.nan, "p_value": np.nan, "cohens_d": np.nan})

        group_rows.append(entry)

    group_df = pd.DataFrame(group_rows)

    if HAS_STATSMODELS and group_df["p_value"].notna().any():
        valid = group_df["p_value"].notna()
        _, fdr_vals, _, _ = multipletests(group_df.loc[valid, "p_value"], method="fdr_bh")
        group_df["p_fdr"] = np.nan
        group_df.loc[valid, "p_fdr"] = fdr_vals
    else:
        group_df["p_fdr"] = np.nan

    header = {
        "analysis": f"group_outcome_from_synch -- synchrony predicts {label}",
        "test": "one-sample t-test on Fisher-z(peak_r), FDR corrected",
        "unit": "one observation = one rater-direction within a dyad",
        "label": label,
    }
    csv_path = out_dir / f"{slug}_from_synch_summary.csv"
    write_csv_with_header(group_df, csv_path, header)

    print("\n  Top dimensions by effect size:")
    top = group_df.dropna(subset=["cohens_d"]).reindex(
        group_df["cohens_d"].abs().sort_values(ascending=False).index
    ).head(15)
    for _, row in top.iterrows():
        sig = "*" if row.get("p_fdr", 1) < 0.05 else ""
        fdr_str = f"  p_fdr={row['p_fdr']:.4f}" if np.isfinite(row.get("p_fdr", np.nan)) else ""
        print(f"    [{row['method']:12s}] {row['dimension']:40s}  "
              f"mean_r={row['mean_r']:+.4f}  t={row['t_stat']:+.3f}  "
              f"p={row['p_value']:.4f}{fdr_str}  d={row['cohens_d']:+.3f}  "
              f"n={row['n_obs']}{sig}")

    return group_df


# ---------------------------------------------------------------------------
# Analysis: method comparison
# ---------------------------------------------------------------------------

def run_method_comparison(stacked, out_dir, label):
    """Compare mean |peak_r| across synchrony reduction methods."""
    print(f"\n{'='*60}")
    print("  Method comparison")
    print(f"{'='*60}")

    obs_key = stacked.apply(lambda r: f"{r['dyad']}_{r['rater']}", axis=1)
    stacked = stacked.copy()
    stacked["obs_id"] = obs_key

    per_obs = (
        stacked.groupby(["obs_id", "method"])["peak_r"]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="mean_abs_r")
    )

    methods = sorted(per_obs["method"].unique())

    method_stats = []
    for method in methods:
        vals = per_obs[per_obs["method"] == method]["mean_abs_r"]
        method_stats.append({
            "method": method,
            "n_obs": len(vals),
            "mean_abs_r": vals.mean(),
            "sd_abs_r": vals.std(),
            "median_abs_r": vals.median(),
        })
    method_df = pd.DataFrame(method_stats)

    friedman_stat = friedman_p = np.nan
    if len(methods) >= 2:
        pivot = per_obs.pivot(index="obs_id", columns="method", values="mean_abs_r")
        pivot_clean = pivot.dropna()
        if len(pivot_clean) >= 2:
            if len(methods) >= 3:
                try:
                    arrays = [pivot_clean[m].values for m in pivot_clean.columns]
                    friedman_stat, friedman_p = stats.friedmanchisquare(*arrays)
                except Exception:
                    pass
            else:
                try:
                    w, friedman_p = stats.wilcoxon(
                        pivot_clean.iloc[:, 0], pivot_clean.iloc[:, 1]
                    )
                    friedman_stat = w
                except ValueError:
                    pass

    method_df["friedman_stat"] = friedman_stat
    method_df["friedman_p"] = friedman_p

    header = {
        "analysis": f"group_outcome_from_synch -- method comparison ({label})",
        "test": "Friedman test" if len(methods) >= 3 else "Wilcoxon signed-rank",
        "friedman_stat": f"{friedman_stat:.4f}" if np.isfinite(friedman_stat) else "N/A",
        "friedman_p": f"{friedman_p:.4f}" if np.isfinite(friedman_p) else "N/A",
    }
    csv_path = out_dir / "method_comparison.csv"
    write_csv_with_header(method_df, csv_path, header)

    print("\n  Mean |peak_r| per method:")
    for _, row in method_df.iterrows():
        print(f"    {row['method']:15s}  M={row['mean_abs_r']:.4f}  "
              f"SD={row['sd_abs_r']:.4f}  n={row['n_obs']}")
    if np.isfinite(friedman_p):
        test_name = "Friedman" if len(methods) >= 3 else "Wilcoxon"
        print(f"  {test_name} test: stat={friedman_stat:.3f}, p={friedman_p:.4f}")

    return per_obs


# ---------------------------------------------------------------------------
# Analysis: lag direction
# ---------------------------------------------------------------------------

def run_lag_analysis(stacked, out_dir, label):
    """Analyze whether peak lags are predominantly positive or negative."""
    print(f"\n{'='*60}")
    print("  Lag direction analysis")
    print(f"{'='*60}")

    lags = stacked["peak_lag"].dropna()
    if len(lags) < 2:
        print("  Too few observations for lag analysis")
        return

    n_pos = (lags > 0).sum()
    n_neg = (lags < 0).sum()
    n_zero = (lags == 0).sum()
    result = stats.binomtest(max(n_pos, n_neg), n_pos + n_neg, 0.5)
    binom_p = result.pvalue

    print(f"\n  Peak lag distribution:")
    print(f"    Mean = {lags.mean():.2f}s, Median = {lags.median():.2f}s, SD = {lags.std():.2f}s")
    print(f"    Positive (synch precedes {label} change): {n_pos}")
    print(f"    Negative ({label} change precedes synch): {n_neg}")
    print(f"    Zero: {n_zero}")
    print(f"    Binomial test (directional bias): p = {binom_p:.4f}")

    t_stat, t_p = stats.ttest_1samp(lags, 0)
    print(f"    One-sample t-test (lag != 0): t={t_stat:.3f}, p={t_p:.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_forest(stacked, summary_df, out_dir, label):
    """Forest plot for top synchrony metrics showing per-observation peak r."""
    if stacked.empty or summary_df.empty:
        return

    top = summary_df.dropna(subset=["cohens_d"]).reindex(
        summary_df["cohens_d"].abs().sort_values(ascending=False).index
    ).head(10)

    if top.empty:
        return

    n_panels = len(top)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.2 * n_panels), squeeze=False)

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i, 0]
        dim = row["dimension"]
        method = row["method"]
        sub = stacked[(stacked["dimension"] == dim) & (stacked["method"] == method)]
        sub = sub.sort_values("peak_r")
        n = len(sub)
        y_pos = np.arange(n)

        labels = [f"{r['dyad']}:{r['rater']}" for _, r in sub.iterrows()]
        ax.scatter(sub["peak_r"], y_pos, color="#457b9d", s=35, zorder=5, alpha=0.8)

        mean_r = sub["peak_r"].mean()
        ax.scatter(mean_r, n + 0.5, marker="D", color="#e63946", s=70, zorder=10,
                   edgecolors="black", linewidths=0.8)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(list(y_pos) + [n + 0.5])
        ax.set_yticklabels(labels + ["Group mean"], fontsize=7)
        ax.set_xlabel("Peak r", fontsize=9)
        sig_str = "*" if row.get("p_fdr", 1) < 0.05 else ""
        ax.set_title(f"{dim} [{method}] (d={row['cohens_d']:+.2f}){sig_str}",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle(f"Forest Plot: Synchrony-{label} Correlations (Top Metrics)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "forest_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_method_comparison(per_obs, out_dir, label):
    """Bar plot of mean |peak_r| per method."""
    if per_obs.empty:
        return

    methods = sorted(per_obs["method"].unique())
    if len(methods) < 2:
        print("  Skipping method comparison plot (fewer than 2 methods)")
        return

    method_means = per_obs.groupby("method")["mean_abs_r"].agg(["mean", "std", "count"])
    method_means = method_means.reindex(methods)
    se = method_means["std"] / np.sqrt(method_means["count"])

    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
              "#264653", "#6d6875", "#b5838d"]

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), 5))
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, method_means["mean"], yerr=se, capsize=4,
                  color=colors[:len(methods)], alpha=0.7, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Mean |Peak r|", fontsize=11)
    ax.set_title(f"Which Synchrony Method Best Predicts {label}?", fontsize=13)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    path = out_dir / "method_comparison_barplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_lag_histogram(stacked, out_dir, label):
    """Histogram of peak lags with directional annotations."""
    lags = stacked["peak_lag"].dropna()
    if lags.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lags, bins=40, density=True, alpha=0.5, color="#457b9d", edgecolor="white")

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

    n_pos = (lags > 0).sum()
    n_neg = (lags < 0).sum()
    ax.text(0.02, 0.95,
            f"Positive (synch precedes {label} change): {n_pos}\n"
            f"Negative ({label} change precedes synch): {n_neg}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Peak Lag (seconds)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Distribution of Peak Lags: Synchrony-{label} Correlations", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = out_dir / "lag_direction_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# Moderator analysis
# ---------------------------------------------------------------------------

def run_moderator_analysis(stacked, subjects_path, scale_cols, out_dir, label):
    """Correlate rater and target covariate scores with peak_r."""
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

    grouped_dims = stacked.groupby(["dimension", "method"]).filter(
        lambda x: len(x) >= 4
    )
    if grouped_dims.empty:
        print("  Too few observations per dimension for moderator analysis")
        return

    dim_method_pairs = grouped_dims[["dimension", "method"]].drop_duplicates()

    mod_rows = []
    for _, dm in dim_method_pairs.iterrows():
        dim, method = dm["dimension"], dm["method"]
        sub = stacked[(stacked["dimension"] == dim) & (stacked["method"] == method)].copy()

        for role, id_col in [("rater", "rater"), ("target", "target_id")]:
            merged = sub.merge(
                q_df[["subject"] + scale_cols],
                left_on=id_col, right_on="subject", how="inner",
            )
            if len(merged) < 4:
                continue

            for scale in scale_cols:
                valid = merged[["peak_r", scale]].dropna()
                if len(valid) < 4:
                    continue
                rho, p_val = stats.spearmanr(valid["peak_r"], valid[scale])
                mod_rows.append({
                    "dimension": dim,
                    "method": method,
                    "role": role,
                    "scale": scale,
                    "spearman_r": rho,
                    "p_value": p_val,
                    "n": len(valid),
                })

    if not mod_rows:
        print("  No valid moderator correlations (too few overlapping subjects)")
        return

    mod_df = pd.DataFrame(mod_rows)
    header = {
        "analysis": f"group_outcome_from_synch -- moderator correlations ({label})",
        "test": "Spearman rank correlation (questionnaire score x peak_r)",
        "caveat": "N is small; moderator analyses are exploratory",
        "roles": f"rater = does rater empathy moderate synch-{label} link; "
                 f"target = does target personality moderate how synchrony predicts {label}",
    }
    csv_path = out_dir / "moderator_correlations.csv"
    write_csv_with_header(mod_df, csv_path, header)

    print(f"\n  Moderator correlations ({len(mod_df)} pairs tested):")
    top = mod_df.reindex(mod_df["spearman_r"].abs().sort_values(ascending=False).index).head(8)
    for _, row in top.iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        print(f"    [{row['role']:6s}] {row['dimension']:30s} x {row['scale']:15s}  "
              f"rho={row['spearman_r']:+.3f}  p={row['p_value']:.4f}  n={row['n']}{sig}")

    _plot_moderator_scatter(top, stacked, q_df, scale_cols, out_dir)


def _plot_moderator_scatter(top_mods, stacked, q_df, scale_cols, out_dir):
    """Scatter matrix for top moderator correlations."""
    n_panels = min(len(top_mods), 6)
    if n_panels == 0:
        return

    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, (_, row) in enumerate(top_mods.head(n_panels).iterrows()):
        ax = axes[idx // ncols, idx % ncols]
        dim, method, role, scale = row["dimension"], row["method"], row["role"], row["scale"]

        sub = stacked[(stacked["dimension"] == dim) & (stacked["method"] == method)].copy()
        id_col = "rater" if role == "rater" else "target_id"
        merged = sub.merge(
            q_df[["subject", scale]], left_on=id_col, right_on="subject", how="inner",
        )
        valid = merged.dropna(subset=["peak_r", scale])

        ax.scatter(valid[scale], valid["peak_r"], alpha=0.7, s=40, color="#457b9d")
        if len(valid) >= 2:
            z = np.polyfit(valid[scale], valid["peak_r"], 1)
            x_line = np.linspace(valid[scale].min(), valid[scale].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), color="#e63946", linewidth=1.5,
                    linestyle="--")
        ax.set_xlabel(f"{scale} ({role})", fontsize=9)
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

def print_summary_report(stacked, summary_df, out_dir, label):
    """Print terminal summary with reference benchmarks and save report."""
    print(f"\n{'='*60}")
    print("  Summary Report")
    print(f"{'='*60}")

    report_lines = [
        f"Group {label}-from-Synchrony Summary Report",
        "=" * 40,
        "",
    ]

    if not stacked.empty:
        r_abs = stacked["peak_r"].abs()
        lags = stacked["peak_lag"].dropna()

        print(f"\n  Overall |peak_r| across all observations:")
        print(f"    Mean = {r_abs.mean():.4f}, SD = {r_abs.std():.4f}, "
              f"Median = {r_abs.median():.4f}")
        print(f"    Range = [{r_abs.min():.4f}, {r_abs.max():.4f}]")

        print(f"\n  Peak lag distribution:")
        print(f"    Mean = {lags.mean():.2f}s, SD = {lags.std():.2f}s, "
              f"Median = {lags.median():.2f}s")

        report_lines.extend([
            f"N dyads: {stacked['dyad'].nunique()}",
            f"N observations (dyad x rater): {stacked.groupby(['dyad', 'rater']).ngroups if not stacked.empty else 0}",
            f"Methods: {', '.join(sorted(stacked['method'].unique()))}",
            f"Dimensions: {stacked['dimension'].nunique()}",
            "",
            f"Overall |peak_r|: M = {r_abs.mean():.4f}, SD = {r_abs.std():.4f}",
            f"Peak lag: M = {lags.mean():.2f}s, SD = {lags.std():.2f}s",
            "",
        ])

        if not summary_df.empty:
            sig = summary_df[summary_df.get("p_fdr", pd.Series(dtype=float)) < 0.05]
            report_lines.append(f"FDR-significant dimensions: {len(sig)}")
            report_lines.append("")

    print(f"\n  Reference benchmarks:")
    print(f"    Synchrony-outcome correlations:")
    print(f"      Typical r = 0.10-0.35 for behavioral synchrony predicting rapport/ratings")
    print(f"      (Bernieri & Rosenthal, 1991; Ramseyer & Tschacher, 2011)")
    print(f"    Expected peak lag: 2-8 seconds")
    print(f"      (rating dial perceptual delay + synchrony window smoothing)")

    report_lines.extend([
        "Reference benchmarks:",
        "  Synchrony-outcome: r = 0.10-0.35 (Bernieri & Rosenthal, 1991; Ramseyer & Tschacher, 2011)",
        "  Expected peak lag: 2-8s (perceptual delay + window smoothing)",
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

    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "_batch" / "group_outcome_from_synch"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = list(out_dir.glob("*_from_synch_*.csv"))
        if existing:
            print(f"  Output files already exist in {out_dir}")
            print(f"  Use --overwrite to replace them")
            sys.exit(0)

    print(f"Group-level analysis: does synchrony predict {label}?")
    print(f"  Input: {input_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Label: {label}")
    print(f"  Method filter: {args.reduce_method}")

    print(f"\nDiscovering trust_from_synch files...")
    corr_files = discover_trust_synch_files(input_dir, args.reduce_method)
    if not corr_files:
        sys.exit("Error: no trust_from_synch correlation files found. "
                 "Run correlate.py on synchrony_timeseries.csv first.")

    print(f"\nStacking peak correlations...")
    stacked = stack_all_peaks(corr_files)
    if stacked.empty:
        sys.exit("Error: no valid peak correlations extracted")

    header = {
        "analysis": f"group_outcome_from_synch -- all observations stacked ({label})",
        "label": label,
        "n_files": len(corr_files),
        "n_dyads": stacked["dyad"].nunique(),
        "methods": ",".join(sorted(stacked["method"].unique())),
    }
    csv_path = out_dir / f"{slug}_from_synch_all.csv"
    write_csv_with_header(stacked, csv_path, header)

    summary_df = run_group_tests(stacked, out_dir, label, slug)
    per_obs = run_method_comparison(stacked, out_dir, label)
    run_lag_analysis(stacked, out_dir, label)

    print(f"\nGenerating plots...")
    plot_forest(stacked, summary_df, out_dir, label)
    plot_method_comparison(per_obs, out_dir, label)
    plot_lag_histogram(stacked, out_dir, label)

    if args.subjects and args.covariates:
        from analysis.utils import load_subjects_df, get_covariate_columns
        s_df = load_subjects_df(args.subjects)
        cov_cols = get_covariate_columns(s_df, args.covariates)
        if cov_cols:
            run_moderator_analysis(stacked, args.subjects, cov_cols, out_dir, label)

    print_summary_report(stacked, summary_df, out_dir, label)

    orientation_note = "applied (subjects.csv provided)" if args.subjects else "not applied"
    print(f"  Orientation normalization: {orientation_note}")
    with open(out_dir / "summary_report.txt", "a") as f:
        f.write(f"\nOrientation normalization: {orientation_note}\n")

    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
