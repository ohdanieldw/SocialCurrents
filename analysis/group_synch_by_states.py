#!/usr/bin/env python3
"""
Group-level analysis: do behavioral states differ in synchrony?

Scans the output directory for all dyad-level synch_by_states results
(map_states_summary.csv and map_states_pairwise.csv), computes per-dyad
effect sizes (eta-squared), and tests whether the state effect on
synchrony is reliably greater than zero across dyads.

States are NOT aligned across dyads (State 1 means different things in
different dyads), so analysis focuses on within-dyad effect sizes rather
than cross-dyad state comparisons.
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
    write_csv_with_header,
    HAS_STATSMODELS,
)

if HAS_STATSMODELS:
    from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Group-level analysis: do behavioral states differ in synchrony?"
    )
    p.add_argument("--input-dir", required=True,
                   help="Study output root containing "
                        "dyad*/sub*_sub*/synch_by_states/")
    p.add_argument("--reduce-method",
                   choices=["grouped", "pca", "fa", "ica", "cca",
                            "grouped-pca", "cluster", "all"],
                   default="all",
                   help="Which synchrony reduction method to analyze (default: all)")
    p.add_argument("--label", default="Synchrony",
                   help="Label for plot titles and report text (default: Synchrony)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: {input-dir}/_batch/group_synch_by_states/)")
    p.add_argument("--subjects", default=None,
                   help="Path to subjects.csv with orientation, demographics, and "
                        "questionnaire data (used for orientation note)")
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


def _parse_dyad_from_path(path):
    """Extract dyad, sub_a, sub_b from path."""
    dyad = sub_a = sub_b = None
    for part in path.parts:
        if part.startswith("dyad"):
            dyad = part
        m = _DYAD_PAIR_RE.match(part)
        if m:
            sub_a, sub_b = m.group(1), m.group(2)
    return dyad, sub_a, sub_b


def _detect_method(path):
    """Detect reduction method from directory structure."""
    for part in reversed(path.parts):
        if part in _METHOD_FOLDERS:
            return part
        if part == "synch_by_states":
            return "unknown"
    return "unknown"


def discover_summary_files(input_dir, reduce_method):
    """Find all map_states_summary.csv files in synch_by_states dirs."""
    all_files = discover_outputs(input_dir, "map_states_summary.csv")
    results = []
    for p in all_files:
        if "synch_by_states" not in str(p):
            continue
        if "_batch" in str(p):
            continue
        dyad, sub_a, sub_b = _parse_dyad_from_path(p)
        method = _detect_method(p)
        if reduce_method != "all" and method != reduce_method:
            continue
        pairwise = p.parent / "map_states_pairwise.csv"
        results.append({
            "summary_path": p,
            "pairwise_path": pairwise if pairwise.exists() else None,
            "dyad": dyad or "unknown",
            "sub_a": sub_a or "unknown",
            "sub_b": sub_b or "unknown",
            "method": method,
        })

    print(f"  Discovered {len(results)} map_states_summary.csv files")
    methods_found = sorted(set(r["method"] for r in results))
    print(f"  Methods: {', '.join(methods_found)}")
    dyads = sorted(set(r["dyad"] for r in results))
    print(f"  Dyads: {len(dyads)} ({', '.join(dyads[:10])}{'...' if len(dyads) > 10 else ''})")
    return results


# ---------------------------------------------------------------------------
# Per-dyad effect size computation
# ---------------------------------------------------------------------------

def compute_eta_squared(summary_df):
    """Compute eta-squared from state summary statistics.

    eta_sq = SS_between / SS_total
    SS_between = sum(n_k * (mean_k - grand_mean)^2)
    SS_total   = SS_between + SS_within
    SS_within  = sum((n_k - 1) * sd_k^2)
    """
    if summary_df.empty or "mean" not in summary_df.columns:
        return np.nan

    ns = summary_df["n_bins"].values.astype(float)
    means = summary_df["mean"].values.astype(float)
    sds = summary_df["sd"].values.astype(float)

    valid = np.isfinite(ns) & np.isfinite(means) & np.isfinite(sds) & (ns > 0)
    if valid.sum() < 2:
        return np.nan

    ns, means, sds = ns[valid], means[valid], sds[valid]
    n_total = ns.sum()
    grand_mean = np.average(means, weights=ns)
    ss_between = np.sum(ns * (means - grand_mean) ** 2)
    ss_within = np.sum((ns - 1) * sds ** 2)
    ss_total = ss_between + ss_within

    if ss_total == 0:
        return np.nan
    return ss_between / ss_total


def process_dyad(finfo):
    """Process one dyad's summary and pairwise files, return effect size row."""
    try:
        summary_df = pd.read_csv(finfo["summary_path"], comment="#")
    except Exception as e:
        print(f"  Warning: skipping summary in {finfo['dyad']} {finfo['method']} ({e})")
        return None

    n_states = len(summary_df)
    eta_sq = compute_eta_squared(summary_df)

    max_d = np.nan
    n_sig_pairs = 0
    n_total_pairs = 0

    if finfo["pairwise_path"] is not None:
        try:
            pairwise_df = pd.read_csv(finfo["pairwise_path"])
            n_total_pairs = len(pairwise_df)
            if "cohens_d" in pairwise_df.columns:
                max_d = pairwise_df["cohens_d"].abs().max()
            if "p_fdr" in pairwise_df.columns:
                n_sig_pairs = (pairwise_df["p_fdr"] < 0.05).sum()
            elif "significant" in pairwise_df.columns:
                n_sig_pairs = pairwise_df["significant"].sum()
        except Exception as e:
            print(f"  Warning: skipping pairwise in {finfo['dyad']} {finfo['method']} ({e})")

    return {
        "dyad": finfo["dyad"],
        "sub_a": finfo["sub_a"],
        "sub_b": finfo["sub_b"],
        "method": finfo["method"],
        "n_states": n_states,
        "eta_squared": eta_sq,
        "max_cohens_d": max_d,
        "n_sig_pairs": n_sig_pairs,
        "n_total_pairs": n_total_pairs,
    }


# ---------------------------------------------------------------------------
# Group-level analysis
# ---------------------------------------------------------------------------

def run_group_analysis(stacked, out_dir, label):
    """Per method: one-sample t-test on eta_squared, proportion with sig pairs."""
    print(f"\n{'='*60}")
    print(f"  Group-level state-{label} analysis")
    print(f"{'='*60}")

    group_rows = []
    for method, sub in stacked.groupby("method"):
        eta_vals = sub["eta_squared"].dropna()
        n = len(eta_vals)

        entry = {
            "method": method,
            "n_dyads": n,
            "mean_eta_squared": eta_vals.mean() if n > 0 else np.nan,
            "sd_eta_squared": eta_vals.std() if n > 1 else np.nan,
            "mean_max_cohens_d": sub["max_cohens_d"].dropna().mean(),
            "mean_n_states": sub["n_states"].mean(),
        }

        if n >= 2:
            t_stat, p_val = stats.ttest_1samp(eta_vals, 0)
            sd = eta_vals.std(ddof=1)
            cohens_d = eta_vals.mean() / sd if sd > 0 else np.nan
            entry.update({
                "t_stat": t_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
            })
        else:
            entry.update({"t_stat": np.nan, "p_value": np.nan, "cohens_d": np.nan})

        dyads_with_sig = (sub["n_sig_pairs"] > 0).sum()
        entry["n_dyads_with_sig_pair"] = dyads_with_sig
        entry["prop_dyads_with_sig_pair"] = dyads_with_sig / n if n > 0 else np.nan

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
        "analysis": f"group_synch_by_states -- state effect on {label}",
        "test": "one-sample t-test on eta-squared (H0: eta_sq = 0)",
        "note": "States are NOT aligned across dyads; eta_sq is a within-dyad effect size",
    }
    csv_path = out_dir / "state_effect_summary.csv"
    write_csv_with_header(group_df, csv_path, header)

    print(f"\n  State effect on {label} per method:")
    for _, row in group_df.iterrows():
        sig = "*" if row.get("p_value", 1) < 0.05 else ""
        fdr_str = f"  p_fdr={row['p_fdr']:.4f}" if np.isfinite(row.get("p_fdr", np.nan)) else ""
        print(f"    {row['method']:15s}  mean_eta2={row['mean_eta_squared']:.4f}  "
              f"t={row['t_stat']:+.3f}  p={row['p_value']:.4f}{fdr_str}  "
              f"prop_sig={row['prop_dyads_with_sig_pair']:.0%}  n={row['n_dyads']}{sig}")

    return group_df


# ---------------------------------------------------------------------------
# Method comparison
# ---------------------------------------------------------------------------

def run_method_comparison(stacked, out_dir, label):
    """Compare eta_squared across reduction methods."""
    print(f"\n{'='*60}")
    print("  Method comparison")
    print(f"{'='*60}")

    methods = sorted(stacked["method"].unique())
    if len(methods) < 2:
        print("  Fewer than 2 methods, skipping comparison")
        return

    pivot = stacked.pivot(index="dyad", columns="method", values="eta_squared")
    pivot_clean = pivot.dropna()

    friedman_stat = friedman_p = np.nan
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

    print(f"\n  Mean eta-squared per method:")
    for method in methods:
        vals = stacked[stacked["method"] == method]["eta_squared"].dropna()
        print(f"    {method:15s}  M={vals.mean():.4f}  SD={vals.std():.4f}  n={len(vals)}")
    if np.isfinite(friedman_p):
        test_name = "Friedman" if len(methods) >= 3 else "Wilcoxon"
        print(f"  {test_name} test: stat={friedman_stat:.3f}, p={friedman_p:.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_method_comparison(stacked, out_dir, label):
    """Box plots of eta-squared per dyad, one box per method."""
    methods = sorted(stacked["method"].unique())
    if len(methods) < 2:
        print("  Skipping method comparison plot (fewer than 2 methods)")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), 5))
    data = [stacked[stacked["method"] == m]["eta_squared"].dropna().values
            for m in methods]
    bp = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.5)
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
              "#264653", "#6d6875", "#b5838d"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Eta-squared", fontsize=11)
    ax.set_xlabel("Reduction Method", fontsize=11)
    ax.set_title(f"State Effect on {label}: Method Comparison", fontsize=13)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    path = out_dir / "method_comparison_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_dyad_barplot(stacked, out_dir, label):
    """Bar plot showing each dyad's eta-squared and max Cohen's d."""
    if stacked.empty:
        return

    methods = sorted(stacked["method"].unique())

    for method in methods:
        sub = stacked[stacked["method"] == method].sort_values("eta_squared", ascending=False)
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(sub) * 0.4)))

        ax1 = axes[0]
        y_pos = np.arange(len(sub))
        ax1.barh(y_pos, sub["eta_squared"].fillna(0), color="#457b9d", edgecolor="white")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sub["dyad"], fontsize=9)
        ax1.set_xlabel("Eta-squared", fontsize=10)
        ax1.set_title(f"State effect size ({method})", fontsize=11)
        ax1.grid(True, alpha=0.2, axis="x")

        ax2 = axes[1]
        bar_colors = ["#2a9d8f" if sig > 0 else "#adb5bd"
                      for sig in sub["n_sig_pairs"].fillna(0)]
        ax2.barh(y_pos, sub["max_cohens_d"].fillna(0), color=bar_colors, edgecolor="white")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sub["dyad"], fontsize=9)
        ax2.set_xlabel("Max |Cohen's d|", fontsize=10)
        ax2.set_title(f"Largest pairwise effect ({method})", fontsize=11)
        ax2.grid(True, alpha=0.2, axis="x")

        fig.suptitle(f"Per-Dyad State-{label} Effects ({method})", fontsize=13, y=1.01)
        fig.tight_layout()

        suffix = f"_{method}" if len(methods) > 1 else ""
        path = out_dir / f"dyad_barplot{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary_report(stacked, summary_df, out_dir, label):
    """Print terminal summary and save report."""
    print(f"\n{'='*60}")
    print("  Summary Report")
    print(f"{'='*60}")

    report_lines = [
        f"Group Synch-by-States Summary Report ({label})",
        "=" * 40,
        "",
    ]

    if not stacked.empty:
        eta_vals = stacked["eta_squared"].dropna()
        d_vals = stacked["max_cohens_d"].dropna()
        dyads_with_sig = (stacked["n_sig_pairs"] > 0).sum()
        n_total = len(stacked)

        print(f"\n  Overall eta-squared:")
        print(f"    Mean = {eta_vals.mean():.4f}, SD = {eta_vals.std():.4f}, "
              f"Median = {eta_vals.median():.4f}")
        print(f"\n  Overall max |Cohen's d|:")
        print(f"    Mean = {d_vals.mean():.4f}, SD = {d_vals.std():.4f}")
        print(f"\n  Dyad-method pairs with >= 1 sig pairwise comparison: "
              f"{dyads_with_sig}/{n_total} ({dyads_with_sig/n_total:.0%})")

        report_lines.extend([
            f"N dyads: {stacked['dyad'].nunique()}",
            f"Methods: {', '.join(sorted(stacked['method'].unique()))}",
            "",
            f"Overall eta-squared: M = {eta_vals.mean():.4f}, SD = {eta_vals.std():.4f}",
            f"Overall max |Cohen's d|: M = {d_vals.mean():.4f}, SD = {d_vals.std():.4f}",
            f"Dyad-method pairs with sig pairwise: {dyads_with_sig}/{n_total} ({dyads_with_sig/n_total:.0%})",
            "",
        ])

    print(f"\n  Reference benchmarks:")
    print(f"    Eta-squared interpretation:")
    print(f"      Small = 0.01, Medium = 0.06, Large = 0.14 (Cohen, 1988)")
    print(f"    State-dependent synchrony:")
    print(f"      States should modulate synchrony if behavioral coordination")
    print(f"      varies across interaction phases (Coco & Dale, 2014)")

    report_lines.extend([
        "Reference benchmarks:",
        "  Eta-squared: small=0.01, medium=0.06, large=0.14 (Cohen, 1988)",
        "  State-dependent synchrony: Coco & Dale, 2014",
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

    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "_batch" / "group_synch_by_states"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = list(out_dir.glob("state_effect_*.csv"))
        if existing:
            print(f"  Output files already exist in {out_dir}")
            print(f"  Use --overwrite to replace them")
            sys.exit(0)

    print(f"Group-level analysis: do behavioral states differ in {label}?")
    print(f"  Input: {input_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Label: {label}")
    print(f"  Method filter: {args.reduce_method}")

    print(f"\nDiscovering synch_by_states files...")
    files = discover_summary_files(input_dir, args.reduce_method)
    if not files:
        sys.exit("Error: no synch_by_states files found. Run map_states.py first.")

    print(f"\nComputing per-dyad effect sizes...")
    rows = []
    for finfo in files:
        result = process_dyad(finfo)
        if result is not None:
            rows.append(result)

    if not rows:
        sys.exit("Error: no valid effect sizes computed")

    stacked = pd.DataFrame(rows)

    header = {
        "analysis": f"group_synch_by_states -- per-dyad effect sizes ({label})",
        "note": "States are NOT aligned across dyads",
        "eta_squared": "SS_between / SS_total from state means and n_bins",
        "n_files": len(files),
    }
    csv_path = out_dir / "state_effect_sizes.csv"
    write_csv_with_header(stacked, csv_path, header)
    print(f"  Computed {len(stacked)} dyad-method effect sizes")

    summary_df = run_group_analysis(stacked, out_dir, label)
    run_method_comparison(stacked, out_dir, label)

    print(f"\nGenerating plots...")
    plot_method_comparison(stacked, out_dir, label)
    plot_dyad_barplot(stacked, out_dir, label)

    print_summary_report(stacked, summary_df, out_dir, label)

    orientation_note = "applied (subjects.csv provided)" if args.subjects else "not applied"
    print(f"  Orientation normalization: {orientation_note}")
    with open(out_dir / "summary_report.txt", "a") as f:
        f.write(f"\nOrientation normalization: {orientation_note}\n")

    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
