#!/usr/bin/env python3
"""
Group-level analysis: which behavioral features drive interpersonal synchrony?

Scans the output directory for all dyad-level synch_from_features correlation
results (multi-mode correlate.py output), applies a feature-vote counting
approach across dyads, and cross-references with outcome prediction results
to identify features that drive both synchrony and outcome ratings.

The --label flag controls what the outcome is called when cross-referencing
with group_correlate results (default: Trustworthiness).
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
    feature_to_modality,
    parse_csv_header,
    write_csv_with_header,
    HAS_STATSMODELS,
)


def _slugify(label):
    return label.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Group-level analysis: which behavioral features drive "
                    "interpersonal synchrony?"
    )
    p.add_argument("--input-dir", required=True,
                   help="Study output root containing "
                        "dyad*/sub*_sub*/synch_from_features/")
    p.add_argument("--reduce-method",
                   choices=["grouped", "pca", "fa", "ica", "cca",
                            "grouped-pca", "cluster", "all"],
                   default="all",
                   help="Which synchrony reduction method to analyze (default: all)")
    p.add_argument("--label", default="Trustworthiness",
                   help="Outcome label for cross-reference with group_correlate results "
                        "(default: Trustworthiness)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: {input-dir}/_batch/group_synch_from_features/)")
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


def _parse_synch_feat_entry(path):
    """Parse a synch_from_features correlation.csv into metadata."""
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
    parent_name = path.parent.name
    if parent_name in _METHOD_FOLDERS:
        method = parent_name

    return {
        "path": path,
        "dir": path.parent,
        "dyad": dyad or "unknown",
        "sub_a": sub_a or "unknown",
        "sub_b": sub_b or "unknown",
        "method": method,
        "meta": meta,
    }


def discover_synch_feat_files(input_dir, reduce_method):
    """Find all synch_from_features correlation CSVs."""
    all_csvs = discover_outputs(input_dir, "correlation.csv")
    results = []
    for p in all_csvs:
        if "synch_from_features" not in str(p):
            continue
        if "_batch" in str(p):
            continue
        info = _parse_synch_feat_entry(p)
        if reduce_method != "all" and info["method"] != reduce_method:
            continue
        results.append(info)

    print(f"  Discovered {len(results)} synch_from_features correlation files")
    methods_found = sorted(set(r["method"] for r in results))
    print(f"  Methods: {', '.join(methods_found)}")
    dyads = sorted(set(r["dyad"] for r in results))
    print(f"  Dyads: {len(dyads)} ({', '.join(dyads[:10])}{'...' if len(dyads) > 10 else ''})")
    return results


def discover_loadings_files(input_dir, reduce_method):
    """Find all feature loadings CSVs in synch_from_features dirs."""
    all_csvs = discover_outputs(input_dir, "loadings_*.csv")
    loadings = {}
    for p in all_csvs:
        if "synch_from_features" not in str(p):
            continue
        if "_batch" in str(p) or "target_loadings" in p.name:
            continue
        dyad = sub_a = sub_b = None
        for part in p.parts:
            if part.startswith("dyad"):
                dyad = part
            m = _DYAD_PAIR_RE.match(part)
            if m:
                sub_a, sub_b = m.group(1), m.group(2)
        method = p.parent.name if p.parent.name in _METHOD_FOLDERS else "unknown"
        if reduce_method != "all" and method != reduce_method:
            continue
        key = (dyad or "unknown", sub_a or "unknown", sub_b or "unknown", method)
        loadings[key] = p
    print(f"  Discovered {len(loadings)} feature loadings files")
    return loadings


# ---------------------------------------------------------------------------
# Multi-mode peak extraction
# ---------------------------------------------------------------------------

def extract_multi_peaks(corr_df):
    """Extract peak |r| per (dimension, target_component) pair.

    For multi-mode correlation CSVs that have both dimension and
    target_component columns.
    """
    if "target_component" not in corr_df.columns:
        from analysis.utils import extract_peak_correlation
        return extract_peak_correlation(corr_df)

    rows = []
    for (dim, tc), sub in corr_df.groupby(["dimension", "target_component"]):
        sub = sub.dropna(subset=["correlation"])
        if sub.empty:
            continue
        idx = sub["correlation"].abs().idxmax()
        row = sub.loc[idx]
        rows.append({
            "dimension": dim,
            "target_component": tc,
            "peak_r": row["correlation"],
            "peak_lag": row["lag_seconds"],
            "peak_p": row["p_value"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature vote counting
# ---------------------------------------------------------------------------

def run_vote_counting(corr_files, loadings_map, out_dir):
    """Count which original features drive synchrony across dyads."""
    print(f"\n{'='*60}")
    print("  Feature vote counting: which features drive synchrony?")
    print(f"{'='*60}")

    all_votes = []
    n_dyads_with_data = 0

    for finfo in corr_files:
        try:
            corr_df = pd.read_csv(finfo["path"], comment="#")
        except Exception as e:
            print(f"  Warning: skipping {finfo['path'].name} in {finfo['dyad']} ({e})")
            continue

        peaks = extract_multi_peaks(corr_df)
        if peaks.empty:
            continue

        sig = peaks[(peaks["peak_r"].abs() > 0.15) & (peaks["peak_p"] < 0.05)]
        if sig.empty:
            continue

        key = (finfo["dyad"], finfo["sub_a"], finfo["sub_b"], finfo["method"])
        if key not in loadings_map:
            print(f"  Warning: no loadings file for {finfo['dyad']} {finfo['method']}, skipping votes")
            continue

        loadings_df = pd.read_csv(loadings_map[key])
        n_dyads_with_data += 1

        sig_dims = sig["dimension"].unique()
        for dim in sig_dims:
            dim_loadings = loadings_df[loadings_df["component"] == dim].copy()
            if dim_loadings.empty:
                continue
            dim_loadings["abs_loading"] = dim_loadings["loading"].abs()
            top5 = dim_loadings.nlargest(5, "abs_loading")

            dim_peaks = sig[sig["dimension"] == dim]
            best_r = dim_peaks.loc[dim_peaks["peak_r"].abs().idxmax(), "peak_r"]

            for _, lr in top5.iterrows():
                all_votes.append({
                    "feature": lr["feature"],
                    "abs_loading": lr["abs_loading"],
                    "dyad": finfo["dyad"],
                    "method": finfo["method"],
                    "dimension": dim,
                    "peak_r": best_r,
                })

    if not all_votes:
        print("  No significant feature-synchrony links found")
        return pd.DataFrame(), pd.DataFrame()

    votes_df = pd.DataFrame(all_votes)
    n_dyads_total = len(set(f["dyad"] for f in corr_files))

    feature_summary = (
        votes_df.groupby("feature")
        .agg(
            n_votes=("dyad", "count"),
            n_unique_dyads=("dyad", "nunique"),
            mean_abs_loading=("abs_loading", "mean"),
            methods_where_significant=("method", lambda x: ",".join(sorted(set(x)))),
        )
        .reset_index()
    )
    feature_summary["n_dyads_available"] = n_dyads_total
    feature_summary["vote_proportion"] = feature_summary["n_votes"] / max(n_dyads_total, 1)
    feature_summary["modality"] = feature_summary["feature"].apply(feature_to_modality)
    feature_summary = feature_summary.sort_values("n_votes", ascending=False).reset_index(drop=True)

    header = {
        "analysis": "group_synch_from_features -- feature vote counting",
        "criteria": "peak |r| > 0.15 AND peak_p < 0.05, top 5 features per sig dimension",
        "n_dyads_total": n_dyads_total,
        "n_dyads_with_sig": n_dyads_with_data,
    }
    cols = ["feature", "n_votes", "n_dyads_available", "vote_proportion",
            "mean_abs_loading", "methods_where_significant", "modality"]
    csv_path = out_dir / "synch_feature_votes.csv"
    write_csv_with_header(feature_summary[cols], csv_path, header)

    by_method = (
        votes_df.groupby(["feature", "method"])
        .agg(
            n_votes=("dyad", "count"),
            mean_abs_loading=("abs_loading", "mean"),
        )
        .reset_index()
    )
    by_method["modality"] = by_method["feature"].apply(feature_to_modality)
    by_method = by_method.sort_values(["method", "n_votes"], ascending=[True, False])
    csv_path2 = out_dir / "synch_feature_votes_by_method.csv"
    write_csv_with_header(by_method, csv_path2, header)

    print(f"\n  Top 15 features driving synchrony:")
    for _, row in feature_summary.head(15).iterrows():
        print(f"    {row['feature']:40s}  votes={row['n_votes']:3d}  "
              f"dyads={row['n_unique_dyads']}  "
              f"modality={row['modality']}")

    return feature_summary, votes_df


# ---------------------------------------------------------------------------
# Cross-reference with outcome prediction
# ---------------------------------------------------------------------------

def run_overlap_analysis(synch_votes, input_dir, out_dir, label, slug):
    """Find features that predict BOTH outcome AND synchrony."""
    print(f"\n{'='*60}")
    print(f"  Overlap analysis: features driving both {label} and synchrony")
    print(f"{'='*60}")

    outcome_path = input_dir / "_batch" / "group_correlate" / f"group_{slug}_feature_votes.csv"
    if not outcome_path.exists():
        alt_path = input_dir / "_batch" / "group_correlate" / "group_trustworthiness_feature_votes.csv"
        if alt_path.exists():
            outcome_path = alt_path
        else:
            print(f"  No outcome feature votes found at {outcome_path}")
            print(f"  Run group_correlate.py first to enable overlap analysis")
            return pd.DataFrame()

    try:
        outcome_votes = pd.read_csv(outcome_path, comment="#")
    except Exception as e:
        print(f"  Warning: could not load outcome votes ({e})")
        return pd.DataFrame()

    if "feature" not in outcome_votes.columns or synch_votes.empty:
        print("  Cannot compute overlap (missing feature column)")
        return pd.DataFrame()

    synch_features = set(synch_votes["feature"])
    outcome_features = set(outcome_votes["feature"])
    overlap_features = synch_features & outcome_features

    print(f"  Features driving synchrony: {len(synch_features)}")
    print(f"  Features predicting {label}: {len(outcome_features)}")
    print(f"  Features in BOTH: {len(overlap_features)}")

    if not overlap_features:
        print(f"  No overlapping features found")
        overlap_df = pd.DataFrame(columns=[
            "feature", "synch_votes", "synch_vote_proportion",
            f"{slug}_votes", f"{slug}_vote_proportion", "modality",
        ])
        header = {
            "analysis": f"feature overlap: {label} AND synchrony",
            "n_overlap": 0,
            "n_synch_only": len(synch_features - outcome_features),
            f"n_{slug}_only": len(outcome_features - synch_features),
        }
        csv_path = out_dir / f"feature_overlap_{slug}_and_synch.csv"
        write_csv_with_header(overlap_df, csv_path, header)
        return overlap_df

    synch_lookup = synch_votes.set_index("feature")
    outcome_lookup = outcome_votes.set_index("feature")

    overlap_rows = []
    for feat in sorted(overlap_features):
        sr = synch_lookup.loc[feat]
        orow = outcome_lookup.loc[feat]
        overlap_rows.append({
            "feature": feat,
            "synch_votes": sr.get("n_votes", 0),
            "synch_vote_proportion": sr.get("vote_proportion", 0),
            f"{slug}_votes": orow.get("n_votes", 0),
            f"{slug}_vote_proportion": orow.get("vote_proportion", 0),
            "synch_modality": sr.get("modality", "other"),
            "modality": feature_to_modality(feat),
        })

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df["total_votes"] = overlap_df["synch_votes"] + overlap_df[f"{slug}_votes"]
    overlap_df = overlap_df.sort_values("total_votes", ascending=False).reset_index(drop=True)

    header = {
        "analysis": f"feature overlap: {label} AND synchrony",
        "headline": f"These features both drive synchrony AND predict {label}",
        "n_overlap": len(overlap_features),
        "n_synch_only": len(synch_features - outcome_features),
        f"n_{slug}_only": len(outcome_features - synch_features),
    }
    csv_path = out_dir / f"feature_overlap_{slug}_and_synch.csv"
    write_csv_with_header(overlap_df, csv_path, header)

    print(f"\n  Headline features (drive synchrony AND predict {label}):")
    for _, row in overlap_df.head(15).iterrows():
        print(f"    {row['feature']:40s}  "
              f"synch_votes={row['synch_votes']:3d}  "
              f"{slug}_votes={row[f'{slug}_votes']:3d}  "
              f"modality={row['modality']}")

    return overlap_df


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


def plot_feature_votes(vote_summary, out_dir):
    """Horizontal bar chart of top 20 features by vote count."""
    if vote_summary.empty:
        return

    top20 = vote_summary.head(20).iloc[::-1]
    colors = [MODALITY_COLORS.get(m, "#adb5bd") for m in top20["modality"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top20) * 0.35)))
    ax.barh(range(len(top20)), top20["n_votes"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"], fontsize=8)
    ax.set_xlabel("Vote Count", fontsize=11)
    ax.set_title("Top 20 Synchrony-Driving Features (Vote Count)", fontsize=13)
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


def plot_overlap(synch_votes, overlap_df, input_dir, out_dir, label, slug):
    """Bar plot showing features by category: outcome-only, synch-only, both."""
    outcome_path = input_dir / "_batch" / "group_correlate" / f"group_{slug}_feature_votes.csv"
    if not outcome_path.exists():
        alt_path = input_dir / "_batch" / "group_correlate" / "group_trustworthiness_feature_votes.csv"
        if alt_path.exists():
            outcome_path = alt_path
        else:
            return

    try:
        outcome_votes = pd.read_csv(outcome_path, comment="#")
    except Exception:
        return

    if synch_votes.empty:
        return

    synch_set = set(synch_votes["feature"])
    outcome_set = set(outcome_votes["feature"])
    overlap_set = synch_set & outcome_set
    synch_only = synch_set - outcome_set
    outcome_only = outcome_set - synch_set

    categories = [f"{label} only", "Both", "Synchrony only"]
    counts = [len(outcome_only), len(overlap_set), len(synch_only)]
    colors_bar = ["#e63946", "#2a9d8f", "#457b9d"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 2]})

    ax1 = axes[0]
    ax1.bar(categories, counts, color=colors_bar, edgecolor="white")
    ax1.set_ylabel("Number of Features", fontsize=11)
    ax1.set_title("Feature Category Counts", fontsize=12)
    for i, (cat, cnt) in enumerate(zip(categories, counts)):
        ax1.text(i, cnt + 0.5, str(cnt), ha="center", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    ax2 = axes[1]
    if not overlap_df.empty and len(overlap_df) > 0:
        top_overlap = overlap_df.head(15).iloc[::-1]
        y_pos = np.arange(len(top_overlap))
        bar_width = 0.35

        ax2.barh(y_pos - bar_width / 2, top_overlap["synch_votes"],
                 bar_width, color="#457b9d", label="Synchrony votes", edgecolor="white")
        ax2.barh(y_pos + bar_width / 2, top_overlap[f"{slug}_votes"],
                 bar_width, color="#e63946", label=f"{label} votes", edgecolor="white")

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_overlap["feature"], fontsize=8)
        ax2.set_xlabel("Vote Count", fontsize=11)
        ax2.set_title(f"Top Overlap Features: Drive Synchrony AND Predict {label}", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2, axis="x")
    else:
        ax2.text(0.5, 0.5, "No overlapping features found",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Overlap Features", fontsize=12)

    fig.tight_layout()
    path = out_dir / "overlap_venn_or_barplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary_report(synch_votes, overlap_df, out_dir, label, slug):
    """Print terminal summary and save report."""
    print(f"\n{'='*60}")
    print("  Summary Report")
    print(f"{'='*60}")

    report_lines = [
        "Group Synch-from-Features Summary Report",
        "=" * 40,
        "",
    ]

    if not synch_votes.empty:
        print(f"\n  Total unique features driving synchrony: {len(synch_votes)}")
        top_modalities = synch_votes.groupby("modality")["n_votes"].sum().sort_values(ascending=False)
        print(f"  Votes by modality:")
        for mod, cnt in top_modalities.items():
            print(f"    {mod:20s}  {cnt} votes")

        report_lines.extend([
            f"Unique features driving synchrony: {len(synch_votes)}",
            "",
            "Votes by modality:",
        ])
        for mod, cnt in top_modalities.items():
            report_lines.append(f"  {mod}: {cnt} votes")
        report_lines.append("")

    if not overlap_df.empty:
        print(f"\n  Features driving BOTH synchrony and {label}: {len(overlap_df)}")
        report_lines.append(f"Features in both synchrony and {label}: {len(overlap_df)}")
        report_lines.append("")
        report_lines.append("Top overlap features:")
        for _, row in overlap_df.head(10).iterrows():
            report_lines.append(
                f"  {row['feature']}: synch={row['synch_votes']}, "
                f"{slug}={row[f'{slug}_votes']}"
            )

    report_lines.extend([
        "",
        "Reference:",
        "  Features that appear in both lists provide converging evidence:",
        "  the same behavioral cues that drive interpersonal synchrony",
        f"  also predict {label} impressions.",
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

    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "_batch" / "group_synch_from_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = list(out_dir.glob("synch_feature_votes*.csv"))
        if existing:
            print(f"  Output files already exist in {out_dir}")
            print(f"  Use --overwrite to replace them")
            sys.exit(0)

    print(f"Group-level analysis: which features drive synchrony?")
    print(f"  Input: {input_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Label (for overlap): {label}")
    print(f"  Method filter: {args.reduce_method}")

    print(f"\nDiscovering synch_from_features files...")
    corr_files = discover_synch_feat_files(input_dir, args.reduce_method)
    if not corr_files:
        sys.exit("Error: no synch_from_features correlation files found. "
                 "Run correlate.py --mode multi on synchrony_timeseries.csv first.")

    loadings_map = discover_loadings_files(input_dir, args.reduce_method)

    synch_votes, votes_raw = run_vote_counting(corr_files, loadings_map, out_dir)

    overlap_df = pd.DataFrame()
    if not synch_votes.empty:
        overlap_df = run_overlap_analysis(synch_votes, input_dir, out_dir, label, slug)

    print(f"\nGenerating plots...")
    plot_feature_votes(synch_votes, out_dir)
    plot_overlap(synch_votes, overlap_df, input_dir, out_dir, label, slug)

    print_summary_report(synch_votes, overlap_df, out_dir, label, slug)

    orientation_note = "applied (subjects.csv provided)" if args.subjects else "not applied"
    print(f"  Orientation normalization: {orientation_note}")
    with open(out_dir / "summary_report.txt", "a") as f:
        f.write(f"\nOrientation normalization: {orientation_note}\n")

    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
