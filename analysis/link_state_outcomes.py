#!/usr/bin/env python3
"""
Link HMM state sequences to continuous external signals.

Answers: "which behavioral states are associated with higher/lower
signal values (e.g., trustworthiness ratings, heart rate, fNIRS)?"
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.utils import (
    load_target,
    bin_timeseries,
    align_bins,
    apply_fdr,
    write_csv_with_header,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Link behavioral states (from segment.py) to continuous signals"
    )
    p.add_argument("--states", required=True,
                   help="Path to segment.py segments.csv (must have time_seconds and state columns)")
    p.add_argument("--signal", required=True,
                   help="Path to signal CSV (rating, HR, fNIRS, etc.)")
    p.add_argument("--signal-col", default="Value",
                   help="Column name for signal values (default: Value)")
    p.add_argument("--signal-label", default="Signal",
                   help="Human-readable label for plots (default: Signal)")
    p.add_argument("--rater", default=None,
                   help="Rater subject ID for output filenames")
    p.add_argument("--target", default=None,
                   help="Target subject ID for output filenames")
    p.add_argument("--bin-size", type=float, default=0.5,
                   help="Time bin in seconds, must match segment.py (default: 0.5)")
    p.add_argument("-o", "--output-dir", default="./analysis_output",
                   help="Output directory (default: ./analysis_output)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_states(path):
    """Load segment.py state timeline CSV."""
    df = pd.read_csv(path, comment="#")
    if "state" not in df.columns:
        sys.exit("Error: states CSV must contain a 'state' column")

    # Detect time column
    if "time_seconds" in df.columns:
        pass
    elif "bin_time" in df.columns:
        df["time_seconds"] = df["bin_time"]
    else:
        sys.exit("Error: states CSV must contain 'time_seconds' or 'bin_time' column")

    unique_states = sorted(df["state"].unique())
    print(f"  Loaded states: {len(df)} bins, {len(unique_states)} states ({unique_states})")
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_state_signal_summary(states, signal, signal_label):
    """Compute per-state signal statistics."""
    rows = []
    for s in sorted(states["state"].unique()):
        mask = states["state"] == s
        vals = signal.loc[mask].dropna().values
        n = len(vals)
        if n == 0:
            rows.append({"state": s, "n_bins": 0, "mean": np.nan, "sd": np.nan,
                         "sem": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
            continue

        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = sd / np.sqrt(n) if n > 1 else 0.0
        ci_hw = 1.96 * sem

        rows.append({
            "state": s,
            "n_bins": int(n),
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "ci_lower": mean - ci_hw,
            "ci_upper": mean + ci_hw,
        })

    return pd.DataFrame(rows)


def compute_pairwise_tests(states, signal):
    """Pairwise Mann-Whitney U tests between all state pairs, with FDR and Cohen's d."""
    unique_states = sorted(states["state"].unique())
    rows = []

    for i, sa in enumerate(unique_states):
        for sb in unique_states[i + 1:]:
            vals_a = signal.loc[states["state"] == sa].dropna().values
            vals_b = signal.loc[states["state"] == sb].dropna().values

            if len(vals_a) < 2 or len(vals_b) < 2:
                rows.append({
                    "state_a": sa, "state_b": sb,
                    "u_stat": np.nan, "p_value": np.nan, "cohens_d": np.nan,
                })
                continue

            u_stat, p_val = sp_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")

            # Cohen's d
            pooled_std = np.sqrt(
                ((len(vals_a) - 1) * np.var(vals_a, ddof=1) +
                 (len(vals_b) - 1) * np.var(vals_b, ddof=1)) /
                (len(vals_a) + len(vals_b) - 2)
            )
            d = (np.mean(vals_a) - np.mean(vals_b)) / pooled_std if pooled_std > 0 else 0.0

            rows.append({
                "state_a": sa, "state_b": sb,
                "u_stat": float(u_stat), "p_value": float(p_val), "cohens_d": float(d),
            })

    df = pd.DataFrame(rows)
    if not df.empty and df["p_value"].notna().any():
        df = apply_fdr(df)
        df["significant"] = df["p_fdr"] < 0.05
    else:
        df["p_fdr"] = np.nan
        df["significant"] = False

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STATE_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                "#9c755f", "#bab0ac"]


def plot_state_outcomes(states_df, signal_series, summary_df, pairwise_df,
                        signal_label, output_path):
    """Two-panel figure: boxplot + timeline."""
    fig, (ax_box, ax_time) = plt.subplots(2, 1, figsize=(14, 8),
                                           gridspec_kw={"height_ratios": [1, 1.2]})

    unique_states = sorted(states_df["state"].unique())

    # --- Top panel: boxplot ---
    box_data = []
    box_colors = []
    for s in unique_states:
        vals = signal_series.loc[states_df["state"] == s].dropna().values
        box_data.append(vals)
        box_colors.append(STATE_COLORS[(s - 1) % len(STATE_COLORS)])

    bp = ax_box.boxplot(box_data, tick_labels=[str(s) for s in unique_states],
                        patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add significance brackets
    sig_pairs = pairwise_df[pairwise_df.get("significant", pd.Series(dtype=bool)) == True]
    if not sig_pairs.empty:
        y_max = max(v.max() for v in box_data if len(v) > 0)
        y_range = y_max - min(v.min() for v in box_data if len(v) > 0)
        bracket_y = y_max + 0.05 * y_range

        for _, row in sig_pairs.iterrows():
            xa = unique_states.index(row["state_a"]) + 1
            xb = unique_states.index(row["state_b"]) + 1
            ax_box.plot([xa, xa, xb, xb], [bracket_y, bracket_y + 0.02 * y_range,
                                            bracket_y + 0.02 * y_range, bracket_y],
                        color="black", linewidth=1)
            ax_box.text((xa + xb) / 2, bracket_y + 0.03 * y_range, "*",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
            bracket_y += 0.08 * y_range

    ax_box.set_xlabel("State", fontsize=11)
    ax_box.set_ylabel(signal_label, fontsize=11)
    ax_box.set_title(f"{signal_label} by Behavioral State", fontsize=13)
    ax_box.grid(True, alpha=0.3, axis="y")

    # --- Bottom panel: timeline ---
    bin_times = states_df["time_seconds"].values
    state_vals = states_df["state"].values

    # Color-coded state bars
    for s in unique_states:
        color = STATE_COLORS[(s - 1) % len(STATE_COLORS)]
        mask = state_vals == s
        ax_time.fill_between(bin_times, 0, 1, where=mask, color=color,
                             alpha=0.3, step="mid", label=f"State {s}",
                             transform=ax_time.get_xaxis_transform())

    # Signal overlay
    ax2 = ax_time.twinx()
    ax2.plot(bin_times, signal_series.values, color="black", linewidth=0.8, alpha=0.8)
    ax2.set_ylabel(signal_label, fontsize=11, color="black")

    ax_time.set_xlabel("Time (seconds)", fontsize=11)
    ax_time.set_yticks([])
    ax_time.set_title("State Sequence with Signal Overlay", fontsize=13)
    ax_time.legend(fontsize=8, ncol=min(len(unique_states), 6), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.rater}_rates_{args.target}_" if args.rater and args.target else ""

    # Check overwrite
    summary_path = out_dir / f"{prefix}state_outcomes_summary.csv"
    if not args.overwrite and summary_path.exists():
        print(f"  Already exists: {summary_path}")
        print("  Skipping (use --overwrite to replace)")
        sys.exit(0)

    # Load
    print("Loading states...")
    states_df = load_states(args.states)

    print("Loading signal...")
    signal_df = load_target(args.signal, value_col=args.signal_col)

    # Bin signal
    print("\nBinning signal...")
    binned_signal = bin_timeseries(signal_df, "time_seconds", args.bin_size)
    print(f"  Binned signal: {binned_signal.shape[0]} bins")

    # Bin states (they should already be binned from segment.py, but ensure alignment)
    binned_states = bin_timeseries(states_df[["time_seconds", "state"]], "time_seconds", args.bin_size)
    # Round state to nearest integer after binning (mean of identical ints = same int)
    binned_states["state"] = binned_states["state"].round().astype(int)
    print(f"  Binned states: {binned_states.shape[0]} bins")

    # Align
    binned_states, binned_signal, common = align_bins(binned_states, binned_signal, args.bin_size)
    signal_series = binned_signal["Value"]
    print(f"  Common bins: {len(common)}")

    # Compute
    print(f"\nComputing {args.signal_label} by state...")
    summary = compute_state_signal_summary(binned_states, signal_series, args.signal_label)

    print("  Per-state means:")
    for _, row in summary.iterrows():
        print(f"    State {int(row['state'])}: mean={row['mean']:.3f} "
              f"(SD={row['sd']:.3f}, n={int(row['n_bins'])})")

    print("\nComputing pairwise tests...")
    pairwise = compute_pairwise_tests(binned_states, signal_series)

    n_sig = pairwise["significant"].sum() if "significant" in pairwise.columns else 0
    n_pairs = len(pairwise)
    print(f"  {n_sig}/{n_pairs} pairs significantly different (FDR < 0.05)")

    # Save
    print("\nSaving outputs...")
    header = {
        "signal": args.signal_label,
        "signal_col": args.signal_col,
        "rater": args.rater or "(not specified)",
        "target": args.target or "(not specified)",
        "bin_size": f"{args.bin_size}s",
        "states_file": args.states,
        "signal_file": args.signal,
    }
    write_csv_with_header(summary, summary_path, header)

    pairwise_path = out_dir / f"{prefix}state_outcomes_pairwise.csv"
    pairwise.to_csv(pairwise_path, index=False, float_format="%.6f")
    print(f"  Saved: {pairwise_path} ({len(pairwise)} pairs)")

    # Plot
    print("\nGenerating plot...")
    plot_path = out_dir / f"{prefix}state_outcomes_plot.png"
    plot_state_outcomes(binned_states, signal_series, summary, pairwise,
                        args.signal_label, plot_path)

    print(f"\nDone. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
