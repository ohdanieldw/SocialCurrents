#!/usr/bin/env python3
"""
Conversation segmentation using latent state models.

Segments a conversation into distinct behavioral states based on
multimodal timeseries features.  Supports Hidden Markov Models (HMM),
changepoint detection, and windowed k-means clustering.

Supports single-subject and batch modes.
"""

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure repo root on path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.utils import (
    load_features,
    bin_timeseries,
    filter_timevarying_columns,
    compute_grouped_dimensions,
    run_reduction,
    write_csv_with_header,
    resolve_facing,
)

# Optional dependencies — graceful fallback
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

try:
    import ruptures
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Segment a conversation into behavioral states"
    )
    p.add_argument("-f", "--features", required=True,
                   help="Path to timeseries_features.csv or directory (batch mode)")
    p.add_argument("-o", "--output-dir", default="./analysis_output",
                   help="Output directory (default: ./analysis_output)")
    p.add_argument("--method", choices=["hmm", "changepoint", "kmeans-window"],
                   default="hmm", help="Segmentation method (default: hmm)")
    p.add_argument("--n-states", default="auto",
                   help="Number of states, or 'auto' (default: auto)")
    p.add_argument("--min-states", type=int, default=2,
                   help="Min states to test when auto (default: 2)")
    p.add_argument("--max-states", type=int, default=10,
                   help="Max states to test when auto (default: 10)")
    p.add_argument("--state-selection", choices=["min", "elbow"], default="elbow",
                   help="State selection criterion: 'min' = lowest BIC (default), "
                        "'elbow' = kneedle elbow detection on BIC curve")
    p.add_argument("--reduce-features",
                   choices=["pca", "fa", "ica", "grouped", "every", "all"],
                   default="pca", help="Feature reduction (default: pca)")
    p.add_argument("--n-components", type=int, default=5,
                   help="Components for PCA/FA/ICA (default: 5)")
    p.add_argument("--min-duration", type=float, default=2.0,
                   help="Min state duration in seconds; merge shorter visits (default: 2.0)")
    p.add_argument("--time-resolution", type=float, default=0.5,
                   help="Resampling bin size in seconds (default: 0.5)")
    p.add_argument("--window-size", type=int, default=10,
                   help="Window size in bins for kmeans-window (default: 10)")
    p.add_argument("--n-restarts", type=int, default=10,
                   help="Random restarts for HMM fitting (default: 10)")
    p.add_argument("--covariance-type", choices=["full", "diag", "tied", "spherical"],
                   default="full", help="HMM covariance type (default: full)")
    p.add_argument("--no-zscore", action="store_true",
                   help="Skip z-scoring before analysis")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--subjects", default=None,
                   help="Path to subjects.csv for orientation normalization "
                        "(columns: dyad, subject, seat_position, facing_direction)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Feature preparation (reuses utils.py)
# ---------------------------------------------------------------------------

def prepare_features(features_df, args):
    """Load, bin, filter, and reduce features. Returns (data_array, col_names, bin_times)."""
    print("\nBinning features...")
    binned = bin_timeseries(features_df, "time_seconds", args.time_resolution)
    print(f"  Binned: {binned.shape[0]} bins")

    print("\nFiltering features...")
    filtered = filter_timevarying_columns(binned)

    method = args.reduce_features if args.reduce_features != "all" else "pca"

    if method in ("pca", "fa", "ica"):
        print(f"\nReducing features ({method.upper()})...")
        comp_df, labels, loadings_df = run_reduction(filtered, method, args.n_components)
        data_df = comp_df
    elif method == "grouped":
        print("\nComputing grouped dimensions...")
        grouped, _ = compute_grouped_dimensions(features_df, filtered, args.time_resolution)
        data_df = pd.DataFrame(grouped, index=filtered.index)
    elif method == "every":
        data_df = filtered
    else:
        data_df = filtered

    # Z-score unless disabled (reduction methods already z-score internally)
    if method in ("grouped", "every") and not args.no_zscore:
        scaler = StandardScaler()
        data_df = pd.DataFrame(
            scaler.fit_transform(data_df),
            index=data_df.index,
            columns=data_df.columns,
        )

    col_names = list(data_df.columns)
    bin_times = data_df.index.values * args.time_resolution
    data_array = data_df.values

    # Handle NaN
    data_array = np.nan_to_num(data_array, nan=0.0)

    print(f"  Feature matrix: {data_array.shape[0]} bins x {data_array.shape[1]} dimensions")
    return data_array, col_names, bin_times


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def _hmm_n_params(n_states, n_features, covariance_type):
    """Count free parameters in a GaussianHMM."""
    # Transition matrix: n_states * (n_states - 1)
    # Start probs: n_states - 1
    # Means: n_states * n_features
    k = n_states * (n_states - 1) + (n_states - 1) + n_states * n_features
    if covariance_type == "full":
        k += n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        k += n_states * n_features
    elif covariance_type == "tied":
        k += n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        k += n_states
    return k


def _fit_hmm(data, n_states, n_restarts, covariance_type, seed):
    """Fit GaussianHMM with multiple random restarts, return best model."""
    best_model = None
    best_ll = -np.inf

    for restart in range(n_restarts):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=200,
                random_state=seed + restart,
            )
            model.fit(data)
            ll = model.score(data)
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception:
            continue

    return best_model, best_ll


def _find_bic_kneedle(selection_df):
    """Find elbow in BIC curve using the kneedle algorithm.

    Uses the kneed package if available, otherwise falls back to the
    geometric max-distance method: normalize k and BIC to [0,1], then
    find the point with maximum perpendicular distance from the line
    connecting the first and last points.

    Returns (selected_k, method_label) or (None, None) if detection fails.
    """
    df = selection_df.sort_values("n_states").reset_index(drop=True)
    if len(df) < 3:
        return None, None

    k_vals = df["n_states"].values.astype(float)
    bic_vals = df["BIC"].values

    # Try kneed package first
    if HAS_KNEED:
        kl = KneeLocator(k_vals, bic_vals, curve="convex", direction="decreasing")
        if kl.knee is not None:
            return int(kl.knee), "elbow_kneedle"

    # Geometric distance fallback
    k_range = k_vals[-1] - k_vals[0]
    bic_range = bic_vals.max() - bic_vals.min()
    if k_range <= 0 or bic_range <= 0:
        return None, None

    # Normalize to [0,1]
    k_norm = (k_vals - k_vals[0]) / k_range
    bic_norm = (bic_vals - bic_vals.min()) / bic_range

    # Line from first point to last point
    x0, y0 = k_norm[0], bic_norm[0]
    x1, y1 = k_norm[-1], bic_norm[-1]

    # Perpendicular distance from each point to the line
    line_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if line_len == 0:
        return None, None

    distances = np.abs((y1 - y0) * k_norm - (x1 - x0) * bic_norm + x1 * y0 - y1 * x0) / line_len

    best_idx = np.argmax(distances)
    return int(k_vals[best_idx]), "elbow_geometric"


def run_hmm(data, args):
    """Run HMM segmentation. Returns (states, probs, model, selection_df|None)."""
    if not HAS_HMMLEARN:
        sys.exit("Error: hmmlearn not installed. Run: pip install hmmlearn")

    n_samples, n_features = data.shape
    auto = args.n_states == "auto"
    selection_df = None

    if auto:
        print(f"\nAuto-selecting states ({args.min_states}–{args.max_states})...")
        records = []
        best_bic = np.inf
        best_n = args.min_states

        for n in range(args.min_states, args.max_states + 1):
            model, ll = _fit_hmm(data, n, args.n_restarts, args.covariance_type, args.seed)
            if model is None:
                continue
            k = _hmm_n_params(n, n_features, args.covariance_type)
            bic = -2 * ll + k * np.log(n_samples)
            aic = -2 * ll + 2 * k
            records.append({"n_states": n, "log_likelihood": ll, "BIC": bic, "AIC": aic, "n_params": k})
            print(f"    n={n}: LL={ll:.1f}  BIC={bic:.1f}  AIC={aic:.1f}")
            if bic < best_bic:
                best_bic = bic
                best_n = n

        selection_df = pd.DataFrame(records)

        if args.state_selection == "elbow":
            elbow_n, elbow_method = _find_bic_kneedle(selection_df)
            if elbow_n is not None:
                n_states = elbow_n
                sel_bic = float(selection_df.loc[selection_df["n_states"] == elbow_n, "BIC"].values[0])
                selection_method = elbow_method
                print(f"  Selected: {n_states} states ({elbow_method}, BIC={sel_bic:.1f})")
            else:
                n_states = best_n
                selection_method = "elbow_fallback_min"
                print(f"  Selected: {n_states} states (argmin BIC={best_bic:.1f}, elbow detection failed)")
        else:
            n_states = best_n
            selection_method = "min_bic"
            print(f"  Selected: {n_states} states (min BIC={best_bic:.1f})")
    else:
        n_states = int(args.n_states)
        selection_method = "fixed"

    print(f"\nFitting final HMM with {n_states} states...")
    model, ll = _fit_hmm(data, n_states, args.n_restarts, args.covariance_type, args.seed)
    if model is None:
        sys.exit("Error: HMM fitting failed for all restarts")

    states = model.predict(data)
    probs = model.predict_proba(data)
    print(f"  Log-likelihood: {ll:.1f}")

    return states, probs, model, selection_df, selection_method


# ---------------------------------------------------------------------------
# Changepoint
# ---------------------------------------------------------------------------

def run_changepoint(data, args):
    """Run changepoint detection. Returns states array."""
    if not HAS_RUPTURES:
        sys.exit("Error: ruptures not installed. Run: pip install ruptures")

    auto = args.n_states == "auto"

    if auto:
        print("\nAuto-detecting changepoints (Pelt)...")
        algo = ruptures.Pelt(model="rbf", min_size=max(2, int(args.min_duration / args.time_resolution)))
        algo.fit(data)
        bkps = algo.predict(pen=1.0)
    else:
        n_bkps = int(args.n_states) - 1
        print(f"\nDetecting {n_bkps} changepoints (BinSeg)...")
        algo = ruptures.Binseg(model="rbf", min_size=max(2, int(args.min_duration / args.time_resolution)))
        algo.fit(data)
        bkps = algo.predict(n_bkps=n_bkps)

    # Convert breakpoints to state assignments
    states = np.zeros(len(data), dtype=int)
    prev = 0
    for i, bp in enumerate(bkps):
        bp = min(bp, len(data))
        states[prev:bp] = i
        prev = bp

    n_states = len(set(states))
    print(f"  Detected {n_states} segments ({len(bkps)} breakpoints)")
    return states


# ---------------------------------------------------------------------------
# K-means window
# ---------------------------------------------------------------------------

def run_kmeans_window(data, args):
    """Run windowed k-means clustering. Returns states array."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    window = args.window_size
    n_bins = len(data)

    # Compute windowed means
    n_windows = max(1, n_bins - window + 1)
    windowed = np.array([data[i:i + window].mean(axis=0) for i in range(n_windows)])

    auto = args.n_states == "auto"

    if auto:
        print(f"\nAuto-selecting k ({args.min_states}–{args.max_states}) via silhouette...")
        best_k = args.min_states
        best_sil = -1

        for k in range(args.min_states, min(args.max_states + 1, n_windows)):
            km = KMeans(n_clusters=k, n_init=10, random_state=args.seed)
            labels = km.fit_predict(windowed)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(windowed, labels)
            print(f"    k={k}: silhouette={sil:.3f}")
            if sil > best_sil:
                best_sil = sil
                best_k = k

        n_states = best_k
        print(f"  Selected: k={n_states} (silhouette={best_sil:.3f})")
    else:
        n_states = int(args.n_states)

    km = KMeans(n_clusters=n_states, n_init=10, random_state=args.seed)
    window_labels = km.fit_predict(windowed)

    # Map window labels back to individual bins
    states = np.zeros(n_bins, dtype=int)
    for i in range(n_windows):
        states[i] = window_labels[i]
    # Fill trailing bins
    if n_windows < n_bins:
        states[n_windows:] = window_labels[-1]

    print(f"  Clustered {n_windows} windows into {n_states} states")
    return states


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def apply_min_duration(states, min_dur_seconds, bin_size):
    """Merge state runs shorter than min_duration into neighbors."""
    if min_dur_seconds <= 0:
        return states

    min_bins = max(1, int(min_dur_seconds / bin_size))
    result = states.copy()
    changed = True

    while changed:
        changed = False
        runs = []
        i = 0
        while i < len(result):
            j = i
            while j < len(result) and result[j] == result[i]:
                j += 1
            runs.append((i, j, result[i]))
            i = j

        for idx, (start, end, state) in enumerate(runs):
            if (end - start) < min_bins:
                # Merge into the longer neighbor
                left_len = (runs[idx - 1][1] - runs[idx - 1][0]) if idx > 0 else 0
                right_len = (runs[idx + 1][1] - runs[idx + 1][0]) if idx < len(runs) - 1 else 0
                if left_len >= right_len and idx > 0:
                    result[start:end] = runs[idx - 1][2]
                elif idx < len(runs) - 1:
                    result[start:end] = runs[idx + 1][2]
                changed = True
                break  # restart scan after each merge

    return result


def sort_states_by_energy(states, data):
    """Re-label states so State 1 = quietest, State N = most animated."""
    unique = np.unique(states)
    energies = {}
    for s in unique:
        mask = states == s
        energies[s] = np.mean(np.abs(data[mask]))

    sorted_states = sorted(energies, key=energies.get)
    mapping = {old: new + 1 for new, old in enumerate(sorted_states)}

    return np.array([mapping[s] for s in states]), mapping


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def build_segments_csv(states, probs, data, col_names, bin_times, bin_size):
    """Build segments DataFrame (one row per time bin)."""
    df = pd.DataFrame(data, columns=col_names)
    df.insert(0, "time_seconds", bin_times)
    df.insert(1, "frame_idx", (bin_times / bin_size).astype(int))
    df.insert(2, "state", states)
    if probs is not None:
        # Max probability for assigned state
        state_prob = np.array([probs[i, states[i] - 1] if states[i] - 1 < probs.shape[1] else np.nan
                               for i in range(len(states))])
        df.insert(3, "state_probability", state_prob)
    return df


def build_state_profiles(states, data, col_names, bin_size):
    """Build state profiles DataFrame (one row per state)."""
    rows = []
    for s in sorted(np.unique(states)):
        mask = states == s
        n_tp = mask.sum()
        row = {
            "state": s,
            "n_timepoints": int(n_tp),
            "duration_seconds": float(n_tp * bin_size),
            "proportion": float(n_tp / len(states)),
        }
        for j, col in enumerate(col_names):
            vals = data[mask, j]
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_sd"] = float(np.nanstd(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def build_transition_matrix(states):
    """Build transition probability matrix."""
    unique = sorted(np.unique(states))
    n = len(unique)
    idx_map = {s: i for i, s in enumerate(unique)}
    counts = np.zeros((n, n), dtype=float)

    for i in range(len(states) - 1):
        counts[idx_map[states[i]], idx_map[states[i + 1]]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, where=row_sums > 0, out=np.zeros_like(counts))

    return pd.DataFrame(probs, index=unique, columns=unique)


def build_state_durations(states, bin_times, bin_size):
    """Build state durations DataFrame (one row per continuous visit)."""
    rows = []
    i = 0
    while i < len(states):
        j = i
        while j < len(states) and states[j] == states[i]:
            j += 1
        rows.append({
            "state": int(states[i]),
            "start_time": float(bin_times[i]),
            "end_time": float(bin_times[min(j, len(bin_times)) - 1] + bin_size),
            "duration": float((j - i) * bin_size),
        })
        i = j
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

STATE_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                "#9c755f", "#bab0ac"]


def plot_model_selection(selection_df, selected_n, output_path):
    """Plot BIC/AIC curves with selected model highlighted."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection_df["n_states"], selection_df["BIC"], "o-", label="BIC", color="#4e79a7")
    ax.plot(selection_df["n_states"], selection_df["AIC"], "s--", label="AIC", color="#e15759")

    sel_row = selection_df[selection_df["n_states"] == selected_n]
    if not sel_row.empty:
        ax.scatter([selected_n], sel_row["BIC"].values, s=200, marker="*",
                   color="gold", zorder=10, edgecolors="black", linewidths=1,
                   label=f"Selected: {selected_n} states")

    ax.set_xlabel("Number of states", fontsize=12)
    ax.set_ylabel("Information criterion", fontsize=12)
    ax.set_title("Model Selection (lower = better)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_state_timeline(states, bin_times, output_path, profiles_df=None):
    """Plot color-coded state timeline."""
    fig, ax = plt.subplots(figsize=(14, 2.5))

    unique_states = sorted(np.unique(states))
    for s in unique_states:
        color = STATE_COLORS[(s - 1) % len(STATE_COLORS)]
        mask = states == s
        ax.fill_between(bin_times, 0, 1, where=mask, color=color,
                        alpha=0.85, label=f"State {s}", step="mid")

    ax.set_xlim(bin_times[0], bin_times[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title("Behavioral State Timeline", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, ncol=min(len(unique_states), 5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_state_distributions(states, data, col_names, output_path):
    """Box plots of each feature dimension split by state."""
    n_dims = min(len(col_names), 12)  # cap at 12 for readability
    unique_states = sorted(np.unique(states))

    fig, axes = plt.subplots(1, n_dims, figsize=(max(10, n_dims * 2), 5), sharey=False)
    if n_dims == 1:
        axes = [axes]

    for j, col in enumerate(col_names[:n_dims]):
        ax = axes[j]
        box_data = [data[states == s, j] for s in unique_states]
        bp = ax.boxplot(box_data, tick_labels=[str(s) for s in unique_states],
                        patch_artist=True, widths=0.6)
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(STATE_COLORS[(unique_states[k] - 1) % len(STATE_COLORS)])
            patch.set_alpha(0.7)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("State", fontsize=8)
        if j == 0:
            ax.set_ylabel("Value", fontsize=9)

    fig.suptitle("Feature Distributions by State", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary_report(path, args, states, profiles_df, trans_df, durations_df,
                         selection_df=None, selection_method=None):
    """Write human-readable summary report."""
    n_states = len(np.unique(states))

    lines = [
        "=" * 60,
        "  SocialCurrents — Segmentation Report",
        "=" * 60,
        "",
        f"Method:           {args.method}",
        f"States found:     {n_states}",
        f"Reduce features:  {args.reduce_features}",
        f"Time resolution:  {args.time_resolution}s",
        f"Min duration:     {args.min_duration}s",
    ]

    if selection_df is not None and not selection_df.empty:
        sel_row = selection_df[selection_df["n_states"] == n_states]
        if not sel_row.empty:
            sel_bic = float(sel_row["BIC"].values[0])
        else:
            sel_bic = float(selection_df.loc[selection_df["BIC"].idxmin(), "BIC"])
        method_label = f", {selection_method}" if selection_method else ""
        lines.append(f"Selection:        {n_states} states "
                     f"(BIC={sel_bic:.1f}{method_label})")
    lines.append("")

    # State descriptions
    lines.append("State Profiles:")
    lines.append("-" * 50)
    for _, row in profiles_df.iterrows():
        s = int(row["state"])
        dur = row["duration_seconds"]
        pct = row["proportion"] * 100
        lines.append(f"  State {s}: {dur:.1f}s ({pct:.1f}% of conversation)")

        # Top features (highest absolute mean)
        feat_cols = [c for c in profiles_df.columns if c.endswith("_mean")]
        feat_vals = {c.replace("_mean", ""): row[c] for c in feat_cols}
        top = sorted(feat_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for name, val in top:
            lines.append(f"    {name}: {val:+.3f}")
        lines.append("")

    # Mean durations per state
    lines.append("Mean Visit Durations:")
    lines.append("-" * 50)
    for s in sorted(np.unique(states)):
        sub = durations_df[durations_df["state"] == s]
        if not sub.empty:
            lines.append(f"  State {s}: mean={sub['duration'].mean():.1f}s, "
                         f"median={sub['duration'].median():.1f}s, "
                         f"n_visits={len(sub)}")
    lines.append("")

    # Most common transitions
    lines.append("Transition Matrix:")
    lines.append("-" * 50)
    lines.append(trans_df.to_string(float_format="%.3f"))
    lines.append("")

    text = "\n".join(lines)
    Path(path).write_text(text)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def segment_subject(csv_path, out_dir, args):
    """Run segmentation on one subject. Returns summary dict for batch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive subject/dyad from filename
    stem = Path(csv_path).stem.replace("_timeseries_features", "")
    parts = stem.split("_", 1)
    dyad_id = parts[0] if len(parts) > 1 else ""
    subject_id = parts[1] if len(parts) > 1 else stem

    # Check overwrite
    segments_path = out_dir / "segments.csv"
    if not args.overwrite and segments_path.exists():
        print(f"  Already exists: {segments_path}")
        print("  Skipping (use --overwrite to replace)")
        return None

    # Check method availability
    if args.method == "hmm" and not HAS_HMMLEARN:
        sys.exit("Error: hmmlearn not installed (pip install hmmlearn)")
    if args.method == "changepoint" and not HAS_RUPTURES:
        sys.exit("Error: ruptures not installed (pip install ruptures)")

    facing = resolve_facing(csv_path, getattr(args, "subjects", None))

    print("Loading data...")
    features_df = load_features(csv_path, facing_direction=facing)

    data, col_names, bin_times = prepare_features(features_df, args)

    # Dispatch to method
    probs = None
    selection_df = None
    selection_method = None

    if args.method == "hmm":
        states, probs, model, selection_df, selection_method = run_hmm(data, args)
    elif args.method == "changepoint":
        states = run_changepoint(data, args)
    elif args.method == "kmeans-window":
        states = run_kmeans_window(data, args)
    else:
        sys.exit(f"Unknown method: {args.method}")

    # Post-process
    print("\nPost-processing...")
    states = apply_min_duration(states, args.min_duration, args.time_resolution)
    states, mapping = sort_states_by_energy(states, data)

    # Remap probs to match sorted states
    if probs is not None:
        inv_map = {v: k for k, v in mapping.items()}
        sorted_cols = [inv_map[s] for s in sorted(mapping.values())]
        probs = probs[:, sorted_cols]

    n_states = len(np.unique(states))
    print(f"  Final: {n_states} states (after smoothing and sorting)")

    # Build outputs
    print("\nSaving outputs...")

    segments_df = build_segments_csv(states, probs, data, col_names, bin_times, args.time_resolution)
    header = {
        "method": args.method,
        "n_states": n_states,
        "selection_method": selection_method or "n/a",
        "reduce_features": args.reduce_features,
        "n_components": args.n_components,
        "time_resolution": f"{args.time_resolution}s",
        "min_duration": f"{args.min_duration}s",
        "features_file": str(csv_path),
    }
    write_csv_with_header(segments_df, segments_path, header)

    profiles_df = build_state_profiles(states, data, col_names, args.time_resolution)
    profiles_df.to_csv(out_dir / "state_profiles.csv", index=False, float_format="%.4f")
    print(f"  Saved: {out_dir / 'state_profiles.csv'}")

    trans_df = build_transition_matrix(states)
    trans_df.to_csv(out_dir / "transition_matrix.csv", float_format="%.4f")
    print(f"  Saved: {out_dir / 'transition_matrix.csv'}")

    durations_df = build_state_durations(states, bin_times, args.time_resolution)
    durations_df.to_csv(out_dir / "state_durations.csv", index=False, float_format="%.2f")
    print(f"  Saved: {out_dir / 'state_durations.csv'}")

    if selection_df is not None:
        selection_df.to_csv(out_dir / "model_selection.csv", index=False, float_format="%.2f")
        print(f"  Saved: {out_dir / 'model_selection.csv'}")
        selected_n = n_states
        plot_model_selection(selection_df, selected_n, out_dir / "model_selection.png")

    # Plots
    print("\nGenerating plots...")
    plot_state_timeline(states, bin_times, out_dir / "state_timeline.png", profiles_df)
    plot_state_distributions(states, data, col_names, out_dir / "state_distributions.png")

    # Summary report
    write_summary_report(out_dir / "summary_report.txt", args, states,
                         profiles_df, trans_df, durations_df, selection_df,
                         selection_method)

    print(f"\nDone. {n_states} behavioral states identified.")

    # Return batch summary row
    best_bic = None
    if selection_df is not None and not selection_df.empty:
        best_bic = float(selection_df["BIC"].min())

    return {
        "subject_id": subject_id,
        "dyad_id": dyad_id,
        "n_states": int(n_states),
        "method": args.method,
        "best_bic": best_bic,
        "features_file": str(csv_path),
    }


def main():
    args = parse_args()
    input_path = Path(args.features)

    if input_path.is_file():
        # Single subject
        segment_subject(str(input_path), args.output_dir, args)
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
            parts = stem.split("_", 1)
            if len(parts) > 1:
                sub_dir = out_dir / parts[0] / parts[1]
            else:
                sub_dir = out_dir / stem

            print(f"\n{'=' * 60}")
            print(f"  Processing: {stem}")
            print(f"{'=' * 60}")
            try:
                row = segment_subject(str(csv_path), str(sub_dir), args)
                if row is not None:
                    batch_rows.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")

        if batch_rows:
            batch_df = pd.DataFrame(batch_rows)
            batch_df.to_csv(out_dir / "batch_summary.csv", index=False, float_format="%.2f")
            print(f"\n  Saved: {out_dir / 'batch_summary.csv'} ({len(batch_df)} subjects)")

            print(f"\n\nBatch complete: {len(batch_rows)} subjects processed.")
    else:
        sys.exit(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()
