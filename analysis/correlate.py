#!/usr/bin/env python3
"""
Lagged cross-correlation analysis between pipeline timeseries features
and one or more target signals (ratings, physiology, EEG, fNIRS, etc.).

Supports multiple dimensionality-reduction methods (PCA, FA, ICA,
grouped summary dimensions, or raw features) and both single-channel
and multi-channel target modes.
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `analysis.utils` resolves
# when invoked as `python analysis/correlate.py` from the repo root.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from analysis.utils import (
    load_features,
    load_target,
    bin_timeseries,
    filter_timevarying_columns,
    align_bins,
    compute_grouped_dimensions,
    run_reduction,
    lagged_crosscorrelation,
    apply_fdr,
    plot_single_crosscorr,
    plot_multi_heatmap,
    write_csv_with_header,
    GROUPED_DIMS,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Lagged cross-correlation between pipeline features and target signals"
    )
    # Common
    p.add_argument("-f", "--features", required=True,
                   help="Path to timeseries_features.csv")
    p.add_argument("-t", "--target", required=True,
                   help="Path to target CSV (rating, physiology, EEG, fNIRS, etc.)")
    p.add_argument("-o", "--output-dir", default="./analysis_output",
                   help="Output directory (default: ./analysis_output)")
    p.add_argument("--mode", choices=["single", "multi"], default="single",
                   help="single: one target timeseries; multi: multi-channel target (default: single)")
    p.add_argument("--reduce-features",
                   choices=["pca", "fa", "ica", "grouped", "every", "all"],
                   default="pca",
                   help="Feature reduction method (default: pca)")
    p.add_argument("--n-components", type=int, default=5,
                   help="Number of components for PCA/FA/ICA (default: 5)")
    p.add_argument("--bin-size", type=float, default=0.5,
                   help="Time bin size in seconds (default: 0.5)")
    p.add_argument("--lag-min", type=float, default=-5.0,
                   help="Minimum lag in seconds (default: -5)")
    p.add_argument("--lag-max", type=float, default=15.0,
                   help="Maximum lag in seconds (default: 15)")
    p.add_argument("--rater", default=None,
                   help="Rater subject ID (e.g. sub001)")
    p.add_argument("--target-id", default=None,
                   help="Target subject ID (e.g. sub007)")
    p.add_argument("--target-col", default="Value",
                   help="Column name for target values in single mode (default: Value)")
    p.add_argument("--label", default="Trustworthiness",
                   help="What the target measures, used in plot titles (default: Trustworthiness)")
    p.add_argument("--select-features", default=None,
                   help="Comma-separated column names or glob patterns "
                        "(e.g. 'pearson_r_*,crosscorr_r_*') to select before reduction")
    p.add_argument("--no-zscore", action="store_true",
                   help="Skip z-scoring for grouped/every modes (PCA/FA/ICA always z-score internally)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files")

    # Multi-mode only
    multi = p.add_argument_group("multi-mode options")
    multi.add_argument("--reduce-target",
                       choices=["pca", "fa", "ica", "cca", "roi-average", "none"],
                       default="pca",
                       help="Target reduction method for multi mode (default: pca)")
    multi.add_argument("--cca-joint", action="store_true",
                       help="Use joint CCA (features + target together) instead of target-only CCA")
    multi.add_argument("--n-target-components", type=int, default=5,
                       help="Number of target components (default: 5)")
    multi.add_argument("--roi-config", default=None,
                       help="JSON file mapping ROI names to channel lists (for --reduce-target roi-average)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def _apply_select_features(df, select_str):
    """Filter DataFrame columns by comma-separated names or glob patterns.

    Returns a subset of *df* containing only columns that match at least one
    pattern.  Patterns are matched with ``fnmatch`` so ``*`` and ``?`` work.
    """
    patterns = [p.strip() for p in select_str.split(",") if p.strip()]
    all_cols = list(df.columns)
    matched = []
    for pat in patterns:
        matched.extend(fnmatch.filter(all_cols, pat))
    # Preserve original order, deduplicate
    seen = set()
    selected = [c for c in all_cols if c in set(matched) and not (c in seen or seen.add(c))]
    if not selected:
        sys.exit(f"Error: --select-features matched no columns. "
                 f"Patterns: {patterns}; available: {all_cols[:20]}...")
    print(f"  --select-features: {len(selected)}/{len(all_cols)} columns selected")
    return df[selected]


# ---------------------------------------------------------------------------
# Feature dimension building
# ---------------------------------------------------------------------------

def _build_dimensions_for_method(method, raw_features_df, binned_filtered,
                                 bin_size, n_components, no_zscore, out_dir, prefix):
    """Build feature dimensions for a single reduction method.

    Returns (dimensions_dict, dim_labels, movement_status).
    dimensions_dict maps dimension name -> 1-D numpy array.
    """
    dimensions = {}
    dim_labels = {}
    loadings_df = None
    movement_status = "ok"

    if method in ("pca", "fa", "ica"):
        print(f"\nRunning {method.upper()}...")
        comp_df, labels, loadings_df = run_reduction(binned_filtered, method, n_components)
        for col in comp_df.columns:
            dimensions[col] = comp_df[col].values
        dim_labels.update(labels)

        # Save loadings
        if loadings_df is not None:
            loadings_path = out_dir / f"{prefix}loadings_{method}.csv"
            loadings_df.to_csv(loadings_path, index=False, float_format="%.6f")
            print(f"  Saved loadings: {loadings_path}")

    if method != "every":
        # Add grouped dimensions
        print("\nComputing grouped dimensions...")
        grouped, movement_status = compute_grouped_dimensions(
            raw_features_df, binned_filtered, bin_size
        )
        # For grouped/every: z-score explicitly unless --no-zscore
        if method in ("grouped",) and not no_zscore:
            for name, arr in grouped.items():
                s = np.nanstd(arr)
                if s > 0:
                    grouped[name] = (arr - np.nanmean(arr)) / s
        dimensions.update(grouped)

    if method == "every":
        # Use every filtered feature individually
        print(f"\nUsing all {binned_filtered.shape[1]} features individually...")
        for col in binned_filtered.columns:
            arr = binned_filtered[col].values
            if not no_zscore:
                s = np.nanstd(arr)
                if s > 0:
                    arr = (arr - np.nanmean(arr)) / s
            dimensions[col] = arr

    return dimensions, dim_labels, movement_status


# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------

def run_single(args):
    """Execute single-target cross-correlation analysis."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.rater}_rates_{args.target_id}_" if args.rater and args.target_id else ""

    print("Loading data...")
    features_df = load_features(args.features)
    target_df = load_target(args.target, value_col=args.target_col)

    # Bin & align
    print("\nBinning features...")
    binned_feat = bin_timeseries(features_df, "time_seconds", args.bin_size)
    print(f"  Binned features: {binned_feat.shape[0]} bins")

    print("Binning target...")
    binned_target = bin_timeseries(target_df, "time_seconds", args.bin_size)
    print(f"  Binned target: {binned_target.shape[0]} bins")

    binned_feat, binned_target, _ = align_bins(binned_feat, binned_target, args.bin_size)
    target_signal = binned_target["Value"].values

    # Filter
    print("\nFiltering features...")
    filtered = filter_timevarying_columns(binned_feat)

    if args.select_features:
        filtered = _apply_select_features(filtered, args.select_features)

    # Determine which methods to run
    methods = ["pca", "fa", "ica", "grouped"] if args.reduce_features == "all" else [args.reduce_features]

    for method in methods:
        suffix = f"_{method}" if args.reduce_features == "all" else ""
        print(f"\n{'='*60}")
        print(f"  Method: {method.upper()}")
        print(f"{'='*60}")

        dimensions, dim_labels, movement_status = _build_dimensions_for_method(
            method, features_df, filtered, args.bin_size,
            args.n_components, args.no_zscore, out_dir, prefix,
        )

        # Cross-correlations
        min_lag = int(round(args.lag_min / args.bin_size))
        max_lag = int(round(args.lag_max / args.bin_size))
        print(f"\nComputing lagged cross-correlations "
              f"(lags {args.lag_min}s to {args.lag_max}s)...")

        rows = []
        for name, signal in dimensions.items():
            lags, rs, ps = lagged_crosscorrelation(signal, target_signal, min_lag, max_lag)
            for lag, r, p_val in zip(lags, rs, ps):
                rows.append({
                    "dimension": name,
                    "lag_seconds": lag * args.bin_size,
                    "correlation": r,
                    "p_value": p_val,
                })

        results = apply_fdr(pd.DataFrame(rows))

        # Output header
        header = {
            "rater": args.rater or "(not specified)",
            "target_id": args.target_id or "(not specified)",
            "target_col": args.target_col,
            "label": args.label,
            "reduce_features": method,
            "n_components": args.n_components,
            "z_scored": "yes (internally)" if method in ("pca","fa","ica") else ("no" if args.no_zscore else "yes"),
            "bin_size": f"{args.bin_size}s",
            "lag_range": f"{args.lag_min}s to {args.lag_max}s",
            "features_file": args.features,
            "target_file": args.target,
        }

        csv_path = out_dir / f"{prefix}correlation{suffix}.csv"
        write_csv_with_header(results, csv_path, header)

        plot_path = out_dir / f"{prefix}correlation{suffix}_plot.png"
        plot_single_crosscorr(results, plot_path, label=args.label,
                              rater=args.rater, target_id=args.target_id,
                              dim_labels=dim_labels)

        # Summary
        _print_summary(results, dimensions, movement_status, method)


# ---------------------------------------------------------------------------
# Multi mode
# ---------------------------------------------------------------------------

def _reduce_target_channels(binned_target, method, n_components, feature_matrix=None,
                            roi_config=None, cca_joint=False):
    """Reduce multi-channel target to components.

    Returns (reduced_df, target_labels, target_loadings_df|None).
    """
    # Drop time/index columns
    target_cols = [c for c in binned_target.columns
                   if c not in ("time_seconds", "bin_time", "frame_idx", "VideoTime", "Value")]
    target_data = binned_target[target_cols].select_dtypes(include=[np.number])
    target_data = target_data.ffill().bfill()

    if method == "none":
        return target_data, {c: c for c in target_data.columns}, None

    if method == "roi-average":
        if roi_config is None:
            sys.exit("Error: --roi-config required for --reduce-target roi-average")
        with open(roi_config) as f:
            rois = json.load(f)
        roi_df = pd.DataFrame(index=target_data.index)
        for roi_name, channels in rois.items():
            avail = [c for c in channels if c in target_data.columns]
            if avail:
                roi_df[roi_name] = target_data[avail].mean(axis=1)
            else:
                print(f"  Warning: no channels found for ROI '{roi_name}'")
        print(f"  ROI averages: {list(roi_df.columns)}")
        return roi_df, {c: c for c in roi_df.columns}, None

    if method == "cca":
        from sklearn.cross_decomposition import CCA
        n_comp = min(n_components, target_data.shape[1], target_data.shape[0])
        if feature_matrix is None:
            sys.exit("Error: CCA requires a feature matrix")

        scaler_t = StandardScaler()
        target_scaled = scaler_t.fit_transform(target_data)
        target_scaled = np.nan_to_num(target_scaled, nan=0.0)

        scaler_f = StandardScaler()
        feat_scaled = scaler_f.fit_transform(feature_matrix)
        feat_scaled = np.nan_to_num(feat_scaled, nan=0.0)

        cca = CCA(n_components=n_comp)
        if cca_joint:
            feat_c, target_c = cca.fit_transform(feat_scaled, target_scaled)
            # For joint CCA, return the target canonical variates
            col_names = [f"CCA_T{i+1}" for i in range(n_comp)]
            print(f"  Joint CCA: {n_comp} canonical variates")
        else:
            cca.fit(feat_scaled, target_scaled)
            target_c = cca.transform(feat_scaled, target_scaled)[1]
            col_names = [f"CCA_T{i+1}" for i in range(n_comp)]
            print(f"  Target-only CCA: {n_comp} canonical variates")

        reduced_df = pd.DataFrame(target_c, index=target_data.index, columns=col_names)
        labels = {c: c for c in col_names}

        # Target loadings from CCA weights
        loading_rows = []
        for i in range(n_comp):
            for j, fname in enumerate(target_cols):
                if j < cca.y_weights_.shape[0]:
                    loading_rows.append({
                        "component": col_names[i],
                        "feature": fname,
                        "loading": float(cca.y_weights_[j, i]),
                    })
        loadings_df = pd.DataFrame(loading_rows) if loading_rows else None
        return reduced_df, labels, loadings_df

    # PCA / FA / ICA on target
    if method in ("pca", "fa", "ica"):
        comp_df, labels, loadings_df = run_reduction(target_data, method, n_components)
        # Rename components to avoid collision with feature components
        rename = {c: f"T_{c}" for c in comp_df.columns}
        comp_df = comp_df.rename(columns=rename)
        labels = {rename[k]: v.replace(k, rename[k]) for k, v in labels.items()}
        if loadings_df is not None:
            loadings_df["component"] = loadings_df["component"].map(lambda x: f"T_{x}")
        return comp_df, labels, loadings_df

    sys.exit(f"Unknown target reduction method: {method}")


def run_multi(args):
    """Execute multi-channel target cross-correlation analysis."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.rater}_rates_{args.target_id}_" if args.rater and args.target_id else ""

    print("Loading data...")
    features_df = load_features(args.features)
    target_df = load_target(args.target)  # no value_col for multi-channel

    # Bin & align
    print("\nBinning features...")
    binned_feat = bin_timeseries(features_df, "time_seconds", args.bin_size)
    print(f"  Binned features: {binned_feat.shape[0]} bins")

    print("Binning target...")
    binned_target = bin_timeseries(target_df, "time_seconds", args.bin_size)
    print(f"  Binned target: {binned_target.shape[0]} bins")

    binned_feat, binned_target, _ = align_bins(binned_feat, binned_target, args.bin_size)

    # Filter features
    print("\nFiltering features...")
    filtered = filter_timevarying_columns(binned_feat)

    if args.select_features:
        filtered = _apply_select_features(filtered, args.select_features)

    # Build feature dimensions (only first method for multi)
    method = args.reduce_features if args.reduce_features != "all" else "pca"
    dimensions, dim_labels, movement_status = _build_dimensions_for_method(
        method, features_df, filtered, args.bin_size,
        args.n_components, args.no_zscore, out_dir, prefix,
    )

    # Reduce target
    print(f"\nReducing target ({args.reduce_target})...")
    feature_matrix = filtered if args.reduce_target == "cca" else None
    reduced_target, target_labels, target_loadings = _reduce_target_channels(
        binned_target, args.reduce_target, args.n_target_components,
        feature_matrix=feature_matrix, roi_config=args.roi_config,
        cca_joint=args.cca_joint,
    )

    # Save target loadings
    if target_loadings is not None:
        tl_path = out_dir / f"{prefix}target_loadings.csv"
        target_loadings.to_csv(tl_path, index=False, float_format="%.6f")
        print(f"  Saved target loadings: {tl_path}")

    # Cross-correlations: each feature dim x each target component
    min_lag = int(round(args.lag_min / args.bin_size))
    max_lag = int(round(args.lag_max / args.bin_size))
    print(f"\nComputing cross-correlations "
          f"({len(dimensions)} dims x {reduced_target.shape[1]} targets, "
          f"lags {args.lag_min}s to {args.lag_max}s)...")

    rows = []
    for dim_name, dim_signal in dimensions.items():
        for tc in reduced_target.columns:
            lags, rs, ps = lagged_crosscorrelation(
                dim_signal, reduced_target[tc].values, min_lag, max_lag
            )
            for lag, r, p_val in zip(lags, rs, ps):
                rows.append({
                    "dimension": dim_name,
                    "target_component": tc,
                    "lag_seconds": lag * args.bin_size,
                    "correlation": r,
                    "p_value": p_val,
                })

    results = apply_fdr(pd.DataFrame(rows))

    header = {
        "mode": "multi",
        "rater": args.rater or "(not specified)",
        "target_id": args.target_id or "(not specified)",
        "label": args.label,
        "reduce_features": method,
        "reduce_target": args.reduce_target,
        "n_components": args.n_components,
        "n_target_components": args.n_target_components,
        "bin_size": f"{args.bin_size}s",
        "lag_range": f"{args.lag_min}s to {args.lag_max}s",
        "features_file": args.features,
        "target_file": args.target,
    }

    csv_path = out_dir / f"{prefix}correlation.csv"
    write_csv_with_header(results, csv_path, header)

    # Heatmap
    feat_dims = list(dimensions.keys())
    target_dims = list(reduced_target.columns)
    heatmap_path = out_dir / f"{prefix}heatmap_peak_lag.png"
    plot_multi_heatmap(results, feat_dims, target_dims, heatmap_path, label=args.label)

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Feature dimensions: {len(feat_dims)}")
    print(f"  Target components: {len(target_dims)}")
    print(f"  Total correlations: {len(results)}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_summary(results, dimensions, movement_status, method):
    """Print terminal summary for single mode."""
    from analysis.utils import HAS_STATSMODELS

    print(f"\n--- Summary ({method.upper()}) ---")
    print(f"  Dimensions: {len(dimensions)}")

    _missing_reasons = {
        "no_columns": "GMP_world columns not found in features CSV",
        "all_zero": "pose detection produced no landmarks for this subject",
    }

    print("\n  Peak |r| per dimension:")
    for dim_name in dimensions:
        sub = results[results["dimension"] == dim_name].dropna(subset=["correlation"])
        if sub.empty:
            reason = ""
            if dim_name == "movement_energy" and movement_status != "ok":
                reason = f" ({_missing_reasons.get(movement_status, movement_status)})"
            print(f"    {dim_name:30s}  (no valid correlations){reason}")
            continue
        idx = sub["correlation"].abs().idxmax()
        row = sub.loc[idx]
        fdr_str = f"  p_fdr={row['p_fdr']:.4f}" if HAS_STATSMODELS and np.isfinite(row['p_fdr']) else ""
        print(f"    {dim_name:30s}  r={row['correlation']:+.4f}  "
              f"lag={row['lag_seconds']:+.1f}s  p={row['p_value']:.4f}{fdr_str}")

    if "movement_energy" not in dimensions and movement_status != "ok":
        reason = _missing_reasons.get(movement_status, movement_status)
        print(f"    {'movement_energy':30s}  (skipped: {reason})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.mode == "single":
        run_single(args)
    elif args.mode == "multi":
        run_multi(args)
    else:
        sys.exit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
