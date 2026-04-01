#!/bin/bash
# ============================================================
# run_batch_exhaustive_synchrony.sh
#
# Runs the full exhaustive synchrony pipeline for all dyads:
#   synchrony (7 reduction methods) → trust_from_synch →
#   synch_from_features → synch_by_states
#
# Rating path: data/ratings/trustworthiness/dyadNNN_subXXX_trustworthiness.csv
# ============================================================
set -euo pipefail

SC=~/Dropbox/research/cont-comp-model/pipeline
STUDY=~/Dropbox/research/cont-comp-model
RATINGS=$STUDY/data/ratings/trustworthiness

# Synchrony settings
METHODS="pearson,crosscorr,concordance,rqa,granger,coherence,wavelet"
WINDOW=30
STEP=5
RESOLUTION=0.5
PERMS=100
N_COMP=5
N_COMP_GROUPED=2
N_CLUSTERS=6

REDUCE_METHODS=("grouped" "pca" "ica" "fa" "cca" "grouped-pca" "cluster")

cd "$SC"

# DYAD  SUB_A  SUB_B
DYADS=(
  "dyad001 sub001 sub002"
  "dyad002 sub003 sub004"
  "dyad005 sub009 sub010"
  "dyad006 sub011 sub012"
  "dyad007 sub013 sub014"
  "dyad008 sub015 sub016"
  "dyad009 sub017 sub018"
  "dyad014 sub027 sub028"
  "dyad015 sub029 sub030"
)

TOTAL_DYADS=${#DYADS[@]}
DYAD_NUM=0

for ENTRY in "${DYADS[@]}"; do
  read -r DYAD A B <<< "$ENTRY"
  DYAD_NUM=$((DYAD_NUM + 1))

  echo ""
  echo "############################################################"
  echo "  DYAD $DYAD_NUM/$TOTAL_DYADS: $DYAD ($A + $B)"
  echo "############################################################"

  OUT=$STUDY/output/$DYAD
  FEAT_A=$OUT/$A/extract/${DYAD}_${A}_timeseries_features.csv
  FEAT_B=$OUT/$B/extract/${DYAD}_${B}_timeseries_features.csv
  SYNCH_BASE=$OUT/${A}_${B}/synchrony
  SEGMENTS=$OUT/$B/segments/segments.csv

  # Rating files: dyadNNN_subXXX_trustworthiness.csv
  RATING_A=$RATINGS/${DYAD}_${A}_trustworthiness.csv
  RATING_B=$RATINGS/${DYAD}_${B}_trustworthiness.csv

  # Verify feature inputs exist
  if [ ! -f "$FEAT_A" ]; then
    echo "  WARNING: Missing $FEAT_A — skipping $DYAD"
    continue
  fi
  if [ ! -f "$FEAT_B" ]; then
    echo "  WARNING: Missing $FEAT_B — skipping $DYAD"
    continue
  fi

  HAS_RATING_A=false
  [ -f "$RATING_A" ] && HAS_RATING_A=true
  HAS_RATING_B=false
  [ -f "$RATING_B" ] && HAS_RATING_B=true

  HAS_SEGMENTS=false
  [ -f "$SEGMENTS" ] && HAS_SEGMENTS=true

  # ========================================
  #  STEP 1: Synchrony (all reduction methods)
  # ========================================
  echo ""
  echo "  ── STEP 1: Synchrony ──"

  for METHOD in "${REDUCE_METHODS[@]}"; do
    echo "    reduce: $METHOD"

    case "$METHOD" in
      grouped)      NC_FLAG="" ;;
      grouped-pca)  NC_FLAG="--n-components $N_COMP_GROUPED" ;;
      cluster)      NC_FLAG="--n-components $N_CLUSTERS" ;;
      *)            NC_FLAG="--n-components $N_COMP" ;;
    esac

    conda run --no-capture-output -n pipeline-env \
      python analysis/synchronize.py \
        --person-a "$FEAT_A" --person-b "$FEAT_B" \
        --reduce-features "$METHOD" $NC_FLAG \
        --methods "$METHODS" \
        --window-size "$WINDOW" --step-size "$STEP" \
        --time-resolution "$RESOLUTION" --permutations "$PERMS" \
        -o "$SYNCH_BASE/$METHOD/" \
        --overwrite \
    || echo "    FAILED: $METHOD synchrony"
  done

  # ========================================
  #  STEP 2: Synchrony → Trustworthiness
  #  Run for each rating file that exists
  # ========================================
  echo ""
  echo "  ── STEP 2: trust_from_synch ──"

  for METHOD in "${REDUCE_METHODS[@]}"; do
    SYNCH_TS="$SYNCH_BASE/$METHOD/synchrony_timeseries.csv"
    [ ! -f "$SYNCH_TS" ] && continue

    # sub_A's rating (sub_A was rated by sub_B)
    if [ "$HAS_RATING_A" = true ]; then
      echo "    $METHOD: $A rated by $B"
      conda run --no-capture-output -n pipeline-env \
        python analysis/correlate.py --mode single \
          -f "$SYNCH_TS" \
          -t "$RATING_A" \
          --reduce-features every \
          --rater "$B" --target-id "$A" \
          --label Trustworthiness \
          -o "$OUT/${A}_${B}/trust_from_synch/$METHOD/" \
          --overwrite \
      || echo "    FAILED"
    fi

    # sub_B's rating (sub_B was rated by sub_A)
    if [ "$HAS_RATING_B" = true ]; then
      echo "    $METHOD: $B rated by $A"
      conda run --no-capture-output -n pipeline-env \
        python analysis/correlate.py --mode single \
          -f "$SYNCH_TS" \
          -t "$RATING_B" \
          --reduce-features every \
          --rater "$A" --target-id "$B" \
          --label Trustworthiness \
          -o "$OUT/${A}_${B}/trust_from_synch/${METHOD}_${B}/" \
          --overwrite \
      || echo "    FAILED"
    fi
  done

  # ========================================
  #  STEP 3: Features → Synchrony
  # ========================================
  echo ""
  echo "  ── STEP 3: synch_from_features ──"

  for METHOD in "${REDUCE_METHODS[@]}"; do
    SYNCH_TS="$SYNCH_BASE/$METHOD/synchrony_timeseries.csv"
    [ ! -f "$SYNCH_TS" ] && continue

    echo "    $B features × $METHOD synch"
    conda run --no-capture-output -n pipeline-env \
      python analysis/correlate.py --mode multi \
        -f "$FEAT_B" \
        -t "$SYNCH_TS" \
        --reduce-features pca --n-components 5 \
        --reduce-target pca --n-target-components 3 \
        --label "Synchrony ($METHOD)" \
        -o "$OUT/${A}_${B}/synch_from_features/$METHOD/" \
        --overwrite \
    || echo "    FAILED"
  done

  # ========================================
  #  STEP 4: States × Synchrony
  # ========================================
  if [ "$HAS_SEGMENTS" = true ]; then
    echo ""
    echo "  ── STEP 4: synch_by_states ──"

    for METHOD in "${REDUCE_METHODS[@]}"; do
      WAVELET_TS="$SYNCH_BASE/$METHOD/wavelet_timeseries.csv"
      [ ! -f "$WAVELET_TS" ] && continue

      echo "    states × $METHOD"
      conda run --no-capture-output -n pipeline-env \
        python analysis/map_states.py \
          --states "$SEGMENTS" \
          --signal "$WAVELET_TS" \
          --signal-col mean_coherence \
          --signal-label "Wavelet Coherence ($METHOD)" \
          -o "$OUT/${A}_${B}/synch_by_states/$METHOD/" \
          --overwrite \
      || echo "    FAILED"
    done
  fi

  echo ""
  echo "  ✓ $DYAD complete"
done

# ============================================================
#  Final summary
# ============================================================
echo ""
echo "============================================================"
echo "  ALL DONE — Summary"
echo "============================================================"

for ENTRY in "${DYADS[@]}"; do
  read -r DYAD A B <<< "$ENTRY"
  SYNCH_BASE=$STUDY/output/$DYAD/${A}_${B}/synchrony
  N_METHODS=0
  for METHOD in "${REDUCE_METHODS[@]}"; do
    [ -f "$SYNCH_BASE/$METHOD/synchrony_timeseries.csv" ] && N_METHODS=$((N_METHODS + 1))
  done
  echo "  $DYAD ($A+$B): $N_METHODS/${#REDUCE_METHODS[@]} reduction methods"
done
