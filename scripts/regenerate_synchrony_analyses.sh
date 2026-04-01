#!/bin/bash
# regenerate_synchrony_analyses.sh

set -euo pipefail

SC=~/Dropbox/research/cont-comp-model/pipeline
STUDY=~/Dropbox/research/cont-comp-model

DYAD=dyad005
A=sub009
B=sub010

OUT=$STUDY/output/$DYAD
FEAT_A=$OUT/$A/extract/${DYAD}_${A}_timeseries_features.csv
FEAT_B=$OUT/$B/extract/${DYAD}_${B}_timeseries_features.csv
RATING=$STUDY/data/ratings/trustworthiness/${DYAD}_${A}_trustworthiness.csv
SYNCH_DIR=$OUT/${A}_${B}/synchrony

cd "$SC"
 

# Synchrony → Ratings
echo ""
echo "=========================================="
echo "  STEP 2: Synchrony → Trustworthiness"
echo "=========================================="

conda run --no-capture-output -n pipeline-env \
  python analysis/correlate.py --mode single \
    -f "$SYNCH_DIR/synchrony_timeseries.csv" \
    -t "$RATING" \
    --reduce-features every \
    --label Trustworthiness \
    -o "$OUT/${A}_${B}/trust_from_synch/" \

#    --overwrite

# Synchrony from features
echo ""
echo "=========================================="
echo "  STEP 3: Synchrony from Person B features"
echo "=========================================="

conda run --no-capture-output -n pipeline-env \
  python analysis/correlate.py --mode multi \
    -f "$FEAT_B" \
    -t "$SYNCH_DIR/synchrony_timeseries.csv" \
    --reduce-features pca --n-components 5 \
    --reduce-target pca --n-target-components 3 \
    --label "Interpersonal Synchrony" \
    -o "$OUT/${A}_${B}/synch_from_features/" \

#    --overwrite

# Synchrony by states
if [ "$HAS_SEGMENTS" = true ]; then
  echo ""
  echo "=========================================="
  echo "  STEP 4: Synchrony by states"
  echo "=========================================="

  conda run --no-capture-output -n pipeline-env \
    python analysis/map_states.py \
      --states "$OUT/$B/segments/segments.csv" \
      --signal "$SYNCH_DIR/wavelet_timeseries.csv" \
      --signal-col mean_coherence \
      --signal-label "Wavelet Coherence" \
      -o "$OUT/${A}_${B}/synch_by_states/" \
      --overwrite
fi

echo ""
echo "Done! Results in $OUT/${A}_${B}/"