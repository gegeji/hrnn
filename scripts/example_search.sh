#!/usr/bin/env bash
# Run HRNN search + HAMG / HNSW / SFT / RDT baselines on one dataset.
# Assumes indexes already built (see example_build.sh) and groundtruth
# generated (see scripts/gen_rknn_gt.py).
set -euo pipefail

DATA_DIR=${DATA_DIR:-./DATA}
DATASET=${DATASET:-gist}
K=${K:-10}

BASE="${DATA_DIR}/${DATASET}/${DATASET}_base.fbin"
QUERY="${DATA_DIR}/${DATASET}/${DATASET}_query.fbin"
RKNN_GT="${DATA_DIR}/${DATASET}/${DATASET}_rknn_gt_k${K}.bin"
KNN_GT="${DATA_DIR}/${DATASET}/${DATASET}_knn_gt_k${K}.bin"
HRNN_IDX="${DATA_DIR}/${DATASET}/${DATASET}_hrnn.idx"
HAMG_IDX="${DATA_DIR}/${DATASET}/${DATASET}_hamg.idx"
HNSW_IDX="${DATA_DIR}/${DATASET}/${DATASET}_hnsw.idx"

echo "=== HRNN search (main method) ==="
./build/hrnn_search \
    --index_path "${HRNN_IDX}" \
    --query      "${QUERY}" \
    --gt         "${RKNN_GT}" \
    --k          "${K}" \
    --K_prime    500 \
    --m          "1,3,5,10,20,50" \
    --ef_search  200

echo "=== HAMG search ==="
./build/hamg_search \
    --index_path "${HAMG_IDX}" \
    --query      "${QUERY}" \
    --gt         "${RKNN_GT}" \
    --k          "${K}" \
    --r          "0.5,0.8,0.9,1.0" \
    --ef_search  200

echo "=== HNSW kNN baseline ==="
./build/knn_search \
    --index_path "${HNSW_IDX}" \
    --query      "${QUERY}" \
    --gt         "${KNN_GT}" \
    --k          "${K}" \
    --ef_search  "25,50,100,125,150"

echo "=== Naive SFT on HNSW ==="
./build/rknn_hnsw_sft_search \
    --index_path "${HNSW_IDX}" \
    --query      "${QUERY}" \
    --gt         "${RKNN_GT}" \
    --method     sft \
    --t          "1,3,5,7,10" \
    --k          "${K}"

echo "=== Naive RDT on HNSW ==="
./build/rknn_hnsw_sft_search \
    --index_path "${HNSW_IDX}" \
    --query      "${QUERY}" \
    --gt         "${RKNN_GT}" \
    --method     rdt \
    --t          "1,3,5,7,10" \
    --k          "${K}"
