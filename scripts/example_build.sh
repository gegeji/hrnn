#!/usr/bin/env bash
# Build HRNN + HAMG + HNSW indexes for one dataset.
# Edit DATA_DIR, DATASET, and parameters below to match your environment.
set -euo pipefail

DATA_DIR=${DATA_DIR:-./DATA}
DATASET=${DATASET:-gist}
THREADS=${THREADS:-$(nproc)}

BASE="${DATA_DIR}/${DATASET}/${DATASET}_base.fbin"
HRNN_IDX="${DATA_DIR}/${DATASET}/${DATASET}_hrnn.idx"
HAMG_IDX="${DATA_DIR}/${DATASET}/${DATASET}_hamg.idx"
HNSW_IDX="${DATA_DIR}/${DATASET}/${DATASET}_hnsw.idx"

# Build the project once
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"${THREADS}"

# HRNN (main method)
./build/hrnn_build \
    --base            "${BASE}" \
    --index_path      "${HRNN_IDX}" \
    --M               16 \
    --ef_construction 400 \
    --K_knng          500 \
    --nn_iters        10 \
    --nn_sample       50 \
    --num_threads     "${THREADS}"

# HAMG baseline
./build/hamg_build \
    --base            "${BASE}" \
    --index_path      "${HAMG_IDX}" \
    --M  16 --M0 5000 --C 500 --dm 80 \
    --ef_construction 200 \
    --num_threads     "${THREADS}"

# Plain HNSW (used by knn_search and naive RkNN baselines)
./build/knn_build \
    --base            "${BASE}" \
    --index_path      "${HNSW_IDX}" \
    --M 16 --ef_construction 200 \
    --num_threads     "${THREADS}"
