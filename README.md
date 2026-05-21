# HRNN: A Hybrid Graph Index for Efficent Reverse k-Nearest Neighbor Search on High-Dimensional Vectors

## 1. Repository Overview

This codebase is built on top of [hnswlib](https://github.com/nmslib/hnswlib)
and adds the HRNN index and several RkNN baselines for comparison.

### 1.1 HRNN

`include/hnswlib/hrnn.h` is the main method. Built drivers:

- `hrnn_build` — build an HRNN index over a base set.
- `hrnn_search` — answer RkNN queries on a built HRNN index.
- `hrnn_dynamic_test` — incremental-update experiment (split a dataset
  into initial + delta, build on initial, insert delta, query at each step).

### 1.2 Baselines

- **HAMG** (graph baseline): `include/hnswlib/hamg.h`, drivers `hamg_build`,
  `hamg_search`. Reimplementation of the HAMG index.
- **HNSW kNN** (kNN-only baseline, for reference numbers): drivers
  `knn_build`, `knn_search` over vendored hnswlib.
- **Naive RkNN over HNSW** (SFT / RDT): `include/hnswlib/hnswalg_rknn.h`,
  driver `rknn_hnsw_sft_search` with `--method {sft, rdt}`. Implements
  the naive single-filter (SFT) and reverse-distance-threshold (RDT)
  RkNN strategies on top of a standard HNSW index.

### 1.3 Layout

```
.
├── CMakeLists.txt
├── README.md
├── include/
│   ├── utils.h                # I/O helpers, recall metrics, timing
│   └── hnswlib/
│       ├── hrnn.h             # HRNN — main method
│       ├── hamg.h             # HAMG baseline
│       ├── hnswalg_rknn.h     # SFT / RDT naive RkNN
│       ├── hnswalg.h          # vendored hnswlib
│       ├── hnswlib.h          # vendored hnswlib
│       ├── bruteforce.h       # vendored hnswlib
│       ├── space_l2.h         # vendored hnswlib
│       ├── space_ip.h         # vendored hnswlib
│       ├── stop_condition.h   # vendored hnswlib
│       └── visited_list_pool.h # vendored hnswlib
├── src/                       # one .cpp per binary (8 total)
└── scripts/
    ├── gen_rknn_gt.py         # RkNN groundtruth (single-machine)
    ├── gen_rknn_gt_large.py   # RkNN groundtruth (GPU, ≥1M points)
    ├── example_build.sh       # end-to-end build example
    └── example_search.sh      # end-to-end search example
```

## 2. Prerequisite

### 2.1 Dependencies

- C++20 compiler (GCC ≥ 10 or Clang ≥ 12)
- CMake ≥ 3.14
- OpenMP
- Boost (`program_options`)
- Hardware with AVX2 (the build uses `-march=native`)

For groundtruth generation (`scripts/gen_rknn_gt_large.py`): Python 3.9+,
PyTorch with CUDA, NumPy.

### 2.2 Datasets

Both `.fvecs` and `.fbin` are accepted (auto-detected by file extension).
Place base / query files under any path; pass absolute paths to the binaries.

You may download public datasets from the following website:

- [SIFT, GIST](http://corpus-texmex.irisa.fr/)
- [DEEP](https://github.com/matsui528/deep1b_gt)
- [MSong](http://www.ifs.tuwien.ac.at/mir/msd/MSongsSubset.html)

## 3. Experiment Setup

### 3.1 Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

This produces 8 binaries under `build/`:
`hrnn_build`, `hrnn_search`, `hrnn_dynamic_test`,
`hamg_build`, `hamg_search`, `knn_build`, `knn_search`,
`rknn_hnsw_sft_search`.

### 3.2 Generate RkNN Groundtruth

```bash
# Small / medium datasets (CPU)
python scripts/gen_rknn_gt.py \
    --base  /path/to/{ds}_base.fbin \
    --query /path/to/{ds}_query.fbin \
    --k     10 \
    --out   /path/to/{ds}_rknn_gt_k10.bin

# Large datasets (GPU recommended)
python scripts/gen_rknn_gt_large.py --help
```

### 3.3 Run HRNN (main method)

Build:

```bash
./build/hrnn_build \
    --base            /path/to/{ds}_base.fbin \
    --index_path      /path/to/{ds}_hrnn.idx \
    --M               16 \
    --ef_construction 400 \
    --K_knng          500 \
    --nn_iters        10 \
    --nn_sample       50 \
    --num_threads     $(nproc)
```

Search:

```bash
./build/hrnn_search \
    --index_path /path/to/{ds}_hrnn.idx \
    --query      /path/to/{ds}_query.fbin \
    --gt         /path/to/{ds}_rknn_gt_k10.bin \
    --k          10 \
    --K_prime    500 \
    --m          "1,3,5,10,20,50" \
    --ef_search  200
```

### 3.4 Run HAMG (baseline)

```bash
./build/hamg_build \
    --base            /path/to/{ds}_base.fbin \
    --index_path      /path/to/{ds}_hamg.idx \
    --M  16 --M0 5000 --C 500 --dm 80 \
    --ef_construction 200 --num_threads $(nproc)

./build/hamg_search \
    --index_path /path/to/{ds}_hamg.idx \
    --query      /path/to/{ds}_query.fbin \
    --gt         /path/to/{ds}_rknn_gt_k10.bin \
    --k 10 --r 1.0 --ef_search 200
```

### 3.5 Run HNSW kNN baseline

```bash
./build/knn_build \
    --base       /path/to/{ds}_base.fbin \
    --index_path /path/to/{ds}_hnsw.idx \
    --M 16 --ef_construction 200

./build/knn_search \
    --index_path /path/to/{ds}_hnsw.idx \
    --query      /path/to/{ds}_query.fbin \
    --gt         /path/to/{ds}_knn_gt_k10.bin \
    --k 10 --ef_search "25,50,100,125,150"
```

### 3.6 Run naive RkNN (SFT / RDT)

Both methods reuse a standard HNSW index built by `knn_build`:

```bash
./build/rknn_hnsw_sft_search \
    --index_path /path/to/{ds}_hnsw.idx \
    --query      /path/to/{ds}_query.fbin \
    --gt         /path/to/{ds}_rknn_gt_k10.bin \
    --method     sft \
    --t          "1,3,5,7,10" \
    --k          10
```

### 3.7 Dynamic update experiment

```bash
./build/hrnn_dynamic_test \
    --base   /path/to/{ds}_base.fbin \
    --query  /path/to/{ds}_query.fbin \
    --gt     /path/to/{ds}_rknn_gt_k10.bin \
    --split  0.5 \
    --M 16 --ef_construction 400 --K_knng 500 \
    --k 10 --m 50 --ef_search 200 \
    --num_threads $(nproc)
```

End-to-end shell wrappers for build and search across a fixed parameter
preset live in [`scripts/example_build.sh`](scripts/example_build.sh) and
[`scripts/example_search.sh`](scripts/example_search.sh).



## 5. Citation

```
CITATION_BIBTEX_TBD
```
