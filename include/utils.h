//
// Created by xwx on 3/6/26.
//

#ifndef RKNN_UTILS_H
#define RKNN_UTILS_H

#include <hnswlib/hnswlib.h>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Binary file I/O
// ---------------------------------------------------------------------------

// fbin layout: [n: uint32][d: uint32][n * d * float32]
struct FbinData {
    uint32_t n, d;
    std::vector<float> data;
};

FbinData load_fbin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Cannot open: " + path);
    FbinData fb;
    ifs.read(reinterpret_cast<char*>(&fb.n), 4);
    ifs.read(reinterpret_cast<char*>(&fb.d), 4);
    fb.data.resize(static_cast<size_t>(fb.n) * fb.d);
    ifs.read(reinterpret_cast<char*>(fb.data.data()),
             static_cast<std::streamsize>(fb.data.size() * 4));
    if (!ifs)
        throw std::runtime_error("Read error: " + path);
    return fb;
}

// kNN groundtruth layout: [nq: uint32][k: uint32][nq*k uint32 ids][nq*k float32 dists]
struct KnnGroundtruth {
    uint32_t nq, k;
    std::vector<uint32_t> ids;  // [nq][k]
    std::vector<float> dists;   // [nq][k]
};

KnnGroundtruth load_knn_groundtruth(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Cannot open: " + path);
    KnnGroundtruth gt;
    ifs.read(reinterpret_cast<char*>(&gt.nq), 4);
    ifs.read(reinterpret_cast<char*>(&gt.k), 4);
    size_t total = static_cast<size_t>(gt.nq) * gt.k;
    gt.ids.resize(total);
    gt.dists.resize(total);
    ifs.read(reinterpret_cast<char*>(gt.ids.data()), total * 4);
    ifs.read(reinterpret_cast<char*>(gt.dists.data()), total * 4);
    if (!ifs)
        throw std::runtime_error("Read error: " + path);
    return gt;
}

// RkNN groundtruth (variable-length per query)
// Layout: [nq: uint32][(nq+1) uint32 offsets][total uint32 ids][total float32 dists]
// offsets[q]..offsets[q+1] gives the index range for query q's results.
struct RknnGroundtruth {
    uint32_t nq;
    std::vector<uint32_t> offsets;  // size nq+1
    std::vector<uint32_t> ids;
    std::vector<float> dists;

    size_t count(size_t q) const { return offsets[q + 1] - offsets[q]; }
    const uint32_t* ids_for(size_t q) const { return ids.data() + offsets[q]; }
    const float* dists_for(size_t q) const { return dists.data() + offsets[q]; }
    size_t total() const { return offsets[nq]; }
};

RknnGroundtruth load_rknn_groundtruth(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Cannot open: " + path);
    RknnGroundtruth gt;
    ifs.read(reinterpret_cast<char*>(&gt.nq), 4);
    gt.offsets.resize(static_cast<size_t>(gt.nq) + 1);
    ifs.read(reinterpret_cast<char*>(gt.offsets.data()),
             static_cast<std::streamsize>((gt.nq + 1) * 4));
    size_t total = gt.offsets[gt.nq];
    gt.ids.resize(total);
    gt.dists.resize(total);
    ifs.read(reinterpret_cast<char*>(gt.ids.data()), total * 4);
    ifs.read(reinterpret_cast<char*>(gt.dists.data()), total * 4);
    if (!ifs)
        throw std::runtime_error("Read error: " + path);
    return gt;
}

// ---------------------------------------------------------------------------
// Recall@k
// ---------------------------------------------------------------------------

// kNN recall: |results[q] ∩ GT top-k| / k  (averaged over queries)
double compute_knn_recall(const KnnGroundtruth& gt,
                          const std::vector<std::vector<hnswlib::labeltype>>& results,
                          int k) {
    size_t gt_k = std::min(static_cast<uint32_t>(k), gt.k);
    size_t found = 0, total = 0;

    for (size_t q = 0; q < gt.nq; ++q) {
        std::unordered_set<uint32_t> gt_set(gt.ids.begin() + q * gt.k,
                                            gt.ids.begin() + q * gt.k + gt_k);

        for (auto label : results[q]) {
            if (gt_set.count(static_cast<uint32_t>(label)))
                ++found;
        }
        total += gt_k;
    }
    return static_cast<double>(found) / static_cast<double>(total);
}

// RkNN recall: |results[q] ∩ GT[q]| / |GT[q]|  (averaged over queries with GT > 0)
double compute_rknn_recall(const RknnGroundtruth& gt,
                           const std::vector<std::vector<hnswlib::labeltype>>& results) {
    double recall_sum = 0;
    size_t counted = 0;

    for (size_t q = 0; q < gt.nq; ++q) {
        size_t gt_cnt = gt.count(q);
        if (gt_cnt == 0)
            continue;

        const uint32_t* gt_ids = gt.ids_for(q);
        std::unordered_set<uint32_t> gt_set(gt_ids, gt_ids + gt_cnt);

        size_t found = 0;
        for (auto label : results[q]) {
            if (gt_set.count(static_cast<uint32_t>(label)))
                ++found;
        }
        recall_sum += static_cast<double>(found) / static_cast<double>(gt_cnt);
        ++counted;
    }
    return counted > 0 ? recall_sum / static_cast<double>(counted) : 0.0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::milliseconds;
using Us = std::chrono::microseconds;

static long elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
}

static long elapsed_us(Clock::time_point t0) {
    return std::chrono::duration_cast<Us>(Clock::now() - t0).count();
}

#endif  // RKNN_UTILS_H
