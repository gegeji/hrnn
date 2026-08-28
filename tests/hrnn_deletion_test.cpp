#include "hnswlib/hrnn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Index = hnswlib::HRNN<float>;
using hnswlib::labeltype;
using hnswlib::tableint;

constexpr size_t kDimension = 2;
constexpr size_t kPointCount = 12;
constexpr size_t kStoredK = 4;
constexpr size_t kQueryK = 3;

struct RowEntry {
    tableint neighbor;
    float distance;

    bool operator==(const RowEntry& other) const {
        return neighbor == other.neighbor && distance == other.distance;
    }
};

using Rows = std::vector<std::vector<RowEntry>>;
using Graph = std::vector<std::vector<std::vector<tableint>>>;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Function>
void requireThrows(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, message);
}

std::vector<std::vector<float>> makePoints() {
    std::vector<std::vector<float>> points;
    for (size_t i = 0; i < kPointCount; i++) {
        points.push_back({
            static_cast<float>(i),
            static_cast<float>((i * 7) % 11)});
    }
    return points;
}

void writeDeterministicKNNG(Index& index) {
    const size_t entry_size = sizeof(tableint) + sizeof(float);
    for (tableint owner = 0; owner < kPointCount; owner++) {
        auto* row = index.get_knng_linklist(owner);
        char* entries = reinterpret_cast<char*>(row + 1);
        for (unsigned rank = 0; rank < kStoredK; rank++) {
            const tableint neighbor =
                static_cast<tableint>((owner + rank + 1) % kPointCount);
            const float distance = static_cast<float>(rank + 1);
            const size_t offset = rank * entry_size;
            memcpy(entries + offset, &neighbor, sizeof(tableint));
            memcpy(entries + offset + sizeof(tableint), &distance, sizeof(float));
        }
        index.setListCount(row, kStoredK);
    }
    index.knng_built_ = true;
    index.buildRKNNG();
}

void populate(Index& index, const std::vector<std::vector<float>>& points) {
    for (size_t i = 0; i < points.size(); i++) {
        index.addPoint(points[i].data(), static_cast<labeltype>(i));
    }
    writeDeterministicKNNG(index);
}

Rows snapshotRows(const Index& index) {
    Rows rows(kPointCount);
    for (tableint owner = 0; owner < kPointCount; owner++) {
        const unsigned count =
            index.getListCount(index.get_knng_linklist(owner));
        for (unsigned rank = 0; rank < count; rank++) {
            rows[owner].push_back({
                index.get_knng_neighbor(owner, rank),
                index.get_knng_dist(owner, rank)});
        }
    }
    return rows;
}

Graph snapshotGraph(const Index& index) {
    Graph graph(kPointCount);
    for (tableint owner = 0; owner < kPointCount; owner++) {
        for (int level = 0; level <= index.element_levels_[owner]; level++) {
            auto* links = index.get_linklist_at_level(owner, level);
            const unsigned count = index.getListCount(links);
            const tableint* neighbors = reinterpret_cast<const tableint*>(links + 1);
            graph[owner].emplace_back(neighbors, neighbors + count);
        }
    }
    return graph;
}

Rows filterRows(
    const Rows& before, const std::unordered_set<tableint>& deleted) {
    Rows expected(before.size());
    for (tableint owner = 0; owner < before.size(); owner++) {
        if (deleted.count(owner)) {
            // Deleted owners are excluded from query verification and reverse
            // transposition, so their private ranked rows need not be rewritten.
            expected[owner] = before[owner];
            continue;
        }
        for (const RowEntry& entry : before[owner]) {
            if (!deleted.count(entry.neighbor))
                expected[owner].push_back(entry);
        }
    }
    return expected;
}

void verifyReverseTranspose(const Index& index, const Rows& rows) {
    std::vector<std::vector<std::pair<unsigned, tableint>>> expected(kPointCount);
    for (tableint owner = 0; owner < rows.size(); owner++) {
        if (index.isMarkedDeleted(owner))
            continue;
        for (unsigned rank = 0; rank < rows[owner].size(); rank++) {
            expected[rows[owner][rank].neighbor].push_back({rank, owner});
        }
    }
    for (auto& postings : expected)
        std::sort(postings.begin(), postings.end());

    for (tableint target = 0; target < kPointCount; target++) {
        const uint64_t begin = index.rknng_offsets_[target];
        const uint64_t end = index.rknng_offsets_[target + 1];
        require(end - begin == expected[target].size(),
                "reverse CSR posting count mismatch");
        for (size_t offset = 0; offset < expected[target].size(); offset++) {
            uint64_t packed = 0;
            uint64_t mask = 0;
            if (index.rknng_wide_) {
                packed = index.rknng_entries_64_[begin + offset];
                mask = index.rknng_rank_mask_64_;
            } else {
                packed = index.rknng_entries_32_[begin + offset];
                mask = index.rknng_rank_mask_32_;
            }
            const unsigned rank = static_cast<unsigned>(packed & mask);
            const tableint owner =
                static_cast<tableint>(packed >> index.rknng_rank_bits_);
            require(
                std::make_pair(rank, owner) == expected[target][offset],
                "reverse CSR rank/owner mismatch");
        }
    }
}

std::vector<labeltype> query(
    const Index& index,
    const std::vector<float>& point,
    size_t k) {
    auto result = index.searchRknn(
        point.data(), kPointCount, 0.0f, k, kStoredK, 32);
    std::sort(result.begin(), result.end());
    return result;
}

void verifyQueryContract(
    const Index& index,
    const std::vector<std::vector<float>>& points,
    const std::unordered_set<tableint>& deleted) {
    for (size_t k : {size_t{1}, kQueryK, kStoredK}) {
        const auto result = query(index, points[5], k);
        for (labeltype label : result) {
            const tableint owner = static_cast<tableint>(label);
            require(!deleted.count(owner), "query returned a deleted label");
            require(
                index.getListCount(index.get_knng_linklist(owner)) >= k,
                "query returned an owner without a k-th surviving neighbor");
        }
    }

    bool oversized_k_rejected = false;
    try {
        (void)index.searchRknn(
            points[5].data(), 4, 0.0f,
            kStoredK + 1, kStoredK, 32);
    } catch (const std::runtime_error&) {
        oversized_k_rejected = true;
    }
    require(oversized_k_rejected, "query accepted k greater than stored K");
}

void verifyRoundTrip(
    const std::vector<labeltype>& labels,
    size_t expected_min_depth) {
    hnswlib::L2Space space(kDimension);
    Index index(&space, kPointCount, 4, 40, 100, false, kStoredK);
    const auto points = makePoints();
    populate(index, points);

    const Rows before_rows = snapshotRows(index);
    const Graph before_graph = snapshotGraph(index);

    const long distances_before = index.metric_distance_computations.load();
    const auto stats = index.deleteBatch(labels, kQueryK);
    const long distances_after = index.metric_distance_computations.load();
    require(
        distances_after == distances_before,
        "lightweight deletion performed distance computations");

    std::unordered_set<tableint> deleted;
    for (labeltype label : labels)
        deleted.insert(static_cast<tableint>(label));
    const Rows expected_rows = filterRows(before_rows, deleted);
    require(snapshotRows(index) == expected_rows,
            "KNNG is not the stable live-entry filter of the original rows");
    require(snapshotGraph(index) == before_graph,
            "lightweight deletion rewired the HNSW graph");
    verifyReverseTranspose(index, expected_rows);

    size_t unservable = 0;
    size_t min_depth = kStoredK;
    for (tableint owner = 0; owner < kPointCount; owner++) {
        if (deleted.count(owner))
            continue;
        const size_t depth = expected_rows[owner].size();
        min_depth = std::min(min_depth, depth);
        if (depth < kQueryK) {
            unservable++;
            require(
                std::isinf(index.getVerifyKdistSq(owner, kQueryK - 1))
                    && index.getVerifyKdistSq(owner, kQueryK - 1) < 0,
                "short row did not receive a negative-infinity verification radius");
        }
    }
    require(min_depth == expected_min_depth, "fixture minimum depth mismatch");
    require(stats.requested == labels.size(), "requested count mismatch");
    require(stats.newly_deleted == deleted.size(), "newly deleted count mismatch");
    require(stats.removed_entries > 0, "no deleted neighbor entries were removed");
    require(stats.unservable_rows == unservable, "unservable row count mismatch");
    verifyQueryContract(index, points, deleted);

    const Rows rows_before_rejected_refinement = snapshotRows(index);
    const auto offsets_before_rejected_refinement = index.rknng_offsets_;
    const auto narrow_before_rejected_refinement = index.rknng_entries_32_;
    const auto wide_before_rejected_refinement = index.rknng_entries_64_;
    requireThrows(
        [&] { index.refineKNNG(1, 2); },
        "tombstoned index accepted KNNG refinement");
    requireThrows(
        [&] { index.buildRKNNG_mutable(); },
        "tombstoned index accepted mutable reverse maintenance");
    require(snapshotRows(index) == rows_before_rejected_refinement,
            "rejected refinement changed ranked rows");
    require(index.rknng_offsets_ == offsets_before_rejected_refinement
                && index.rknng_entries_32_ == narrow_before_rejected_refinement
                && index.rknng_entries_64_ == wide_before_rejected_refinement,
            "rejected refinement changed reverse CSR");

    std::vector<std::vector<labeltype>> before_save_results;
    for (size_t k : {size_t{1}, kQueryK, kStoredK})
        before_save_results.push_back(query(index, points[5], k));

    const auto path = std::filesystem::temp_directory_path()
        / ("hrnn_deletion_roundtrip_"
           + std::to_string(expected_min_depth)
           + "_"
           + std::to_string(reinterpret_cast<std::uintptr_t>(&index))
           + ".idx");
    index.saveIndex(path.string());
    {
        Index reloaded(&space, path.string());
        require(reloaded.getDeletedCount() == deleted.size(),
                "round trip lost tombstones");
        require(snapshotRows(reloaded) == expected_rows,
                "round trip changed filtered KNNG rows");
        verifyReverseTranspose(reloaded, expected_rows);
        verifyQueryContract(reloaded, points, deleted);
        for (size_t i = 0; i < before_save_results.size(); i++) {
            const size_t k = i == 0 ? 1 : (i == 1 ? kQueryK : kStoredK);
            require(query(reloaded, points[5], k) == before_save_results[i],
                    "round trip changed a query result digest");
        }
    }
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);
    require(!remove_error, "failed to remove round-trip test index");

    bool insertion_rejected = false;
    const std::vector<float> new_point = {100.0f, 100.0f};
    try {
        index.addPoint(new_point.data(), static_cast<labeltype>(kPointCount));
    } catch (const std::runtime_error&) {
        insertion_rejected = true;
    }
    require(insertion_rejected, "index accepted insertion after deletion");

    bool direct_mark_rejected = false;
    try {
        index.markDelete(5);
    } catch (const std::runtime_error&) {
        direct_mark_rejected = true;
    }
    require(direct_mark_rejected, "direct markDelete bypassed batch maintenance");

    bool revive_rejected = false;
    try {
        index.unmarkDelete(labels.front());
    } catch (const std::runtime_error&) {
        revive_rejected = true;
    }
    require(revive_rejected, "unmarkDelete revived a deleted label");

    bool compact_rejected = false;
    try {
        index.compactForServing(kQueryK);
    } catch (const std::runtime_error&) {
        compact_rejected = true;
    }
    require(compact_rejected, "tombstoned index entered compact serving mode");

    const auto idempotent_stats =
        index.deleteBatch({labels.front(), labels.front()}, kQueryK);
    require(idempotent_stats.requested == 2 && idempotent_stats.newly_deleted == 0,
            "already-deleted labels were not idempotent");

    const auto second_stats = index.deleteBatch({5, 5}, kQueryK);
    require(second_stats.requested == 2 && second_stats.newly_deleted == 1,
            "second deletion batch was not idempotent over duplicate labels");
}

void verifyPreflightFailures() {
    hnswlib::L2Space space(kDimension);
    Index index(&space, kPointCount, 4, 40, 100, false, kStoredK);
    const auto points = makePoints();
    populate(index, points);
    const Rows before_rows = snapshotRows(index);

    requireThrows(
        [&] { index.deleteBatch({999}, kQueryK); },
        "unknown label was accepted");
    require(index.getDeletedCount() == 0 && snapshotRows(index) == before_rows,
            "unknown-label rejection mutated the index");

    requireThrows(
        [&] {
            index.deleteBatch(
                {0, 1, 2, 3, 4, 5, 6, 7, 8},
                kQueryK);
        },
        "batch leaving at most k active points was accepted");
    require(index.getDeletedCount() == 0 && snapshotRows(index) == before_rows,
            "oversized-batch rejection mutated the index");

    require(index.deleteBatch({}, kQueryK).requested == 0,
            "empty deletion batch was not a no-op");
}
void verifyServingShortRowsMatchFullIndex() {
    hnswlib::L2Space space(kDimension);
    Index full(&space, kPointCount, 4, 40, 100, false, kStoredK);
    Index compact(&space, kPointCount, 4, 40, 100, false, kStoredK);
    const auto points = makePoints();
    populate(full, points);
    populate(compact, points);

    full.setListCount(full.get_knng_linklist(0), kQueryK - 1);
    compact.setListCount(compact.get_knng_linklist(0), kQueryK - 1);
    full.buildRKNNG();
    compact.buildRKNNG();

    std::vector<std::vector<labeltype>> expected;
    for (const auto& point : points)
        expected.push_back(query(full, point, kQueryK));

    compact.compactForServing(kQueryK);
    require(
        std::isinf(compact.getVerifyKdistSq(0, kQueryK - 1))
            && compact.getVerifyKdistSq(0, kQueryK - 1) < 0,
        "compact serving mode did not preserve the short-row sentinel");
    for (size_t query_id = 0; query_id < points.size(); query_id++) {
        require(query(compact, points[query_id], kQueryK) == expected[query_id],
                "compact serving mode changed short-row query results");
    }
    requireThrows(
        [&] {
            (void)compact.searchRknn(
                points[0].data(), 4, 0.0f,
                kStoredK, kStoredK, 32);
        },
        "compact serving mode accepted k beyond its materialized depth");
}

void verifyEmptyReverseCSR() {
    constexpr size_t point_count = 4;
    constexpr size_t stored_k = 2;
    hnswlib::L2Space space(kDimension);
    Index index(&space, point_count, 4, 40, 100, false, stored_k);
    const std::vector<std::vector<float>> points = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {2.0f, 0.0f},
        {3.0f, 0.0f}};
    for (size_t id = 0; id < point_count; id++)
        index.addPoint(points[id].data(), static_cast<labeltype>(id));

    const std::vector<std::vector<tableint>> neighbors = {
        {1, 2},
        {0, 2},
        {0, 1},
        {0, 1}};
    const size_t entry_size = sizeof(tableint) + sizeof(float);
    for (tableint owner = 0; owner < point_count; owner++) {
        auto* row = index.get_knng_linklist(owner);
        char* entries = reinterpret_cast<char*>(row + 1);
        for (unsigned rank = 0; rank < stored_k; rank++) {
            const tableint neighbor = neighbors[owner][rank];
            const float distance = static_cast<float>(rank + 1);
            const size_t offset = rank * entry_size;
            memcpy(entries + offset, &neighbor, sizeof(tableint));
            memcpy(entries + offset + sizeof(tableint), &distance, sizeof(float));
        }
        index.setListCount(row, stored_k);
    }
    index.knng_built_ = true;
    index.buildRKNNG();

    const auto stats = index.deleteBatch({0, 1}, 1);
    require(stats.unservable_rows == 2,
            "empty-reverse fixture did not produce two short live rows");
    require(index.getListCount(index.get_knng_linklist(2)) == 0
                && index.getListCount(index.get_knng_linklist(3)) == 0,
            "empty-reverse fixture retained a deleted neighbor");
    require(std::all_of(
                index.rknng_offsets_.begin(),
                index.rknng_offsets_.end(),
                [](uint64_t offset) { return offset == 0; }),
            "empty reverse CSR has a nonzero offset");
    require(index.rknng_entries_32_.empty()
                && index.rknng_entries_64_.empty(),
            "empty reverse CSR retained entries");
    require(index.searchRknn(
                points[2].data(), point_count, 0.0f,
                1, stored_k, 16).empty(),
            "empty reverse CSR returned an RkNN result");
}

void verifyParallelDeterminism() {
    hnswlib::L2Space space(kDimension);
    Index single(&space, kPointCount, 4, 40, 100, false, kStoredK);
    Index parallel(&space, kPointCount, 4, 40, 100, false, kStoredK);
    const auto points = makePoints();
    populate(single, points);
    populate(parallel, points);

    int original_threads = 1;
#ifdef _OPENMP
    original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    single.deleteBatch({1, 2, 3, 4}, kQueryK);
#ifdef _OPENMP
    omp_set_num_threads(std::min(4, omp_get_num_procs()));
#endif
    parallel.deleteBatch({1, 2, 3, 4}, kQueryK);

    require(snapshotRows(single) == snapshotRows(parallel),
            "thread count changed filtered KNNG rows");
    require(single.rknng_offsets_ == parallel.rknng_offsets_,
            "thread count changed reverse CSR offsets");
    require(single.rknng_entries_32_ == parallel.rknng_entries_32_,
            "thread count changed narrow reverse CSR entries");
    require(single.rknng_entries_64_ == parallel.rknng_entries_64_,
            "thread count changed wide reverse CSR entries");
    require(query(single, points[5], kQueryK)
                == query(parallel, points[5], kQueryK),
            "thread count changed query results");
#ifdef _OPENMP
    omp_set_num_threads(original_threads);
    omp_set_dynamic(original_dynamic);
#endif
}

}  // namespace

int main() {
    verifyPreflightFailures();
    verifyServingShortRowsMatchFullIndex();
    verifyEmptyReverseCSR();
    verifyRoundTrip({1, 2, 3}, 1);
    verifyRoundTrip({1, 2, 3, 4}, 0);
    verifyParallelDeterminism();
    std::cout << "hrnn_deletion_test: PASS\n";
    return 0;
}
