#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <cmath>
#include <algorithm>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
class HierarchicalNSWNaiveRKNN : public HierarchicalNSW<dist_t> {
public:
    HierarchicalNSWNaiveRKNN(SpaceInterface<dist_t> *s) {}

    HierarchicalNSWNaiveRKNN(SpaceInterface<dist_t> *s,
                             const std::string &location,
                             bool nmslib = false,
                             size_t max_elements = 0,
                             bool allow_replace_deleted = false)
        : HierarchicalNSW<dist_t>(s, location, nmslib, max_elements, allow_replace_deleted) {}

    HierarchicalNSWNaiveRKNN(SpaceInterface<dist_t> *s,
                             size_t max_elements,
                             size_t M = 16,
                             size_t ef_construction = 200,
                             size_t random_seed = 100,
                             bool allow_replace_deleted = false)
        : HierarchicalNSW<dist_t>(
              s, max_elements, M, ef_construction, random_seed, allow_replace_deleted) {}

    ~HierarchicalNSWNaiveRKNN() {}

    // ========================================================================
    // HNSW-SFT: Simple Fetch Then Verify (Singh 2003)
    // ========================================================================
    std::priority_queue<std::pair<dist_t, labeltype>> searchRknnSFT(
        const void *query_data, size_t k, size_t k_bar) const {

        std::priority_queue<std::pair<dist_t, labeltype>> empty_result;
        if (this->cur_element_count == 0) return empty_result;

        k_bar = std::min(k_bar, static_cast<size_t>(this->cur_element_count));
        size_t k_eff = std::min(k, static_cast<size_t>(this->cur_element_count - 1));

        // Phase 1: Fetch k_bar nearest neighbors of q
        auto candidates_pq = this->searchKnn(query_data, k_bar);

        // Convert max-heap (dist, label) to sorted vector (dist, internal_id), ascending
        std::vector<std::pair<dist_t, tableint>> candidates;
        candidates.reserve(candidates_pq.size());
        while (!candidates_pq.empty()) {
            auto [dist, label] = candidates_pq.top();
            candidates_pq.pop();
            candidates.emplace_back(dist, getInternalIdByLabel(label));
        }
        std::reverse(candidates.begin(), candidates.end());

        // Phase 2: SFT Filtering Step I — witness counting among candidates
        // For each candidate c, count how many other candidates y have d(y,c) < d(q,c)
        // If witness_count >= k, q cannot be in c's k-NN → reject
        std::vector<std::pair<dist_t, tableint>> surviving;
        surviving.reserve(candidates.size());

        for (size_t i = 0; i < candidates.size(); i++) {
            auto [d_qc, c] = candidates[i];
            size_t witness_count = 0;
            for (size_t j = 0; j < candidates.size() && witness_count < k_eff; j++) {
                if (i == j) continue;
                tableint y = candidates[j].second;
                dist_t d_yc = this->fstdistfunc_(
                    this->getDataByInternalId(y),
                    this->getDataByInternalId(c),
                    this->dist_func_param_);
                if (d_yc < d_qc) {
                    witness_count++;
                }
            }
            if (witness_count < k_eff) {
                surviving.push_back(candidates[i]);
            }
        }

        // Phase 3: Verification
        return verifyCandidates(query_data, surviving, k_eff);
    }

    // ========================================================================
    // HNSW-RDT: Reverse Distance Threshold (Casanova et al. 2017)
    // Algorithm 1 with RDT+ optimization
    // ========================================================================
    std::priority_queue<std::pair<dist_t, labeltype>> searchRknnRDT(
        const void *query_data, size_t k, int t) const {

        std::priority_queue<std::pair<dist_t, labeltype>> result;
        size_t n = this->cur_element_count;
        if (n == 0) return result;

        size_t k_eff = std::min(k, n - 1);
        size_t k_bar = std::min(n, static_cast<size_t>(std::pow(2, t) * k));

        // Step 1: Navigate upper layers to find entry point for base layer
        tableint currObj = this->enterpoint_node_;
        dist_t curdist = this->fstdistfunc_(
            query_data,
            this->getDataByInternalId(this->enterpoint_node_),
            this->dist_func_param_);

        for (int level = this->maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data = (unsigned int *)this->get_linklist(currObj, level);
                int size = this->getListCount(data);
                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = this->fstdistfunc_(
                        query_data,
                        this->getDataByInternalId(cand),
                        this->dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        // Search base layer with ef = k_bar
        auto top_candidates = this->template searchBaseLayerST<true>(
            currObj, query_data, k_bar);

        // Convert to sorted vector (ascending distance)
        std::vector<std::pair<dist_t, tableint>> all_neighbors;
        all_neighbors.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            all_neighbors.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::reverse(all_neighbors.begin(), all_neighbors.end());

        // Step 2: RDT filter with witness counting + dimensional testing
        struct FilterEntry {
            tableint id;
            dist_t d_qx;
            size_t W;
        };

        std::vector<FilterEntry> F;
        F.reserve(std::min(k_bar, (size_t)1024));
        std::unordered_set<tableint> accepted;
        double omega = std::numeric_limits<double>::infinity();

        for (size_t s_idx = 0; s_idx < all_neighbors.size(); s_idx++) {
            auto [d_qv, v] = all_neighbors[s_idx];
            size_t rank_s = s_idx + 1;  // 1-based rank

            // Termination
            if (static_cast<double>(d_qv) > omega) break;
            if (rank_s > k_bar) break;

            size_t W_v = 0;

            // Update witnesses (Casanova Algorithm 1, lines 9-18)
            // Use boolean vector for batch removal (avoids O(n) erase during iteration)
            std::vector<bool> to_remove(F.size(), false);

            for (size_t i = 0; i < F.size(); i++) {
                if (to_remove[i]) continue;

                dist_t d_vx = this->fstdistfunc_(
                    this->getDataByInternalId(v),
                    this->getDataByInternalId(F[i].id),
                    this->dist_func_param_);

                // Line 10-11: d(v,x) < d(q,x) → v is closer to x than q → v witnesses against x
                if (F[i].d_qx > d_vx) {
                    F[i].W++;
                }

                // Line 13-14: d(v,x) < d(q,v) → x is closer to v than q → x witnesses against v
                if (d_qv > d_vx) {
                    W_v++;
                }

                // Line 16-17: Lazy accept
                if (F[i].W < k_eff && d_qv >= 2 * F[i].d_qx) {
                    accepted.insert(F[i].id);
                    to_remove[i] = true;
                    continue;
                }

                // RDT+ lazy reject: W(x) >= k
                if (F[i].W >= k_eff) {
                    to_remove[i] = true;
                }
            }

            // Compact F (remove marked entries)
            size_t write = 0;
            for (size_t read = 0; read < F.size(); read++) {
                if (!to_remove[read]) {
                    if (write != read) F[write] = F[read];
                    write++;
                }
            }
            F.resize(write);

            // Add v to filter set
            F.push_back({v, d_qv, W_v});

            // Dimensional test (Casanova Theorem 1, lines 21-22)
            if (rank_s > k_eff && d_qv > 0) {
                double ratio = std::pow(
                    static_cast<double>(rank_s) / static_cast<double>(k_eff),
                    1.0 / static_cast<double>(t));
                if (ratio > 1.0) {
                    double new_omega =
                        static_cast<double>(d_qv) / (ratio - 1.0);
                    omega = std::min(omega, new_omega);
                }
            }
        }

        // Step 3: Verify unresolved candidates (F with W < k)
        std::vector<std::pair<dist_t, tableint>> unresolved;
        for (const auto &entry : F) {
            if (entry.W < k_eff) {
                unresolved.emplace_back(entry.d_qx, entry.id);
            }
        }
        std::sort(unresolved.begin(), unresolved.end());

        result = verifyCandidates(query_data, unresolved, k_eff);

        // Add lazy-accepted results
        for (tableint x_id : accepted) {
            dist_t d = distToQuery(query_data, x_id);
            result.emplace(d, this->getExternalLabel(x_id));
        }

        return result;
    }

private:
    // ========================================================================
    // Helper methods (reused from hamg.h patterns)
    // ========================================================================

    tableint getInternalIdByLabel(labeltype label) const {
        auto it = this->label_lookup_.find(label);
        if (it == this->label_lookup_.end()) {
            throw std::runtime_error("Label not found in label_lookup_");
        }
        return it->second;
    }

    inline dist_t distToQuery(const void *query_data, tableint node) const {
        return this->fstdistfunc_(
            query_data, this->getDataByInternalId(node), this->dist_func_param_);
    }

    // kNN search returning (dist, internal_id) sorted ascending, excluding self
    std::vector<std::pair<dist_t, tableint>> searchKnnInternal(
        const void *query_data, size_t k) const {

        auto label_res = this->searchKnn(query_data, k);
        std::vector<std::pair<dist_t, tableint>> out;
        out.reserve(label_res.size());
        while (!label_res.empty()) {
            auto item = label_res.top();
            label_res.pop();
            out.emplace_back(item.first, getInternalIdByLabel(item.second));
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    // Verification: for each candidate c, check if d(q,c) <= d(c, c_k)
    // Pattern: hamg.h lines 1302-1322
    std::priority_queue<std::pair<dist_t, labeltype>> verifyCandidates(
        const void * /*query_data*/,
        const std::vector<std::pair<dist_t, tableint>> &candidates,
        size_t k) const {

        std::priority_queue<std::pair<dist_t, labeltype>> result;

        for (const auto &[d_cq, c_id] : candidates) {
            // Search k-NN of candidate c (k+2 for safety, skip self)
            auto knn_c = searchKnnInternal(this->getDataByInternalId(c_id), k + 2);

            // Collect distances, excluding self
            std::vector<dist_t> dist_c;
            dist_c.reserve(knn_c.size());
            for (const auto &[d, node] : knn_c) {
                if (node == c_id) continue;
                dist_c.push_back(d);
            }
            if (dist_c.empty()) continue;

            // k-th NN distance of c
            dist_t kth_dist = dist_c.size() >= k ? dist_c[k - 1] : dist_c.back();

            // Verify: is q in c's k-NN?
            if (d_cq <= kth_dist) {
                result.emplace(d_cq, this->getExternalLabel(c_id));
            }
        }

        return result;
    }
};
}  // namespace hnswlib
