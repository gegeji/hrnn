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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t> class HRNNAblation;     // forward declaration (exp_050)

template<typename dist_t>
class HRNN : public AlgorithmInterface<dist_t> {
    template<typename> friend class HRNNAblation;
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    // ---- KNNG / RKNNG fields ----
    size_t K_knng_{0};              // KNN graph depth (e.g. 500)
    size_t size_knng_slot_{0};      // bytes for KNNG slot per node
    size_t offsetKNNG_{0};          // offset to KNNG in level-0 memory
    bool knng_built_{false};
    bool rknng_built_{false};

    // RKNNG CSR storage — dual-width: narrow (uint32) or wide (uint64)
    bool rknng_wide_{false};
    std::vector<uint64_t> rknng_offsets_;     // always uint64_t (only n+1 elements)
    unsigned rknng_rank_bits_{0};

    // Narrow mode (node_bits + rank_bits <= 32)
    std::vector<uint32_t> rknng_entries_32_;
    uint32_t rknng_rank_mask_32_{0};

    // Wide mode (node_bits + rank_bits > 32)
    std::vector<uint64_t> rknng_entries_64_;
    uint64_t rknng_rank_mask_64_{0};

    // Mutable RKNNG storage (alternative to CSR, for dynamic updates)
    std::vector<std::vector<uint32_t>> rknng_lists_;
    bool rknng_mutable_{false};

    // ---- Serve mode: compact element layout (KNNG removed) ----
    bool serve_mode_{false};
    float* kdist_matrix_{nullptr};    // dense [n × max_k_serve_] row-major distances
    size_t max_k_serve_{0};


    HRNN(SpaceInterface<dist_t> *s) {
    }


    HRNN(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HRNN(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false,
        size_t K_knng = 0)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

        // KNNG memory layout: insert KNNG slot between HNSW links and vector data
        K_knng_ = K_knng;
        if (K_knng_ > 0) {
            size_knng_slot_ = sizeof(linklistsizeint) + K_knng_ * (sizeof(tableint) + sizeof(float));
            offsetKNNG_ = size_links_level0_;
            offsetData_ = offsetKNNG_ + size_knng_slot_;
        } else {
            size_knng_slot_ = 0;
            offsetKNNG_ = 0;
            offsetData_ = size_links_level0_;
        }
        label_offset_ = offsetData_ + data_size_;
        size_data_per_element_ = label_offset_ + sizeof(labeltype);
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HRNN failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HRNN() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        free(kdist_matrix_);
        kdist_matrix_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    // ---- KNNG accessors ----

    // KNNG link list pointer (count + K entries of {tableint, float})
    linklistsizeint* get_knng_linklist(tableint id) const {
        return (linklistsizeint*)(data_level0_memory_ + id * size_data_per_element_ + offsetKNNG_);
    }

    // Get k-th neighbor's distance (0-indexed) — this IS kdist_sq when k_idx = k_rknn - 1
    float get_knng_dist(tableint id, unsigned k_idx) const {
        char* base = (char*)(get_knng_linklist(id) + 1);  // skip count
        float d;
        memcpy(&d, base + k_idx * (sizeof(tableint) + sizeof(float)) + sizeof(tableint), sizeof(float));
        return d;
    }

    // Get k-th neighbor's ID (0-indexed)
    tableint get_knng_neighbor(tableint id, unsigned k_idx) const {
        char* base = (char*)(get_knng_linklist(id) + 1);
        tableint nbr;
        memcpy(&nbr, base + k_idx * (sizeof(tableint) + sizeof(float)), sizeof(tableint));
        return nbr;
    }

    // ---- Serve mode: compact element layout ----

    // Get kdist_sq for verification (works in both serve and normal mode)
    inline float getVerifyKdistSq(tableint id, unsigned k_idx) const {
        if (serve_mode_) {
            return kdist_matrix_[static_cast<size_t>(id) * max_k_serve_ + k_idx];
        }
        unsigned cnt = getListCount(get_knng_linklist(id));
        if (k_idx >= cnt) return std::numeric_limits<float>::max();
        return get_knng_dist(id, k_idx);
    }

    // Compact the in-memory element layout by removing the KNNG slot.
    // Extracts a dense kdist matrix for k = 1..max_k_serve and repacks
    // elements as [HNSW links | Vector | Label] (no KNNG).
    // The on-disk index is NOT modified — this is a load-time optimization.
    void compactForServing(size_t max_k_serve) {
        if (serve_mode_) return;
        if (K_knng_ == 0 || !knng_built_)
            throw std::runtime_error("compactForServing: KNNG not built");
        if (max_k_serve == 0 || max_k_serve > K_knng_)
            throw std::runtime_error("compactForServing: max_k_serve must be in [1, K_knng]");

        size_t n = cur_element_count;
        max_k_serve_ = max_k_serve;
        size_t old_elem_size = size_data_per_element_;

        // 1. Extract dense kdist matrix [n × max_k_serve]
        kdist_matrix_ = (float*)malloc(n * max_k_serve * sizeof(float));
        if (!kdist_matrix_)
            throw std::runtime_error("compactForServing: OOM for kdist_matrix");
        for (size_t i = 0; i < n; i++) {
            unsigned cnt = getListCount(get_knng_linklist((tableint)i));
            for (size_t j = 0; j < max_k_serve; j++) {
                kdist_matrix_[i * max_k_serve + j] =
                    (j < cnt) ? get_knng_dist((tableint)i, (unsigned)j)
                              : std::numeric_limits<float>::max();
            }
        }

        // 2. Compute new compact layout: [HNSW links | Vector | Label]
        size_t new_offset_data = size_links_level0_;
        size_t new_label_offset = new_offset_data + data_size_;
        size_t new_elem_size = new_label_offset + sizeof(labeltype);

        char* compact = (char*)malloc(max_elements_ * new_elem_size);
        if (!compact)
            throw std::runtime_error("compactForServing: OOM for compact buffer");

        for (size_t i = 0; i < n; i++) {
            char* src = data_level0_memory_ + i * size_data_per_element_;
            char* dst = compact + i * new_elem_size;
            // HNSW links (offset 0, size_links_level0_ bytes)
            memcpy(dst, src, size_links_level0_);
            // Vector data + label (skip KNNG slot)
            memcpy(dst + size_links_level0_,
                   src + offsetData_,
                   data_size_ + sizeof(labeltype));
        }

        free(data_level0_memory_);
        data_level0_memory_ = compact;

        // 3. Update offsets
        offsetData_ = new_offset_data;
        label_offset_ = new_label_offset;
        size_data_per_element_ = new_elem_size;
        offsetKNNG_ = 0;
        size_knng_slot_ = 0;
        serve_mode_ = true;

        fprintf(stderr, "[compactForServing] max_k_serve=%zu  "
                "elem_size: %zu → %zu (%.1fx)  "
                "level0_mem: %.1f GB → %.1f GB  "
                "kdist_matrix: %.1f MB\n",
                max_k_serve,
                old_elem_size, new_elem_size,
                (double)old_elem_size / new_elem_size,
                (double)(n * old_elem_size) / (1024.0 * 1024.0 * 1024.0),
                (double)(n * new_elem_size) / (1024.0 * 1024.0 * 1024.0),
                (double)(n * max_k_serve * sizeof(float)) / (1024.0 * 1024.0));
    }


    // Hint kernel to use transparent huge pages for data_level0_memory_.
    // Call after loadIndex() or compactForServing().
    void enableHugepages() {
#ifdef __linux__
        size_t sz = max_elements_ * size_data_per_element_;
        if (data_level0_memory_ && sz > 0)
            madvise(data_level0_memory_, sz, MADV_HUGEPAGE);
        if (kdist_matrix_ && max_k_serve_ > 0) {
            size_t ksz = cur_element_count * max_k_serve_ * sizeof(float);
            madvise(kdist_matrix_, ksz, MADV_HUGEPAGE);
        }
        fprintf(stderr, "[enableHugepages] madvise(MADV_HUGEPAGE) on %.1f GB\n",
                (double)sz / (1024.0 * 1024.0 * 1024.0));
#endif
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);

        // Magic header
        const uint32_t magic = 0x4E4E5248;  // "HRNN" in little-endian
        writeBinaryPOD(output, magic);

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        // KNNG/RKNNG header fields
        writeBinaryPOD(output, K_knng_);
        writeBinaryPOD(output, size_knng_slot_);
        writeBinaryPOD(output, offsetKNNG_);
        writeBinaryPOD(output, knng_built_);
        writeBinaryPOD(output, rknng_built_);

        // Level-0 data (includes KNNG slots in the per-element layout)
        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }

        // RKNNG CSR data (dual-width format)
        if (rknng_built_) {
            uint8_t entry_bytes = rknng_wide_ ? 8 : 4;
            writeBinaryPOD(output, entry_bytes);
            writeBinaryPOD(output, rknng_rank_bits_);

            size_t offsets_size = rknng_offsets_.size();
            writeBinaryPOD(output, offsets_size);
            output.write(reinterpret_cast<const char*>(rknng_offsets_.data()),
                         offsets_size * sizeof(uint64_t));

            if (rknng_wide_) {
                size_t entries_size = rknng_entries_64_.size();
                writeBinaryPOD(output, entries_size);
                output.write(reinterpret_cast<const char*>(rknng_entries_64_.data()),
                             entries_size * sizeof(uint64_t));
                writeBinaryPOD(output, rknng_rank_mask_64_);
            } else {
                size_t entries_size = rknng_entries_32_.size();
                writeBinaryPOD(output, entries_size);
                output.write(reinterpret_cast<const char*>(rknng_entries_32_.data()),
                             entries_size * sizeof(uint32_t));
                writeBinaryPOD(output, rknng_rank_mask_32_);
            }
        }

        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();

        // Read and verify magic
        uint32_t magic;
        readBinaryPOD(input, magic);
        if (magic != 0x4E4E5248)
            throw std::runtime_error("Not an HRNN index file (bad magic)");

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        // KNNG/RKNNG header fields
        readBinaryPOD(input, K_knng_);
        readBinaryPOD(input, size_knng_slot_);
        readBinaryPOD(input, offsetKNNG_);
        readBinaryPOD(input, knng_built_);
        readBinaryPOD(input, rknng_built_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        // Read level-0 data (includes KNNG slots)
        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        // Load RKNNG CSR data (dual-width format)
        if (rknng_built_) {
            uint8_t entry_bytes;
            readBinaryPOD(input, entry_bytes);
            rknng_wide_ = (entry_bytes == 8);
            readBinaryPOD(input, rknng_rank_bits_);

            size_t offsets_size;
            readBinaryPOD(input, offsets_size);
            rknng_offsets_.resize(offsets_size);
            input.read(reinterpret_cast<char*>(rknng_offsets_.data()),
                       offsets_size * sizeof(uint64_t));

            size_t entries_size;
            readBinaryPOD(input, entries_size);
            if (rknng_wide_) {
                rknng_entries_64_.resize(entries_size);
                input.read(reinterpret_cast<char*>(rknng_entries_64_.data()),
                           entries_size * sizeof(uint64_t));
                readBinaryPOD(input, rknng_rank_mask_64_);
            } else {
                rknng_entries_32_.resize(entries_size);
                input.read(reinterpret_cast<char*>(rknng_entries_32_.data()),
                           entries_size * sizeof(uint32_t));
                readBinaryPOD(input, rknng_rank_mask_32_);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }

                // Save KNNG candidates at level 0 (before HNSW pruning consumes them)
                if (level == 0 && K_knng_ > 0) {
                    auto knng_candidates = top_candidates;  // copy the heap
                    std::vector<std::pair<dist_t, tableint>> sorted_cands;
                    while (!knng_candidates.empty()) {
                        sorted_cands.push_back(knng_candidates.top());
                        knng_candidates.pop();
                    }
                    std::sort(sorted_cands.begin(), sorted_cands.end());  // ascending by dist

                    linklistsizeint* knng_ll = get_knng_linklist(cur_c);
                    char* entry_base = (char*)(knng_ll + 1);
                    unsigned count = 0;
                    for (auto& [dist, nbr_id] : sorted_cands) {
                        if (nbr_id == cur_c) continue;
                        if (count >= K_knng_) break;
                        size_t off = count * (sizeof(tableint) + sizeof(float));
                        memcpy(entry_base + off, &nbr_id, sizeof(tableint));
                        memcpy(entry_base + off + sizeof(tableint), &dist, sizeof(float));
                        count++;
                    }
                    setListCount(knng_ll, count);
                }

                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    // ========================================================================
    // KNNG refinement via NNDescent-style local join
    // ========================================================================

    bool try_update_knng(tableint u, tableint w, dist_t d) {
        std::lock_guard<std::mutex> lock(link_list_locks_[u]);
        linklistsizeint* ll = get_knng_linklist(u);
        unsigned cnt = getListCount(ll);
        char* base = (char*)(ll + 1);
        const size_t entry_size = sizeof(tableint) + sizeof(float);

        // Check if w already in list
        for (unsigned i = 0; i < cnt; i++) {
            tableint existing;
            memcpy(&existing, base + i * entry_size, sizeof(tableint));
            if (existing == w) return false;
        }

        if (cnt < K_knng_) {
            // List not full: append
            size_t off = cnt * entry_size;
            memcpy(base + off, &w, sizeof(tableint));
            memcpy(base + off + sizeof(tableint), &d, sizeof(float));
            setListCount(ll, cnt + 1);
            // Insertion sort from the back to maintain distance order
            for (unsigned i = cnt; i > 0; i--) {
                float d_i, d_prev;
                memcpy(&d_prev, base + (i-1) * entry_size + sizeof(tableint), sizeof(float));
                memcpy(&d_i, base + i * entry_size + sizeof(tableint), sizeof(float));
                if (d_i < d_prev) {
                    char tmp[sizeof(tableint) + sizeof(float)];
                    memcpy(tmp, base + i * entry_size, entry_size);
                    memcpy(base + i * entry_size, base + (i-1) * entry_size, entry_size);
                    memcpy(base + (i-1) * entry_size, tmp, entry_size);
                } else {
                    break;
                }
            }
            return true;
        }

        // List full: check if d < farthest neighbor's distance
        float farthest_d;
        memcpy(&farthest_d, base + (cnt-1) * entry_size + sizeof(tableint), sizeof(float));
        if (d >= farthest_d) return false;

        // Replace farthest, then insertion sort from back
        size_t off = (cnt-1) * entry_size;
        memcpy(base + off, &w, sizeof(tableint));
        memcpy(base + off + sizeof(tableint), &d, sizeof(float));
        for (unsigned i = cnt - 1; i > 0; i--) {
            float d_i, d_prev;
            memcpy(&d_prev, base + (i-1) * entry_size + sizeof(tableint), sizeof(float));
            memcpy(&d_i, base + i * entry_size + sizeof(tableint), sizeof(float));
            if (d_i < d_prev) {
                char tmp[sizeof(tableint) + sizeof(float)];
                memcpy(tmp, base + i * entry_size, entry_size);
                memcpy(base + i * entry_size, base + (i-1) * entry_size, entry_size);
                memcpy(base + (i-1) * entry_size, tmp, entry_size);
            } else {
                break;
            }
        }
        return true;
    }


    // Like try_update_knng but returns the evicted neighbor's ID.
    // Returns {success, evicted_id}. evicted_id is valid only when success==true
    // and the list was full (an entry was replaced).
    static constexpr tableint INVALID_KNNG_ID = std::numeric_limits<tableint>::max();

    std::pair<bool, tableint> try_update_knng_evict(tableint u, tableint w, dist_t d) {
        std::lock_guard<std::mutex> lock(link_list_locks_[u]);
        linklistsizeint* ll = get_knng_linklist(u);
        unsigned cnt = getListCount(ll);
        char* base = (char*)(ll + 1);
        const size_t entry_size = sizeof(tableint) + sizeof(float);

        // Dedup check
        for (unsigned i = 0; i < cnt; i++) {
            tableint existing;
            memcpy(&existing, base + i * entry_size, sizeof(tableint));
            if (existing == w) return {false, INVALID_KNNG_ID};
        }

        if (cnt < K_knng_) {
            // List not full: append, no eviction
            size_t off = cnt * entry_size;
            memcpy(base + off, &w, sizeof(tableint));
            memcpy(base + off + sizeof(tableint), &d, sizeof(float));
            setListCount(ll, cnt + 1);
            for (unsigned i = cnt; i > 0; i--) {
                float d_i, d_prev;
                memcpy(&d_prev, base + (i - 1) * entry_size + sizeof(tableint), sizeof(float));
                memcpy(&d_i, base + i * entry_size + sizeof(tableint), sizeof(float));
                if (d_i < d_prev) {
                    char tmp[sizeof(tableint) + sizeof(float)];
                    memcpy(tmp, base + i * entry_size, entry_size);
                    memcpy(base + i * entry_size, base + (i - 1) * entry_size, entry_size);
                    memcpy(base + (i - 1) * entry_size, tmp, entry_size);
                } else {
                    break;
                }
            }
            return {true, INVALID_KNNG_ID};
        }

        // List full: check if d < farthest
        float farthest_d;
        memcpy(&farthest_d, base + (cnt - 1) * entry_size + sizeof(tableint), sizeof(float));
        if (d >= farthest_d) return {false, INVALID_KNNG_ID};

        // Capture evicted ID before overwriting
        tableint evicted;
        memcpy(&evicted, base + (cnt - 1) * entry_size, sizeof(tableint));

        // Replace farthest, then insertion sort
        size_t off = (cnt - 1) * entry_size;
        memcpy(base + off, &w, sizeof(tableint));
        memcpy(base + off + sizeof(tableint), &d, sizeof(float));
        for (unsigned i = cnt - 1; i > 0; i--) {
            float d_i, d_prev;
            memcpy(&d_prev, base + (i - 1) * entry_size + sizeof(tableint), sizeof(float));
            memcpy(&d_i, base + i * entry_size + sizeof(tableint), sizeof(float));
            if (d_i < d_prev) {
                char tmp[sizeof(tableint) + sizeof(float)];
                memcpy(tmp, base + i * entry_size, entry_size);
                memcpy(base + i * entry_size, base + (i - 1) * entry_size, entry_size);
                memcpy(base + (i - 1) * entry_size, tmp, entry_size);
            } else {
                break;
            }
        }
        return {true, evicted};
    }


    // Insert w into u's KNNG with new-flag tracking for NNDescent.
    // Returns true if insertion succeeded. Uses link_list_locks_[u].
    bool try_update_knng_nnd(tableint u, tableint w, dist_t d,
                              std::vector<std::vector<char>>& is_new_flags) {
        std::lock_guard<std::mutex> lock(link_list_locks_[u]);
        linklistsizeint* ll = get_knng_linklist(u);
        unsigned cnt = getListCount(ll);
        char* base = (char*)(ll + 1);
        const size_t entry_size = sizeof(tableint) + sizeof(float);

        // Early rejection: if full and d >= farthest, no improvement
        if (cnt >= K_knng_) {
            float farthest_d;
            memcpy(&farthest_d, base + (cnt-1) * entry_size + sizeof(tableint), sizeof(float));
            if (d >= farthest_d) return false;
        }

        // Dedup check
        for (unsigned i = 0; i < cnt; i++) {
            tableint existing;
            memcpy(&existing, base + i * entry_size, sizeof(tableint));
            if (existing == w) return false;
        }

        if (cnt < K_knng_) {
            // Not full: append + insertion sort
            size_t off = cnt * entry_size;
            memcpy(base + off, &w, sizeof(tableint));
            memcpy(base + off + sizeof(tableint), &d, sizeof(float));
            setListCount(ll, cnt + 1);
            is_new_flags[u].push_back(1);
            for (unsigned i = cnt; i > 0; i--) {
                float d_i, d_prev;
                memcpy(&d_prev, base + (i-1) * entry_size + sizeof(tableint), sizeof(float));
                memcpy(&d_i, base + i * entry_size + sizeof(tableint), sizeof(float));
                if (d_i < d_prev) {
                    char tmp[sizeof(tableint) + sizeof(float)];
                    memcpy(tmp, base + i * entry_size, entry_size);
                    memcpy(base + i * entry_size, base + (i-1) * entry_size, entry_size);
                    memcpy(base + (i-1) * entry_size, tmp, entry_size);
                    std::swap(is_new_flags[u][i], is_new_flags[u][i-1]);
                } else break;
            }
            return true;
        }

        // Full: replace farthest + insertion sort
        size_t off = (cnt-1) * entry_size;
        memcpy(base + off, &w, sizeof(tableint));
        memcpy(base + off + sizeof(tableint), &d, sizeof(float));
        is_new_flags[u][cnt-1] = 1;
        for (unsigned i = cnt - 1; i > 0; i--) {
            float d_i, d_prev;
            memcpy(&d_prev, base + (i-1) * entry_size + sizeof(tableint), sizeof(float));
            memcpy(&d_i, base + i * entry_size + sizeof(tableint), sizeof(float));
            if (d_i < d_prev) {
                char tmp[sizeof(tableint) + sizeof(float)];
                memcpy(tmp, base + i * entry_size, entry_size);
                memcpy(base + i * entry_size, base + (i-1) * entry_size, entry_size);
                memcpy(base + (i-1) * entry_size, tmp, entry_size);
                std::swap(is_new_flags[u][i], is_new_flags[u][i-1]);
            } else break;
        }
        return true;
    }


    void refineKNNG(int num_iters = 3, int sample_size = 20) {
        size_t n = cur_element_count;
        unsigned S = (unsigned)sample_size;
        unsigned R = S;  // reverse neighbor cap

        std::cout << "    refineKNNG (NNDescent): n=" << n << " K=" << K_knng_
                  << " iters=" << num_iters << " S=" << S << std::endl;

        // Side structures for NNDescent local join
        std::vector<std::vector<tableint>> nn_new(n);
        std::vector<std::vector<tableint>> nn_old(n);
        std::vector<std::vector<tableint>> rnn_new(n);
        std::vector<std::vector<tableint>> rnn_old(n);

        // Per-entry "new" flags aligned with KNNG entries
        // Protected by link_list_locks_ during join phase
        std::vector<std::vector<char>> is_new(n);

        // INIT: all existing neighbors are "new"
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            unsigned cnt = getListCount(get_knng_linklist((tableint)i));
            is_new[i].assign(cnt, 1);
        }

        for (int iter = 0; iter < num_iters; iter++) {
            auto iter_start = std::chrono::high_resolution_clock::now();
            size_t join_updates = 0;
            size_t join_dists = 0;

            // ==== PHASE 1: UPDATE — classify neighbors, build reverse lists ====

            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                nn_new[i].clear();
                nn_old[i].clear();
                rnn_new[i].clear();
                rnn_old[i].clear();
            }

            #pragma omp parallel for schedule(dynamic, 256)
            for (size_t i = 0; i < n; i++) {
                unsigned cnt = getListCount(get_knng_linklist((tableint)i));
                if (cnt == 0) continue;

                for (unsigned j = 0; j < cnt; j++) {
                    tableint nbr = get_knng_neighbor((tableint)i, j);
                    float dist_ij = get_knng_dist((tableint)i, j);
                    bool entry_new = (j < is_new[i].size() && is_new[i][j]);

                    if (entry_new) {
                        if (nn_new[i].size() < S)
                            nn_new[i].push_back(nbr);
                        is_new[i][j] = 0;  // clear flag

                        // Reverse neighbor: if dist(i->nbr) > nbr's worst,
                        // nbr probably doesn't have i in its KNNG
                        unsigned cnt_nbr = getListCount(get_knng_linklist(nbr));
                        if (cnt_nbr > 0) {
                            float nbr_worst = (cnt_nbr >= K_knng_)
                                ? get_knng_dist(nbr, cnt_nbr - 1)
                                : std::numeric_limits<float>::max();
                            if (dist_ij > nbr_worst) {
                                std::lock_guard<std::mutex> lk(link_list_locks_[nbr]);
                                if (rnn_new[nbr].size() < R)
                                    rnn_new[nbr].push_back((tableint)i);
                            }
                        }
                    } else {
                        if (nn_old[i].size() < S)
                            nn_old[i].push_back(nbr);

                        unsigned cnt_nbr = getListCount(get_knng_linklist(nbr));
                        if (cnt_nbr > 0) {
                            float nbr_worst = (cnt_nbr >= K_knng_)
                                ? get_knng_dist(nbr, cnt_nbr - 1)
                                : std::numeric_limits<float>::max();
                            if (dist_ij > nbr_worst) {
                                std::lock_guard<std::mutex> lk(link_list_locks_[nbr]);
                                if (rnn_old[nbr].size() < R)
                                    rnn_old[nbr].push_back((tableint)i);
                            }
                        }
                    }
                }
            }

            // Merge reverse neighbors into forward lists
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                nn_new[i].insert(nn_new[i].end(), rnn_new[i].begin(), rnn_new[i].end());
                nn_old[i].insert(nn_old[i].end(), rnn_old[i].begin(), rnn_old[i].end());
                if (nn_old[i].size() > 2 * R)
                    nn_old[i].resize(2 * R);
            }

            // ==== PHASE 2: JOIN — local join on (new×new) + (new×old) pairs ====

            #pragma omp parallel for schedule(dynamic, 256) reduction(+:join_updates,join_dists)
            for (size_t node = 0; node < n; node++) {
                auto& nw = nn_new[node];
                auto& ol = nn_old[node];

                // new × new pairs
                for (size_t i = 0; i < nw.size(); i++) {
                    for (size_t j = i + 1; j < nw.size(); j++) {
                        tableint a = nw[i], b = nw[j];
                        if (a == b) continue;
                        dist_t d = fstdistfunc_(getDataByInternalId(a),
                                                getDataByInternalId(b), dist_func_param_);
                        join_dists++;
                        if (try_update_knng_nnd(a, b, d, is_new)) join_updates++;
                        if (try_update_knng_nnd(b, a, d, is_new)) join_updates++;
                    }
                }

                // new × old pairs
                for (size_t i = 0; i < nw.size(); i++) {
                    for (size_t j = 0; j < ol.size(); j++) {
                        tableint a = nw[i], b = ol[j];
                        if (a == b) continue;
                        dist_t d = fstdistfunc_(getDataByInternalId(a),
                                                getDataByInternalId(b), dist_func_param_);
                        join_dists++;
                        if (try_update_knng_nnd(a, b, d, is_new)) join_updates++;
                        if (try_update_knng_nnd(b, a, d, is_new)) join_updates++;
                    }
                }
            }

            auto iter_end = std::chrono::high_resolution_clock::now();
            long iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                iter_end - iter_start).count();
            std::cout << "      iter " << iter << ": " << join_updates << " updates, "
                      << join_dists << " dists  (" << iter_ms << " ms)" << std::endl;
        }
        knng_built_ = true;
    }


    // ========================================================================
    // RKNNG construction (CSR transpose of KNNG)
    // ========================================================================

    void buildRKNNG() {
        if (!knng_built_)
            throw std::runtime_error("buildRKNNG: KNNG not built yet");
        size_t n = cur_element_count;

        // Compute bit-packing parameters
        rknng_rank_bits_ = 0;
        while ((1u << rknng_rank_bits_) < K_knng_) rknng_rank_bits_++;
        unsigned node_bits = 0;
        while ((1ull << node_bits) < n) node_bits++;

        rknng_wide_ = (node_bits + rknng_rank_bits_ > 32);

        if (node_bits + rknng_rank_bits_ > 64)
            throw std::runtime_error("buildRKNNG: exceeds 64-bit packing (n=" + std::to_string(n)
                + " needs " + std::to_string(node_bits) + " bits, K=" + std::to_string(K_knng_)
                + " needs " + std::to_string(rknng_rank_bits_) + " bits)");

        std::cout << "    buildRKNNG: node_bits=" << node_bits
                  << " rank_bits=" << rknng_rank_bits_
                  << " wide=" << rknng_wide_ << std::endl;

        // Count reverse references (offsets always uint64_t)
        rknng_offsets_.assign(n + 1, 0);
        for (size_t i = 0; i < n; i++) {
            unsigned cnt = getListCount(get_knng_linklist((tableint)i));
            for (unsigned j = 0; j < cnt; j++)
                rknng_offsets_[get_knng_neighbor((tableint)i, j) + 1]++;
        }
        // Prefix sum
        for (size_t i = 1; i <= n; i++)
            rknng_offsets_[i] += rknng_offsets_[i - 1];

        // Fill and sort entries (templated on entry width)
        if (rknng_wide_)
            buildRKNNG_fill<uint64_t>(rknng_entries_64_, rknng_rank_mask_64_);
        else
            buildRKNNG_fill<uint32_t>(rknng_entries_32_, rknng_rank_mask_32_);

        rknng_built_ = true;
        std::cout << "    buildRKNNG: total_entries=" << rknng_offsets_[cur_element_count] << std::endl;
    }

    template <typename EntryT>
    void buildRKNNG_fill(std::vector<EntryT>& entries, EntryT& rank_mask) {
        size_t n = cur_element_count;
        rank_mask = (EntryT(1) << rknng_rank_bits_) - 1;
        entries.resize(rknng_offsets_[n]);
        std::vector<uint64_t> pos(rknng_offsets_.begin(), rknng_offsets_.end());

        for (size_t i = 0; i < n; i++) {
            unsigned cnt = getListCount(get_knng_linklist((tableint)i));
            for (unsigned j = 0; j < cnt; j++) {
                tableint nbr = get_knng_neighbor((tableint)i, j);
                entries[pos[nbr]++] = (EntryT(i) << rknng_rank_bits_) | j;
            }
        }

        // Sort each node's reverse list by rank (for K' prefix scan)
        for (size_t b = 0; b < n; b++) {
            auto* beg = entries.data() + rknng_offsets_[b];
            auto* en  = entries.data() + rknng_offsets_[b + 1];
            std::sort(beg, en, [mask = rank_mask](EntryT a, EntryT b) {
                return (a & mask) < (b & mask);
            });
        }
    }


    // ========================================================================
    // Mutable RKNNG: per-node vector (for dynamic updates)
    // ========================================================================

    void buildRKNNG_mutable() {
        if (!knng_built_)
            throw std::runtime_error("buildRKNNG_mutable: KNNG not built yet");
        size_t n = cur_element_count;
        // Pre-allocate to max capacity so insertPointDynamic never resizes
        rknng_lists_.assign(max_elements_, {});
        size_t total = 0;
        for (size_t i = 0; i < n; i++) {
            unsigned cnt = getListCount(get_knng_linklist((tableint)i));
            for (unsigned j = 0; j < cnt; j++) {
                tableint nbr = get_knng_neighbor((tableint)i, j);
                rknng_lists_[nbr].push_back((uint32_t)i);
                total++;
            }
        }
        rknng_mutable_ = true;
        std::cout << "    buildRKNNG_mutable: n=" << n << " total_entries=" << total << std::endl;
    }


    // ========================================================================
    // RkNN search: three-phase with thread_local flat score buffer
    // ========================================================================

    std::vector<labeltype> searchRknn(
        const void* query_data, size_t m, float threshold,
        size_t k_rknn, size_t K_prime, size_t ef_search) const
    {
        if (rknng_wide_)
            return searchRknn_impl(query_data, m, threshold, k_rknn, K_prime, ef_search,
                                   rknng_entries_64_, rknng_rank_mask_64_);
        else
            return searchRknn_impl(query_data, m, threshold, k_rknn, K_prime, ef_search,
                                   rknng_entries_32_, rknng_rank_mask_32_);
    }

    template <typename EntryT>
    std::vector<labeltype> searchRknn_impl(
        const void* query_data, size_t m, float threshold,
        size_t k_rknn, size_t K_prime, size_t ef_search,
        const std::vector<EntryT>& rknng_entries,
        EntryT rknng_rank_mask) const
    {
        if (!rknng_built_)
            throw std::runtime_error("searchRknn: RKNNG not built");
        if (cur_element_count == 0) return {};

        thread_local std::vector<float> score_buf;
        if (score_buf.size() < max_elements_) score_buf.resize(max_elements_, 0.0f);

        // Phase A: HNSW greedy descent to base layer
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        // Base layer search for m proxies
        auto top_candidates = searchBaseLayerST<true>(
            currObj, query_data, std::max(ef_search, m));
        while (top_candidates.size() > m)
            top_candidates.pop();

        // Phase B: Reverse lookup with K' truncation + binary scoring
        std::vector<uint32_t> candidates;
        while (!top_candidates.empty()) {
            tableint proxy = top_candidates.top().second;
            top_candidates.pop();

            uint64_t begin = rknng_offsets_[proxy];
            uint64_t end = rknng_offsets_[proxy + 1];
            for (uint64_t idx = begin; idx < end; idx++) {
                EntryT packed = rknng_entries[idx];
                uint32_t rank = (uint32_t)(packed & rknng_rank_mask);
                if (rank >= K_prime) break;  // sorted by rank → prefix scan
                uint32_t cand = (uint32_t)(packed >> rknng_rank_bits_);
                if (score_buf[cand] == 0.0f) candidates.push_back(cand);
                score_buf[cand] += 1.0f;
            }
        }

        // Phase C: Verification (sort + prefetch optimized)
        const unsigned k_idx = static_cast<unsigned>(k_rknn - 1);

        // Pass 1: threshold filter + score_buf cleanup
        {
            size_t write = 0;
            for (size_t i = 0; i < candidates.size(); ++i) {
                uint32_t c = candidates[i];
                bool keep = score_buf[c] >= threshold;
                score_buf[c] = 0.0f;
                if (keep) candidates[write++] = c;
            }
            candidates.resize(write);
        }

        // Pass 2: sort by ID for memory locality, then prefetch + verify
        std::sort(candidates.begin(), candidates.end());

        std::vector<labeltype> results;
        for (size_t i = 0; i < candidates.size(); ++i) {
#ifdef USE_SSE
            if (i + 8 < candidates.size())
                _mm_prefetch(getDataByInternalId(candidates[i + 8]), _MM_HINT_T0);
#endif
            uint32_t cand = candidates[i];
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
            float kdist_sq = getVerifyKdistSq(cand, k_idx);
            if (d <= kdist_sq) {
                results.push_back(getExternalLabel(cand));
            }
        }
        return results;
    }

    // ========================================================================
    // Profiled RkNN search — returns per-phase timings alongside results.
    // ========================================================================

    struct RknnProfile {
        double phase1_us = 0;  // HNSW routing + base layer proxy retrieval
        double phase2_us = 0;  // Reverse posting list scan + scoring
        double phase3_us = 0;  // Verification (L2 + kdist compare)
        size_t n_candidates = 0;
        size_t n_verified = 0; // candidates passing threshold
        size_t n_results = 0;
    };

    std::vector<labeltype> searchRknn_profiled(
        const void* query_data, size_t m, float threshold,
        size_t k_rknn, size_t K_prime, size_t ef_search,
        RknnProfile& prof) const
    {
        if (rknng_wide_)
            return searchRknn_profiled_impl(query_data, m, threshold, k_rknn, K_prime, ef_search,
                                            rknng_entries_64_, rknng_rank_mask_64_, prof);
        else
            return searchRknn_profiled_impl(query_data, m, threshold, k_rknn, K_prime, ef_search,
                                            rknng_entries_32_, rknng_rank_mask_32_, prof);
    }

    template <typename EntryT>
    std::vector<labeltype> searchRknn_profiled_impl(
        const void* query_data, size_t m, float threshold,
        size_t k_rknn, size_t K_prime, size_t ef_search,
        const std::vector<EntryT>& rknng_entries,
        EntryT rknng_rank_mask,
        RknnProfile& prof) const
    {
        using HRC = std::chrono::high_resolution_clock;

        if (!rknng_built_)
            throw std::runtime_error("searchRknn_profiled: RKNNG not built");
        if (cur_element_count == 0) return {};

        thread_local std::vector<float> score_buf;
        if (score_buf.size() < max_elements_) score_buf.resize(max_elements_, 0.0f);

        // ── Phase 1: HNSW routing + proxy retrieval ──
        auto t1 = HRC::now();

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        auto top_candidates = searchBaseLayerST<true>(
            currObj, query_data, std::max(ef_search, m));
        while (top_candidates.size() > m)
            top_candidates.pop();

        auto t2 = HRC::now();

        // ── Phase 2: Reverse lookup with K' truncation ──
        std::vector<uint32_t> candidates;
        while (!top_candidates.empty()) {
            tableint proxy = top_candidates.top().second;
            top_candidates.pop();
            uint64_t begin = rknng_offsets_[proxy];
            uint64_t end = rknng_offsets_[proxy + 1];
            for (uint64_t idx = begin; idx < end; idx++) {
                EntryT packed = rknng_entries[idx];
                uint32_t rank = (uint32_t)(packed & rknng_rank_mask);
                if (rank >= K_prime) break;
                uint32_t cand = (uint32_t)(packed >> rknng_rank_bits_);
                if (score_buf[cand] == 0.0f) candidates.push_back(cand);
                score_buf[cand] += 1.0f;
            }
        }

        auto t3 = HRC::now();

        // ── Phase 3: Verification (sort + prefetch optimized) ──
        const unsigned k_idx = static_cast<unsigned>(k_rknn - 1);

        // Pass 1: threshold filter + score_buf cleanup
        {
            size_t write = 0;
            for (size_t i = 0; i < candidates.size(); ++i) {
                uint32_t c = candidates[i];
                bool keep = score_buf[c] >= threshold;
                score_buf[c] = 0.0f;
                if (keep) candidates[write++] = c;
            }
            candidates.resize(write);
        }

        // Pass 2: sort by ID, prefetch + verify
        std::sort(candidates.begin(), candidates.end());

        std::vector<labeltype> results;
        size_t n_verified = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
#ifdef USE_SSE
            if (i + 8 < candidates.size())
                _mm_prefetch(getDataByInternalId(candidates[i + 8]), _MM_HINT_T0);
#endif
            uint32_t cand = candidates[i];
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
            float kdist_sq = getVerifyKdistSq(cand, k_idx);
            if (d <= kdist_sq) {
                results.push_back(getExternalLabel(cand));
            }
        }

        auto t4 = HRC::now();

        prof.phase1_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
        prof.phase2_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
        prof.phase3_us = std::chrono::duration<double, std::micro>(t4 - t3).count();
        prof.n_candidates = candidates.size();
        prof.n_verified = n_verified;
        prof.n_results = results.size();

        return results;
    }


    // ========================================================================
    // Phase A+B only: return candidates that pass score threshold (no verification)
    // Used by HRNNRabitqIndex for two-pass verification.
    // ========================================================================

    std::vector<uint32_t> searchRknn_candidates(
        const void* query_data, size_t m, float threshold,
        size_t K_prime, size_t ef_search) const
    {
        if (rknng_wide_)
            return searchRknn_candidates_impl(query_data, m, threshold, K_prime, ef_search,
                                              rknng_entries_64_, rknng_rank_mask_64_);
        else
            return searchRknn_candidates_impl(query_data, m, threshold, K_prime, ef_search,
                                              rknng_entries_32_, rknng_rank_mask_32_);
    }

    template <typename EntryT>
    std::vector<uint32_t> searchRknn_candidates_impl(
        const void* query_data, size_t m, float threshold,
        size_t K_prime, size_t ef_search,
        const std::vector<EntryT>& rknng_entries,
        EntryT rknng_rank_mask) const
    {
        if (!rknng_built_)
            throw std::runtime_error("searchRknn_candidates: RKNNG not built");
        if (cur_element_count == 0) return {};

        thread_local std::vector<float> score_buf;
        if (score_buf.size() < max_elements_) score_buf.resize(max_elements_, 0.0f);

        // Phase A: HNSW greedy descent to base layer
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        // Base layer search for m proxies
        auto top_candidates = searchBaseLayerST<true>(
            currObj, query_data, std::max(ef_search, m));
        while (top_candidates.size() > m)
            top_candidates.pop();

        // Phase B: Reverse lookup with K' truncation + binary scoring
        std::vector<uint32_t> candidates;
        while (!top_candidates.empty()) {
            tableint proxy = top_candidates.top().second;
            top_candidates.pop();

            uint64_t begin = rknng_offsets_[proxy];
            uint64_t end = rknng_offsets_[proxy + 1];
            for (uint64_t idx = begin; idx < end; idx++) {
                EntryT packed = rknng_entries[idx];
                uint32_t rank = (uint32_t)(packed & rknng_rank_mask);
                if (rank >= K_prime) break;
                uint32_t cand = (uint32_t)(packed >> rknng_rank_bits_);
                if (score_buf[cand] == 0.0f) candidates.push_back(cand);
                score_buf[cand] += 1.0f;
            }
        }

        // Filter by score threshold and cleanup
        std::vector<uint32_t> result;
        result.reserve(candidates.size());
        for (uint32_t cand : candidates) {
            if (score_buf[cand] >= threshold)
                result.push_back(cand);
            score_buf[cand] = 0.0f;
        }
        return result;
    }

    // ========================================================================
    // Phase B only: reverse posting lookup given proxies, return candidates
    // passing score threshold. Used by HRNNRabitqIndex when HNSW search is
    // done externally with RaBitQ.
    // ========================================================================

    std::vector<uint32_t> reversePostingLookup(
        const std::vector<tableint>& proxies,
        float threshold, size_t K_prime) const
    {
        if (rknng_wide_)
            return reversePostingLookup_impl(proxies, threshold, K_prime,
                                             rknng_entries_64_, rknng_rank_mask_64_);
        else
            return reversePostingLookup_impl(proxies, threshold, K_prime,
                                             rknng_entries_32_, rknng_rank_mask_32_);
    }

    template <typename EntryT>
    std::vector<uint32_t> reversePostingLookup_impl(
        const std::vector<tableint>& proxies,
        float threshold, size_t K_prime,
        const std::vector<EntryT>& rknng_entries,
        EntryT rknng_rank_mask) const
    {
        if (!rknng_built_) return {};

        thread_local std::vector<float> score_buf;
        if (score_buf.size() < max_elements_) score_buf.resize(max_elements_, 0.0f);

        std::vector<uint32_t> candidates;
        for (tableint proxy : proxies) {
            uint64_t begin = rknng_offsets_[proxy];
            uint64_t end = rknng_offsets_[proxy + 1];
            for (uint64_t idx = begin; idx < end; idx++) {
                EntryT packed = rknng_entries[idx];
                uint32_t rank = (uint32_t)(packed & rknng_rank_mask);
                if (rank >= K_prime) break;
                uint32_t cand = (uint32_t)(packed >> rknng_rank_bits_);
                if (score_buf[cand] == 0.0f) candidates.push_back(cand);
                score_buf[cand] += 1.0f;
            }
        }

        std::vector<uint32_t> result;
        result.reserve(candidates.size());
        for (uint32_t cand : candidates) {
            if (score_buf[cand] >= threshold)
                result.push_back(cand);
            score_buf[cand] = 0.0f;
        }
        return result;
    }

    // ========================================================================
    // Dynamic INSERT: HNSW + KNNG + mutable RKNNG maintenance
    // ========================================================================

    // m_update: number of proxies for self-query (0 = use all KNNG neighbors, legacy behavior)
    // bidirectional: if true, also improve x_new's KNNG from self-query candidates (Opt-2)
    tableint insertPointDynamic(const void* data_point, labeltype label,
                                unsigned m_update = 0,
                                bool bidirectional = false) {
        // Phase 1+2: HNSW insert + KNNG slot init (from searchBaseLayer candidates)
        // addPoint is already thread-safe (uses link_list_locks_ + global mutex)
        tableint x_new = addPoint(data_point, label, -1);

        // Read x_new's KNNG neighbors (set by addPoint from searchBaseLayer)
        unsigned cnt = getListCount(get_knng_linklist(x_new));
        std::vector<tableint> all_neighbors(cnt);
        for (unsigned j = 0; j < cnt; j++)
            all_neighbors[j] = get_knng_neighbor(x_new, j);

        // Phase 3: Register x_new in each neighbor's reverse list
        // (always uses ALL neighbors, not truncated — x_new's full KNNG is correct)
        for (tableint y : all_neighbors) {
            std::lock_guard<std::mutex> lk(link_list_locks_[y]);
            rknng_lists_[y].push_back(x_new);
        }

        // Opt-1: Truncate proxy set for self-query (neighbors are sorted by distance)
        unsigned proxy_cnt = cnt;
        if (m_update > 0 && m_update < cnt) {
            proxy_cnt = m_update;
        }

        // Phase 4: Self-query — collect candidates via flat visited buffer (no hash table)
        thread_local std::vector<uint8_t> visited_buf;
        if (visited_buf.size() < max_elements_) visited_buf.resize(max_elements_, 0);

        std::vector<uint32_t> candidates;
        candidates.reserve(proxy_cnt * 256);
        // Mark x_new and all_neighbors as visited (exclude from candidates)
        visited_buf[x_new] = 1;
        for (tableint y : all_neighbors) visited_buf[y] = 1;

        for (unsigned p = 0; p < proxy_cnt; p++) {
            tableint y = all_neighbors[p];
            std::lock_guard<std::mutex> lk(link_list_locks_[y]);
            for (uint32_t x_i : rknng_lists_[y]) {
                if (!visited_buf[x_i]) {
                    visited_buf[x_i] = 1;
                    candidates.push_back(x_i);
                }
            }
        }
        // Reset visited buffer (only touch entries we set)
        visited_buf[x_new] = 0;
        for (tableint y : all_neighbors) visited_buf[y] = 0;
        for (uint32_t c : candidates) visited_buf[c] = 0;

        // Opt-2: Maintain a top-K heap for x_new's KNNG improvement.
        using DistPair = std::pair<dist_t, tableint>;
        std::priority_queue<DistPair> xnew_heap;  // max-heap by distance
        if (bidirectional) {
            for (unsigned j = 0; j < cnt; j++) {
                dist_t d_j = get_knng_dist(x_new, j);
                xnew_heap.push({d_j, all_neighbors[j]});
            }
        }

        // Phase 5: Update affected points' KNNG + reverse lists
        // Also collects distances into x_new's heap (Opt-2).
        for (size_t idx = 0; idx < candidates.size(); idx++) {
            uint32_t x_i = candidates[idx];

            // Prefetch next candidate's vector data + KNNG slot
            if (idx + 4 < candidates.size()) {
                _mm_prefetch(getDataByInternalId(candidates[idx + 4]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(
                    get_knng_linklist(candidates[idx + 4])), _MM_HINT_T0);
            }

            dist_t d_new =
                fstdistfunc_(data_point, getDataByInternalId(x_i), dist_func_param_);

            // Opt-2: Feed distance into x_new's top-K heap
            if (bidirectional) {
                if (xnew_heap.size() < K_knng_) {
                    xnew_heap.push({d_new, static_cast<tableint>(x_i)});
                } else if (d_new < xnew_heap.top().first) {
                    xnew_heap.pop();
                    xnew_heap.push({d_new, static_cast<tableint>(x_i)});
                }
            }

            // Optimistic early rejection: unlocked read of farthest distance.
            // Data race is benign — worst case we enter the locked path unnecessarily
            // or skip a marginal improvement.
            unsigned cnt_i = getListCount(get_knng_linklist(x_i));
            if (cnt_i >= K_knng_) {
                float farthest_d = get_knng_dist(x_i, cnt_i - 1);
                if (d_new >= farthest_d) continue;
            }

            // Only ~1-2% of candidates reach here
            auto [success, evicted] = try_update_knng_evict(x_i, x_new, d_new);
            if (success) {
                {
                    std::lock_guard<std::mutex> lk(link_list_locks_[x_new]);
                    rknng_lists_[x_new].push_back(x_i);
                }
                if (evicted != INVALID_KNNG_ID) {
                    std::lock_guard<std::mutex> lk(link_list_locks_[evicted]);
                    auto& rl = rknng_lists_[evicted];
                    // Swap-and-pop: O(find) + O(1) removal
                    auto it = std::find(rl.begin(), rl.end(), x_i);
                    if (it != rl.end()) {
                        *it = rl.back();
                        rl.pop_back();
                    }
                }
            }
        }

        // Opt-2 finalization: Replace x_new's KNNG with heap contents.
        if (bidirectional)
        // Only ADD x_new to newly gained neighbors' reverse lists.
        // Don't remove from dropped neighbors' reverse lists — stale entries
        // are benign (produce extra candidates that get filtered by verification).
        {
            std::vector<DistPair> new_knng;
            new_knng.reserve(xnew_heap.size());
            while (!xnew_heap.empty()) {
                new_knng.push_back(xnew_heap.top());
                xnew_heap.pop();
            }
            std::sort(new_knng.begin(), new_knng.end());

            std::vector<tableint> newly_added;
            {
                std::lock_guard<std::mutex> lk_xnew(link_list_locks_[x_new]);

                // Save old neighbor IDs before overwriting
                unsigned old_cnt = getListCount(get_knng_linklist(x_new));
                std::vector<tableint> old_ids(old_cnt);
                for (unsigned j = 0; j < old_cnt; j++)
                    old_ids[j] = get_knng_neighbor(x_new, j);

                // Mark old neighbors in visited_buf
                for (tableint id : old_ids) visited_buf[id] = 1;

                // Write new KNNG, track newly added neighbors
                linklistsizeint* knng_ll = get_knng_linklist(x_new);
                char* entry_base = (char*)(knng_ll + 1);
                unsigned new_cnt = 0;
                for (auto& [dist, nbr_id] : new_knng) {
                    if (nbr_id == x_new) continue;
                    if (new_cnt >= K_knng_) break;
                    if (visited_buf[nbr_id] == 2) continue;  // dedup
                    size_t off = new_cnt * (sizeof(tableint) + sizeof(float));
                    memcpy(entry_base + off, &nbr_id, sizeof(tableint));
                    memcpy(entry_base + off + sizeof(tableint), &dist, sizeof(float));
                    if (visited_buf[nbr_id] == 0) {
                        newly_added.push_back(nbr_id);
                    }
                    visited_buf[nbr_id] = 2;
                    new_cnt++;
                }
                setListCount(knng_ll, new_cnt);

                // Reset visited_buf
                for (tableint id : old_ids) visited_buf[id] = 0;
                for (auto& [dist, nbr_id] : new_knng) visited_buf[nbr_id] = 0;
            }

            // Register x_new in newly added neighbors' reverse lists
            for (tableint id : newly_added) {
                std::lock_guard<std::mutex> lk(link_list_locks_[id]);
                rknng_lists_[id].push_back(x_new);
            }
        }

        return x_new;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};
}  // namespace hnswlib
