#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <memory>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
class HAMG : public AlgorithmInterface<dist_t> {
public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;
    static constexpr uint64_t HAMG_INDEX_MAGIC = 0x48414D4749445831ULL;  // "HAMGIDX1"
    static constexpr uint64_t HAMG_INDEX_VERSION = 3;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t hamg_C_{0};
    size_t hamg_dm_{80};  // minimum out-degree
    size_t build_num_threads_{0};
    size_t build_seed_{0};
    std::string build_base_path_;
    mutable bool hamg_query_model_ready_{false};
    mutable size_t hamg_model_E_{0};
    mutable size_t hamg_model_samples_{0};
    mutable std::vector<double> hamg_alpha_;           // 1-based
    mutable std::vector<std::vector<double>> hamg_e_;  // [i][a], 1-based
    mutable std::vector<double> hamg_beta_;            // 1-based
    mutable std::vector<double> hamg_e1_avg_;          // e(j,1,j-1), 1-based
    mutable std::vector<double> hamg_pr_cache_;        // 1-based by hop
    mutable size_t hamg_pr_cache_k_{0};
    mutable size_t hamg_pr_cache_H_{0};
    mutable double hamg_avg_out_degree_{0.0};
    mutable std::mutex hamg_query_model_lock_;
    size_t forced_k_prime_{0};  // when > 0, override getHopCountHAMG result
    bool no_ps1_{false};       // when true, set ED = infinity (disable PS1)
    size_t ef_construction_{0};
    size_t ef_{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

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

    bool allow_replace_deleted_ =
        false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;               // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    HAMG(SpaceInterface<dist_t> *s) {}

    HAMG(SpaceInterface<dist_t> *s,
         const std::string &location,
         bool nmslib = false,
         size_t max_elements = 0,
         bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }

    HAMG(SpaceInterface<dist_t> *s,
         size_t max_elements,
         size_t M = 16,
         size_t ef_construction = 200,
         size_t random_seed = 100,
         bool allow_replace_deleted = false,
         size_t maxM0 = 0,
         size_t hamg_C = 0,
         size_t hamg_dm = 80)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
          link_list_locks_(max_elements),
          element_levels_(max_elements),
          allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if (M <= 10000) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects."
                    << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing."
                    << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = (maxM0 > 0) ? maxM0 : (M_ * 2);
        if (maxM0_ > std::numeric_limits<unsigned short>::max()) {
            throw std::runtime_error("maxM0 exceeds link-list capacity (65535)");
        }
        hamg_C_ = (hamg_C > 0) ? hamg_C : maxM0_;
        hamg_dm_ = hamg_dm;
        if (hamg_C_ == 0) {
            throw std::runtime_error("HAMG parameter C must be > 0");
        }
        if (hamg_C_ > maxM0_) {
            throw std::runtime_error("HAMG parameter C cannot exceed maxM0");
        }
        ef_construction_ = std::max(std::max(ef_construction, M_), hamg_C_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HAMG failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    ~HAMG() { clear(); }

    void setBuildMetadata(const std::string &base_path, size_t num_threads, size_t seed) {
        build_base_path_ = base_path;
        build_num_threads_ = num_threads;
        build_seed_ = seed;
    }

    const std::string &getBuildBasePath() const { return build_base_path_; }

    size_t getBuildNumThreads() const { return build_num_threads_; }

    size_t getBuildSeed() const { return build_seed_; }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
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
        constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                  std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };

    static void writeBinaryString(std::ostream &out, const std::string &value) {
        uint64_t size = static_cast<uint64_t>(value.size());
        writeBinaryPOD(out, size);
        if (size > 0) {
            out.write(value.data(), static_cast<std::streamsize>(size));
        }
    }

    static void readBinaryString(std::istream &in, std::string &value) {
        uint64_t size = 0;
        readBinaryPOD(in, size);
        value.resize(static_cast<size_t>(size));
        if (size > 0) {
            in.read(value.data(), static_cast<std::streamsize>(size));
        }
    }

    void setEf(size_t ef) { ef_ = ef; }

    void setForcedKPrime(size_t k) { forced_k_prime_ = k; }

    void setNoPS1(bool v) { no_ps1_ = v; }

    size_t getM() const { return M_; }

    size_t getM0() const { return maxM0_; }

    size_t getHAMGC() const { return hamg_C_; }

    size_t getHAMGDm() const { return hamg_dm_; }

    size_t getEfConstruction() const { return ef_construction_; }

    inline std::mutex &getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label,
               (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
               sizeof(labeltype));
        return return_label;
    }

    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
               &label,
               sizeof(labeltype));
    }

    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                             label_offset_);
    }

    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    size_t getMaxElements() const { return max_elements_; }

    size_t getCurrentElementCount() const { return cur_element_count; }

    size_t getDeletedCount() const { return num_deleted_; }

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidateSet;

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

            std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int *)get_linklist0(curNodeNum);
            } else {
                data = (int *)get_linklist(curNodeNum, layer);
                //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) *
                //                    size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
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

    // bare_bone_search means there is no check for deletions and stop condition is ignored in
    // return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayerST(tableint ep_id,
                      const void *data_point,
                      size_t ef,
                      BaseFilterFunctor *isIdAllowed = nullptr,
                      BaseSearchStopCondition<dist_t> *stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || (!isMarkedDeleted(ep_id) &&
                                 ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char *ep_data = getDataByInternalId(ep_id);
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
                    flag_stop_search =
                        stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint *)data);
            //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_,
                         _MM_HINT_T0);
            _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(
                    data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                    _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate =
                            stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ +
                                         candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,  ///////////
                                     _MM_HINT_T0);       ////////////////////////
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) &&
                             ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(
                                    getExternalLabel(candidate_id), currObj1, dist);
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
                                stop_condition->remove_point_from_result(
                                    getExternalLabel(id), getDataByInternalId(id), dist);
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

    void getNeighborsByHeuristic2(std::priority_queue<std::pair<dist_t, tableint>,
                                                      std::vector<std::pair<dist_t, tableint>>,
                                                      CompareByFirst> &top_candidates,
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
                dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
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
        return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                                   offsetLevel0_);
    }

    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                                   offsetLevel0_);
    }

    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    // wrapper of L2 squared distance
    dist_t distBetweenNodes(tableint lhs, tableint rhs) const {
        return fstdistfunc_(getDataByInternalId(lhs), getDataByInternalId(rhs), dist_func_param_);
    }

    std::vector<tableint> getBottomNeighbors(tableint node) const {
        linklistsizeint *ll = get_linklist0(node);
        size_t sz = getListCount(ll);
        tableint *data = (tableint *)(ll + 1);
        return std::vector<tableint>(data, data + sz);
    }

    std::vector<tableint> sortBottomNeighbors(tableint base,
                                              const std::vector<tableint> &neighbors) const {
        std::vector<tableint> ordered = neighbors;
        std::sort(ordered.begin(), ordered.end(), [&](tableint lhs, tableint rhs) {
            if (lhs == rhs) {
                return false;
            }
            dist_t dl = distBetweenNodes(base, lhs);
            dist_t dr = distBetweenNodes(base, rhs);
            if (dl != dr) {
                return dl < dr;
            }
            return lhs < rhs;
        });

        std::vector<tableint> uniq;
        uniq.reserve(ordered.size());
        std::unordered_set<tableint> seen;
        for (tableint node : ordered) {
            if (node == base) {
                continue;
            }
            if (seen.insert(node).second) {
                uniq.push_back(node);
            }
        }
        return uniq;
    }

    void setBottomNeighbors(tableint node, const std::vector<tableint> &neighbors) {
        std::vector<tableint> ordered = sortBottomNeighbors(node, neighbors);
        if (ordered.size() > maxM0_) {
            throw std::runtime_error("HAMG level-0 out-degree exceeds maxM0");
        }

        linklistsizeint *ll = get_linklist0(node);
        setListCount(ll, ordered.size());
        tableint *data = (tableint *)(ll + 1);
        for (size_t i = 0; i < ordered.size(); i++) {
            data[i] = ordered[i];
        }
    }

    bool passHAMGNeighborSelection(tableint base,
                                   tableint candidate,
                                   const std::vector<tableint> &selected) const {
        dist_t dist_bc = distBetweenNodes(base, candidate);  // dist(a, c) in paper
        for (tableint b : selected) {
            if (distBetweenNodes(b, candidate) < dist_bc) {  // dist(b, c) < dist(a, c)
                return false;
            }
        }
        return true;
    }

    std::vector<tableint> incrNeighborSelHAMG(
        tableint v,
        const std::vector<tableint> &Nv,
        tableint a) const {  // trying to add a to v's neighbors
        std::vector<tableint> Lv;
        dist_t dist_va = distBetweenNodes(v, a);

        for (tableint u : Nv) {
            if (distBetweenNodes(v, u) < dist_va) {  // todo: store distance together?
                Lv.push_back(u);
            }
        }

        if (!passHAMGNeighborSelection(v, a, Lv)) {
            return Nv;
        }

        Lv.push_back(a);
        for (tableint u : Nv) {
            if (distBetweenNodes(v, u) >= dist_va &&
                passHAMGNeighborSelection(v, u, Lv)) {  // todo: duplicated dist computation
                Lv.push_back(u);
            }
        }
        return Lv;
    }

    tableint mutuallyConnectNewElementBottomHAMG(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates,
        bool isUpdate) {
        (void)data_point;

        std::vector<std::pair<dist_t, tableint>> ranked;
        ranked.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            ranked.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

        std::vector<tableint> R;
        R.reserve(std::min(hamg_C_, ranked.size()));
        std::unordered_set<tableint> seen;
        for (const auto &entry : ranked) {
            tableint cand = entry.second;
            if (cand == cur_c) {
                continue;
            }
            if (seen.insert(cand).second) {
                R.push_back(cand);
                if (R.size() >= hamg_C_) {
                    break;
                }
            }
        }

        std::vector<tableint> La;
        La.reserve(R.size());
        for (tableint c : R) {
            if (passHAMGNeighborSelection(cur_c, c, La)) {
                La.push_back(c);
            }
        }
        std::vector<tableint> orderedLa = sortBottomNeighbors(cur_c, La);

        {
            std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur = get_linklist0(cur_c);
            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setBottomNeighbors(cur_c, orderedLa);
        }

        for (tableint v : orderedLa) {
            std::unique_lock<std::mutex> lock(link_list_locks_[v]);
            std::vector<tableint> Nv = getBottomNeighbors(v);
            if (Nv.size() < hamg_dm_) {
                Nv.push_back(cur_c);
                setBottomNeighbors(v, Nv);
            } else {
                std::vector<tableint> Lv = incrNeighborSelHAMG(v, Nv, cur_c);
                setBottomNeighbors(v, Lv);
            }
        }

        return orderedLa.empty() ? cur_c : orderedLa.front();
    }

    tableint mutuallyConnectNewElement(const void *data_point,
                                       tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>,
                                                           std::vector<std::pair<dist_t, tableint>>,
                                                           CompareByFirst> &top_candidates,
                                       int level,
                                       bool isUpdate) {
        if (level == 0 && !isUpdate) {
            return mutuallyConnectNewElementBottomHAMG(data_point, cur_c, top_candidates, isUpdate);
        }

        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error(
                "Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        if (selectedNeighbors.empty()) {
            return cur_c;
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
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
            tableint *data = (tableint *)(ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

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

            tableint *data = (tableint *)(ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of
            // `selectedNeighbors[idx]` then no need to modify any connections or run the
            // heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                                getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]),
                                                        getDataByInternalId(selectedNeighbors[idx]),
                                                        dist_func_param_),
                                           data[j]);
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
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
                    getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) { indx = j;
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

    tableint getInternalIdByLabel(labeltype label) const {
        auto it = label_lookup_.find(label);
        if (it == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        return it->second;
    }

    std::vector<std::pair<dist_t, tableint>> searchKnnInternal(const void *query_data,
                                                               size_t k) const {
        std::priority_queue<std::pair<dist_t, labeltype>> label_res = searchKnn(query_data, k);
        std::vector<std::pair<dist_t, tableint>> out;
        out.reserve(label_res.size());
        while (!label_res.empty()) {
            std::pair<dist_t, labeltype> item = label_res.top();
            label_res.pop();
            out.emplace_back(item.first, getInternalIdByLabel(item.second));
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    inline dist_t distToQuery(const void *query_data, tableint node) const {
        return fstdistfunc_(query_data, getDataByInternalId(node), dist_func_param_);
    }

    // MRNG neighbor selection condition applied to query q (treated as virtual node).
    // Returns true iff no already-selected node b is closer to candidate than q is.
    // Condition: ∀b ∈ selected, Γ(b, candidate) ≥ Γ(q, candidate)
    // This mirrors passHAMGNeighborSelection but uses query→candidate distance.
    bool passHAMGNeighborSelectionQuery(const void *query_data,
                                        tableint candidate,
                                        const std::vector<tableint> &selected) const {
        dist_t dqc = distToQuery(query_data, candidate);  // Γ(q, candidate)
        for (tableint b : selected) {
            if (distBetweenNodes(b, candidate) <
                dqc) {  // Γ(b, candidate) < Γ(q, candidate) → reject
                return false;
            }
        }
        return true;
    }

    // Identify 1-hop nodes of q by simulating HAMG insertion (Algorithm 3 applied to query).
    // Paper §IV-C: "The 1-hop nodes of q are obtained via the HAMG insertion algorithm"
    // Steps: 1) Find C nearest candidates via HNSW kANNS search
    //        2) Apply MRNG-style selection → survivors are q's 1-hop neighbors
    std::vector<tableint> getOneHopNodesByHAMGInsert(const void *query_data) const {
        size_t n = getCurrentElementCount();
        if (n == 0) {
            return {};
        }
        size_t c_eff = std::min(hamg_C_, n);
        // Step 1: R ← C nearest nodes of q, sorted by ascending distance
        std::vector<std::pair<dist_t, tableint>> ranked = searchKnnInternal(query_data, c_eff);

        // Step 2: MRNG selection — same as Algorithm 3 lines 3-9 but using q as base
        std::vector<tableint> one_hop;
        one_hop.reserve(ranked.size());
        for (const auto &item : ranked) {
            tableint cand = item.second;
            if (passHAMGNeighborSelectionQuery(query_data, cand, one_hop)) {
                one_hop.push_back(cand);
            }
        }
        return one_hop;
    }

    static double chooseDouble(size_t n, size_t r) {
        if (r > n) {
            return 0.0;
        }
        if (r == 0 || r == n) {
            return 1.0;
        }
        r = std::min(r, n - r);
        long double res = 1.0;
        for (size_t i = 1; i <= r; i++) {
            res *= static_cast<long double>(n - r + i);
            res /= static_cast<long double>(i);
        }
        return static_cast<double>(res);
    }

    inline double clamp01(double x) const {
        if (x < 0.0) {
            return 0.0;
        }
        if (x > 1.0) {
            return 1.0;
        }
        return x;
    }

    double avgOutDegreeBottom() const {
        if (cur_element_count == 0) {
            return 1.0;
        }
        double total = 0.0;
        for (size_t i = 0; i < cur_element_count; i++) {
            total += static_cast<double>(getListCount(get_linklist0(i)));
        }
        return std::max(1.0, total / static_cast<double>(cur_element_count));
    }

    // Paper §IV-A "Values of α, e and β": Sampling-based estimation of statistics
    // needed to compute pr(p, h) — the probability a ground truth is at hop h.
    //
    // Outputs (all 1-indexed):
    //   α_i        = Pr[p's i-th NN is directly connected to p in HAMG]
    //   e(i,a,i-1) = avg probability that p_i is connected to a node in [p_a, p_{i-1}]
    //   β(1,b)     = avg probability that a node in [p_1, p_b] is connected to p
    //              = (Σ_{i=1}^{b} α_i) / b
    //   φ(a,b)     = avg of e(j,1,j-1) for j ∈ [a,b]  (stored as hamg_e1_avg_)
    void prepareHAMGQueryModel(size_t E, size_t sample_count) {
        std::unique_lock<std::mutex> model_lock(hamg_query_model_lock_);
        if (hamg_query_model_ready_ && hamg_model_E_ >= E && hamg_model_samples_ >= sample_count) {
            return;  // model already computed with sufficient parameters
        }

        if (cur_element_count <= 1) {
            throw std::runtime_error("Index is too small for HAMG query model");
        }
        if (E == 0) {
            throw std::runtime_error("HAMG query model E must be > 0");
        }

        size_t e_eff = std::min(E, static_cast<size_t>(cur_element_count - 1));
        size_t samples_eff = std::min(sample_count, static_cast<size_t>(cur_element_count));
        if (samples_eff == 0) {
            samples_eff = 1;
        }

        // Accumulators for sampling
        std::vector<double> alpha_sum(e_eff + 1, 0.0), alpha_cnt(e_eff + 1, 0.0);
        std::vector<std::vector<double>> e_sum(e_eff + 1, std::vector<double>(e_eff + 1, 0.0));
        std::vector<std::vector<double>> e_cnt(e_eff + 1, std::vector<double>(e_eff + 1, 0.0));

        std::mt19937 rng(20260306u);
        std::uniform_int_distribution<size_t> uid(0, static_cast<size_t>(cur_element_count - 1));

        // Sampling loop: for each sampled node p, find its E nearest neighbors
        for (size_t s = 0; s < samples_eff; s++) {
            tableint p = static_cast<tableint>(uid(rng));
            // kANNS query to get p's nearest neighbor set {v_1, ..., v_E}
            std::vector<std::pair<dist_t, tableint>> knn =
                searchKnnInternal(getDataByInternalId(p), e_eff + 2);

            // nn[i-1] = p's i-th nearest neighbor (1-indexed: nn[0]=p_1, nn[1]=p_2, ...)
            std::vector<tableint> nn;
            nn.reserve(e_eff);
            for (const auto &it : knn) {
                if (it.second == p) {
                    continue;  // exclude self
                }
                nn.push_back(it.second);
                if (nn.size() >= e_eff) {
                    break;
                }
            }
            if (nn.size() < e_eff) {
                continue;
            }

            for (size_t i = 1; i <= e_eff; i++) {
                tableint vi = nn[i - 1];  // p_i = p's i-th nearest neighbor

                // α_i: check if p_i is connected to p in the HAMG bottom graph
                bool connected = false;
                std::vector<tableint> n_vi = getBottomNeighbors(vi);
                for (tableint nb : n_vi) {
                    if (nb == p) {
                        connected = true;
                        break;
                    }
                }
                alpha_sum[i] += connected ? 1.0 : 0.0;
                alpha_cnt[i] += 1.0;

                // e(i, a, i-1): for each a in [1, i-1], compute the ratio of nodes
                // in [p_a, p_{i-1}] that p_i is connected to.
                // Denominator = |[p_a, p_{i-1}]| = i - a
                if (i >= 2) {
                    std::unordered_set<tableint> set_vi(n_vi.begin(), n_vi.end());
                    for (size_t a = 1; a <= i - 1; a++) {
                        size_t denom = i - a;  // |[p_a, p_{i-1}]|
                        size_t hit = 0;
                        for (size_t t = a; t <= i - 1; t++) {
                            if (set_vi.find(nn[t - 1]) != set_vi.end()) {
                                hit++;  // p_i is connected to p_t
                            }
                        }
                        e_sum[i][a] += static_cast<double>(hit) / static_cast<double>(denom);
                        e_cnt[i][a] += 1.0;
                    }
                }
            }
        }

        // Aggregate: compute final statistics from accumulated samples
        hamg_alpha_.assign(e_eff + 1, 0.0);
        hamg_e_.assign(e_eff + 1, std::vector<double>(e_eff + 1, 0.0));
        hamg_beta_.assign(e_eff + 1, 0.0);
        hamg_e1_avg_.assign(e_eff + 1, 0.0);

        for (size_t i = 1; i <= e_eff; i++) {
            // α_i = (# samples where p_i connected to p) / (# samples)
            if (alpha_cnt[i] > 0) {
                hamg_alpha_[i] = clamp01(alpha_sum[i] / alpha_cnt[i]);
            }
            if (i >= 2) {
                // e(i, a, i-1) = average ratio across samples
                for (size_t a = 1; a <= i - 1; a++) {
                    if (e_cnt[i][a] > 0) {
                        hamg_e_[i][a] = clamp01(e_sum[i][a] / e_cnt[i][a]);
                    }
                }
                // hamg_e1_avg_[i] = e(i, 1, i-1), used in φ(a,b) computation
                hamg_e1_avg_[i] = hamg_e_[i][1];
            }
        }

        // β(1, b) = (Σ_{i=1}^{b} α_i) / b
        // Interpretation: avg probability that a node in [p_1, p_b] is connected to p
        double pref = 0.0;
        for (size_t b = 1; b <= e_eff; b++) {
            pref += hamg_alpha_[b];
            hamg_beta_[b] = clamp01(pref / static_cast<double>(b));
        }

        hamg_avg_out_degree_ = avgOutDegreeBottom();  // ω = avg out-degree of HAMG bottom graph
        hamg_model_E_ = e_eff;
        hamg_model_samples_ = samples_eff;
        hamg_query_model_ready_ = true;
        hamg_pr_cache_.clear();
        hamg_pr_cache_k_ = 0;
        hamg_pr_cache_H_ = 0;
    }

    // Returns e(i, a, i-1): avg probability that p_i is connected to a node in [p_a, p_{i-1}]
    inline double getEStat(size_t i, size_t a) const {
        if (i >= hamg_e_.size() || a >= hamg_e_[i].size()) {
            return 0.0;
        }
        return hamg_e_[i][a];
    }

    // Returns β(1, b) = (Σ_{i=1}^{b} α_i) / b
    inline double getBetaStat(size_t b) const {
        if (b == 0 || b >= hamg_beta_.size()) {
            return 0.0;
        }
        return hamg_beta_[b];
    }

    // Returns φ(a, b) = avg of e(j, 1, j-1) for j ∈ [a, b]
    // Paper: "φ(2, i-1) is the average probability that the node in [p_2, p_{i-1}]
    //  is connected to the node that may appear in its next hop"
    // φ(a, b) = Σ_{j=a}^{b} e(j, 1, j-1) / (b - a + 1)
    double getPhiStat(size_t a, size_t b) const {
        if (a > b) {
            return 1.0;
        }
        double sum = 0.0;
        size_t cnt = 0;
        for (size_t j = a; j <= b && j < hamg_e1_avg_.size(); j++) {
            if (j < 2) {
                continue;  // e(j, 1, j-1) undefined for j < 2
            }
            sum += hamg_e1_avg_[j];  // hamg_e1_avg_[j] = e(j, 1, j-1)
            cnt++;
        }
        if (cnt == 0) {
            return 0.0;
        }
        return clamp01(sum / static_cast<double>(cnt));
    }

    // Paper Equation (2): I(p_i, p, h) — probability that an h-hop MP exists from p_i to p
    //
    //   h=1: I(p_i, p, 1) = α_i
    //   h=2: I(p_i, p, 2) = 1/(i-1) · e(i, 1, i-1) · β(1, i-1)
    //   h≥3: I(p_i, p, h) = 1/C(i-1,h-1) · e(i, h-1, i-1) · β(1, i-h+1) · Λ
    //         where Λ = Π_{s=1}^{h-2} φ(s+1, i-h+s+1)
    //
    // MP = monotonic path: p_i → u_1 → ... → u_{h-1} → p
    // C(i-1,h-1) = number of potential h-hop paths (h-1 intermediate nodes from [p_1, p_{i-1}])
    double estimateI(size_t i, size_t h) const {
        if (i == 0 || i >= hamg_alpha_.size()) {
            return 0.0;
        }
        // Case h=1: I(p_i, p, 1) = α_i (direct connection probability)
        if (h == 1) {
            return hamg_alpha_[i];
        }
        if (h > i) {
            return 0.0;  // need at least h-1 intermediate nodes from [p_1, p_{i-1}]
        }
        // Case h=2: I(p_i, p, 2) = 1/(i-1) · e(i, 1, i-1) · β(1, i-1)
        // Single intermediate node u_1 ∈ [p_1, p_{i-1}]
        if (h == 2) {
            if (i < 2) {
                return 0.0;
            }
            double e_term = getEStat(i, 1);         // e(i, 1, i-1): p_i connected to u_1
            double beta_term = getBetaStat(i - 1);  // β(1, i-1): u_1 connected to p
            return clamp01((1.0 / static_cast<double>(i - 1)) * e_term * beta_term);
        }

        // Case h≥3: I(p_i, p, h) = 1/C(i-1,h-1) · e(i, h-1, i-1) · β(1, i-h+1) · Λ
        if (i <= h - 1) {
            return 0.0;  // C(i-1, h-1) = 0
        }
        double comb = chooseDouble(i - 1, h - 1);  // C(i-1, h-1)
        if (comb <= 0.0) {
            return 0.0;
        }

        double e_term = getEStat(i, h - 1);         // e(i, h-1, i-1): p_i → u_1
        double beta_term = getBetaStat(i - h + 1);  // β(1, i-h+1): u_{h-1} → p
        // Λ = Π_{s=1}^{h-2} φ(s+1, i-h+s+1)
        // Joint probability that intermediate nodes u_1..u_{h-2} connect to their successors
        double lambda = 1.0;
        for (size_t s = 1; s <= h - 2; s++) {
            size_t a = s + 1;
            size_t b = i - h + s + 1;
            lambda *= getPhiStat(a, b);
        }
        return clamp01((1.0 / comb) * e_term * beta_term * lambda);
    }

    // Paper §IV-A: pr[gd(p_i, p) = h] — probability that shortest MP from p_i to p is h hops
    //
    //   h=1: pr[gd(p_i, p) = 1] = α_i
    //   h=2, i=2: pr[gd(p_2, p) = 2] = 1 - α_2
    //   h=2, i>2: pr[gd(p_i, p) = 2] = (1 - α_i) · I(p_i, p, 2)
    //   h≥3, h≠i and h≠H:
    //     pr[gd(p_i, p) = h] = (1 - α_i) · I(p_i, p, h) · Π_{x=2}^{h-1} [1 - I(p_i, p, x)]
    //     Joint of: (1) not directly connected, (2) h-hop MP exists, (3) no shorter MP exists
    //   h=i or h=H (boundary):
    //     pr[gd(p_i, p) = h] = (1 - α_i) · Π_{x=2}^{h-1} [1 - I(p_i, p, x)]
    //     (must be h-hop since no shorter path possible)
    double estimatePrGd(size_t i, size_t h, size_t H) const {
        if (i == 0 || i >= hamg_alpha_.size() || h == 0 || h > H) {
            return 0.0;
        }
        double alpha_i = hamg_alpha_[i];
        if (h == 1) {
            return alpha_i;  // pr[gd(p_i, p) = 1] = α_i
        }

        // Π_{x=2}^{h-1} [1 - I(p_i, p, x)] : probability no shorter MP (2..h-1 hops) exists
        double no_shorter_mp_prob = 1.0;
        for (size_t x = 2; x <= h - 1; x++) {
            no_shorter_mp_prob *= (1.0 - estimateI(i, x));
        }
        no_shorter_mp_prob = clamp01(no_shorter_mp_prob);

        // Boundary case: h=i or h=H → must be at this hop (no need for I(p_i,p,h) factor)
        if (h == i || h == H) {
            return clamp01((1.0 - alpha_i) * no_shorter_mp_prob);
        }
        // General case: (1-α_i) · I(p_i, p, h) · Π[1 - I(p_i, p, x)]
        return clamp01((1.0 - alpha_i) * estimateI(i, h) * no_shorter_mp_prob);
    }

    // Paper Equation (1): pr(p, h) = (1/k) · Σ_{i=1}^{k} pr[gd(p_i, p) = h]
    // Expected probability of encountering a ground truth node at the h-th hop of q.
    // Cached per (k, H) pair for reuse across queries.
    void buildPrCache(size_t k, size_t H) const {
        std::unique_lock<std::mutex> lock(hamg_query_model_lock_);
        if (hamg_pr_cache_k_ == k && hamg_pr_cache_H_ == H && hamg_pr_cache_.size() == H + 1) {
            return;
        }
        hamg_pr_cache_.assign(H + 1, 0.0);
        for (size_t h = 1; h <= H; h++) {
            double sum = 0.0;
            for (size_t i = 1; i <= k && i < hamg_alpha_.size(); i++) {
                sum += estimatePrGd(i, h, H);  // pr[gd(p_i, p) = h]
            }
            hamg_pr_cache_[h] = clamp01(sum / static_cast<double>(k));  // pr(p, h)
        }
        hamg_pr_cache_k_ = k;
        hamg_pr_cache_H_ = H;
    }

    // Algorithm 1: getHopCount(q, k, r)
    // Determines the minimum traversed hop count k' such that expected recall ≥ r.
    //   H = min(k, ⌈log_ω |D|⌉)      — maximum possible hop count
    //   Loop: k' += 1; cur_r += pr(p, k'); stop when cur_r ≥ r or k' = H
    size_t getHopCountHAMG(size_t k, double desired_recall) const {
        if (k == 0) {
            return 0;
        }
        double omega = std::max(1.000001, hamg_avg_out_degree_);  // ω = avg out-degree
        // H = min(k, ⌈log_ω |D|⌉) — Paper Algorithm 1 Line 3
        double h_est =
            std::ceil(std::log(static_cast<double>(cur_element_count)) / std::log(omega));
        if (!std::isfinite(h_est) || h_est < 1.0) {
            h_est = 1.0;
        }
        size_t H = std::min(k, static_cast<size_t>(h_est));
        H = std::max<size_t>(1, H);

        buildPrCache(k, H);  // precompute pr(p, h) for h = 1..H

        // Algorithm 1 Lines 4-10: accumulate pr(p, k') until cur_r ≥ r
        double r = std::max(0.0, std::min(1.0, desired_recall));
        size_t k_prime = 0;
        double cur_r = 0.0;
        while (cur_r < r) {
            k_prime += 1;
            if (k_prime == H) {
                return k_prime;  // Line 6-7: reached maximum hop count
            }
            cur_r += hamg_pr_cache_[k_prime];  // Line 9: cur_r += pr(p, k')
        }
        return std::max<size_t>(1, std::min(k_prime, H));
    }

    // Compute ED_{k'}: distance threshold for PS1 pruning.
    // Paper §IV-B PS1: "the expected maximum distance between q and its result nodes
    // can be estimated by the distance within k' hops of q"
    //
    // Base case (k'=1): ED_1 = max Γ(q, v) for v in 1-hop nodes
    // General case:     ED_{k'} = ED_1 · ω^{k'/d}  (Parzen density estimator, EUCLIDEAN)
    //
    // hnswlib L2Space's distance function returns ‖x-y‖² (squared L2). Both `md` and the
    // returned ED are squared. Squaring the paper formula gives ED² = ED_1² · ω^{2k'/d},
    // hence the 2.0× in the exponent below.
    dist_t computeEDHAMG(const void *query_data,
                         const std::vector<tableint> &one_hop_nodes,
                         size_t k_prime) const {
        if (one_hop_nodes.empty()) {
            return std::numeric_limits<dist_t>::max();
        }
        // ED_1² = max squared distance from q to any 1-hop node
        dist_t md = 0;
        for (tableint node : one_hop_nodes) {
            md = std::max(md, distToQuery(query_data, node));
        }
        if (k_prime <= 1) {
            return md;
        }
        // ED_{k'}² = ED_1² · ω^{2k'/d}
        size_t dim = *((size_t *)dist_func_param_);
        double omega = std::max(1.000001, hamg_avg_out_degree_);
        double lambda =
            std::pow(omega, 2.0 * static_cast<double>(k_prime) / static_cast<double>(dim));
        return static_cast<dist_t>(static_cast<double>(md) * lambda);
    }

    struct HAMGQueryDiag {
        double omega;                  // avg out-degree ω
        size_t H;                      // max hop count
        size_t k_prime;                // actual hop count returned by getHopCount
        size_t one_hop_count;          // |1-hop nodes|
        float ED_1;                    // max dist to 1-hop nodes
        float ED_kprime;               // PS1 distance threshold
        double lambda;                 // ED_{k'} / ED_1
        size_t candidates_ps1;         // |candidates after BFS+PS1|
        size_t candidates_ps2;         // |candidates after PS2|
        size_t anns_verifications;     // |candidates sent to inner kANNS verification|
        size_t verified;               // |results after verification|
        double time_prepare_ms;        // prepareHAMGQueryModel
        double time_hop_count_ms;      // getHopCountHAMG
        double time_one_hop_ms;        // getOneHopNodesByHAMGInsert
        double time_ed_ms;             // computeEDHAMG
        double time_ps1_ms;            // BFS + PS1 pruning
        double time_ps2_ms;            // PS2 pruning
        double time_verification_ms;   // final verification stage
        double time_total_ms;          // total searchRknnHAMG time
        std::vector<double> pr_cache;  // pr(p, h) for h=1..H
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Algorithm 2: query(q, k, r) — RkANNS query on HAMG
    //
    // Three phases:
    //   Phase 1 (BFS + PS1): collect candidate nodes within k' hops of q
    //   Phase 2 (PS2):       prune candidates using neighbor distance comparison
    //   Phase 3 (Verify):    full kANNS verification for surviving candidates
    // ═══════════════════════════════════════════════════════════════════════════
    std::priority_queue<std::pair<dist_t, labeltype>> searchRknnHAMG(const void *query_data,
                                                                     size_t k,
                                                                     double desired_recall,
                                                                     size_t model_E = 0,
                                                                     size_t sample_count = 2000,
                                                                     HAMGQueryDiag *diag = nullptr,
                                                                     bool run_verification = true) {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (diag != nullptr) {
            *diag = {};
        }
        using QueryClock = std::chrono::steady_clock;
        auto total_start = QueryClock::now();
        auto finalize_diag_total = [&]() {
            if (diag != nullptr) {
                diag->time_total_ms =
                    std::chrono::duration<double, std::milli>(QueryClock::now() - total_start)
                        .count();
            }
        };
        if (cur_element_count == 0 || k == 0) {
            finalize_diag_total();
            return result;
        }

        // Prepare probability model (α, e, β statistics) if not already computed
        size_t k_eff = std::min(k, static_cast<size_t>(cur_element_count - 1));
        size_t E = std::max(k_eff, (model_E > 0 ? model_E : std::max<size_t>(k_eff, 32)));
        E = std::min(E, static_cast<size_t>(cur_element_count - 1));
        auto phase_start = QueryClock::now();
        prepareHAMGQueryModel(E, sample_count);
        if (diag != nullptr) {
            diag->time_prepare_ms =
                std::chrono::duration<double, std::milli>(QueryClock::now() - phase_start).count();
        }

        if (diag != nullptr) {
            diag->omega = hamg_avg_out_degree_;

            double h_est = std::ceil(std::log(static_cast<double>(cur_element_count)) /
                                     std::log(std::max(1.000001, diag->omega)));
            if (!std::isfinite(h_est) || h_est < 1.0) {
                h_est = 1.0;
            }
            diag->H = std::min(k_eff, static_cast<size_t>(h_est));
            diag->H = std::max<size_t>(1, diag->H);
        }

        // Algorithm 2, Line 1: k' ← getHopCount(q, k, r)
        phase_start = QueryClock::now();
        size_t k_prime = (forced_k_prime_ > 0) ? forced_k_prime_
                                                : getHopCountHAMG(k_eff, desired_recall);
        if (diag != nullptr) {
            diag->time_hop_count_ms =
                std::chrono::duration<double, std::milli>(QueryClock::now() - phase_start).count();
            diag->k_prime = k_prime;
            diag->pr_cache.assign(hamg_pr_cache_.begin(), hamg_pr_cache_.end());
        }

        // Algorithm 2, Line 2: identify 1-hop nodes via simulated HAMG insertion of q
        phase_start = QueryClock::now();
        std::vector<tableint> one_hop = getOneHopNodesByHAMGInsert(query_data);
        if (diag != nullptr) {
            diag->time_one_hop_ms =
                std::chrono::duration<double, std::milli>(QueryClock::now() - phase_start).count();
            diag->one_hop_count = one_hop.size();
            dist_t md = 0;
            for (tableint node : one_hop) {
                md = std::max(md, distToQuery(query_data, node));
            }
            diag->ED_1 = md;
        }
        if (one_hop.empty()) {
            finalize_diag_total();
            return result;
        }

        // Algorithm 2, Line 3: ED_{k'} ← distance threshold for PS1
        phase_start = QueryClock::now();
        dist_t ED_kprime = no_ps1_ ? std::numeric_limits<dist_t>::max()
                                   : computeEDHAMG(query_data, one_hop, k_prime);
        if (diag != nullptr) {
            diag->time_ed_ms =
                std::chrono::duration<double, std::milli>(QueryClock::now() - phase_start).count();
            diag->ED_kprime = ED_kprime;
            diag->lambda = (diag->ED_1 > 0)
                               ? static_cast<double>(ED_kprime) / static_cast<double>(diag->ED_1)
                               : 0.0;
        }

        // ── Phase 1: BFS expansion with PS1 pruning (Algorithm 2 Lines 5-12) ──
        // Starting from 1-hop nodes, expand to k' hops.
        // PS1: only visit neighbor v if Γ(q, v) ≤ ED_{k'}
        std::unordered_set<tableint> candidate_set;
        std::unordered_map<tableint, size_t> hop_map;  // tracks min hop count per candidate
        std::vector<tableint> cur_hop_set = one_hop;
        for (tableint n : one_hop) {
            candidate_set.insert(n);
            hop_map[n] = 1;
        }

        phase_start = QueryClock::now();
        size_t h = 1;
        while (h < k_prime && !cur_hop_set.empty()) {
            h += 1;
            std::vector<tableint> next_hop_set;
            std::unordered_set<tableint> next_seen;
            for (tableint p : cur_hop_set) {
                std::vector<tableint> neigh = getBottomNeighbors(p);
                for (tableint v : neigh) {
                    // PS1: Γ(q, v) ≤ ED_{k'} — prune nodes too far from q
                    if (distToQuery(query_data, v) <= ED_kprime) {
                        candidate_set.insert(v);
                        auto it = hop_map.find(v);
                        if (it == hop_map.end() || h < it->second) {
                            hop_map[v] = h;  // record minimum hop count for PS2
                        }
                        if (next_seen.insert(v).second) {
                            next_hop_set.push_back(v);
                        }
                    }
                }
            }
            cur_hop_set.swap(next_hop_set);
        }
        if (diag != nullptr) {
            diag->time_ps1_ms =
                std::chrono::duration<double, std::milli>(QueryClock::now() - phase_start).count();
            diag->candidates_ps1 = candidate_set.size();
        }

        // ── Phase 2: PS2 pruning (Algorithm 2 Lines 14-18, Lemma 4) ──
        // For each candidate o at hop h_o:
        //   f ≤ (h-1)/(C-h+2) · ω_o     (upper bound on neighbors of o within MP)
        //   l = min(k - h + 1 + f, k)     (tightened neighbor index)
        //   If Γ(o, v_l) < Γ(o, q), prune o as non-result  (Lemma 4)
        std::vector<tableint> vset;
        vset.reserve(candidate_set.size());
        phase_start = QueryClock::now();
        for (tableint o : candidate_set) {
            size_t h_o = 1;  // hop count at which o was discovered
            auto hit = hop_map.find(o);
            if (hit != hop_map.end()) {
                h_o = hit->second;
            }
            std::vector<tableint> o_neighbors = getBottomNeighbors(o);  // sorted by distance
            size_t omega_o = o_neighbors.size();                        // ω_o = out-degree of o

            // f ≤ ω_o · (h-1) / (C - h + 2)  (Lemma 4 upper bound)
            long long denom_ll =
                static_cast<long long>(hamg_C_) - static_cast<long long>(h_o) + 2LL;
            double denom = denom_ll > 0 ? static_cast<double>(denom_ll) : 1.0;
            double f_est = std::ceil((static_cast<double>(omega_o) *
                                      std::max<long long>(0, static_cast<long long>(h_o) - 1LL)) /
                                     denom);

            // l = min(k - h + 1 + ⌈f⌉, k)
            double l_raw = static_cast<double>(k_eff) - static_cast<double>(h_o) + 1.0 + f_est;
            size_t l =
                static_cast<size_t>(std::max(1.0, std::min(static_cast<double>(k_eff), l_raw)));

            // Get Γ(o, v_l) — distance from o to its l-th nearest neighbor
            dist_t d_ol;
            if (o_neighbors.empty()) {
                d_ol = std::numeric_limits<dist_t>::max();
            } else if (l <= o_neighbors.size()) {
                d_ol = distBetweenNodes(o, o_neighbors[l - 1]);  // v_l (0-indexed)
            } else {
                // Remark 2: when l > actual out-degree g, Parzen density gives the
                // EUCLIDEAN estimate Γ(o, v_l) ≈ (l/g)^{1/d} · Γ(o, v_g).
                // distBetweenNodes returns ‖·‖²; squaring yields the 2/d exponent.
                size_t g = o_neighbors.size();
                dist_t d_og = distBetweenNodes(o, o_neighbors[g - 1]);
                size_t dim = *((size_t *)dist_func_param_);
                double scale = std::pow(static_cast<double>(l) / static_cast<double>(g),
                                        2.0 / static_cast<double>(dim));
                d_ol = static_cast<dist_t>(scale * static_cast<double>(d_og));
            }

            // Lemma 4 check: if Γ(o, q) ≤ Γ(o, v_l), o survives PS2
            if (distToQuery(query_data, o) <= d_ol) {
                vset.push_back(o);
            }
        }
        if (diag != nullptr) {
            diag->time_ps2_ms =
                std::chrono::duration<double, std::milli>(QueryClock::now() - phase_start).count();
            diag->candidates_ps2 = vset.size();
            diag->anns_verifications = vset.size();
        }

        if (!run_verification) {
            finalize_diag_total();
            return result;
        }

        // ── Phase 3: Verification (Algorithm 2 Lines 20-22) ──
        // For each surviving candidate u, run a full kANNS query centered at u.
        // If q is among u's k nearest neighbors → u is a true RkNN result.
        // Check: Γ(u, q) ≤ Γ(u, u_k)  (u_k = u's k-th nearest neighbor)
        size_t verified = 0;
        phase_start = QueryClock::now();
        for (tableint u : vset) {
            dist_t d_uq = distToQuery(query_data, u);  // Γ(u, q)
            std::vector<std::pair<dist_t, tableint>> knn_u =
                searchKnnInternal(getDataByInternalId(u), k_eff + 2);  // kANNS(u, k)

            std::vector<dist_t> dist_u;
            dist_u.reserve(knn_u.size());
            for (const auto &it : knn_u) {
                if (it.second == u) {
                    continue;  // exclude self
                }
                dist_u.push_back(it.first);
            }
            if (dist_u.empty()) {
                continue;
            }
            dist_t kth_dist = dist_u.size() >= k_eff ? dist_u[k_eff - 1] : dist_u.back();
            // Γ(u, q) ≤ Γ(u, u_k) → q is in u's k-NN → u is RkNN result
            if (d_uq <= kth_dist) {
                result.emplace(d_uq, getExternalLabel(u));
                ++verified;
            }
        }
        if (diag != nullptr) {
            diag->time_verification_ms =
                std::chrono::duration<double, std::milli>(QueryClock::now() - phase_start).count();
            diag->verified = verified;
        }

        finalize_diag_total();
        return result;
    }

    // ── Diagnostic query: same as searchRknnHAMG but collects per-query stats ──
    HAMGQueryDiag diagRknnHAMG(const void *query_data,
                               size_t k,
                               double desired_recall,
                               size_t model_E = 0,
                               size_t sample_count = 2000) {
        HAMGQueryDiag diag = {};
        searchRknnHAMG(query_data, k, desired_recall, model_E, sample_count, &diag);
        return diag;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Ablation: searchRknnHAMG with selective disabling of H cap / PS1 / PS2
    // ═══════════════════════════════════════════════════════════════════════════

    struct AblationFlags {
        bool no_h_cap = false;  // H = k instead of min(k, ⌈log_ω|D|⌉)
        bool no_ps1 = false;    // ED_{k'} = MAX_FLOAT (disable distance pruning)
        bool no_ps2 = false;    // skip Lemma 4 pruning
    };

    // getHopCount variant that accepts an explicit H override
    size_t getHopCountHAMGAblation(size_t k, double desired_recall, size_t H_override) const {
        if (k == 0) {
            return 0;
        }
        size_t H = std::max<size_t>(1, H_override);

        buildPrCache(k, H);

        double r = std::max(0.0, std::min(1.0, desired_recall));
        size_t k_prime = 0;
        double cur_r = 0.0;
        while (cur_r < r) {
            k_prime += 1;
            if (k_prime == H) {
                return k_prime;
            }
            cur_r += hamg_pr_cache_[k_prime];
        }
        return std::max<size_t>(1, std::min(k_prime, H));
    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchRknnHAMGAblation(
        const void *query_data,
        size_t k,
        double desired_recall,
        const AblationFlags &flags,
        size_t model_E = 0,
        size_t sample_count = 2000) {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0 || k == 0) {
            return result;
        }

        size_t k_eff = std::min(k, static_cast<size_t>(cur_element_count - 1));
        size_t E = std::max(k_eff, (model_E > 0 ? model_E : std::max<size_t>(k_eff, 32)));
        E = std::min(E, static_cast<size_t>(cur_element_count - 1));
        prepareHAMGQueryModel(E, sample_count);

        // --- Hop count (with optional H override) ---
        size_t k_prime;
        if (flags.no_h_cap) {
            k_prime = getHopCountHAMGAblation(k_eff, desired_recall, k_eff);
        } else {
            k_prime = getHopCountHAMG(k_eff, desired_recall);
        }

        std::vector<tableint> one_hop = getOneHopNodesByHAMGInsert(query_data);
        if (one_hop.empty()) {
            return result;
        }

        // --- ED_{k'} (with optional PS1 disable) ---
        dist_t ED_kprime;
        if (flags.no_ps1) {
            ED_kprime = std::numeric_limits<dist_t>::max();
        } else {
            ED_kprime = computeEDHAMG(query_data, one_hop, k_prime);
        }

        // ── Phase 1: BFS + PS1 ──
        std::unordered_set<tableint> candidate_set;
        std::unordered_map<tableint, size_t> hop_map;
        std::vector<tableint> cur_hop_set = one_hop;
        for (tableint n : one_hop) {
            candidate_set.insert(n);
            hop_map[n] = 1;
        }

        size_t h = 1;
        while (h < k_prime && !cur_hop_set.empty()) {
            h += 1;
            std::vector<tableint> next_hop_set;
            std::unordered_set<tableint> next_seen;
            for (tableint p : cur_hop_set) {
                std::vector<tableint> neigh = getBottomNeighbors(p);
                for (tableint v : neigh) {
                    if (distToQuery(query_data, v) <= ED_kprime) {
                        candidate_set.insert(v);
                        auto it = hop_map.find(v);
                        if (it == hop_map.end() || h < it->second) {
                            hop_map[v] = h;
                        }
                        if (next_seen.insert(v).second) {
                            next_hop_set.push_back(v);
                        }
                    }
                }
            }
            cur_hop_set.swap(next_hop_set);
        }

        // ── Phase 2: PS2 (with optional disable) ──
        std::vector<tableint> vset;
        if (flags.no_ps2) {
            // Skip PS2: all PS1 candidates go directly to verification
            vset.assign(candidate_set.begin(), candidate_set.end());
        } else {
            vset.reserve(candidate_set.size());
            for (tableint o : candidate_set) {
                size_t h_o = 1;
                auto hit = hop_map.find(o);
                if (hit != hop_map.end()) {
                    h_o = hit->second;
                }
                std::vector<tableint> o_neighbors = getBottomNeighbors(o);
                size_t omega_o = o_neighbors.size();

                long long denom_ll =
                    static_cast<long long>(hamg_C_) - static_cast<long long>(h_o) + 2LL;
                double denom = denom_ll > 0 ? static_cast<double>(denom_ll) : 1.0;
                double f_est =
                    std::ceil((static_cast<double>(omega_o) *
                               std::max<long long>(0, static_cast<long long>(h_o) - 1LL)) /
                              denom);

                double l_raw = static_cast<double>(k_eff) - static_cast<double>(h_o) + 1.0 + f_est;
                size_t l =
                    static_cast<size_t>(std::max(1.0, std::min(static_cast<double>(k_eff), l_raw)));

                dist_t d_ol;
                if (o_neighbors.empty()) {
                    d_ol = std::numeric_limits<dist_t>::max();
                } else if (l <= o_neighbors.size()) {
                    d_ol = distBetweenNodes(o, o_neighbors[l - 1]);
                } else {
                    size_t g = o_neighbors.size();
                    dist_t d_og = distBetweenNodes(o, o_neighbors[g - 1]);
                    size_t dim = *((size_t *)dist_func_param_);
                    double scale = std::pow(static_cast<double>(l) / static_cast<double>(g),
                                            2.0 / static_cast<double>(dim));
                    d_ol = static_cast<dist_t>(scale * static_cast<double>(d_og));
                }

                if (distToQuery(query_data, o) <= d_ol) {
                    vset.push_back(o);
                }
            }
        }

        // ── Phase 3: Verification ──
        for (tableint u : vset) {
            dist_t d_uq = distToQuery(query_data, u);
            std::vector<std::pair<dist_t, tableint>> knn_u =
                searchKnnInternal(getDataByInternalId(u), k_eff + 2);

            std::vector<dist_t> dist_u;
            dist_u.reserve(knn_u.size());
            for (const auto &it : knn_u) {
                if (it.second == u) {
                    continue;
                }
                dist_u.push_back(it.first);
            }
            if (dist_u.empty()) {
                continue;
            }
            dist_t kth_dist = dist_u.size() >= k_eff ? dist_u[k_eff - 1] : dist_u.back();
            if (d_uq <= kth_dist) {
                result.emplace(d_uq, getExternalLabel(u));
            }
        }

        return result;
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error(
                "Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char *data_level0_memory_new =
            (char *)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char **linkLists_new = (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate other layers");
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
        size += sizeof(HAMG_INDEX_MAGIC);
        size += sizeof(HAMG_INDEX_VERSION);
        size += sizeof(hamg_C_);
        size += sizeof(hamg_dm_);
        size += sizeof(build_num_threads_);
        size += sizeof(build_seed_);
        size += sizeof(uint64_t);
        size += build_base_path_.size();

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize =
                element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

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
        writeBinaryPOD(output, HAMG_INDEX_MAGIC);
        writeBinaryPOD(output, HAMG_INDEX_VERSION);
        writeBinaryPOD(output, hamg_C_);
        writeBinaryPOD(output, hamg_dm_);
        writeBinaryPOD(output, build_num_threads_);
        writeBinaryPOD(output, build_seed_);
        writeBinaryString(output, build_base_path_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize =
                element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }

    void loadIndex(const std::string &location,
                   SpaceInterface<dist_t> *s,
                   size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

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
        uint64_t index_magic = 0;
        uint64_t index_version = 0;
        readBinaryPOD(input, index_magic);
        readBinaryPOD(input, index_version);
        if (index_magic != HAMG_INDEX_MAGIC || index_version != HAMG_INDEX_VERSION) {
            throw std::runtime_error(
                "Unsupported HAMG index format. Rebuild the HAMG index with the current code.");
        }
        readBinaryPOD(input, hamg_C_);
        readBinaryPOD(input, hamg_dm_);
        readBinaryPOD(input, build_num_threads_);
        readBinaryPOD(input, build_seed_);
        readBinaryString(input, build_base_path_);
        if (hamg_C_ == 0 || hamg_C_ > maxM0_) {
            throw std::runtime_error("Loaded HAMG index has invalid parameter C");
        }

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        hamg_query_model_ready_ = false;
        hamg_model_E_ = 0;
        hamg_model_samples_ = 0;
        hamg_pr_cache_.clear();
        hamg_pr_cache_k_ = 0;
        hamg_pr_cache_H_ = 0;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *)malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error(
                        "Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_)
                    deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }

    template <typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char *data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *)dist_func_param_);
        std::vector<data_t> data;
        data_t *data_ptr = (data_t *)data_ptrv;
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
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
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
     * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost
     * all cases.
     */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
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
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
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
                std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
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
        unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    unsigned short int getListCount(linklistsizeint *ptr) const {
        return *((unsigned short int *)ptr);
    }

    void setListCount(linklistsizeint *ptr, unsigned short int size) const {
        *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
    }

    /*
     * Adds point. Updates the point if it is already in the index.
     * If replacement of deleted elements is enabled: replaces previously deleted point if any,
     * updating it with new point
     */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        hamg_query_model_ready_ = false;
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
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

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
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
        // If point to be updated is entry point and graph just contains single element then just
        // return.
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

            for (auto &&elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto &&elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto &&neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    candidates;
                size_t size = sCand.find(neigh) == sCand.end()
                                  ? sCand.size()
                                  : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto &&cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(
                        getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
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
                    std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *)(ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }

    void repairConnectionsForUpdate(const void *dataPoint,
                                    tableint entryPointInternalId,
                                    tableint dataPointInternalId,
                                    int dataPointLevel,
                                    int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist =
                fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d =
                            fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
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
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                topCandidates = searchBaseLayer(currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases
            // where `topCandidates` could just contains entry point itself. To prevent self loops,
            // the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(
                        fstdistfunc_(
                            dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_),
                        entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(
                    dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }

    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *)(data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }

    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error(
                            "Can't use addPoint to update deleted elements if replacement of "
                            "deleted elements is enabled.");
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

        std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock<std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_,
               0,
               size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist =
                    fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *)(data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(
                                data_point, getDataByInternalId(cand), dist_func_param_);
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

                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    top_candidates = searchBaseLayer(currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(
                        fstdistfunc_(
                            data_point, getDataByInternalId(enterpoint_copy), dist_func_param_),
                        enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj =
                    mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
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

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist =
            fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d =
                        fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates =
                searchBaseLayerST<true>(currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates =
                searchBaseLayerST<false>(currObj, query_data, std::max(ef_, k), isIdAllowed);
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

    std::vector<std::pair<dist_t, labeltype>> searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t> &stop_condition,
        BaseFilterFunctor *isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist =
            fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d =
                        fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        top_candidates =
            searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

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
        std::vector<int> inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *)(ll_cur + 1);
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
            for (int i = 0; i < cur_element_count; i++) {
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
