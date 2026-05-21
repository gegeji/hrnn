#include "utils.h"
#include "hnswlib/hrnn.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>

/*
 * Canonical streaming-insert test harness: builds HRNN on initial split_n
 * points, then streams the remaining via insertPointDynamic() in an OpenMP
 * parallel-for loop, then evaluates recall and per-insert latency.
 *
 * Historical note: an experimental "batch_adaptive" mode (frontier repair
 * via NNDescent-style local join, proposed in #46) lived here until #48
 * falsified it on K=500 KNN graphs (cascade covers 31-53% of graph, 134x
 * slower than incremental at identical recall). Pruned from the source;
 * keep incremental as the only maintenance path.
 */

using HRNN = hnswlib::HRNN<float>;
using TableId = hnswlib::tableint;

struct EvalMetrics {
    double macro_recall{0.0};
    double micro_recall{0.0};
    double avg_results{0.0};
    long search_ms{0};
};

static EvalMetrics evaluate_index(HRNN& index,
                                  const FbinData& query,
                                  const RknnGroundtruth& gt,
                                  int m_proxy,
                                  int k_rknn,
                                  int ef_search) {
    index.setEf(static_cast<size_t>(ef_search));
    size_t nq = query.n;
    std::vector<std::vector<hnswlib::labeltype>> all_results(nq);

    auto t0 = Clock::now();
    size_t k_prime = index.K_knng_;
    for (size_t q = 0; q < nq; ++q) {
        const float* qvec = query.data.data() + q * query.d;
        all_results[q] = index.searchRknn(
            qvec, static_cast<size_t>(m_proxy), 1.0f,
            static_cast<size_t>(k_rknn), k_prime,
            static_cast<size_t>(ef_search));
    }
    long search_ms = elapsed_ms(t0);

    double macro_recall = compute_rknn_recall(gt, all_results);

    size_t found = 0;
    size_t total_gt = 0;
    for (size_t q = 0; q < nq; ++q) {
        size_t gt_cnt = gt.count(q);
        if (gt_cnt == 0) {
            continue;
        }
        total_gt += gt_cnt;
        const uint32_t* gt_ids = gt.ids_for(q);
        std::unordered_set<uint32_t> gt_set(gt_ids, gt_ids + gt_cnt);
        for (auto label : all_results[q]) {
            if (gt_set.count(static_cast<uint32_t>(label))) {
                ++found;
            }
        }
    }

    double micro_recall =
        total_gt > 0 ? static_cast<double>(found) / static_cast<double>(total_gt) : 0.0;

    double avg_results = 0.0;
    for (const auto& results : all_results) {
        avg_results += static_cast<double>(results.size());
    }
    avg_results /= static_cast<double>(nq);

    return EvalMetrics{
        .macro_recall = macro_recall,
        .micro_recall = micro_recall,
        .avg_results = avg_results,
        .search_ms = search_ms,
    };
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string base_path, query_path, gt_path, snapshot_path, oracle_snapshot_path;
    int M, ef_construction, K_knng, nn_iters, nn_sample, num_threads;
    int k_rknn, m_proxy, ef_search;
    int batch_size_cli, m_update, bidirectional_flag;
    double split_ratio;

    po::options_description desc("HRNN Dynamic Update Test");
    desc.add_options()("help,h", "Show help")

        ("base", po::value<std::string>(&base_path)->required(),
         "Path to base vectors (.fbin)")

        ("query", po::value<std::string>(&query_path)->required(),
         "Path to query vectors (.fbin)")

        ("gt", po::value<std::string>(&gt_path)->required(),
         "Path to RkNN groundtruth (.bin)")

        ("snapshot", po::value<std::string>(&snapshot_path)->default_value(""),
         "Snapshot index path. If file exists, load it (skip build). "
         "If not, build from scratch and save here.")

        ("oracle_snapshot", po::value<std::string>(&oracle_snapshot_path)->default_value(""),
         "Optional full-build oracle HRNN snapshot for recall-gap comparison")

        ("batch_size", po::value<int>(&batch_size_cli)->default_value(0),
         "Inserted batch size. 0 = use all points after split")

        ("M", po::value<int>(&M)->default_value(16), "HNSW M parameter")

        ("ef_construction", po::value<int>(&ef_construction)->default_value(200),
         "HNSW ef_construction")

        ("K_knng", po::value<int>(&K_knng)->default_value(500),
         "KNNG depth")

        ("nn_iters", po::value<int>(&nn_iters)->default_value(3),
         "NNDescent refinement iterations for initial build")

        ("nn_sample", po::value<int>(&nn_sample)->default_value(20),
         "NNDescent sample size")

        ("num_threads", po::value<int>(&num_threads)->default_value(64),
         "Number of threads")

        ("k", po::value<int>(&k_rknn)->default_value(10),
         "RkNN k parameter")

        ("m", po::value<int>(&m_proxy)->default_value(50),
         "Proxy count for search")

        ("ef_search", po::value<int>(&ef_search)->default_value(200),
         "HNSW ef_search")

        ("split", po::value<double>(&split_ratio)->default_value(0.5),
         "Fraction of data for initial build (e.g. 0.5 = 50%)")

        ("m_update", po::value<int>(&m_update)->default_value(0),
         "Incremental: proxy count for self-query (0 = use all KNNG neighbors)")

        ("bidirectional", po::value<int>(&bidirectional_flag)->default_value(0),
         "Incremental: if 1, also improve x_new's KNNG from self-query candidates (Opt-2)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    std::cout << "=== HRNN Dynamic Update Test ===" << std::endl;
    std::cout << "  M=" << M
              << "  ef_construction=" << ef_construction
              << "  K_knng=" << K_knng << std::endl;
    std::cout << "  split=" << split_ratio << "  k=" << k_rknn
              << "  m=" << m_proxy << "  ef_search=" << ef_search
              << "  m_update=" << m_update
              << "  bidirectional=" << bidirectional_flag << std::endl;

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
#endif

    std::cout << "\n[1] Loading data" << std::endl;
    auto t0 = Clock::now();
    auto base = load_fbin(base_path);
    std::cout << "      base: n=" << base.n << "  d=" << base.d
              << "  (" << elapsed_ms(t0) << " ms)" << std::endl;

    t0 = Clock::now();
    auto query = load_fbin(query_path);
    std::cout << "      query: n=" << query.n << "  d=" << query.d
              << "  (" << elapsed_ms(t0) << " ms)" << std::endl;

    t0 = Clock::now();
    auto gt = load_rknn_groundtruth(gt_path);
    std::cout << "      GT: nq=" << gt.nq << "  total=" << gt.total()
              << "  (" << elapsed_ms(t0) << " ms)" << std::endl;

    size_t n = base.n;
    size_t split_n = static_cast<size_t>(static_cast<double>(n) * split_ratio);

    std::vector<uint32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(perm.begin(), perm.end(), rng);

    size_t insert_n_total = n - split_n;
    size_t batch_n = batch_size_cli > 0
        ? std::min<size_t>(static_cast<size_t>(batch_size_cli), insert_n_total)
        : insert_n_total;

    std::cout << "\n[2] Split: initial=" << split_n
              << "  incremental_total=" << insert_n_total
              << "  active_batch=" << batch_n << std::endl;

    hnswlib::L2Space space(base.d);
    std::unique_ptr<HRNN> index;
    long total_build_ms = 0;

    bool snapshot_exists = !snapshot_path.empty()
        && std::filesystem::exists(snapshot_path);

    if (snapshot_exists) {
        std::cout << "\n[3] Loading snapshot: " << snapshot_path << std::endl;
        t0 = Clock::now();
        index = std::make_unique<HRNN>(
            &space, snapshot_path, /*nmslib=*/false, /*max_elements=*/n);
        long load_ms = elapsed_ms(t0);
        std::cout << "      Loaded (" << load_ms << " ms)"
                  << "  cur_elements=" << index->getCurrentElementCount()
                  << "  K_knng=" << index->K_knng_
                  << "  knng_built=" << index->knng_built_ << std::endl;

        std::cout << "\n[3b] Building mutable RKNNG from snapshot" << std::endl;
        t0 = Clock::now();
        index->buildRKNNG_mutable();
        long rknng_ms = elapsed_ms(t0);
        std::cout << "      RKNNG: " << rknng_ms << " ms" << std::endl;
        total_build_ms = load_ms + rknng_ms;
    } else {
        std::cout << "\n[3] Building HRNN on initial " << split_n << " points" << std::endl;
        index = std::make_unique<HRNN>(
            &space, n, M, ef_construction, /*random_seed=*/100,
            /*allow_replace_deleted=*/false, /*K_knng=*/static_cast<size_t>(K_knng));

        t0 = Clock::now();
        {
            std::atomic<int> progress{0};
            int total_init = static_cast<int>(split_n);
            int report_interval = std::max(1, total_init / 10);
#pragma omp parallel for schedule(dynamic, 512)
            for (int i = 0; i < total_init; ++i) {
                uint32_t orig_id = perm[i];
                index->addPoint(base.data.data() + static_cast<size_t>(orig_id) * base.d,
                                static_cast<hnswlib::labeltype>(orig_id));
                int done = progress.fetch_add(1, std::memory_order_relaxed) + 1;
                if (done % report_interval == 0 || done == total_init) {
#pragma omp critical
                    {
                        std::cout << "\r      addPoint: " << done << " / " << total_init
                                  << " (" << (100 * done / total_init) << "%)"
                                  << "  [" << elapsed_ms(t0) << " ms]" << std::flush;
                    }
                }
            }
        }
        std::cout << std::endl;
        long build_hnsw_ms = elapsed_ms(t0);
        std::cout << "      HNSW + initial KNNG: " << build_hnsw_ms << " ms" << std::endl;

        std::cout << "\n[3b] NNDescent refinement (" << nn_iters << " iters)" << std::endl;
        t0 = Clock::now();
        index->refineKNNG(nn_iters, nn_sample);
        long refine_ms = elapsed_ms(t0);
        std::cout << "      Refine: " << refine_ms << " ms" << std::endl;

        if (!snapshot_path.empty()) {
            std::cout << "\n[3s] Saving snapshot to " << snapshot_path << std::endl;
            t0 = Clock::now();
            index->saveIndex(snapshot_path);
            std::cout << "      Saved (" << elapsed_ms(t0) << " ms)" << std::endl;
        }

        std::cout << "\n[3c] Building mutable RKNNG" << std::endl;
        t0 = Clock::now();
        index->buildRKNNG_mutable();
        long rknng_ms = elapsed_ms(t0);
        std::cout << "      RKNNG: " << rknng_ms << " ms" << std::endl;

        total_build_ms = build_hnsw_ms + refine_ms + rknng_ms;
    }

    std::cout << "\n      Total initial build/load: " << total_build_ms << " ms" << std::endl;

    std::vector<uint32_t> batch_orig_ids;
    batch_orig_ids.reserve(batch_n);
    for (size_t i = 0; i < batch_n; ++i) {
        batch_orig_ids.push_back(perm[split_n + i]);
    }

    long maintenance_ms = 0;
    double per_insert_ms = 0.0;

    std::vector<long> per_insert_us;
    std::cout << "\n[4] Inserting " << batch_n << " points incrementally" << std::endl;
    per_insert_us.assign(batch_n, 0);
    t0 = Clock::now();
    {
        std::atomic<int> insert_progress{0};
        int total_insert = static_cast<int>(batch_n);
        int insert_report = std::max(1, std::max(1, total_insert) / 20);
#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < total_insert; ++i) {
            uint32_t orig_id = batch_orig_ids[i];
            auto ts = Clock::now();
            index->insertPointDynamic(
                base.data.data() + static_cast<size_t>(orig_id) * base.d,
                static_cast<hnswlib::labeltype>(orig_id),
                static_cast<unsigned>(m_update),
                bidirectional_flag != 0);
            per_insert_us[i] = elapsed_us(ts);

            int done = insert_progress.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % insert_report == 0 || done == total_insert) {
#pragma omp critical
                {
                    std::cout << "\r      insert: " << done << " / " << total_insert
                              << " (" << (100 * done / std::max(1, total_insert)) << "%)"
                              << "  [" << elapsed_ms(t0) << " ms]" << std::flush;
                }
            }
        }
    }
    std::cout << std::endl;
    maintenance_ms = elapsed_ms(t0);
    per_insert_ms = batch_n > 0
        ? static_cast<double>(maintenance_ms) / static_cast<double>(batch_n)
        : 0.0;
    std::cout << "      Total insert: " << maintenance_ms << " ms" << std::endl;

    if (!per_insert_us.empty()) {
        std::vector<long> sorted_us = per_insert_us;
        std::sort(sorted_us.begin(), sorted_us.end());
        size_t n_ins = sorted_us.size();
        auto pct = [&](double p) {
            size_t idx = static_cast<size_t>(p * static_cast<double>(n_ins));
            if (idx >= n_ins) idx = n_ins - 1;
            return sorted_us[idx];
        };
        double mean_us = static_cast<double>(
            std::accumulate(sorted_us.begin(), sorted_us.end(), 0LL)) /
            static_cast<double>(n_ins);
        std::cout << "      Per-insert wall (measured, μs):"
                  << "  mean=" << std::fixed << std::setprecision(1) << mean_us
                  << "  p50=" << pct(0.50)
                  << "  p95=" << pct(0.95)
                  << "  p99=" << pct(0.99)
                  << "  max=" << sorted_us.back()
                  << std::endl;
    }

    std::cout << "      Per-insert: " << std::fixed << std::setprecision(3)
              << per_insert_ms << " ms" << std::endl;
    std::cout << "      Index size after maintenance: " << index->getCurrentElementCount()
              << std::endl;

    std::cout << "\n[5] Searching (m=" << m_proxy << ", threshold=1, ef_search="
              << ef_search << ")" << std::endl;
    t0 = Clock::now();
    index->buildRKNNG();
    std::cout << "      Flat RKNNG for search: " << elapsed_ms(t0) << " ms" << std::endl;
    EvalMetrics eval = evaluate_index(*index, query, gt, m_proxy, k_rknn, ef_search);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "      Macro Recall = " << eval.macro_recall << std::endl;
    std::cout << "      Micro Recall = " << eval.micro_recall << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "      Avg results/query = " << eval.avg_results << std::endl;
    std::cout << "      Search time = " << eval.search_ms << " ms ("
              << static_cast<double>(query.n) * 1000.0 /
                     static_cast<double>(std::max<long>(1, eval.search_ms))
              << " QPS)" << std::endl;

    bool has_oracle = false;
    EvalMetrics oracle_eval;
    if (!oracle_snapshot_path.empty()) {
        std::cout << "\n[6] Loading oracle snapshot: " << oracle_snapshot_path << std::endl;
        t0 = Clock::now();
        auto oracle = std::make_unique<HRNN>(
            &space, oracle_snapshot_path, /*nmslib=*/false, /*max_elements=*/n);
        long oracle_load_ms = elapsed_ms(t0);
        std::cout << "      Oracle loaded (" << oracle_load_ms << " ms)" << std::endl;

        t0 = Clock::now();
        oracle->buildRKNNG();
        long oracle_rknng_ms = elapsed_ms(t0);
        std::cout << "      Oracle flat RKNNG: " << oracle_rknng_ms << " ms"
                  << std::endl;

        oracle_eval = evaluate_index(*oracle, query, gt, m_proxy, k_rknn, ef_search);
        has_oracle = true;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "      Oracle Macro Recall = " << oracle_eval.macro_recall
                  << std::endl;
        std::cout << "      Oracle Micro Recall = " << oracle_eval.micro_recall
                  << std::endl;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "  Initial build/load (" << split_n << " pts): "
              << total_build_ms << " ms" << std::endl;
    std::cout << "  Maintained batch size: " << batch_n << std::endl;
    std::cout << "  Maintenance time: " << maintenance_ms << " ms" << std::endl;
    std::cout << "  Per-insert latency: " << std::fixed << std::setprecision(3)
              << per_insert_ms << " ms" << std::endl;
    std::cout << "  Macro Recall = " << std::fixed << std::setprecision(6)
              << eval.macro_recall << std::endl;
    std::cout << "  Micro Recall = " << std::fixed << std::setprecision(6)
              << eval.micro_recall << std::endl;

    if (has_oracle) {
        std::cout << "  Oracle macro gap = " << std::fixed << std::setprecision(6)
                  << (eval.macro_recall - oracle_eval.macro_recall) << std::endl;
        std::cout << "  Oracle micro gap = " << std::fixed << std::setprecision(6)
                  << (eval.micro_recall - oracle_eval.micro_recall) << std::endl;
    }

    return 0;
}
