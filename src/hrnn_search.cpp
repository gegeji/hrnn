#include "utils.h"
#include "hnswlib/hrnn.h"

#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Statistics helpers (same as rknn_relaxed_proxy.cpp)
// ---------------------------------------------------------------------------

struct LatencyStats {
    double mean_ms = 0;
    double p50_ms = 0;
    double p95_ms = 0;
    double p99_ms = 0;
    double max_ms = 0;
};

LatencyStats compute_latency_stats(std::vector<double>& times_ms) {
    LatencyStats s;
    if (times_ms.empty()) return s;
    std::sort(times_ms.begin(), times_ms.end());
    double total = 0;
    for (double t : times_ms) total += t;
    s.mean_ms = total / static_cast<double>(times_ms.size());
    auto pct = [&](double p) -> double {
        double rank = p * static_cast<double>(times_ms.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(rank));
        size_t hi = static_cast<size_t>(std::ceil(rank));
        double w = rank - static_cast<double>(lo);
        return times_ms[lo] + (times_ms[hi] - times_ms[lo]) * w;
    };
    s.p50_ms = pct(0.50);
    s.p95_ms = pct(0.95);
    s.p99_ms = pct(0.99);
    s.max_ms = times_ms.back();
    return s;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string index_path, query_path, gt_path;
    std::string m_str, threshold_str;
    int k, K_prime, ef_search, max_queries, max_k_serve;

    po::options_description desc("HRNN Search Options");
    desc.add_options()("help,h", "Show help")

        ("index_path", po::value<std::string>(&index_path)->required(),
         "HRNN index path")

        ("query", po::value<std::string>(&query_path)->required(),
         "Query vectors (.fbin)")

        ("gt", po::value<std::string>(&gt_path)->required(),
         "RkNN groundtruth file")

        ("k", po::value<int>(&k)->default_value(10),
         "RkNN k parameter")

        ("K_prime", po::value<int>(&K_prime)->default_value(500),
         "KNNG K' truncation (use first K' reverse entries)")

        ("m", po::value<std::string>(&m_str)->default_value("1,3,5,10,20,50"),
         "Proxy counts to sweep (comma-separated)")

        ("threshold,t", po::value<std::string>(&threshold_str)->default_value("1"),
         "Binary score thresholds (comma-separated)")

        ("ef_search", po::value<int>(&ef_search)->default_value(200),
         "HNSW ef_search")

        ("max_queries", po::value<int>(&max_queries)->default_value(0),
         "0=all")

        ("max_k_serve", po::value<int>(&max_k_serve)->default_value(0),
         "Enable serve mode: compact elements, dense kdist matrix for k<=max_k_serve (0=off)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    // Parse m values
    std::vector<int> m_values;
    {
        std::stringstream ss(m_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) m_values.push_back(std::stoi(item));
        }
    }
    if (m_values.empty()) m_values.push_back(20);

    // Parse threshold values
    std::vector<float> threshold_values;
    {
        std::stringstream ss(threshold_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) threshold_values.push_back(std::stof(item));
        }
    }
    if (threshold_values.empty()) threshold_values.push_back(1.0f);

    std::cout << "=== HRNN RkNN Search ===" << std::endl;
    std::cout << "  k=" << k << "  K'=" << K_prime << "  ef_search=" << ef_search << std::endl;
    std::cout << "  m values:";
    for (int m : m_values) std::cout << " " << m;
    std::cout << std::endl;
    std::cout << "  thresholds:";
    for (float t : threshold_values) std::cout << " " << t;
    std::cout << std::endl;

    // ---- 1. Load queries ----
    std::cout << "\n[1] Loading queries " << query_path << std::endl;
    auto query = load_fbin(query_path);
    std::cout << "      nq=" << query.n << "  d=" << query.d << std::endl;

    // ---- 2. Load groundtruth ----
    std::cout << "[2] Loading RkNN GT " << gt_path << std::endl;
    auto gt = load_rknn_groundtruth(gt_path);
    std::cout << "      nq=" << gt.nq << "  total=" << gt.total() << std::endl;

    // ---- 3. Load HRNN index ----
    std::cout << "[3] Loading HRNN index " << index_path << std::endl;
    auto t0 = Clock::now();
    auto space = std::make_unique<hnswlib::L2Space>(query.d);
    auto index = std::make_unique<hnswlib::HRNN<float>>(
        space.get(), index_path);
    index->setEf(static_cast<size_t>(ef_search));
    std::cout << "      Done (" << elapsed_ms(t0) << " ms)" << std::endl;
    std::cout << "      K_knng=" << index->K_knng_
              << "  knng_built=" << index->knng_built_
              << "  rknng_built=" << index->rknng_built_ << std::endl;

    if (max_k_serve > 0) {
        std::cout << "[3b] Compacting for serve mode (max_k_serve=" << max_k_serve << ")" << std::endl;
        auto tc = Clock::now();
        index->compactForServing(static_cast<size_t>(max_k_serve));
        std::cout << "      Done (" << elapsed_ms(tc) << " ms)" << std::endl;
    }

    // ---- 4. Search sweep ----
    size_t nq = query.n;
    if (max_queries > 0)
        nq = std::min<size_t>(static_cast<size_t>(max_queries), query.n);

    using HRC = std::chrono::high_resolution_clock;
    using DurUs = std::chrono::duration<double, std::micro>;

    for (int m : m_values) {
        for (float threshold : threshold_values) {
            std::cout << "\n[4] Search: nq=" << nq << "  m=" << m
                      << "  threshold=" << threshold << std::endl;

            std::vector<std::vector<hnswlib::labeltype>> all_results(nq);
            std::vector<double> times_ms(nq);

            for (size_t q = 0; q < nq; q++) {
                const float* qvec = query.data.data() + q * query.d;

                auto ta = HRC::now();
                all_results[q] = index->searchRknn(
                    qvec, static_cast<size_t>(m), threshold,
                    static_cast<size_t>(k), static_cast<size_t>(K_prime),
                    static_cast<size_t>(ef_search));
                auto tb = HRC::now();

                times_ms[q] = std::chrono::duration_cast<DurUs>(tb - ta).count() / 1000.0;
            }

            // Macro recall (per-query average)
            if (nq == query.n) {
                double recall = compute_rknn_recall(gt, all_results);
                std::cout << "      Macro Recall = " << std::fixed << std::setprecision(6)
                          << recall << std::endl;
            }

            // Micro recall
            if (nq == query.n) {
                size_t found = 0, total_gt = 0;
                for (size_t q = 0; q < nq; q++) {
                    size_t gt_cnt = gt.count(q);
                    if (gt_cnt == 0) continue;
                    total_gt += gt_cnt;
                    const uint32_t* gt_ids = gt.ids_for(q);
                    std::unordered_set<uint32_t> gt_set(gt_ids, gt_ids + gt_cnt);
                    for (auto label : all_results[q]) {
                        if (gt_set.count(static_cast<uint32_t>(label)))
                            found++;
                    }
                }
                double micro = total_gt > 0
                    ? static_cast<double>(found) / static_cast<double>(total_gt) : 0.0;
                std::cout << "      Micro Recall = " << std::fixed << std::setprecision(6)
                          << micro << std::endl;
            }

            // Avg results per query
            {
                double avg = 0;
                for (size_t q = 0; q < nq; q++)
                    avg += static_cast<double>(all_results[q].size());
                avg /= static_cast<double>(nq);
                std::cout << "      Avg results/query = " << std::fixed << std::setprecision(2)
                          << avg << std::endl;
            }

            // Latency stats
            auto ls = compute_latency_stats(times_ms);
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "      Latency (ms): mean=" << ls.mean_ms
                      << "  p50=" << ls.p50_ms << "  p95=" << ls.p95_ms
                      << "  p99=" << ls.p99_ms << "  max=" << ls.max_ms << std::endl;

            double total_time = 0;
            for (double t : times_ms) total_time += t;
            double qps = static_cast<double>(nq) * 1000.0 / total_time;
            std::cout << "      QPS = " << std::fixed << std::setprecision(2) << qps << std::endl;
        }
    }

    return 0;
}
