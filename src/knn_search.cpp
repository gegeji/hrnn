#include "utils.h"
#include <hnswlib/hnswlib.h>

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    std::string base_path, query_path, gt_path, index_path, ef_search_str;
    int top_k;
    std::vector<int> ef_searchs;

    po::options_description desc("HNSW Baseline Options");
    desc.add_options()("help,h", "Show help")

        // Required
        ("index_path", po::value<std::string>(&index_path)->required(), "Index path to save")(
            "query",
            po::value<std::string>(&query_path)->required(),
            "Path to query vectors (.fbin)")(
            "gt", po::value<std::string>(&gt_path)->required(), "Path to groundtruth file")
        // Optional
        ("ef_search",
         po::value<std::string>(&ef_search_str)->default_value("25,50,100,125,150"),
         "HNSW ef at search time")("k",
                                   po::value<int>(&top_k)->default_value(10),
                                   "Number of nearest neighbors to retrieve");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    // parse ef_searchs
    {
        std::stringstream ss(ef_search_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                int ef = std::stoi(item);
                ef_searchs.push_back(ef);
            } catch (const std::exception &e) {
                std::cerr << "Invalid ef_search value: " << item << std::endl;
                return 1;
            }
        }
    }

    std::cout << "=== HNSW Baseline Search ===" << std::endl;
    std::cout << "  k=" << top_k << std::endl;
#ifdef _OPENMP
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
#endif

    // ---- 1. Load queries ----
    std::cout << "[1] Loading query " << query_path << std::endl;
    auto query = load_fbin(query_path);
    std::cout << "      nq=" << query.n << "  d=" << query.d << std::endl;

    // --- 2. Load index ---
    std::cout << "\n[2]  loading index from " << index_path << std::endl;
    if (!std::filesystem::exists(index_path))
        throw std::runtime_error("Index file does not exist: " + index_path);

    auto t0 = Clock::now();
    auto space = std::make_unique<hnswlib::L2Space>(query.d);
    auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
    long build_ms = elapsed_ms(t0);
    std::cout << "\n  Load index Done  (" << build_ms << " ms)" << std::endl;

    // ---- 3. Search ----
    for (auto ef_search : ef_searchs) {
        std::cout << "\n[3] Searching nq=" << query.n << "  k=" << top_k << "  ef=" << ef_search
                  << " ..." << std::endl;
        index->setEf(ef_search);

        std::vector<std::vector<hnswlib::labeltype>> results(query.n);
        t0 = Clock::now();

#pragma omp parallel for schedule(dynamic, 32)
        for (int q = 0; q < static_cast<int>(query.n); ++q) {
            auto pq = index->searchKnn(query.data.data() + static_cast<size_t>(q) * query.d, top_k);
            auto &r = results[q];
            r.resize(pq.size());
            // pq is a max-heap: pop gives largest distance first → store reversed
            for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
                r[i] = pq.top().second;
                pq.pop();
            }
        }
        long search_ms = elapsed_ms(t0);
        double qps = static_cast<double>(query.n) * 1000.0 / search_ms;
        std::cout << "      Done  (" << search_ms << " ms  " << qps << " QPS)" << std::endl;

        // ---- 4. Recall ----
        std::cout << "\n[4] Loading groundtruth " << gt_path << std::endl;
        auto knn_gt = load_knn_groundtruth(gt_path);
        std::cout << "      nq=" << knn_gt.nq << "  k=" << knn_gt.k << std::endl;

        if (knn_gt.nq != query.n)
            throw std::runtime_error("GT nq != query nq");

        double recall = compute_knn_recall(knn_gt, results, top_k);

        std::cout << "\n=== Results ==============================" << std::endl;
        std::cout << "  Recall@" << top_k << "          = " << recall << std::endl;
        std::cout << "  Build time          = " << build_ms << " ms" << std::endl;
        std::cout << "  Search time         = " << search_ms << " ms" << std::endl;
        std::cout << "  QPS                 = " << qps << std::endl;
        std::cout << "=========================================" << std::endl;
    }
    return 0;
}
