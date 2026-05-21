#include "hnswlib/hnswalg_rknn.h"
#include "utils.h"
#include <hnswlib/hnswlib.h>


int main(int argc, char* argv[]) {
    std::string base_path, query_path, gt_path, index_path, t_str, method;
    int top_k, verify_ef;
    std::vector<int> ts;

    po::options_description desc("HNSW-SFT / HNSW-RDT Baseline Options");
    desc.add_options()("help,h", "Show help")

        // Required
        ("index_path", po::value<std::string>(&index_path)->required(), "Index path to load")(
            "query",
            po::value<std::string>(&query_path)->required(),
            "Path to query vectors (.fbin)")(
            "gt", po::value<std::string>(&gt_path)->required(), "Path to groundtruth file")
        // Optional
        ("method",
         po::value<std::string>(&method)->default_value("sft"),
         "Method: sft (HNSW-SFT) or rdt (HNSW-RDT)")(
            "t",
            po::value<std::string>(&t_str)->default_value("1,2,3,4,5,6,7,8,9,10"),
            "Parameter t (controls candidate set size: k_bar = 2^t * k)")(
            "k",
            po::value<int>(&top_k)->default_value(10),
            "Number of nearest neighbors for RkNN")(
            "verify_ef",
            po::value<int>(&verify_ef)->default_value(200),
            "ef for per-candidate verification kNN search (decoupled from k_bar). "
            "Candidate generation always uses k_bar regardless. Use <=0 for the old "
            "coupled behavior ef=max(200,k_bar).");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    if (method != "sft" && method != "rdt") {
        std::cerr << "Invalid method: " << method << ". Use 'sft' or 'rdt'." << std::endl;
        return 1;
    }

    // parse t values
    {
        std::stringstream ss(t_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                int t = std::stoi(item);
                ts.push_back(t);
            } catch (const std::exception& e) {
                std::cerr << "Invalid t value: " << item << std::endl;
                return 1;
            }
        }
    }

    std::string method_label = (method == "sft") ? "HNSW-SFT" : "HNSW-RDT";
    std::cout << "=== " << method_label << " Baseline Search ===" << std::endl;
    std::cout << "  k=" << top_k << std::endl;
#ifdef _OPENMP
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
#endif

    // ---- 1. Load queries ----
    std::cout << "[1] Loading query " << query_path << std::endl;
    auto query = load_fbin(query_path);
    std::cout << "      nq=" << query.n << "  d=" << query.d << std::endl;

    // --- 2. Load index ---
    std::cout << "\n[2] Loading index from " << index_path << std::endl;
    if (!std::filesystem::exists(index_path))
        throw std::runtime_error("Index file does not exist: " + index_path);

    auto t0 = Clock::now();
    auto space = std::make_unique<hnswlib::L2Space>(query.d);
    auto index =
        std::make_unique<hnswlib::HierarchicalNSWNaiveRKNN<float>>(space.get(), index_path);
    long build_ms = elapsed_ms(t0);
    std::cout << "  Load index Done  (" << build_ms << " ms)" << std::endl;

    // ---- 3. Load groundtruth ----
    std::cout << "\n[3] Loading RkNN groundtruth " << gt_path << std::endl;
    auto rknn_gt = load_rknn_groundtruth(gt_path);
    std::cout << "      nq=" << rknn_gt.nq << "  total_results=" << rknn_gt.total()
              << "  avg_per_query=" << (double)rknn_gt.total() / rknn_gt.nq << std::endl;
    if (rknn_gt.nq != query.n)
        throw std::runtime_error("GT nq != query nq");

    // ---- 4. Search with varying t ----
    for (auto t : ts) {
        size_t k_bar = static_cast<size_t>(std::pow(2, t) * top_k);
        k_bar = std::min(static_cast<size_t>(index->max_elements_), k_bar);

        std::cout << "\n[4] " << method_label << "  t=" << t << "  k_bar=" << k_bar
                  << "  k=" << top_k << "  nq=" << query.n << std::endl;

        // Verification ef. searchKnn uses max(ef_, k), so candidate generation
        // (searchKnn(q,k_bar) for SFT; searchBaseLayerST(.,k_bar) for RDT) always
        // uses k_bar; only per-candidate verification searchKnn(c,k+2) is governed
        // by ef_. verify_ef<=0 reproduces the old coupled behavior ef=max(200,k_bar).
        size_t eff_ef = (verify_ef > 0) ? static_cast<size_t>(verify_ef)
                                        : std::max((size_t)200, k_bar);
        index->setEf(eff_ef);

        std::vector<std::vector<hnswlib::labeltype>> results(query.n);
        t0 = Clock::now();

#pragma omp parallel for schedule(dynamic, 32)
        for (int q = 0; q < static_cast<int>(query.n); ++q) {
            const void* query_ptr = query.data.data() + static_cast<size_t>(q) * query.d;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> pq;

            if (method == "sft") {
                pq = index->searchRknnSFT(query_ptr, top_k, k_bar);
            } else {
                pq = index->searchRknnRDT(query_ptr, top_k, t);
            }

            auto& r = results[q];
            r.resize(pq.size());
            // pq is a max-heap: pop gives largest distance first → store reversed
            for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
                r[i] = pq.top().second;
                pq.pop();
            }
        }
        long search_ms = elapsed_ms(t0);
        double qps = static_cast<double>(query.n) * 1000.0 / search_ms;

        double recall = compute_rknn_recall(rknn_gt, results);

        double avg_result = 0;
        for (auto& r : results)
            avg_result += r.size();
        avg_result /= query.n;

        std::cout << "=== Results (" << method_label << ", t=" << t << ") ===" << std::endl;
        std::cout << "  verify_ef           = " << eff_ef << std::endl;
        std::cout << "  RkNN Recall         = " << recall << std::endl;
        std::cout << "  Avg results         = " << avg_result << std::endl;
        std::cout << "  Search time         = " << search_ms << " ms" << std::endl;
        std::cout << "  QPS                 = " << qps << std::endl;
        std::cout << "=========================================" << std::endl;
    }
    return 0;
}
