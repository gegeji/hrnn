#include "utils.h"

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    std::string base_path, query_path, gt_path, index_path;
    int M, ef_construction, top_k, num_threads = 64;

    po::options_description desc("HNSW Baseline Options");
    desc.add_options()("help,h", "Show help")(

        // Required
        "base",
        po::value<std::string>(&base_path)->required(),
        "Path to base vectors (.fbin)")

        ("index_path", po::value<std::string>(&index_path)->required(), "Index path to save")

        // Optional
        ("M", po::value<int>(&M)->default_value(16), "HNSW M parameter")

            ("num_threads", po::value<int>(&num_threads)->default_value(64), "number of threads")

                ("ef_construction",
                 po::value<int>(&ef_construction)->default_value(200),
                 "HNSW ef_construction parameter");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::cout << "=== HNSW Baseline Build ===" << std::endl;
    std::cout << "  M=" << M << "  ef_construction=" << ef_construction << "  k=" << top_k
              << std::endl;
#ifdef _OPENMP

    omp_set_num_threads(num_threads);
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
#endif

    // ---- 1. Load base ----
    std::cout << "\n[1] Loading base  " << base_path << std::endl;
    auto t0 = Clock::now();
    auto base = load_fbin(base_path);
    std::cout << "      n=" << base.n << "  d=" << base.d << "  (" << elapsed_ms(t0) << " ms)"
              << std::endl;

    // ---- 2. Build HNSW ----
    std::cout << "\n[2] Building HNSW index ..." << std::endl;
    hnswlib::L2Space space(base.d);
    auto index =
        std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, base.n, M, ef_construction);

    t0 = Clock::now();
#pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < static_cast<int>(base.n); ++i) {
        index->addPoint(base.data.data() + static_cast<size_t>(i) * base.d,
                        static_cast<hnswlib::labeltype>(i));
    }
    long build_ms = elapsed_ms(t0);
    std::cout << "      Done  (" << build_ms << " ms)" << std::endl;

    // ---- 3. Save HNSW ----
    index->saveIndex(index_path);
    std::cout << "\n[3] Index saved to " << index_path << std::endl;

    return 0;
}
