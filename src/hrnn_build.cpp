#include "utils.h"
#include "hnswlib/hrnn.h"

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    std::string base_path, index_path;
    int M, ef_construction, K_knng, nn_iters, nn_sample, num_threads;

    po::options_description desc("HRNN Build Options");
    desc.add_options()("help,h", "Show help")

        ("base", po::value<std::string>(&base_path)->required(),
         "Path to base vectors (.fbin)")

        ("index_path", po::value<std::string>(&index_path)->required(),
         "Index path to save")

        ("M", po::value<int>(&M)->default_value(16), "HNSW M parameter")

        ("ef_construction", po::value<int>(&ef_construction)->default_value(200),
         "HNSW ef_construction parameter")

        ("K_knng", po::value<int>(&K_knng)->default_value(500),
         "KNNG depth (K nearest neighbors to store per node)")

        ("nn_iters", po::value<int>(&nn_iters)->default_value(3),
         "NNDescent refinement iterations")

        ("nn_sample", po::value<int>(&nn_sample)->default_value(20),
         "NNDescent sample size per node per iteration")

        ("num_threads", po::value<int>(&num_threads)->default_value(64),
         "Number of threads");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    std::cout << "=== HRNN Index Build ===" << std::endl;
    std::cout << "  M=" << M << "  ef_construction=" << ef_construction
              << "  K_knng=" << K_knng << std::endl;
    std::cout << "  nn_iters=" << nn_iters << "  nn_sample=" << nn_sample << std::endl;
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
#endif

    // ---- 1. Load base ----
    std::cout << "\n[1] Loading base  " << base_path << std::endl;
    auto t0 = Clock::now();
    auto base = load_fbin(base_path);
    std::cout << "      n=" << base.n << "  d=" << base.d
              << "  (" << elapsed_ms(t0) << " ms)" << std::endl;

    // ---- 2. Build HNSW + initial KNNG ----
    std::cout << "\n[2] Building HNSW + initial KNNG (addPoint saves candidates to KNNG slot) ..."
              << std::endl;
    hnswlib::L2Space space(base.d);
    auto index = std::make_unique<hnswlib::HRNN<float>>(
        &space, base.n, M, ef_construction, /*random_seed=*/100,
        /*allow_replace_deleted=*/false, /*K_knng=*/static_cast<size_t>(K_knng));

    t0 = Clock::now();
    std::atomic<int> progress{0};
    const int total = static_cast<int>(base.n);
    const int report_interval = std::max(1, total / 20);  // ~5% increments
#pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < total; ++i) {
        index->addPoint(base.data.data() + static_cast<size_t>(i) * base.d,
                        static_cast<hnswlib::labeltype>(i));
        int done = progress.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % report_interval == 0 || done == total) {
            #pragma omp critical
            {
                std::cout << "\r      HNSW addPoint: " << done << " / " << total
                          << " (" << (100 * done / total) << "%)"
                          << "  [" << elapsed_ms(t0) << " ms]" << std::flush;
            }
        }
    }
    std::cout << std::endl;
    long build_ms = elapsed_ms(t0);
    std::cout << "      HNSW + initial KNNG done  (" << build_ms << " ms)" << std::endl;

    // ---- 3. Refine KNNG ----
    std::cout << "\n[3] Refining KNNG (NNDescent local join, " << nn_iters
              << " iterations) ..." << std::endl;
    t0 = Clock::now();
    index->refineKNNG(nn_iters, nn_sample);
    std::cout << "      Refine done  (" << elapsed_ms(t0) << " ms)" << std::endl;

    // ---- 4. Build RKNNG ----
    std::cout << "\n[4] Building RKNNG (CSR transpose) ..." << std::endl;
    t0 = Clock::now();
    index->buildRKNNG();
    std::cout << "      RKNNG done  (" << elapsed_ms(t0) << " ms)" << std::endl;

    // ---- 5. Save index ----
    std::cout << "\n[5] Saving index to " << index_path << std::endl;
    t0 = Clock::now();
    index->saveIndex(index_path);
    std::cout << "      Saved  (" << elapsed_ms(t0) << " ms)" << std::endl;

    return 0;
}
