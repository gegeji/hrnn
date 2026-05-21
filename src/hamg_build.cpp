#include "hnswlib/hamg.h"
#include "utils.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    std::string base_path, index_path;
    int M, M0, C, dm, ef_construction, seed, num_threads = 64;

    po::options_description desc("HAMG Build Options");
    desc.add_options()("help,h", "Show help")(

        // Required
        "base",
        po::value<std::string>(&base_path)->required(),
        "Path to base vectors (.fbin)")

        ("index_path", po::value<std::string>(&index_path)->required(), "Index path to save")

        // Optional
        ("M", po::value<int>(&M)->default_value(16), "Upper-layer HNSW M parameter")

            ("M0", po::value<int>(&M0)->default_value(5000), "Bottom-layer max out-degree capacity")

                ("C", po::value<int>(&C)->default_value(500), "HAMG candidate neighbor size")

                    ("dm",
                     po::value<int>(&dm)->default_value(80),
                     "HAMG minimum out-degree threshold")

                        ("ef_construction",
                         po::value<int>(&ef_construction)->default_value(200),
                         "HNSW ef_construction parameter for insertion candidate search")

                            ("num_threads",
                             po::value<int>(&num_threads)->default_value(64),
                             "number of threads")

                                ("seed", po::value<int>(&seed)->default_value(100), "Random seed");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    if (M <= 0 || M0 <= 0 || C <= 0 || dm < 0 || ef_construction <= 0) {
        throw std::runtime_error(
            "Invalid parameters: require M>0, M0>0, C>0, dm>=0, ef_construction>0.");
    }

    std::cout << "=== HAMG Build (HNSW upper + HAMG bottom in hamg.h) ===" << std::endl;
    std::cout << "  M=" << M << "  M0=" << M0 << "  C=" << C << "  dm=" << dm
              << "  ef_construction=" << ef_construction << "  seed=" << seed << std::endl;
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
#endif

    // ---- 1. Load base ----
    std::cout << "\n[1] Loading base " << base_path << std::endl;
    auto t0 = Clock::now();
    auto base = load_fbin(base_path);
    std::cout << "      n=" << base.n << "  d=" << base.d << " (" << elapsed_ms(t0) << " ms)"
              << std::endl;

    // ---- 2. Build HAMG ----
    std::cout << "\n[2] Building HAMG index ..." << std::endl;
    hnswlib::L2Space space(base.d);
    auto index = std::make_unique<hnswlib::HAMG<float>>(&space,
                                                        base.n,
                                                        static_cast<size_t>(M),
                                                        static_cast<size_t>(ef_construction),
                                                        static_cast<size_t>(seed),
                                                        false,
                                                        static_cast<size_t>(M0),
                                                        static_cast<size_t>(C),
                                                        static_cast<size_t>(dm));
    index->setBuildMetadata(base_path, static_cast<size_t>(num_threads), static_cast<size_t>(seed));

    std::vector<long> per_insert_us(base.n, 0);
    t0 = Clock::now();
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < static_cast<int>(base.n); ++i) {
        auto ts = Clock::now();
        index->addPoint(base.data.data() + static_cast<size_t>(i) * base.d,
                        static_cast<hnswlib::labeltype>(i));
        per_insert_us[i] = elapsed_us(ts);

        if ((i + 1) % 100000 == 0 || i + 1 == static_cast<int>(base.n)) {
            std::cout << "      inserted " << (i + 1) << "/" << base.n << std::endl;
        }
    }
    long build_ms = elapsed_ms(t0);
    std::cout << "      Done (" << build_ms << " ms)" << std::endl;

    {
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

    // ---- Graph stats ----
    {
        double avg_deg = index->avgOutDegreeBottom();
        size_t max_deg = 0, min_deg = SIZE_MAX;
        size_t n = index->cur_element_count;
        for (size_t i = 0; i < n; i++) {
            size_t deg = index->getBottomNeighbors(static_cast<hnswlib::tableint>(i)).size();
            if (deg > max_deg) max_deg = deg;
            if (deg < min_deg) min_deg = deg;
        }
        std::cout << "      Graph stats: avg_degree=" << avg_deg << "  min_degree=" << min_deg
                  << "  max_degree=" << max_deg << std::endl;
    }

    // ---- 3. Save ----
    index->saveIndex(index_path);
    std::cout << "\n[3] Index saved to " << index_path << std::endl;

    return 0;
}
