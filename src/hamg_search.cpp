#include "hnswlib/hamg.h"
#include "utils.h"

#include <cmath>
#include <iomanip>

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

namespace {

using QueryDiag = hnswlib::HAMG<float>::HAMGQueryDiag;

struct CountSummary {
    size_t count = 0;
    size_t total = 0;
    size_t min = 0;
    size_t max = 0;
    double avg = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

double percentileFromSorted(const std::vector<size_t>& values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    if (values.size() == 1) {
        return static_cast<double>(values.front());
    }

    double rank = percentile * static_cast<double>(values.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(rank));
    size_t upper = static_cast<size_t>(std::ceil(rank));
    double weight = rank - static_cast<double>(lower);
    double lower_value = static_cast<double>(values[lower]);
    double upper_value = static_cast<double>(values[upper]);
    return lower_value + (upper_value - lower_value) * weight;
}

CountSummary summarizeCounts(std::vector<size_t> values) {
    CountSummary summary;
    summary.count = values.size();
    if (values.empty()) {
        return summary;
    }

    std::sort(values.begin(), values.end());
    summary.min = values.front();
    summary.max = values.back();
    for (size_t value : values) {
        summary.total += value;
    }

    summary.avg = static_cast<double>(summary.total) / static_cast<double>(summary.count);
    summary.mean = summary.avg;

    double sq_sum = 0.0;
    for (size_t value : values) {
        double centered = static_cast<double>(value) - summary.mean;
        sq_sum += centered * centered;
    }
    summary.stddev = std::sqrt(sq_sum / static_cast<double>(summary.count));
    summary.p50 = percentileFromSorted(values, 0.50);
    summary.p90 = percentileFromSorted(values, 0.90);
    summary.p95 = percentileFromSorted(values, 0.95);
    summary.p99 = percentileFromSorted(values, 0.99);
    return summary;
}

void printCountSummary(const std::string& label, const CountSummary& summary) {
    auto old_flags = std::cout.flags();
    auto old_precision = std::cout.precision();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "      " << label << " total=" << summary.total << "  avg=" << summary.avg
              << "  mean=" << summary.mean << "  std=" << summary.stddev << "  min=" << summary.min
              << "  p50=" << summary.p50 << "  p90=" << summary.p90 << "  p95=" << summary.p95
              << "  p99=" << summary.p99 << "  max=" << summary.max << std::endl;

    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

struct BreakdownTotals {
    double prepare_ms = 0.0;
    double hop_count_ms = 0.0;
    double one_hop_ms = 0.0;
    double ed_ms = 0.0;
    double ps1_ms = 0.0;
    double ps2_ms = 0.0;
    double verification_ms = 0.0;
    double total_ms = 0.0;
};

BreakdownTotals accumulateBreakdown(const std::vector<QueryDiag>& diags, size_t begin_index) {
    BreakdownTotals totals;
    if (begin_index >= diags.size()) {
        return totals;
    }

    for (size_t i = begin_index; i < diags.size(); ++i) {
        totals.prepare_ms += diags[i].time_prepare_ms;
        totals.hop_count_ms += diags[i].time_hop_count_ms;
        totals.one_hop_ms += diags[i].time_one_hop_ms;
        totals.ed_ms += diags[i].time_ed_ms;
        totals.ps1_ms += diags[i].time_ps1_ms;
        totals.ps2_ms += diags[i].time_ps2_ms;
        totals.verification_ms += diags[i].time_verification_ms;
        totals.total_ms += diags[i].time_total_ms;
    }
    return totals;
}

void printBreakdownLine(const std::string& label, double avg_ms, double avg_total_ms) {
    double pct = avg_total_ms > 0.0 ? (avg_ms / avg_total_ms) * 100.0 : 0.0;
    std::cout << "        " << std::left << std::setw(16) << label << std::right << std::fixed
              << std::setprecision(3) << avg_ms << " ms/query"
              << "  (" << std::setprecision(1) << pct << "%)" << std::endl;
}

void printBreakdown(const std::string& title, const BreakdownTotals& totals, size_t query_count) {
    if (query_count == 0) {
        return;
    }

    double denom = static_cast<double>(query_count);
    double avg_total_ms = totals.total_ms / denom;
    double avg_known_ms = (totals.prepare_ms + totals.hop_count_ms + totals.one_hop_ms +
                           totals.ed_ms + totals.ps1_ms + totals.ps2_ms + totals.verification_ms) /
                          denom;
    double avg_other_ms = std::max(0.0, avg_total_ms - avg_known_ms);

    auto old_flags = std::cout.flags();
    auto old_precision = std::cout.precision();

    std::cout << "      " << title << " (" << query_count << " queries)" << std::endl;
    printBreakdownLine("total", avg_total_ms, avg_total_ms);
    printBreakdownLine("prepare_model", totals.prepare_ms / denom, avg_total_ms);
    printBreakdownLine("get_hop_count", totals.hop_count_ms / denom, avg_total_ms);
    printBreakdownLine("one_hop", totals.one_hop_ms / denom, avg_total_ms);
    printBreakdownLine("compute_ed", totals.ed_ms / denom, avg_total_ms);
    printBreakdownLine("ps1_bfs", totals.ps1_ms / denom, avg_total_ms);
    printBreakdownLine("ps2_prune", totals.ps2_ms / denom, avg_total_ms);
    printBreakdownLine("verification", totals.verification_ms / denom, avg_total_ms);
    printBreakdownLine("other", avg_other_ms, avg_total_ms);

    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string query_path, gt_path, index_path, recall_str;
    int top_k, ef_search, model_E, sample_count, max_queries, force_hops;
    bool candidate_stats_only = false, print_breakdown = false, no_ps1 = false;

    po::options_description desc("HAMG Query Options");
    desc.add_options()("help,h", "Show help")

        // Required
        ("index_path", po::value<std::string>(&index_path)->required(), "Index path")(
            "query",
            po::value<std::string>(&query_path)->required(),
            "Path to query vectors (.fbin)")(
            "gt", po::value<std::string>(&gt_path)->required(), "Path to groundtruth file")

        // Optional
        ("k", po::value<int>(&top_k)->default_value(10), "Reverse-k parameter")(
            "r",
            po::value<std::string>(&recall_str)->default_value("1.0"),
            "Desired recall list, comma-separated (e.g., 0.8,0.9,1.0)")(
            "ef_search",
            po::value<int>(&ef_search)->default_value(200),
            "ef used by inner kANNS searches")(
            "model_E",
            po::value<int>(&model_E)->default_value(64),
            "Max nearest-neighbor rank E used in HAMG sampling model")(
            "sample_count",
            po::value<int>(&sample_count)->default_value(2000),
            "Number of sampled nodes for HAMG query model")(
            "candidate_stats_only",
            po::bool_switch(&candidate_stats_only),
            "Only collect ANNS verification candidate stats and skip inner verification")(
            "max_queries",
            po::value<int>(&max_queries)->default_value(0),
            "Run only the first N queries (0=all)")(
            "force_hops",
            po::value<int>(&force_hops)->default_value(0),
            "Force k' to this value, bypassing getHopCount (0=auto)")(
            "no_ps1",
            po::bool_switch(&no_ps1),
            "Disable PS1 distance pruning (ED = infinity)")(
            "print_breakdown",
            po::bool_switch(&print_breakdown),
            "Print phase-level latency breakdown");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    if (top_k <= 0 || ef_search <= 0 || model_E <= 0 || sample_count <= 0 || max_queries < 0) {
        throw std::runtime_error(
            "Invalid parameters: require k>0, ef_search>0, model_E>0, sample_count>0, "
            "max_queries>=0.");
    }

    std::vector<double> recalls;
    {
        std::stringstream ss(recall_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) {
                continue;
            }
            double r = std::stod(item);
            r = std::max(0.0, std::min(1.0, r));
            recalls.push_back(r);
        }
    }
    if (recalls.empty()) {
        recalls.push_back(1.0);
    }

    std::cout << "=== HAMG Query (Algorithm 1/2 + PS1/PS2) ===" << std::endl;
    std::cout << "  k=" << top_k << "  ef_search=" << ef_search << "  model_E=" << model_E
              << "  sample_count=" << sample_count << std::endl;
    std::cout << "  candidate_stats_only=" << candidate_stats_only
              << "  max_queries=" << max_queries << "  force_hops=" << force_hops
              << "  no_ps1=" << no_ps1 << "  print_breakdown=" << print_breakdown << std::endl;

    // ---- 1. Load queries ----
    std::cout << "[1] Loading query " << query_path << std::endl;
    auto query = load_fbin(query_path);
    std::cout << "      nq=" << query.n << "  d=" << query.d << std::endl;

    // ---- 2. Load RkNN GT ----
    std::cout << "[2] Loading RkNN groundtruth " << gt_path << std::endl;
    auto rknn_gt = load_rknn_groundtruth(gt_path);
    std::cout << "      nq=" << rknn_gt.nq << "  total_results=" << rknn_gt.total()
              << "  avg_per_query=" << (double)rknn_gt.total() / rknn_gt.nq << std::endl;
    if (rknn_gt.nq != query.n) {
        throw std::runtime_error("GT nq != query nq");
    }

    // ---- 3. Load index ----
    std::cout << "[3] Loading index from " << index_path << std::endl;
    if (!std::filesystem::exists(index_path)) {
        throw std::runtime_error("Index file does not exist: " + index_path);
    }

    auto t0 = Clock::now();
    auto space = std::make_unique<hnswlib::L2Space>(query.d);
    auto index = std::make_unique<hnswlib::HAMG<float>>(space.get(), index_path);
    index->setEf(static_cast<size_t>(ef_search));
    if (force_hops > 0) {
        index->setForcedKPrime(static_cast<size_t>(force_hops));
    }
    if (no_ps1) {
        index->setNoPS1(true);
    }
    long load_ms = elapsed_ms(t0);
    std::cout << "      Load index done (" << load_ms << " ms)" << std::endl;
    std::cout << "      Index metadata:"
              << " M=" << index->getM()
              << " M0=" << index->getM0()
              << " C=" << index->getHAMGC()
              << " dm=" << index->getHAMGDm()
              << " ef_construction=" << index->getEfConstruction()
              << " build_threads=" << index->getBuildNumThreads()
              << " seed=" << index->getBuildSeed() << std::endl;
    std::cout << "      Index base path: " << index->getBuildBasePath() << std::endl;

    size_t nq_to_run = query.n;
    if (max_queries > 0) {
        nq_to_run = std::min<size_t>(static_cast<size_t>(max_queries), query.n);
    }
    if (nq_to_run != query.n) {
        std::cout << "      Using first " << nq_to_run << " / " << query.n << " queries"
                  << std::endl;
    }

    // ---- 4. Search ----
    for (double desired_recall : recalls) {
        std::cout << "\n[4] Searching nq=" << nq_to_run << "  k=" << top_k
                  << "  desired_recall=" << desired_recall << " ..." << std::endl;

        std::vector<std::vector<hnswlib::labeltype>> results;
        if (!candidate_stats_only) {
            results.resize(nq_to_run);
        }
        std::vector<size_t> anns_verification_counts(nq_to_run, 0);
        std::vector<QueryDiag> query_diags;
        if (print_breakdown) {
            query_diags.resize(nq_to_run);
        }
        t0 = Clock::now();
        // Warmup: trigger one-time model preparation on main thread
        {
            QueryDiag warmup_diag;
            index->searchRknnHAMG(query.data.data(), static_cast<size_t>(top_k), desired_recall,
                                  static_cast<size_t>(model_E),
                                  static_cast<size_t>(sample_count), &warmup_diag, false);
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t q = 0; q < nq_to_run; ++q) {
            QueryDiag diag;
            auto pq = index->searchRknnHAMG(query.data.data() + static_cast<size_t>(q) * query.d,
                                            static_cast<size_t>(top_k),
                                            desired_recall,
                                            static_cast<size_t>(model_E),
                                            static_cast<size_t>(sample_count),
                                            &diag,
                                            !candidate_stats_only);
            anns_verification_counts[q] = diag.anns_verifications;
            if (print_breakdown) {
                query_diags[q] = diag;
            }

            if (!candidate_stats_only) {
                auto& r = results[q];
                r.resize(pq.size());
                for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
                    r[i] = pq.top().second;
                    pq.pop();
                }
            }
        }
        long search_ms = elapsed_ms(t0);
        double qps = static_cast<double>(nq_to_run) * 1000.0 / static_cast<double>(search_ms);
        double avg_latency_ms =
            static_cast<double>(search_ms) / static_cast<double>(std::max<size_t>(1, nq_to_run));
        CountSummary anns_summary = summarizeCounts(anns_verification_counts);

        std::cout << "      Done (" << search_ms << " ms, " << qps << " QPS)" << std::endl;
        std::cout << "      Avg latency/query = " << avg_latency_ms << " ms" << std::endl;
        if (!candidate_stats_only) {
            // count avg result size
            double avg_result = 0;
            for (auto& r : results)
                avg_result += r.size();
            avg_result /= nq_to_run;

            if (nq_to_run == query.n) {
                double recall = compute_rknn_recall(rknn_gt, results);
                std::cout << "      RkNN Recall   = " << recall << std::endl;
            } else {
                std::cout << "      RkNN Recall   = skipped (partial query subset)" << std::endl;
            }
            std::cout << "      Avg results   = " << avg_result << std::endl;
        }
        printCountSummary("ANNS verification candidates/query", anns_summary);
        if (print_breakdown) {
            printBreakdown("Breakdown avg over all queries",
                           accumulateBreakdown(query_diags, 0),
                           query_diags.size());
            if (query_diags.size() > 1) {
                printBreakdown("Breakdown avg excluding q0 warmup",
                               accumulateBreakdown(query_diags, 1),
                               query_diags.size() - 1);
            }
        }
    }

    return 0;
}
