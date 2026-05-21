/**
 * TurboQuant OpenCL — runtime autotuner scaffolding.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace tq {

struct TuneJob {
    std::string kernel_name;
    std::vector<size_t> candidate_wg_sizes;
    std::vector<int> candidate_items_per_thread;
    int warmup_runs = 3;
    int bench_runs = 5;
    std::function<void(size_t, int)> configure;
    std::function<double()> measure;
};

struct TuneResult {
    size_t best_wg_size = 0;
    int best_items_per_thread = 1;
    double best_time_us = 0.0;
    std::vector<std::tuple<size_t, int, double>> all_results;
};

class AutoTuner {
public:
    TuneResult auto_tune(const TuneJob& job) const;
};

} // namespace tq
