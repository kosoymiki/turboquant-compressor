/**
 * TurboQuant OpenCL — runtime autotuner scaffolding.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_autotuner.h"

#include <limits>

namespace tq {

TuneResult AutoTuner::auto_tune(const TuneJob& job) const {
    TuneResult result;
    result.best_time_us = std::numeric_limits<double>::max();

    for (size_t wg : job.candidate_wg_sizes) {
        for (int items : job.candidate_items_per_thread) {
            if (job.configure) job.configure(wg, items);
            for (int i = 0; i < job.warmup_runs; ++i) {
                if (job.measure) (void)job.measure();
            }

            double total_us = 0.0;
            for (int i = 0; i < job.bench_runs; ++i) {
                total_us += job.measure ? job.measure() : 0.0;
            }
            double avg_us = job.bench_runs > 0 ? total_us / job.bench_runs : 0.0;
            result.all_results.emplace_back(wg, items, avg_us);
            if (avg_us > 0.0 && avg_us < result.best_time_us) {
                result.best_time_us = avg_us;
                result.best_wg_size = wg;
                result.best_items_per_thread = items;
            }
        }
    }

    if (result.best_time_us == std::numeric_limits<double>::max()) {
        result.best_time_us = 0.0;
    }
    return result;
}

} // namespace tq
