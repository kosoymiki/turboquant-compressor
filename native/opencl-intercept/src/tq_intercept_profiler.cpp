/**
 * TurboQuant OpenCL-Intercept-Layer — Performance Profiler
 */

#include "../include/tq_intercept.h"
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

typedef struct {
    char func_name[128];
    uint64_t call_count;
    uint64_t total_duration_ns;
    uint64_t min_duration_ns;
    uint64_t max_duration_ns;
    double avg_duration_ns;
} api_stat_entry_t;

typedef struct {
    char kernel_name[128];
    uint64_t call_count;
    uint64_t total_duration_ns;
    size_t total_work_items;
} kernel_stat_entry_t;

static pthread_mutex_t g_profiler_lock = PTHREAD_MUTEX_INITIALIZER;

#define MAX_API_STATS 256
#define MAX_KERNEL_STATS 128

static api_stat_entry_t g_api_stats[MAX_API_STATS];
static uint32_t g_api_stats_count = 0;

static kernel_stat_entry_t g_kernel_stats[MAX_KERNEL_STATS];
static uint32_t g_kernel_stats_count = 0;

static uint64_t g_total_mem_ops = 0;
static uint64_t g_total_mem_bytes = 0;
static uint64_t g_total_mem_time_ns = 0;

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static api_stat_entry_t* find_or_create_api_stat(const char* func_name) {
    for (uint32_t i = 0; i < g_api_stats_count; i++) {
        if (strcmp(g_api_stats[i].func_name, func_name) == 0) {
            return &g_api_stats[i];
        }
    }
    if (g_api_stats_count < MAX_API_STATS) {
        api_stat_entry_t* entry = &g_api_stats[g_api_stats_count++];
        strncpy(entry->func_name, func_name, sizeof(entry->func_name) - 1);
        entry->call_count = 0;
        entry->total_duration_ns = 0;
        entry->min_duration_ns = 0;
        entry->max_duration_ns = 0;
        return entry;
    }
    return NULL;
}

static kernel_stat_entry_t* find_or_create_kernel_stat(const char* kernel_name) {
    for (uint32_t i = 0; i < g_kernel_stats_count; i++) {
        if (strcmp(g_kernel_stats[i].kernel_name, kernel_name) == 0) {
            return &g_kernel_stats[i];
        }
    }
    if (g_kernel_stats_count < MAX_KERNEL_STATS) {
        kernel_stat_entry_t* entry = &g_kernel_stats[g_kernel_stats_count++];
        strncpy(entry->kernel_name, kernel_name, sizeof(entry->kernel_name) - 1);
        entry->call_count = 0;
        entry->total_duration_ns = 0;
        entry->total_work_items = 0;
        return entry;
    }
    return NULL;
}

void tq_profiler_record_api_call(const char* func_name, uint64_t duration_ns) {
    pthread_mutex_lock(&g_profiler_lock);

    api_stat_entry_t* stat = find_or_create_api_stat(func_name);
    if (stat) {
        stat->call_count++;
        stat->total_duration_ns += duration_ns;
        if (stat->min_duration_ns == 0 || duration_ns < stat->min_duration_ns) {
            stat->min_duration_ns = duration_ns;
        }
        if (duration_ns > stat->max_duration_ns) {
            stat->max_duration_ns = duration_ns;
        }
        if (stat->call_count > 0) {
            stat->avg_duration_ns = (double)stat->total_duration_ns / stat->call_count;
        }
    }

    pthread_mutex_unlock(&g_profiler_lock);
}

void tq_profiler_record_kernel_exec(const char* kernel_name, uint64_t duration_ns, size_t work_items) {
    pthread_mutex_lock(&g_profiler_lock);

    kernel_stat_entry_t* stat = find_or_create_kernel_stat(kernel_name);
    if (stat) {
        stat->call_count++;
        stat->total_duration_ns += duration_ns;
        stat->total_work_items += work_items;
    }

    pthread_mutex_unlock(&g_profiler_lock);
}

void tq_profiler_record_mem_op(tq_mem_op_type_t op_type, size_t size, uint64_t duration_ns) {
    pthread_mutex_lock(&g_profiler_lock);

    g_total_mem_ops++;
    g_total_mem_bytes += size;
    g_total_mem_time_ns += duration_ns;

    pthread_mutex_unlock(&g_profiler_lock);
}

tq_perf_metrics_t tq_profiler_get_metrics() {
    tq_perf_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));

    pthread_mutex_lock(&g_profiler_lock);

    uint64_t total_api_calls = 0;
    uint64_t total_api_time = 0;
    for (uint32_t i = 0; i < g_api_stats_count; i++) {
        total_api_calls += g_api_stats[i].call_count;
        total_api_time += g_api_stats[i].total_duration_ns;
    }

    uint64_t total_kernel_execs = 0;
    uint64_t total_kernel_time = 0;
    for (uint32_t i = 0; i < g_kernel_stats_count; i++) {
        total_kernel_execs += g_kernel_stats[i].call_count;
        total_kernel_time += g_kernel_stats[i].total_duration_ns;
    }

    metrics.total_api_calls = total_api_calls;
    metrics.total_api_time_ns = total_api_time;
    metrics.total_kernel_execs = total_kernel_execs;
    metrics.total_kernel_time_ns = total_kernel_time;
    metrics.total_mem_ops = g_total_mem_ops;
    metrics.total_mem_time_ns = g_total_mem_time_ns;

    if (total_api_calls > 0) {
        metrics.avg_api_call_ns = (double)total_api_time / total_api_calls;
    }
    if (g_total_mem_ops > 0) {
        metrics.avg_mem_op_ns = (double)g_total_mem_time_ns / g_total_mem_ops;
    }
    if (total_kernel_execs > 0) {
        metrics.avg_kernel_exec_ns = (double)total_kernel_time / total_kernel_execs;
    }

    uint64_t max_api = 0;
    for (uint32_t i = 0; i < g_api_stats_count; i++) {
        if (g_api_stats[i].max_duration_ns > max_api) {
            max_api = g_api_stats[i].max_duration_ns;
        }
    }
    metrics.max_api_call_ns = max_api;

    uint64_t max_kernel = 0;
    for (uint32_t i = 0; i < g_kernel_stats_count; i++) {
        if (g_kernel_stats[i].total_duration_ns > max_kernel &&
            g_kernel_stats[i].call_count > 0) {
            uint64_t avg = g_kernel_stats[i].total_duration_ns / g_kernel_stats[i].call_count;
            if (avg > max_kernel) max_kernel = avg;
        }
    }
    metrics.max_kernel_exec_ns = max_kernel;

    pthread_mutex_unlock(&g_profiler_lock);
    return metrics;
}

void tq_profiler_print_report() {
    pthread_mutex_lock(&g_profiler_lock);

    fprintf(stderr, "\n[TQ_PROFILER] === Performance Report ===\n\n");

    fprintf(stderr, "API Calls Summary:\n");
    fprintf(stderr, "  Total calls: %llu\n", (unsigned long long)tq_profiler_get_metrics().total_api_calls);
    fprintf(stderr, "  Total time: %lluns\n", (unsigned long long)tq_profiler_get_metrics().total_api_time_ns);
    fprintf(stderr, "  Avg time: %.2fns\n\n", tq_profiler_get_metrics().avg_api_call_ns);

    fprintf(stderr, "Top API Functions:\n");
    for (uint32_t i = 0; i < g_api_stats_count && i < 10; i++) {
        fprintf(stderr, "  %-40s calls=%llu avg=%.2fns max=%lluns\n",
                g_api_stats[i].func_name,
                (unsigned long long)g_api_stats[i].call_count,
                g_api_stats[i].avg_duration_ns,
                (unsigned long long)g_api_stats[i].max_duration_ns);
    }

    fprintf(stderr, "\nKernel Executions:\n");
    fprintf(stderr, "  Total kernels: %llu\n", (unsigned long long)tq_profiler_get_metrics().total_kernel_execs);
    fprintf(stderr, "  Total time: %lluns\n", (unsigned long long)tq_profiler_get_metrics().total_kernel_time_ns);
    fprintf(stderr, "  Avg time: %.2fns\n\n", tq_profiler_get_metrics().avg_kernel_exec_ns);

    fprintf(stderr, "Top Kernels:\n");
    for (uint32_t i = 0; i < g_kernel_stats_count && i < 10; i++) {
        double avg = g_kernel_stats[i].call_count > 0 ?
            (double)g_kernel_stats[i].total_duration_ns / g_kernel_stats[i].call_count : 0;
        fprintf(stderr, "  %-40s calls=%llu avg=%.2fns items=%zu\n",
                g_kernel_stats[i].kernel_name,
                (unsigned long long)g_kernel_stats[i].call_count,
                avg,
                g_kernel_stats[i].total_work_items);
    }

    fprintf(stderr, "\nMemory Operations:\n");
    fprintf(stderr, "  Total ops: %llu\n", (unsigned long long)g_total_mem_ops);
    fprintf(stderr, "  Total bytes: %llu\n", (unsigned long long)g_total_mem_bytes);
    fprintf(stderr, "  Total time: %lluns\n", (unsigned long long)g_total_mem_time_ns);
    fprintf(stderr, "  Avg time: %.2fns\n\n", tq_profiler_get_metrics().avg_mem_op_ns);

    pthread_mutex_unlock(&g_profiler_lock);
}

void tq_profiler_reset() {
    pthread_mutex_lock(&g_profiler_lock);

    memset(g_api_stats, 0, sizeof(g_api_stats));
    g_api_stats_count = 0;

    memset(g_kernel_stats, 0, sizeof(g_kernel_stats));
    g_kernel_stats_count = 0;

    g_total_mem_ops = 0;
    g_total_mem_bytes = 0;
    g_total_mem_time_ns = 0;

    pthread_mutex_unlock(&g_profiler_lock);
}