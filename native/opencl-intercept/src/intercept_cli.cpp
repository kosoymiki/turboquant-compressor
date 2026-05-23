/**
 * TurboQuant OpenCL-Intercept-Layer — CLI Tool
 */

#include "../include/tq_intercept.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_version() {
    printf("TurboQuant OpenCL-Intercept-Layer v%s\n", tq_intercept_version_string());
    printf("Built with LLVM/clang-cl %s\n", __VERSION__);
}

static void print_usage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  -v, --version     Show version\n");
    printf("  -h, --help       Show this help\n");
    printf("  -i, --init       Initialize intercept layer\n");
    printf("  -s, --shutdown   Shutdown intercept layer\n");
    printf("  -m, --metrics    Show performance metrics\n");
    printf("  -d, --dump-mem   Dump memory tracking\n");
    printf("  -k, --dump-kernel Dump kernel tracking\n");
    printf("  -r, --reset      Reset all metrics\n");
    printf("  -f, --flush      Flush logs\n");
    printf("  -t, --trace      Enable call tracing\n");
    printf("  -p, --perf       Enable performance profiling\n");
    printf("  -o, --log FILE   Log file path\n");
}

static tq_intercept_config_t build_config(int trace_calls, int perf, const char* log_file) {
    tq_intercept_config_t config = {0};

    config.flags = TQ_INTERCEPT_FLAG_THREAD_SAFE;
    if (trace_calls) {
        config.flags |= TQ_INTERCEPT_FLAG_TRACE_CALLS;
        config.flags |= TQ_INTERCEPT_FLAG_TRACE_ARGS;
        config.flags |= TQ_INTERCEPT_FLAG_TRACE_RETURNS;
    }
    if (perf) {
        config.flags |= TQ_INTERCEPT_FLAG_PROFILE_PERF;
        config.flags |= TQ_INTERCEPT_FLAG_LOG_MEMORY;
        config.flags |= TQ_INTERCEPT_FLAG_LOG_KERNELS;
    }

    config.log_buffer_size = 1024 * 1024;
    config.max_mem_ops = 1024;
    config.max_kernel_execs = 256;
    config.thread_count = 4;

    if (log_file) {
        config.log_file_path = log_file;
    }

    return config;
}

static void print_metrics() {
    tq_perf_metrics_t metrics;
    cl_int err = tq_intercept_get_metrics(&metrics);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get metrics: %d\n", err);
        return;
    }

    printf("\n=== Performance Metrics ===\n\n");
    printf("API Calls:\n");
    printf("  Total: %llu\n", (unsigned long long)metrics.total_api_calls);
    printf("  Total time: %llu ns\n", (unsigned long long)metrics.total_api_time_ns);
    printf("  Avg time: %.2f ns\n", metrics.avg_api_call_ns);
    printf("  Max time: %llu ns\n\n", (unsigned long long)metrics.max_api_call_ns);

    printf("Memory Operations:\n");
    printf("  Total: %llu\n", (unsigned long long)metrics.total_mem_ops);
    printf("  Total time: %llu ns\n", (unsigned long long)metrics.total_mem_time_ns);
    printf("  Avg time: %.2f ns\n\n", metrics.avg_mem_op_ns);

    printf("Kernel Executions:\n");
    printf("  Total: %llu\n", (unsigned long long)metrics.total_kernel_execs);
    printf("  Total time: %llu ns\n", (unsigned long long)metrics.total_kernel_time_ns);
    printf("  Avg time: %.2f ns\n", metrics.avg_kernel_exec_ns);
    printf("  Max time: %llu ns\n", (unsigned long long)metrics.max_kernel_exec_ns);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    int trace_calls = 0;
    int perf = 0;
    const char* log_file = NULL;
    int init_done = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
            print_version();
            return 0;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--trace") == 0) {
            trace_calls = 1;
        }
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--perf") == 0) {
            perf = 1;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--log") == 0) {
            if (i + 1 < argc) {
                log_file = argv[++i];
            }
        }
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--init") == 0) {
            tq_intercept_config_t config = build_config(trace_calls, perf, log_file);
            cl_int err = tq_intercept_init(&config);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to init intercept: %d\n", err);
                return 1;
            }
            printf("Intercept layer initialized\n");
            init_done = 1;
        }
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--shutdown") == 0) {
            cl_int err = tq_intercept_shutdown();
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to shutdown intercept: %d\n", err);
                return 1;
            }
            printf("Intercept layer shutdown complete\n");
        }
        else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--metrics") == 0) {
            print_metrics();
        }
        else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--reset") == 0) {
            cl_int err = tq_intercept_reset_metrics();
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to reset metrics: %d\n", err);
                return 1;
            }
            printf("Metrics reset\n");
        }
        else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--flush") == 0) {
            cl_int err = tq_intercept_flush_logs();
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to flush logs: %d\n", err);
                return 1;
            }
            printf("Logs flushed\n");
        }
        else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dump-mem") == 0) {
            fprintf(stderr, "[TQ_CLI] Memory dump requested\n");
        }
        else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--dump-kernel") == 0) {
            fprintf(stderr, "[TQ_CLI] Kernel dump requested\n");
        }
    }

    return 0;
}