/**
 * TurboQuant OpenCL-Intercept-Layer — Test Suite
 */

#include "../include/tq_intercept.h"
#include <stdio.h>
#include <assert.h>

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    do { \
        fprintf(stderr, "[TEST] %s...", #name); \
        if (name()) { \
            fprintf(stderr, " PASS\n"); \
            g_tests_passed++; \
        } else { \
            fprintf(stderr, " FAIL\n"); \
            g_tests_failed++; \
        } \
    } while(0)

int test_init_shutdown() {
    tq_intercept_config_t config = {0};
    config.flags = TQ_INTERCEPT_FLAG_THREAD_SAFE;
    config.log_buffer_size = 1024 * 1024;
    config.max_mem_ops = 256;
    config.max_kernel_execs = 64;

    cl_int err = tq_intercept_init(&config);
    if (err != CL_SUCCESS) return 0;

    cl_bool active = tq_intercept_is_active();
    if (active != CL_TRUE) return 0;

    err = tq_intercept_shutdown();
    if (err != CL_SUCCESS) return 0;

    active = tq_intercept_is_active();
    return active == CL_FALSE;
}

int test_metrics() {
    tq_intercept_config_t config = {0};
    config.flags = TQ_INTERCEPT_FLAG_THREAD_SAFE | TQ_INTERCEPT_FLAG_PROFILE_PERF;

    cl_int err = tq_intercept_init(&config);
    if (err != CL_SUCCESS) return 0;

    tq_perf_metrics_t metrics;
    err = tq_intercept_get_metrics(&metrics);
    if (err != CL_SUCCESS) {
        tq_intercept_shutdown();
        return 0;
    }

    assert(metrics.total_api_calls == 0);
    assert(metrics.total_kernel_execs == 0);

    err = tq_intercept_reset_metrics();
    if (err != CL_SUCCESS) {
        tq_intercept_shutdown();
        return 0;
    }

    tq_intercept_shutdown();
    return 1;
}

int test_context_tracking() {
    tq_intercept_config_t config = {0};
    config.flags = TQ_INTERCEPT_FLAG_THREAD_SAFE;

    cl_int err = tq_intercept_init(&config);
    if (err != CL_SUCCESS) return 0;

    cl_device_id fake_device = (cl_device_id)0x12345678;
    tq_intercept_context_t* ctx = tq_intercept_get_context(fake_device);
    if (!ctx) {
        tq_intercept_shutdown();
        return 0;
    }

    tq_intercept_shutdown();
    return 1;
}

int test_memory_alloc_tracking() {
    tq_intercept_config_t config = {0};
    config.flags = TQ_INTERCEPT_FLAG_THREAD_SAFE | TQ_INTERCEPT_FLAG_LOG_MEMORY;

    cl_int err = tq_intercept_init(&config);
    if (err != CL_SUCCESS) return 0;

    tq_mem_alloc_info_t allocs[16];
    uint32_t count = 0;

    err = tq_intercept_get_mem_allocations(allocs, &count, 16);
    if (err != CL_SUCCESS) {
        tq_intercept_shutdown();
        return 0;
    }

    tq_intercept_shutdown();
    return 1;
}

int test_layer_chain() {
    tq_intercept_config_t config = {0};
    config.flags = TQ_INTERCEPT_FLAG_THREAD_SAFE | TQ_INTERCEPT_FLAG_CHAIN_LAYERS;

    cl_int err = tq_intercept_init(&config);
    if (err != CL_SUCCESS) return 0;

    tq_intercept_layer_t layer = {0};
    layer.name = "test_layer";

    err = tq_intercept_register_layer(&layer);
    if (err != CL_SUCCESS) {
        tq_intercept_shutdown();
        return 0;
    }

    err = tq_intercept_unregister_layer("test_layer");
    if (err != CL_SUCCESS) {
        tq_intercept_shutdown();
        return 0;
    }

    tq_intercept_shutdown();
    return 1;
}

int test_flush_logs() {
    tq_intercept_config_t config = {0};
    config.flags = TQ_INTERCEPT_FLAG_THREAD_SAFE;

    cl_int err = tq_intercept_init(&config);
    if (err != CL_SUCCESS) return 0;

    err = tq_intercept_flush_logs();
    if (err != CL_SUCCESS) {
        tq_intercept_shutdown();
        return 0;
    }

    tq_intercept_shutdown();
    return 1;
}

int test_version_string() {
    const char* ver = tq_intercept_version_string();
    if (!ver) return 0;
    if (strlen(ver) == 0) return 0;
    fprintf(stderr, "  (version: %s)", ver);
    return 1;
}

int main() {
    fprintf(stderr, "\n=== TurboQuant OpenCL-Intercept-Layer Test Suite ===\n\n");

    TEST(test_version_string);
    TEST(test_init_shutdown);
    TEST(test_metrics);
    TEST(test_context_tracking);
    TEST(test_memory_alloc_tracking);
    TEST(test_layer_chain);
    TEST(test_flush_logs);

    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "Passed: %d\n", g_tests_passed);
    fprintf(stderr, "Failed: %d\n", g_tests_failed);
    fprintf(stderr, "Total:  %d\n\n", g_tests_passed + g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}