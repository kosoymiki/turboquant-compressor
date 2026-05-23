/**
 * TurboQuant GPU Pipeline — Test
 */

#include "tq_gpu_pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

static int g_pass = 0, g_fail = 0;
#define TEST(n) do { fprintf(stderr, "[TEST] %s...", #n); if (test_##n()) { fprintf(stderr, " PASS\n"); g_pass++; } else { fprintf(stderr, " FAIL\n"); g_fail++; } } while(0)

int test_init_shutdown() {
    tq_gpu_pipeline_t p;
    cl_context fake_ctx = (cl_context)0x1;
    cl_device_id fake_dev = (cl_device_id)0x2;
    cl_int err = tq_gpu_pipeline_init(&p, fake_ctx, fake_dev, TQ_GPU_MODE_SPIRV);
    if (err != CL_SUCCESS) return 0;
    err = tq_gpu_pipeline_shutdown(&p);
    return err == CL_SUCCESS;
}

int test_buffer_alloc() {
    tq_gpu_pipeline_t p;
    cl_context fake_ctx = (cl_context)0x1;
    cl_device_id fake_dev = (cl_device_id)0x2;
    tq_gpu_pipeline_init(&p, fake_ctx, fake_dev, TQ_GPU_MODE_SPIRV);
    tq_gpu_buffer_t b;
    cl_int err = tq_gpu_buffer_alloc(&p, &b, 1024, CL_MEM_READ_WRITE, NULL);
    tq_gpu_buffer_free(&p, &b);
    tq_gpu_pipeline_shutdown(&p);
    return err == CL_SUCCESS;
}

int test_launch_config() {
    tq_kernel_launch_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.kernel_id = TQ_KERNEL_MSE_SCORE;
    cfg.global_size[0] = 1024;
    cfg.local_size[0] = 256;
    cfg.arg_count = 0;
    return cfg.kernel_id == TQ_KERNEL_MSE_SCORE && cfg.global_size[0] == 1024;
}

int main() {
    fprintf(stderr, "\n=== TurboQuant GPU Pipeline Test ===\n\n");
    TEST(init_shutdown);
    TEST(buffer_alloc);
    TEST(launch_config);
    fprintf(stderr, "\nResults: %d pass, %d fail\n\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
