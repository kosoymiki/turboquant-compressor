// OpenCL kernel compile test — validates v4 fp16 kernel on real Adreno
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_program;
typedef cl_ulong cl_device_type;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_BUILD_SUCCESS 0
#define CL_BUILD_ERROR -2
#define CL_PROGRAM_BUILD_STATUS 0x1181
#define CL_PROGRAM_BUILD_LOG 0x1183

typedef cl_int (*fn_getPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*fn_getDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*fn_createContext)(void*, cl_uint, cl_device_id*, void*, void*, cl_int*);
typedef cl_program (*fn_createProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*fn_buildProgram)(cl_program, cl_uint, cl_device_id*, const char*, void*, void*);
typedef cl_int (*fn_getProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (*fn_releaseProgram)(cl_program);
typedef cl_int (*fn_releaseContext)(cl_context);

int main() {
    void *lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) { printf("ERR: %s\n", dlerror()); return 1; }

    fn_getPlatformIDs getPlatforms = dlsym(lib, "clGetPlatformIDs");
    fn_getDeviceIDs getDevices = dlsym(lib, "clGetDeviceIDs");
    fn_createContext createCtx = dlsym(lib, "clCreateContext");
    fn_createProgramWithSource createProg = dlsym(lib, "clCreateProgramWithSource");
    fn_buildProgram buildProg = dlsym(lib, "clBuildProgram");
    fn_getProgramBuildInfo getBuildInfo = dlsym(lib, "clGetProgramBuildInfo");
    fn_releaseProgram relProg = dlsym(lib, "clReleaseProgram");
    fn_releaseContext relCtx = dlsym(lib, "clReleaseContext");

    cl_platform_id plat; cl_uint np;
    getPlatforms(1, &plat, &np);
    cl_device_id dev; cl_uint nd;
    getDevices(plat, CL_DEVICE_TYPE_ALL, 1, &dev, &nd);

    cl_int err;
    cl_context ctx = createCtx(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { printf("ERR: context %d\n", err); return 1; }

    // Read kernel source
    FILE *f = fopen("/data/local/tmp/tq_fused_attention_v4_fp16.cl", "r");
    if (!f) { printf("ERR: cannot open kernel file\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *src = malloc(sz + 1);
    fread(src, 1, sz, f); src[sz] = 0; fclose(f);

    const char *srcs[] = { src };
    size_t lens[] = { (size_t)sz };
    cl_program prog = createProg(ctx, 1, srcs, lens, &err);
    if (err != CL_SUCCESS) { printf("ERR: createProgram %d\n", err); free(src); return 1; }

    // Build with fp16 + relaxed math
    cl_int bret = buildProg(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);

    cl_int bstatus;
    getBuildInfo(prog, dev, CL_PROGRAM_BUILD_STATUS, sizeof(bstatus), &bstatus, NULL);

    char log[8192] = {0};
    getBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);

    if (bret == CL_SUCCESS && bstatus == CL_BUILD_SUCCESS) {
        printf("BUILD: SUCCESS\n");
        printf("Kernel: tq_fused_attention_v4_fp16\n");
        printf("Kernel: tq_fwht_fp16\n");
        if (strlen(log) > 0) printf("Warnings: %s\n", log);
    } else {
        printf("BUILD: FAILED (ret=%d status=%d)\n", bret, bstatus);
        printf("Log:\n%s\n", log);
    }

    free(src);
    relProg(prog);
    relCtx(ctx);
    dlclose(lib);
    return bret == CL_SUCCESS ? 0 : 1;
}
