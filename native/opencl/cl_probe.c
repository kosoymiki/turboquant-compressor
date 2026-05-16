// OpenCL fp16 + extensions probe for Adreno
// Compile: clang -o cl_probe cl_probe.c -lOpenCL
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef void* cl_platform_id;
typedef void* cl_device_id;
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VERSION 0x102F
#define CL_SUCCESS 0

typedef cl_int (*fn_getPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*fn_getDeviceIDs)(cl_platform_id, unsigned long, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*fn_getDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);

int main() {
    void *lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) { printf("ERR: %s\n", dlerror()); return 1; }

    fn_getPlatformIDs getPlatforms = dlsym(lib, "clGetPlatformIDs");
    fn_getDeviceIDs getDevices = dlsym(lib, "clGetDeviceIDs");
    fn_getDeviceInfo getInfo = dlsym(lib, "clGetDeviceInfo");

    cl_platform_id plat; cl_uint np;
    if (getPlatforms(1, &plat, &np) != CL_SUCCESS || np == 0) { printf("ERR: no platforms\n"); return 1; }

    cl_device_id dev; cl_uint nd;
    if (getDevices(plat, 0xFFFFFFFF, 1, &dev, &nd) != CL_SUCCESS || nd == 0) { printf("ERR: no devices\n"); return 1; }

    char buf[4096];
    getInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, NULL);
    printf("Device: %s\n", buf);
    getInfo(dev, CL_DEVICE_VERSION, sizeof(buf), buf, NULL);
    printf("Version: %s\n", buf);
    getInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, NULL);
    printf("Extensions: %s\n", buf);

    // Check fp16
    printf("\ncl_khr_fp16: %s\n", strstr(buf, "cl_khr_fp16") ? "YES" : "NO");
    printf("cl_khr_fp64: %s\n", strstr(buf, "cl_khr_fp64") ? "YES" : "NO");
    printf("cl_qcom_ext_host_ptr: %s\n", strstr(buf, "cl_qcom_ext_host_ptr") ? "YES" : "NO");

    dlclose(lib);
    return 0;
}
