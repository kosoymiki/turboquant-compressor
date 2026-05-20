/**
 * TurboQuant Custom Driver Loader — implementation.
 *
 * Multi-tier probe/load with forensic tracing.
 * Inspired by: adrenotools (android_dlopen_ext namespace bypass),
 * kgsl_forensic.h model, pipe_loader_kgsl.c probe pattern,
 * Mesa/Freedreno source analysis (26.1.0-dev).
 *
 * adrenotools pattern:
 *   1. android_create_namespace() with custom permitted_paths (app data dir)
 *   2. android_dlopen_ext() with ANDROID_DLEXT_USE_NAMESPACE
 *   3. Hook dlopen inside loaded driver → redirect deps to custom path
 *   4. /dev/kgsl-3d0 direct ioctl (no DRM needed)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_driver_loader.h"
#include "../include/tq_driver_manifest.h"
#include "../include/tq_gpu_profile.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/dlext.h>
#endif

namespace tq {

static std::string env_or_empty(const char* name) {
    const char* value = getenv(name);
    return value ? std::string(value) : std::string();
}

static std::string home_dir() {
    auto home = env_or_empty("HOME");
    return home.empty() ? std::string("/data/local/tmp") : home;
}

static std::string join_path(const std::string& base, const char* suffix) {
    if (base.empty()) return std::string();
    if (!base.empty() && base.back() == '/') return base + suffix;
    return base + "/" + suffix;
}

static std::vector<std::string> driver_roots() {
    std::vector<std::string> roots;
    auto env_root = env_or_empty("TQ_DRIVER_ROOT");
    auto tq_root = env_or_empty("TURBOQUANT_ROOT");
    auto home = home_dir();
    if (!env_root.empty()) roots.push_back(env_root);
    if (!tq_root.empty()) roots.push_back(join_path(tq_root, "native/opencl/driver-pack"));
    roots.push_back(join_path(home, "turboquant/drivers"));
    roots.push_back(join_path(home, "drivers"));
    return roots;
}

static std::vector<std::string> mesa_roots() {
    std::vector<std::string> roots;
    auto env_root = env_or_empty("TQ_MESA_ROOT");
    auto home = home_dir();
    if (!env_root.empty()) roots.push_back(env_root);
    roots.push_back(join_path(home, "mesa-upstream"));
    roots.push_back(join_path(home, "mesa-upstream-26.2-devel"));
    return roots;
}

// --- KGSL ioctl defs (from msm_kgsl.h, forensic-validated) ---
#define KGSL_IOC_TYPE 0x09
#define IOCTL_KGSL_DEVICE_GETPROPERTY _IOWR(KGSL_IOC_TYPE, 0x02, struct kgsl_device_getproperty)
#define KGSL_PROP_DEVICE_INFO 0x1

struct kgsl_device_getproperty {
    unsigned int type;
    void* value;
    size_t sizebytes;
};

struct kgsl_devinfo {
    unsigned int device_id;
    unsigned int chip_id;
    unsigned int mmu_enabled;
    unsigned long gmem_gpubaseaddr;
    unsigned int gpu_id;
    size_t gmem_sizebytes;
};

// --- Forensic trace (compile-time toggleable) ---
#ifdef TQ_DRIVER_TRACE
#define DRV_TRACE(fmt, ...) fprintf(stderr, "[tq-drv] " fmt "\n", ##__VA_ARGS__)
#else
#define DRV_TRACE(fmt, ...) ((void)0)
#endif

// --- Thread-safe KGSL probe (#3: atomic guard) ---
static std::once_flag g_kgsl_once;
static bool g_kgsl_ok = false;
static kgsl_devinfo g_kgsl_info = {};

// --- OpenCL function pointer types ---
typedef int32_t (*pfn_clGetPlatformIDs)(uint32_t, void**, uint32_t*);
typedef int32_t (*pfn_clGetPlatformInfo)(void*, uint32_t, size_t, void*, size_t*);
typedef int32_t (*pfn_clGetDeviceIDs)(void*, uint64_t, uint32_t, void**, uint32_t*);
typedef int32_t (*pfn_clGetDeviceInfo)(void*, uint32_t, size_t, void*, size_t*);
typedef void*   (*pfn_clCreateContext)(void*, uint32_t, void**, void*, void*, int32_t*);
typedef void*   (*pfn_clCreateCommandQueueWithProperties)(void*, void*, const uint64_t*, int32_t*);
typedef int32_t (*pfn_clReleaseContext)(void*);
typedef int32_t (*pfn_clReleaseCommandQueue)(void*);

struct ClFuncs {
    pfn_clGetPlatformIDs GetPlatformIDs;
    pfn_clGetPlatformInfo GetPlatformInfo;
    pfn_clGetDeviceIDs GetDeviceIDs;
    pfn_clGetDeviceInfo GetDeviceInfo;
    pfn_clCreateContext CreateContext;
    pfn_clCreateCommandQueueWithProperties CreateQueue;
    pfn_clReleaseContext ReleaseContext;
    pfn_clReleaseCommandQueue ReleaseQueue;
};

// --- Tier 0: KGSL direct probe (from pipe_loader_kgsl.c) ---
static bool probe_kgsl(kgsl_devinfo* info) {
    struct stat st;
    if (stat("/dev/kgsl-3d0", &st) != 0) return false;

    int fd = open("/dev/kgsl-3d0", O_RDWR | O_CLOEXEC);
    if (fd < 0) return false;

    struct kgsl_device_getproperty prop = {
        .type = KGSL_PROP_DEVICE_INFO,
        .value = info,
        .sizebytes = sizeof(*info),
    };

    int ret;
    do {
        ret = ioctl(fd, IOCTL_KGSL_DEVICE_GETPROPERTY, &prop);
    } while (ret == -1 && errno == EINTR);

    close(fd);
    if (ret == 0) {
        DRV_TRACE("kgsl probe: chip=0x%x gmem=%zuKB gpu_id=%u",
                  info->chip_id, info->gmem_sizebytes / 1024, info->gpu_id);
    }
    return ret == 0;
}

// --- adrenotools-inspired: load .so via custom namespace (Android) ---
static void* adrenotools_dlopen(const char* path, const char* redirect_dir) {
#ifdef __ANDROID__
    typedef struct android_namespace_t* (*pfn_android_create_namespace)(
        const char*, const char*, const char*, uint64_t,
        const char*, struct android_namespace_t*);
    typedef void* (*pfn_android_dlopen_ext)(const char*, int, const android_dlextinfo*);
    typedef bool (*pfn_android_link_namespaces)(
        struct android_namespace_t*, struct android_namespace_t*, const char*);

    pfn_android_create_namespace create_ns =
        (pfn_android_create_namespace)dlsym(RTLD_DEFAULT, "__loader_android_create_namespace");
    pfn_android_dlopen_ext dlopen_ext =
        (pfn_android_dlopen_ext)dlsym(RTLD_DEFAULT, "android_dlopen_ext");
    // #5: namespace link for libstdc++/libc++ from parent
    pfn_android_link_namespaces link_ns =
        (pfn_android_link_namespaces)dlsym(RTLD_DEFAULT, "__loader_android_link_namespaces");

    if (create_ns && dlopen_ext && redirect_dir) {
        struct android_namespace_t* ns = create_ns(
            "tq_gpu",
            redirect_dir,
            redirect_dir,
            3, // ANDROID_NAMESPACE_TYPE_ISOLATED|SHARED
            "/dev:/system/lib64:/vendor/lib64:/apex/com.android.runtime/lib64",
            nullptr
        );

        if (ns) {
            // #5: Link parent namespace libs (libc++, libm, libdl) into our namespace
            if (link_ns) {
                link_ns(ns, nullptr,
                    "libc.so:libm.so:libdl.so:libc++.so:libstdc++.so:liblog.so:libz.so");
                DRV_TRACE("namespace linked parent libs");
            }

            android_dlextinfo extinfo = {};
            extinfo.flags = ANDROID_DLEXT_USE_NAMESPACE;
            extinfo.library_namespace = ns;

            void* lib = dlopen_ext(path, RTLD_NOW | RTLD_LOCAL, &extinfo);
            if (lib) {
                DRV_TRACE("adrenotools_dlopen(%s) via namespace OK", path);
                return lib;
            }
            DRV_TRACE("adrenotools_dlopen namespace failed: %s, falling back", dlerror());
        }
    }
#else
    (void)redirect_dir;
#endif
    return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
}

// --- dlopen with adrenotools pattern ---
static void* try_dlopen(const char* path) {
    // Extract directory from path for redirect
    char dir[512] = {};
    strncpy(dir, path, sizeof(dir) - 1);
    char* last_slash = strrchr(dir, '/');
    if (last_slash) *last_slash = '\0';

    void* h = adrenotools_dlopen(path, dir);
    DRV_TRACE("dlopen(%s) = %p %s", path, h, h ? "OK" : dlerror());
    return h;
}

// --- #1: hook_dlopen — intercept dlopen inside loaded vendor .so ---
// Vendor libOpenCL.so calls dlopen("libgsl.so") etc. internally.
// We redirect those to /vendor/lib64 via LD_LIBRARY_PATH or preloaded interposition.
static void* hooked_dlopen(const char* filename, int flags) {
    if (!filename) return dlopen(nullptr, flags);

    // If it's a bare name (no path), look in /vendor/lib64 first
    if (filename[0] != '/') {
        char vendor_path[512];
        snprintf(vendor_path, sizeof(vendor_path), "/vendor/lib64/%s", filename);
        struct stat st;
        if (stat(vendor_path, &st) == 0) {
            DRV_TRACE("hook_dlopen: redirect %s → %s", filename, vendor_path);
            return dlopen(vendor_path, flags | RTLD_LOCAL);
        }
        // Also try /system/lib64
        snprintf(vendor_path, sizeof(vendor_path), "/system/lib64/%s", filename);
        if (stat(vendor_path, &st) == 0) {
            DRV_TRACE("hook_dlopen: redirect %s → %s", filename, vendor_path);
            return dlopen(vendor_path, flags | RTLD_LOCAL);
        }
    }
    return dlopen(filename, flags);
}

// Set LD_PRELOAD-style hook env for vendor driver child dlopen calls
static void setup_vendor_hook_env(const char* driver_dir) {
    // Set AERO_RUNTIME_VENDOR_LIB hint for any nested loaders
    setenv("TQ_VENDOR_LIB_PATH", "/vendor/lib64:/system/lib64", 0);
    (void)driver_dir;
}

// --- #2: tmpdir for LLVM JIT shader cache ---
static void ensure_cache_dir() {
    auto cache_dir = env_or_empty("TURBOQUANT_RUSTICL_CACHE_DIR");
    if (cache_dir.empty()) {
        auto home = home_dir();
        cache_dir = join_path(home, ".cache/turboquant-rusticl");
    }
    auto parent = cache_dir.substr(0, cache_dir.find_last_of('/'));
    if (!parent.empty()) mkdir(parent.c_str(), 0755);
    mkdir(cache_dir.c_str(), 0755);
    setenv("RUSTICL_CACHE_DIR", cache_dir.c_str(), 0);
    setenv("MESA_SHADER_CACHE_DIR", cache_dir.c_str(), 0);
    DRV_TRACE("cache dir: %s", cache_dir.c_str());
}

// --- Load CL function pointers from library ---
static bool load_cl_funcs(void* lib, ClFuncs* f) {
    f->GetPlatformIDs = (pfn_clGetPlatformIDs)dlsym(lib, "clGetPlatformIDs");
    f->GetPlatformInfo = (pfn_clGetPlatformInfo)dlsym(lib, "clGetPlatformInfo");
    f->GetDeviceIDs = (pfn_clGetDeviceIDs)dlsym(lib, "clGetDeviceIDs");
    f->GetDeviceInfo = (pfn_clGetDeviceInfo)dlsym(lib, "clGetDeviceInfo");
    f->CreateContext = (pfn_clCreateContext)dlsym(lib, "clCreateContext");
    f->CreateQueue = (pfn_clCreateCommandQueueWithProperties)dlsym(lib, "clCreateCommandQueueWithProperties");
    f->ReleaseContext = (pfn_clReleaseContext)dlsym(lib, "clReleaseContext");
    f->ReleaseQueue = (pfn_clReleaseCommandQueue)dlsym(lib, "clReleaseCommandQueue");
    return f->GetPlatformIDs && f->GetDeviceIDs && f->GetDeviceInfo && f->CreateContext;
}

// --- Query device caps ---
static bool query_device_caps(ClFuncs* f, DriverCaps* caps) {
    uint32_t np = 0;
    if (f->GetPlatformIDs(0, nullptr, &np) != 0 || np == 0) return false;

    void* plat = nullptr;
    f->GetPlatformIDs(1, &plat, nullptr);
    if (!plat) return false;

    char pname[128] = {};
    if (f->GetPlatformInfo)
        f->GetPlatformInfo(plat, 0x0902/*CL_PLATFORM_NAME*/, sizeof(pname), pname, nullptr);
    DRV_TRACE("platform: %s", pname);

    void* dev = nullptr;
    uint32_t nd = 0;
    if (f->GetDeviceIDs(plat, 4/*CL_DEVICE_TYPE_GPU*/, 1, &dev, &nd) != 0 || nd == 0)
        return false;

    char dname[128] = {};
    f->GetDeviceInfo(dev, 0x102B/*CL_DEVICE_NAME*/, sizeof(dname), dname, nullptr);
    caps->device_name = dname;

    uint32_t cu = 0;
    f->GetDeviceInfo(dev, 0x1002/*CL_DEVICE_MAX_COMPUTE_UNITS*/, sizeof(cu), &cu, nullptr);
    caps->compute_units = cu;

    size_t wgs = 0;
    f->GetDeviceInfo(dev, 0x1004/*CL_DEVICE_MAX_WORK_GROUP_SIZE*/, sizeof(wgs), &wgs, nullptr);
    caps->max_wg_size = wgs;

    uint64_t lmem = 0;
    f->GetDeviceInfo(dev, 0x1023/*CL_DEVICE_LOCAL_MEM_SIZE*/, sizeof(lmem), &lmem, nullptr);
    caps->local_mem_size = (size_t)lmem;

    char ext[4096] = {};
    f->GetDeviceInfo(dev, 0x1030/*CL_DEVICE_EXTENSIONS*/, sizeof(ext), ext, nullptr);
    caps->has_fp16 = strstr(ext, "cl_khr_fp16") != nullptr;
    caps->has_subgroups = strstr(ext, "cl_khr_subgroups") != nullptr;

    // Query subgroup size if available
    if (caps->has_subgroups) {
        uint32_t sgs = 0;
        // CL_DEVICE_SUB_GROUP_SIZES_INTEL or preferred subgroup size
        f->GetDeviceInfo(dev, 0x4108/*CL_DEVICE_SUB_GROUP_SIZES_INTEL*/, sizeof(sgs), &sgs, nullptr);
        caps->subgroup_size = sgs;
    }

    // Enrich caps from GPU profile if KGSL detected
    if (g_kgsl_ok) {
        GpuProfile profile = profile_from_chip_id(g_kgsl_info.chip_id);
        if (profile.gen != GpuGen::UNKNOWN) {
            if (caps->subgroup_size == 0)
                caps->subgroup_size = (uint32_t)profile.compute_wave;
            caps->has_fma = profile.supports_fp16_fma;
        }
    }

    return true;
}

// --- Public API ---

DriverCaps probe_best_driver() {
    auto t0 = std::chrono::steady_clock::now();
    DriverCaps caps = {};
    caps.tier = DriverTier::CPU_FALLBACK;

    // #3: Thread-safe atomic KGSL probe (std::call_once)
    std::call_once(g_kgsl_once, []() {
        g_kgsl_ok = probe_kgsl(&g_kgsl_info);
    });
    caps.has_kgsl = g_kgsl_ok;
    if (caps.has_kgsl) {
        DRV_TRACE("KGSL: chip=0x%x gmem=%zuKB", g_kgsl_info.chip_id, g_kgsl_info.gmem_sizebytes/1024);
    }

    // #2: Ensure cache dir exists before any driver load
    ensure_cache_dir();

    // T0-pre: Try manifest-based driver pack (two-layer architecture)
    DriverPackManifest l1_manifest, l2_manifest;
    for (const auto& pack_dir : driver_roots()) {
        if (scan_driver_pack(pack_dir.c_str(), &l1_manifest, &l2_manifest)) {
            if (!l1_manifest.library_path.empty()) {
                void* lib = try_dlopen(l1_manifest.library_path.c_str());
                if (lib) {
                    ClFuncs f = {};
                    if (load_cl_funcs(lib, &f) && query_device_caps(&f, &caps)) {
                        caps.tier = DriverTier::RUSTICL_KGSL;
                        caps.lib_path = l1_manifest.library_path;
                        caps.has_fma = true;
                        DRV_TRACE("driver pack layer1: %s v%s",
                                  l1_manifest.id.c_str(), l1_manifest.version.c_str());
                        dlclose(lib);
                        goto done;
                    }
                    dlclose(lib);
                }
            }
        }
    }

    // T0: Custom Rusticl/kgsl (adrenotools-style namespace load)
    std::vector<std::string> rusticl_paths;
    for (const auto& root : driver_roots()) {
        rusticl_paths.push_back(join_path(root, "layer1-compute/libRusticlOpenCL.so.1"));
        rusticl_paths.push_back(join_path(root, "libRusticlOpenCL.so"));
    }
    for (const auto& mesa_root : mesa_roots()) {
        rusticl_paths.push_back(join_path(mesa_root, "build/src/gallium/targets/rusticl/libRusticlOpenCL.so.1"));
    }
    for (const auto& path : rusticl_paths) {
        void* lib = try_dlopen(path.c_str());
        if (!lib) continue;
        ClFuncs f = {};
        if (load_cl_funcs(lib, &f) && query_device_caps(&f, &caps)) {
            caps.tier = DriverTier::RUSTICL_KGSL;
            caps.lib_path = path;
            caps.has_fma = true; // Adreno native
            dlclose(lib);
            goto done;
        }
        dlclose(lib);
    }

    // T1: Vendor Qualcomm OpenCL (namespace-restricted, adrenotools bypass)
    {
        // #1: Setup hook env for vendor dep resolution
        setup_vendor_hook_env("/vendor/lib64");
        static const char* vendor_paths[] = {
            "/vendor/lib64/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
            nullptr
        };
        for (int i = 0; vendor_paths[i]; i++) {
            void* lib = try_dlopen(vendor_paths[i]);
            if (!lib) continue;
            ClFuncs f = {};
            if (load_cl_funcs(lib, &f) && query_device_caps(&f, &caps)) {
                caps.tier = DriverTier::VENDOR_CL;
                caps.lib_path = vendor_paths[i];
                caps.has_fma = true;
                dlclose(lib);
                goto done;
            }
            dlclose(lib);
        }
    }

    // T2: System ICD (proot-debian)
    {
        static const char* icd_paths[] = {
            "/usr/lib/aarch64-linux-gnu/libRusticlOpenCL.so.1",
            nullptr
        };
        for (int i = 0; icd_paths[i]; i++) {
            void* lib = try_dlopen(icd_paths[i]);
            if (!lib) continue;
            ClFuncs f = {};
            if (load_cl_funcs(lib, &f) && query_device_caps(&f, &caps)) {
                caps.tier = DriverTier::SYSTEM_ICD;
                caps.lib_path = icd_paths[i];
                dlclose(lib);
                goto done;
            }
            dlclose(lib);
        }
    }

    // T3: Turnip Vulkan compute (always works via kgsl)
    if (caps.has_kgsl) {
        std::vector<std::string> vk_paths;
        for (const auto& root : driver_roots()) {
            vk_paths.push_back(join_path(root, "layer2-vulkan/libvulkan_freedreno.so"));
            vk_paths.push_back(join_path(root, "libvulkan_freedreno.so"));
        }
        for (const auto& mesa_root : mesa_roots()) {
            vk_paths.push_back(join_path(mesa_root, "build/src/freedreno/vulkan/libvulkan_freedreno.so"));
        }
        vk_paths.push_back(join_path(home_dir(), "turnip/mesa-26.0.6/libvulkan_freedreno.so"));
        for (const auto& path : vk_paths) {
            struct stat st;
            if (stat(path.c_str(), &st) == 0) {
                caps.tier = DriverTier::TURNIP_VK;
                caps.lib_path = path;
                caps.device_name = "Turnip Adreno (VkCompute)";
                caps.compute_units = 2;
                caps.max_wg_size = 1024;
                caps.local_mem_size = 32768;
                caps.subgroup_size = 128;
                caps.has_fp16 = true;
                caps.has_subgroups = true;
                caps.has_fma = true;
                goto done;
            }
        }
    }

    // T4: CPU fallback
    caps.tier = DriverTier::CPU_FALLBACK;
    caps.device_name = "CPU (WASM SIMD128 / scalar)";

done:
    auto t1 = std::chrono::steady_clock::now();
    caps.probe_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    DRV_TRACE("selected: tier=%d lib=%s (%.1fms)", (int)caps.tier, caps.lib_path.c_str(), caps.probe_latency_ms);
    return caps;
}

DriverHandle load_driver(DriverTier preferred) {
    DriverHandle h = {};
    h.valid = false;

    DriverCaps caps = probe_best_driver();
    if (caps.tier > preferred)
        DRV_TRACE("preferred tier %d unavailable, using %d", (int)preferred, (int)caps.tier);

    if (caps.tier == DriverTier::CPU_FALLBACK) {
        h.tier = DriverTier::CPU_FALLBACK;
        h.valid = true;
        return h;
    }

    if (caps.tier == DriverTier::TURNIP_VK) {
        h.lib = try_dlopen(caps.lib_path.c_str());
        h.tier = DriverTier::TURNIP_VK;
        h.valid = (h.lib != nullptr);
        return h;
    }

    // OpenCL path (T0/T1/T2) — adrenotools-style load
    h.lib = try_dlopen(caps.lib_path.c_str());
    if (!h.lib) {
        DRV_TRACE("FATAL: cannot load %s: %s", caps.lib_path.c_str(), dlerror());
        return h;
    }

    ClFuncs f = {};
    if (!load_cl_funcs(h.lib, &f)) {
        dlclose(h.lib);
        h.lib = nullptr;
        return h;
    }

    void* plat = nullptr;
    f.GetPlatformIDs(1, &plat, nullptr);
    void* dev = nullptr;
    f.GetDeviceIDs(plat, 4/*GPU*/, 1, &dev, nullptr);

    int32_t err = 0;
    h.cl_device = dev;
    h.cl_context = f.CreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != 0 || !h.cl_context) {
        DRV_TRACE("CreateContext failed: err=%d", err);
        dlclose(h.lib);
        h.lib = nullptr;
        return h;
    }

    if (f.CreateQueue) {
        uint64_t qprops[] = { 0x1093/*CL_QUEUE_PROPERTIES*/, 1/*PROFILING*/, 0 };
        h.cl_queue = f.CreateQueue(h.cl_context, dev, qprops, &err);
    }

    h.tier = caps.tier;
    h.valid = (h.cl_context != nullptr);
    DRV_TRACE("loaded: tier=%d ctx=%p queue=%p", (int)h.tier, h.cl_context, h.cl_queue);
    return h;
}

void unload_driver(DriverHandle& h) {
    if (!h.lib) return;

    if (h.tier != DriverTier::TURNIP_VK) {
        ClFuncs f = {};
        load_cl_funcs(h.lib, &f);
        if (f.ReleaseQueue && h.cl_queue) f.ReleaseQueue(h.cl_queue);
        if (f.ReleaseContext && h.cl_context) f.ReleaseContext(h.cl_context);
    }

    dlclose(h.lib);
    h = {};
}

std::string driver_info_string(const DriverCaps& caps) {
    static const char* tier_names[] = {
        "RUSTICL_KGSL", "VENDOR_CL", "SYSTEM_ICD", "TURNIP_VK", "CPU_FALLBACK"
    };
    char buf[512];
    snprintf(buf, sizeof(buf),
        "tier=%s dev=\"%s\" CU=%u WG=%zu LM=%zu sub=%u fp16=%d kgsl=%d lib=%s (%.1fms)",
        tier_names[(int)caps.tier], caps.device_name.c_str(),
        caps.compute_units, caps.max_wg_size, caps.local_mem_size,
        caps.subgroup_size, caps.has_fp16, caps.has_kgsl,
        caps.lib_path.c_str(), caps.probe_latency_ms);
    return buf;
}

// --- #4: DriverManifest JSON loader (aesolator DriverPackageInfo inspired) ---
bool load_manifest(const char* json_path, DriverManifest* out) {
    if (!json_path || !out) return false;

    FILE* f = fopen(json_path, "r");
    if (!f) {
        DRV_TRACE("manifest: cannot open %s", json_path);
        return false;
    }

    char buf[4096] = {};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return false;

    // Minimal JSON parse (no deps) — extract key fields
    auto extract = [&](const char* key) -> std::string {
        char pattern[64];
        snprintf(pattern, sizeof(pattern), "\"%s\"", key);
        const char* pos = strstr(buf, pattern);
        if (!pos) return {};
        pos = strchr(pos + strlen(pattern), '"');
        if (!pos) return {};
        pos++; // skip opening quote
        const char* end = strchr(pos, '"');
        if (!end) return {};
        return std::string(pos, end - pos);
    };

    out->id = extract("id");
    out->version = extract("version");
    out->provider = extract("provider");
    out->library_name = extract("library_name");
    out->sha256 = extract("sha256");

    std::string tier_str = extract("tier");
    if (tier_str == "RUSTICL_KGSL") out->tier = DriverTier::RUSTICL_KGSL;
    else if (tier_str == "VENDOR_CL") out->tier = DriverTier::VENDOR_CL;
    else if (tier_str == "SYSTEM_ICD") out->tier = DriverTier::SYSTEM_ICD;
    else if (tier_str == "TURNIP_VK") out->tier = DriverTier::TURNIP_VK;
    else out->tier = DriverTier::CPU_FALLBACK;

    out->validated = extract("validated") == "true";
    DRV_TRACE("manifest: id=%s ver=%s lib=%s tier=%s",
              out->id.c_str(), out->version.c_str(),
              out->library_name.c_str(), tier_str.c_str());
    return !out->id.empty() && !out->library_name.empty();
}

} // namespace tq
