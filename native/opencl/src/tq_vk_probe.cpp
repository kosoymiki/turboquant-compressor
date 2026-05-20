/**
 * TurboQuant Vulkan Runtime Probe — implementation.
 *
 * Dynamic Vulkan loading via dlopen (no link-time dependency).
 * Probes device properties, extensions, subgroup features.
 * Detects Turnip by driver name / device name heuristics.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_vk_probe.h"
#include <dlfcn.h>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

// Vulkan types (minimal, no vulkan.h dependency)
#define VK_NULL_HANDLE nullptr
#define VK_SUCCESS 0
#define VK_MAKE_VERSION(major, minor, patch) \
    ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | ((uint32_t)(patch)))
#define VK_VERSION_MAJOR(v) ((uint32_t)(v) >> 22)
#define VK_VERSION_MINOR(v) (((uint32_t)(v) >> 12) & 0x3FF)
#define VK_VERSION_PATCH(v) ((uint32_t)(v) & 0xFFF)
#define VK_API_VERSION_1_0 VK_MAKE_VERSION(1, 0, 0)

typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef int32_t VkResult;
typedef uint32_t VkFlags;

struct VkApplicationInfo {
    uint32_t sType;  // VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
};

struct VkInstanceCreateInfo {
    uint32_t sType;  // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};

struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    // limits and sparseProperties follow but we don't need them
    uint8_t _pad[1024];
};

struct VkExtensionProperties {
    char extensionName[256];
    uint32_t specVersion;
};

struct VkQueueFamilyProperties {
    uint32_t queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    uint32_t minImageTransferGranularity[3];
};

// Subgroup properties (Vulkan 1.1)
struct VkPhysicalDeviceSubgroupProperties {
    uint32_t sType;  // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000
    void* pNext;
    uint32_t subgroupSize;
    uint32_t supportedStages;
    uint32_t supportedOperations;
    uint32_t quadOperationsInAllStages;
};

struct VkPhysicalDeviceProperties2 {
    uint32_t sType;  // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
    void* pNext;
    VkPhysicalDeviceProperties properties;
};

// Function pointer types
typedef void* (*PFN_vkGetInstanceProcAddr)(VkInstance, const char*);
typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*);
typedef void (*PFN_vkDestroyInstance)(VkInstance, const void*);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void (*PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties*);
typedef void (*PFN_vkGetPhysicalDeviceProperties2)(VkPhysicalDevice, VkPhysicalDeviceProperties2*);
typedef VkResult (*PFN_vkEnumerateDeviceExtensionProperties)(VkPhysicalDevice, const char*, uint32_t*, VkExtensionProperties*);
typedef void (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
typedef VkResult (*PFN_vkEnumerateInstanceVersion)(uint32_t*);

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

VkProbeResult probe_vulkan(const char* driver_path) {
    auto t0 = std::chrono::steady_clock::now();
    VkProbeResult r = {};

    // Dynamic load
    const char* lib_path = driver_path ? driver_path : "libvulkan.so";
    void* vk_lib = dlopen(lib_path, RTLD_LOCAL | RTLD_NOW);
    if (!vk_lib && !driver_path) {
        // Try alternate paths
        std::vector<std::string> fallbacks;
        auto driver_root = env_or_empty("TQ_DRIVER_ROOT");
        auto tq_root = env_or_empty("TURBOQUANT_ROOT");
        auto mesa_root = env_or_empty("TQ_MESA_ROOT");
        auto home = home_dir();
        if (!driver_root.empty()) {
            fallbacks.push_back(join_path(driver_root, "layer2-vulkan/libvulkan_freedreno.so"));
            fallbacks.push_back(join_path(driver_root, "libvulkan_freedreno.so"));
        }
        if (!tq_root.empty()) {
            fallbacks.push_back(join_path(tq_root, "native/opencl/driver-pack/layer2-vulkan/libvulkan_freedreno.so"));
        }
        if (!mesa_root.empty()) {
            fallbacks.push_back(join_path(mesa_root, "build/src/freedreno/vulkan/libvulkan_freedreno.so"));
        }
        fallbacks.push_back(join_path(home, "turboquant/drivers/layer2-vulkan/libvulkan_freedreno.so"));
        fallbacks.push_back(join_path(home, "mesa-upstream/build/src/freedreno/vulkan/libvulkan_freedreno.so"));
        fallbacks.push_back(join_path(home, "mesa-upstream-26.2-devel/build/src/freedreno/vulkan/libvulkan_freedreno.so"));
        for (const auto& fallback : fallbacks) {
            vk_lib = dlopen(fallback.c_str(), RTLD_LOCAL | RTLD_NOW);
            if (vk_lib) break;
        }
    }
    if (!vk_lib) {
        auto t1 = std::chrono::steady_clock::now();
        r.probe_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return r;
    }

    auto gip = (PFN_vkGetInstanceProcAddr)dlsym(vk_lib, "vkGetInstanceProcAddr");
    auto createInstance = (PFN_vkCreateInstance)dlsym(vk_lib, "vkCreateInstance");
    if (!gip || !createInstance) {
        dlclose(vk_lib);
        auto t1 = std::chrono::steady_clock::now();
        r.probe_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return r;
    }

    // Query max instance API version
    uint32_t instanceApiVersion = VK_API_VERSION_1_0;
    auto enumInstVer = (PFN_vkEnumerateInstanceVersion)gip(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (!enumInstVer) enumInstVer = (PFN_vkEnumerateInstanceVersion)dlsym(vk_lib, "vkEnumerateInstanceVersion");
    if (enumInstVer) enumInstVer(&instanceApiVersion);

    // Create instance
    VkApplicationInfo app_info = {};
    app_info.sType = 0; // VK_STRUCTURE_TYPE_APPLICATION_INFO
    app_info.pApplicationName = "TurboQuantum";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "TQ";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = instanceApiVersion;

    VkInstanceCreateInfo ci = {};
    ci.sType = 1; // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    ci.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = createInstance(&ci, nullptr, &instance);
    if (res != VK_SUCCESS || !instance) {
        dlclose(vk_lib);
        auto t1 = std::chrono::steady_clock::now();
        r.probe_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return r;
    }

    // Get function pointers
    auto destroyInstance = (PFN_vkDestroyInstance)gip(instance, "vkDestroyInstance");
    auto enumPhysDevs = (PFN_vkEnumeratePhysicalDevices)gip(instance, "vkEnumeratePhysicalDevices");
    auto getProps = (PFN_vkGetPhysicalDeviceProperties)gip(instance, "vkGetPhysicalDeviceProperties");
    auto getProps2 = (PFN_vkGetPhysicalDeviceProperties2)gip(instance, "vkGetPhysicalDeviceProperties2");
    auto enumExtProps = (PFN_vkEnumerateDeviceExtensionProperties)gip(instance, "vkEnumerateDeviceExtensionProperties");
    auto getQueueFamilyProps = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gip(instance, "vkGetPhysicalDeviceQueueFamilyProperties");

    if (!enumPhysDevs || !getProps || !destroyInstance) {
        if (destroyInstance) destroyInstance(instance, nullptr);
        dlclose(vk_lib);
        auto t1 = std::chrono::steady_clock::now();
        r.probe_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return r;
    }

    // Enumerate physical devices
    uint32_t pdCount = 0;
    enumPhysDevs(instance, &pdCount, nullptr);
    if (pdCount == 0) {
        destroyInstance(instance, nullptr);
        dlclose(vk_lib);
        auto t1 = std::chrono::steady_clock::now();
        r.probe_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return r;
    }

    VkPhysicalDevice pd = nullptr;
    enumPhysDevs(instance, &pdCount, &pd);

    // Get device properties
    VkPhysicalDeviceProperties props = {};
    getProps(pd, &props);

    r.available = true;
    r.api_version = props.apiVersion;
    r.api_major = VK_VERSION_MAJOR(props.apiVersion);
    r.api_minor = VK_VERSION_MINOR(props.apiVersion);
    r.api_patch = VK_VERSION_PATCH(props.apiVersion);
    r.device_name = props.deviceName;
    r.vendor_id = props.vendorID;
    r.device_id = props.deviceID;
    r.driver_version = props.driverVersion;

    // Detect Turnip
    r.is_turnip = (strstr(props.deviceName, "Turnip") != nullptr ||
                   strstr(props.deviceName, "freedreno") != nullptr ||
                   (props.vendorID == 0x5143 && driver_path && strstr(driver_path, "turnip")));

    // Subgroup properties via VkPhysicalDeviceProperties2
    if (getProps2 && r.api_major >= 1 && r.api_minor >= 1) {
        VkPhysicalDeviceSubgroupProperties subgroup_props = {};
        subgroup_props.sType = 1000094000; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES

        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = 1000059001; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
        props2.pNext = &subgroup_props;

        getProps2(pd, &props2);
        r.subgroup_size = subgroup_props.subgroupSize;

        // Decode supported operations
        r.has_subgroup_basic = (subgroup_props.supportedOperations & 0x01) != 0;
        r.has_subgroup_vote = (subgroup_props.supportedOperations & 0x02) != 0;
        r.has_subgroup_arithmetic = (subgroup_props.supportedOperations & 0x04) != 0;
        r.has_subgroup_shuffle = (subgroup_props.supportedOperations & 0x10) != 0;
    }

    // Enumerate extensions
    if (enumExtProps) {
        uint32_t extCount = 0;
        enumExtProps(pd, nullptr, &extCount, nullptr);
        if (extCount > 0) {
            std::vector<VkExtensionProperties> exts(extCount);
            enumExtProps(pd, nullptr, &extCount, exts.data());
            for (uint32_t i = 0; i < extCount; i++) {
                r.extensions.push_back(exts[i].extensionName);
            }
        }
    }

    // Check key extensions
    r.has_16bit_storage = vk_has_extension(r, "VK_KHR_16bit_storage");
    r.has_8bit_storage = vk_has_extension(r, "VK_KHR_8bit_storage");
    r.has_shader_float16 = vk_has_extension(r, "VK_KHR_shader_float16_int8");
    r.has_shader_int8 = r.has_shader_float16; // same extension
    r.has_descriptor_indexing = vk_has_extension(r, "VK_EXT_descriptor_indexing");
    r.has_buffer_device_address = vk_has_extension(r, "VK_KHR_buffer_device_address");
    r.has_spirv_1_4 = vk_has_extension(r, "VK_KHR_spirv_1_4") || (r.api_minor >= 2);

    // Check for compute queue
    if (getQueueFamilyProps) {
        uint32_t qfCount = 0;
        getQueueFamilyProps(pd, &qfCount, nullptr);
        if (qfCount > 0) {
            std::vector<VkQueueFamilyProperties> qfs(qfCount);
            getQueueFamilyProps(pd, &qfCount, qfs.data());
            for (uint32_t i = 0; i < qfCount; i++) {
                if (qfs[i].queueFlags & 0x02) { // VK_QUEUE_COMPUTE_BIT
                    r.has_compute_queue = true;
                    break;
                }
            }
        }
    }

    // Cleanup
    destroyInstance(instance, nullptr);
    dlclose(vk_lib);

    auto t1 = std::chrono::steady_clock::now();
    r.probe_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

bool vk_has_extension(const VkProbeResult& r, const char* ext_name) {
    for (auto& e : r.extensions) {
        if (e == ext_name) return true;
    }
    return false;
}

} // namespace tq
