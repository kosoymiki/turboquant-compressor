/**
 * TurboQuant CL↔VK Interop Layer — implementation.
 *
 * KGSL-based dmabuf sharing between Rusticl and Turnip.
 * Layer manifest generation for Vulkan loader discovery.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_cl_vk_interop.h"
#include "../include/tq_opencl.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <cerrno>
#include <linux/dma-heap.h>

namespace tq {

namespace {

thread_local int g_last_dmabuf_errno = 0;
thread_local const char* g_last_dmabuf_stage = "none";
thread_local int g_last_cl_import_error = 0;
thread_local const char* g_last_cl_import_path = "none";

void set_last_dmabuf_error(const char* stage, int err) {
    g_last_dmabuf_stage = stage;
    g_last_dmabuf_errno = err;
}

template <typename T>
T load_cl_extension_fn(const char* name) {
    cl_platform_id platform = get_platform();
    if (!platform) return nullptr;
    return reinterpret_cast<T>(clGetExtensionFunctionAddressForPlatform(platform, name));
}

} // namespace

// KGSL ioctls for legacy allocation fallback
#define KGSL_IOC_TYPE 0x09
#define IOCTL_KGSL_GPUMEM_ALLOC_ID _IOWR(KGSL_IOC_TYPE, 0x34, struct kgsl_gpumem_alloc_id)
#define IOCTL_KGSL_GPUMEM_FREE_ID  _IOWR(KGSL_IOC_TYPE, 0x36, struct kgsl_gpumem_free_id)

// KGSL memory flags
#define KGSL_MEMFLAGS_GPUREADONLY  0x01000000
#define KGSL_MEMFLAGS_USE_CPU_MAP  0x10000000

struct kgsl_gpumem_alloc_id {
    unsigned int id;
    unsigned int flags;
    uint64_t size;
    unsigned int mmapsize;
    uint64_t gpuaddr;
};

struct kgsl_gpumem_free_id {
    unsigned int id;
    unsigned int __pad;
};

static int alloc_dmabuf_via_dma_heap(size_t size) {
    int dma_heap = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (dma_heap < 0) {
        set_last_dmabuf_error("open_dma_heap", errno);
        return -1;
    }

    struct dma_heap_allocation_data alloc_data = {
        .len = size,
        .fd = 0,
        .fd_flags = static_cast<__u32>(O_RDWR | O_CLOEXEC),
        .heap_flags = 0,
    };

    int ret;
    do { ret = ioctl(dma_heap, DMA_HEAP_IOCTL_ALLOC, &alloc_data); } while (ret == -1 && errno == EINTR);
    int saved_errno = errno;
    close(dma_heap);
    if (ret != 0) {
        set_last_dmabuf_error("dma_heap_alloc", saved_errno);
        return -1;
    }

    set_last_dmabuf_error("ok", 0);
    return static_cast<int>(alloc_data.fd);
}

static int alloc_dmabuf_via_ion(size_t size) {
    struct ion_allocation_data {
        uint64_t len;
        uint32_t heap_id_mask;
        uint32_t flags;
        uint32_t fd;
        uint32_t unused;
    };

    int ion_heap = open("/dev/ion", O_RDONLY | O_CLOEXEC);
    if (ion_heap < 0) {
        set_last_dmabuf_error("open_ion", errno);
        return -1;
    }

    ion_allocation_data alloc_data = {
        .len = size,
        .heap_id_mask = (1U << 0) | (1U << 25),
        .flags = 0,
        .fd = 0,
        .unused = 0,
    };

    int ret;
    do { ret = ioctl(ion_heap, _IOWR('I', 0, struct ion_allocation_data), &alloc_data); } while (ret == -1 && errno == EINTR);
    int saved_errno = errno;
    close(ion_heap);
    if (ret != 0) {
        set_last_dmabuf_error("ion_alloc", saved_errno);
        return -1;
    }

    set_last_dmabuf_error("ok", 0);
    return static_cast<int>(alloc_data.fd);
}

static int kgsl_alloc_dmabuf(size_t size, InteropBufferType type, unsigned int* out_id) {
    (void)type;
    (void)out_id;
    int fd = alloc_dmabuf_via_dma_heap(size);
    if (fd >= 0) return fd;
    return alloc_dmabuf_via_ion(size);
}

// --- Layer manifest generation ---
std::string generate_layer_manifest(const char* library_path) {
    char buf[2048];
    snprintf(buf, sizeof(buf),
        "{\n"
        "  \"file_format_version\": \"1.0.0\",\n"
        "  \"layer\": {\n"
        "    \"name\": \"VK_LAYER_TQ_cl_vk_interop\",\n"
        "    \"type\": \"GLOBAL\",\n"
        "    \"library_path\": \"%s\",\n"
        "    \"api_version\": \"1.3.0\",\n"
        "    \"implementation_version\": \"1\",\n"
        "    \"description\": \"TurboQuantum OpenCL-Vulkan interop layer\",\n"
        "    \"disable_environment\": {\n"
        "      \"DISABLE_TQ_VK_LAYER\": \"1\"\n"
        "    },\n"
        "    \"enable_environment\": {\n"
        "      \"ENABLE_TQ_VK_LAYER\": \"1\"\n"
        "    }\n"
        "  }\n"
        "}\n",
        library_path);
    return buf;
}

bool install_interop_layer(const InteropLayerConfig& config) {
    // Create layer directory
    mkdir(config.layer_dir.c_str(), 0755);

    // Generate and write manifest
    std::string lib_path = config.layer_dir + "/libtq_vk_layer.so";
    std::string manifest = generate_layer_manifest(lib_path.c_str());

    std::string manifest_path = config.manifest_path.empty()
        ? config.layer_dir + "/VkLayer_TQ_interop.json"
        : config.manifest_path;

    FILE* f = fopen(manifest_path.c_str(), "w");
    if (!f) return false;
    fwrite(manifest.c_str(), 1, manifest.size(), f);
    fclose(f);

    // Set VK_LAYER_PATH to include our layer directory
    const char* existing = getenv("VK_LAYER_PATH");
    std::string new_path = config.layer_dir;
    if (existing && existing[0]) {
        new_path += ":";
        new_path += existing;
    }
    setenv("VK_LAYER_PATH", new_path.c_str(), 1);
    setenv("ENABLE_TQ_VK_LAYER", "1", 1);

    return true;
}

InteropBuffer alloc_interop_buffer(size_t size, InteropBufferType type) {
    InteropBuffer buf = {};
    buf.size = size;
    buf.type = type;
    set_last_dmabuf_error("alloc_begin", 0);

    unsigned int kgsl_id = 0;
    buf.dmabuf_fd = kgsl_alloc_dmabuf(size, type, &kgsl_id);
    if (buf.dmabuf_fd < 0) return buf;

    buf.valid = true;
    return buf;
}

int last_interop_dmabuf_errno() {
    return g_last_dmabuf_errno;
}

const char* last_interop_dmabuf_stage() {
    return g_last_dmabuf_stage;
}

int last_interop_cl_import_error() {
    return g_last_cl_import_error;
}

const char* last_interop_cl_import_path() {
    return g_last_cl_import_path;
}

void* import_dmabuf_to_cl(int dmabuf_fd, size_t size) {
    if (!is_initialized() || dmabuf_fd < 0 || size == 0 || !device_has_external_memory()) {
        g_last_cl_import_error = CL_INVALID_VALUE;
        g_last_cl_import_path = "precheck_rejected";
        return nullptr;
    }
#if defined(CL_MEM_DEVICE_HANDLE_LIST_KHR) && defined(CL_MEM_DEVICE_HANDLE_LIST_END_KHR) && defined(CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR)
    cl_context ctx = get_context();
    cl_device_id dev = get_device();
    if (!ctx || !dev) return nullptr;
    struct Attempt {
        const char* label;
        const cl_mem_properties* props;
    };
    cl_mem_properties props_dma_with_dev[] = {
        CL_MEM_DEVICE_HANDLE_LIST_KHR,
        reinterpret_cast<cl_mem_properties>(dev),
        CL_MEM_DEVICE_HANDLE_LIST_END_KHR,
        CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
        static_cast<cl_mem_properties>(dmabuf_fd),
        0
    };
    cl_mem_properties props_dma_no_dev[] = {
        CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
        static_cast<cl_mem_properties>(dmabuf_fd),
        0
    };
#if defined(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR)
    cl_mem_properties props_opaque_with_dev[] = {
        CL_MEM_DEVICE_HANDLE_LIST_KHR,
        reinterpret_cast<cl_mem_properties>(dev),
        CL_MEM_DEVICE_HANDLE_LIST_END_KHR,
        CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR,
        static_cast<cl_mem_properties>(dmabuf_fd),
        0
    };
    cl_mem_properties props_opaque_no_dev[] = {
        CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR,
        static_cast<cl_mem_properties>(dmabuf_fd),
        0
    };
#endif

    const Attempt attempts[] = {
        {"dma_buf+device_list", props_dma_with_dev},
        {"dma_buf", props_dma_no_dev},
#if defined(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR)
        {"opaque_fd+device_list", props_opaque_with_dev},
        {"opaque_fd", props_opaque_no_dev},
#endif
    };

    cl_int last_err = CL_INVALID_OPERATION;
    for (const Attempt& attempt : attempts) {
        cl_int err = CL_SUCCESS;
        cl_mem mem = clCreateBufferWithProperties(
            ctx, attempt.props, CL_MEM_READ_WRITE, size, nullptr, &err);
        g_last_cl_import_path = attempt.label;
        if (err == CL_SUCCESS && mem) {
            g_last_cl_import_error = CL_SUCCESS;
            return mem;
        }
        last_err = err;
    }
    g_last_cl_import_error = last_err;
    return nullptr;
#else
    (void)dmabuf_fd;
    (void)size;
    g_last_cl_import_error = CL_INVALID_OPERATION;
    g_last_cl_import_path = "compile_time_missing_external_memory";
    return nullptr;
#endif
}

bool import_dmabuf_to_vk(int dmabuf_fd, size_t size, void** out_buffer, void** out_memory) {
    // Requires VK_EXT_external_memory_dma_buf + VK_KHR_external_memory_fd
    // Stub: actual import done via VkImportMemoryFdInfoKHR
    (void)dmabuf_fd;
    (void)size;
    (void)out_buffer;
    (void)out_memory;
    return false;
}

void* import_sync_fd_semaphore_to_cl(int sync_fd) {
    if (!is_initialized() || sync_fd < 0 || !device_has_external_semaphore()) {
        return nullptr;
    }
#if defined(CL_SEMAPHORE_TYPE_KHR) && defined(CL_SEMAPHORE_TYPE_BINARY_KHR) && defined(CL_SEMAPHORE_HANDLE_SYNC_FD_KHR)
    auto fn = load_cl_extension_fn<clCreateSemaphoreWithPropertiesKHR_fn>("clCreateSemaphoreWithPropertiesKHR");
    if (!fn) return nullptr;
    cl_semaphore_properties_khr props[] = {
        CL_SEMAPHORE_TYPE_KHR, CL_SEMAPHORE_TYPE_BINARY_KHR,
        CL_SEMAPHORE_HANDLE_SYNC_FD_KHR, static_cast<cl_semaphore_properties_khr>(sync_fd),
        0
    };
    cl_int err = CL_SUCCESS;
    cl_semaphore_khr sema = fn(get_context(), props, &err);
    return err == CL_SUCCESS ? sema : nullptr;
#else
    (void)sync_fd;
    return nullptr;
#endif
}

bool sync_cl_to_vk(const InteropBuffer& buf) {
    if (!buf.valid || buf.dmabuf_fd < 0) return false;
    // On KGSL: clFinish() on CL side ensures all writes are flushed.
    // The dmabuf guarantees cache coherency across GPU contexts on same device.
    // For explicit sync: use KGSL timestamp wait or DMA_BUF_IOCTL_SYNC.

    // DMA_BUF_IOCTL_SYNC for cache management
    struct dma_buf_sync {
        uint64_t flags;
    };
    #define DMA_BUF_SYNC_START (0 << 2)
    #define DMA_BUF_SYNC_END   (1 << 2)
    #define DMA_BUF_SYNC_READ  (1 << 0)
    #define DMA_BUF_SYNC_WRITE (2 << 0)
    #define DMA_BUF_SYNC_RW    (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
    #define DMA_BUF_BASE 'b'
    #define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)

    struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE };
    int ret;
    do { ret = ioctl(buf.dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync); } while (ret == -1 && errno == EINTR);
    return ret == 0;
}

bool sync_vk_to_cl(const InteropBuffer& buf) {
    if (!buf.valid || buf.dmabuf_fd < 0) return false;

    struct dma_buf_sync {
        uint64_t flags;
    };
    #define DMA_BUF_SYNC_START (0 << 2)
    #define DMA_BUF_SYNC_READ  (1 << 0)
    #define DMA_BUF_BASE 'b'
    #define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)

    struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
    int ret;
    do { ret = ioctl(buf.dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync); } while (ret == -1 && errno == EINTR);
    return ret == 0;
}

void free_interop_buffer(InteropBuffer& buf) {
    if (buf.dmabuf_fd >= 0) {
        close(buf.dmabuf_fd);
    }
    // CL/VK handles should be released by their respective APIs before this
    buf = {};
}

} // namespace tq
