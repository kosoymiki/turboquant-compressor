/**
 * TurboQuant CL↔VK Interop Layer — implementation.
 *
 * KGSL-based dmabuf sharing between Rusticl and Turnip.
 * Layer manifest generation for Vulkan loader discovery.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_cl_vk_interop.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <cerrno>

namespace tq {

// KGSL ioctls for dmabuf export
#define KGSL_IOC_TYPE 0x09
#define IOCTL_KGSL_GPUMEM_ALLOC_ID _IOWR(KGSL_IOC_TYPE, 0x35, struct kgsl_gpumem_alloc_id)
#define IOCTL_KGSL_GPUMEM_GET_FD   _IOWR(KGSL_IOC_TYPE, 0x3C, struct kgsl_gpumem_get_fd)
#define IOCTL_KGSL_GPUMEM_FREE_ID  _IOWR(KGSL_IOC_TYPE, 0x36, struct kgsl_gpumem_free_id)

// KGSL memory flags
#define KGSL_MEMFLAGS_GPUREADONLY  0x01000000
#define KGSL_CACHEMODE_WRITEBACK   0x0C000000
#define KGSL_MEMTYPE_OBJECTANY     0x00000000
#define KGSL_MEMALIGN_4K           (12 << 16)

struct kgsl_gpumem_alloc_id {
    unsigned int id;
    unsigned int flags;
    uint64_t size;
    unsigned int mmapsize;
    uint64_t gpuaddr;
};

struct kgsl_gpumem_get_fd {
    unsigned int id;
    int fd;
};

struct kgsl_gpumem_free_id {
    unsigned int id;
    unsigned int __pad;
};

// --- KGSL dmabuf allocation ---
static int kgsl_alloc_dmabuf(size_t size, InteropBufferType type, unsigned int* out_id) {
    int kgsl_fd = open("/dev/kgsl-3d0", O_RDWR | O_CLOEXEC);
    if (kgsl_fd < 0) return -1;

    unsigned int flags = KGSL_MEMALIGN_4K | KGSL_CACHEMODE_WRITEBACK | KGSL_MEMTYPE_OBJECTANY;
    if (type == InteropBufferType::DEVICE_LOCAL) {
        flags |= 0x00100000; // KGSL_MEMFLAGS_FORCE_GMEM hint (non-standard, best-effort)
    }

    kgsl_gpumem_alloc_id alloc = {};
    alloc.flags = flags;
    alloc.size = size;

    int ret;
    do { ret = ioctl(kgsl_fd, IOCTL_KGSL_GPUMEM_ALLOC_ID, &alloc); } while (ret == -1 && errno == EINTR);
    if (ret != 0) { close(kgsl_fd); return -1; }

    // Export as dmabuf FD
    kgsl_gpumem_get_fd get_fd = {};
    get_fd.id = alloc.id;
    do { ret = ioctl(kgsl_fd, IOCTL_KGSL_GPUMEM_GET_FD, &get_fd); } while (ret == -1 && errno == EINTR);
    if (ret != 0) {
        kgsl_gpumem_free_id free_req = {};
        free_req.id = alloc.id;
        ioctl(kgsl_fd, IOCTL_KGSL_GPUMEM_FREE_ID, &free_req);
        close(kgsl_fd);
        return -1;
    }

    if (out_id) *out_id = alloc.id;
    close(kgsl_fd);
    return get_fd.fd;
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

    unsigned int kgsl_id = 0;
    buf.dmabuf_fd = kgsl_alloc_dmabuf(size, type, &kgsl_id);
    if (buf.dmabuf_fd < 0) return buf;

    buf.valid = true;
    return buf;
}

void* import_dmabuf_to_cl(int dmabuf_fd, size_t size) {
    // This requires the loaded Rusticl to support cl_khr_external_memory
    // Implementation depends on the CL ICD being loaded.
    // Stub: actual import done via clCreateBufferWithProperties with
    // CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR
    (void)dmabuf_fd;
    (void)size;
    // Real implementation would:
    // cl_mem_properties props[] = {
    //     CL_MEM_DEVICE_HANDLE_LIST_KHR, (cl_mem_properties)device, CL_MEM_DEVICE_HANDLE_LIST_END_KHR,
    //     CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, (cl_mem_properties)dmabuf_fd,
    //     0
    // };
    // return clCreateBufferWithProperties(ctx, CL_MEM_READ_WRITE, props, size, NULL, &err);
    return nullptr;
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
