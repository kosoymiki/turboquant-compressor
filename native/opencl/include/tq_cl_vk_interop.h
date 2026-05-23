/**
 * TurboQuant CL↔VK Interop Layer.
 *
 * Zero-copy buffer sharing between Rusticl (OpenCL) and Turnip (Vulkan)
 * via dmabuf file descriptors on KGSL.
 *
 * From GameNative LsfgVkManager pattern:
 *   - Implicit Vulkan layer injection via VK_LAYER_PATH + JSON manifest
 *   - Layer hooks vkAllocateMemory to export dmabuf FDs
 *   - OpenCL side imports via cl_khr_external_memory
 *
 * Two-layer driver architecture:
 *   layer1-compute (Rusticl) ←→ dmabuf FD ←→ layer2-vulkan (Turnip)
 *
 * Use cases:
 *   - Attention output → Vulkan post-processing (upscale/quantize)
 *   - Vulkan decode → OpenCL compute (inference pipeline)
 *   - Shared weight buffers (load once, use from both APIs)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace tq {

enum class InteropBufferType : uint8_t {
    DEVICE_LOCAL = 0,    // GPU-only, fastest
    HOST_VISIBLE = 1,    // CPU-mappable, for staging
    HOST_COHERENT = 2,   // No explicit flush needed
};

struct InteropBuffer {
    int dmabuf_fd = -1;          // DMA-BUF file descriptor (shared handle)
    size_t size = 0;             // Buffer size in bytes
    InteropBufferType type = InteropBufferType::DEVICE_LOCAL;

    // OpenCL side
    void* cl_mem = nullptr;      // cl_mem handle (from clCreateBufferFromDmaBuf or external_memory)

    // Vulkan side
    void* vk_buffer = nullptr;   // VkBuffer handle
    void* vk_memory = nullptr;   // VkDeviceMemory handle

    bool valid = false;
};

struct InteropLayerConfig {
    std::string layer_dir;       // Where libtq_vk_layer.so lives
    std::string manifest_path;   // VkLayer_TQ_interop.json path
    bool enable_validation = false;
    bool enable_timeline_semaphores = false;  // For async CL↔VK sync
};

/**
 * Install the TQ interop Vulkan layer manifest.
 * Creates JSON manifest pointing to libtq_vk_layer.so.
 * Sets VK_LAYER_PATH env var for discovery.
 */
bool install_interop_layer(const InteropLayerConfig& config);

/**
 * Allocate a shared buffer accessible from both CL and VK.
 * Uses KGSL GPUMEM_ALLOC_ID → dmabuf export → import on both sides.
 */
InteropBuffer alloc_interop_buffer(size_t size, InteropBufferType type);
int last_interop_dmabuf_errno();
const char* last_interop_dmabuf_stage();

/**
 * Import an existing dmabuf FD into OpenCL as cl_mem.
 * Requires cl_khr_external_memory or KGSL-specific import.
 */
void* import_dmabuf_to_cl(int dmabuf_fd, size_t size);
int last_interop_cl_import_error();
const char* last_interop_cl_import_path();

/**
 * Import an existing dmabuf FD into Vulkan as VkBuffer+VkDeviceMemory.
 * Requires VK_EXT_external_memory_dma_buf.
 */
bool import_dmabuf_to_vk(int dmabuf_fd, size_t size, void** out_buffer, void** out_memory);

/**
 * Create an importable OpenCL semaphore from a sync FD.
 * Requires cl_khr_external_semaphore + cl_khr_external_semaphore_sync_fd.
 */
void* import_sync_fd_semaphore_to_cl(int sync_fd);

/**
 * Synchronize CL→VK: ensure CL writes are visible to Vulkan.
 * On KGSL this is a cache flush + timestamp wait.
 */
bool sync_cl_to_vk(const InteropBuffer& buf);

/**
 * Synchronize VK→CL: ensure Vulkan writes are visible to OpenCL.
 */
bool sync_vk_to_cl(const InteropBuffer& buf);

/**
 * Release interop buffer (closes dmabuf FD, releases CL/VK handles).
 */
void free_interop_buffer(InteropBuffer& buf);

/**
 * Generate the Vulkan layer manifest JSON content.
 * Compatible with Vulkan loader implicit layer discovery.
 */
std::string generate_layer_manifest(const char* library_path);

} // namespace tq
