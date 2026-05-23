/**
 * TurboQuant OpenCL — program/kernel loader with binary cache and SPIR-V IL path.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_opencl_kernel_manager.h"
#include "../include/tq_kernel_cache.h"
#include "../include/tq_opencl.h"
#include "../include/tq_repo_paths.h"
#include "../include/tq_runtime_scheduler.h"
#include "../include/tq_trace.h"

#include <CL/cl.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tq {

namespace {

std::unordered_map<std::string, cl_program> g_programs;
std::unordered_map<std::string, cl_kernel> g_kernels;
std::unordered_map<std::string, KernelMetadata> g_kernel_metadata;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

bool want_source_only() {
    const char* v = std::getenv("TQ_OPENCL_FORCE_SOURCE");
    return v && std::strcmp(v, "1") == 0;
}

bool binary_cache_disabled() {
    const char* v = std::getenv("TQ_OPENCL_DISABLE_BINARY_CACHE");
    return v && std::strcmp(v, "1") == 0;
}

unsigned async_build_wait_timeout_ms() {
    const char* raw = std::getenv("TQ_OPENCL_ASYNC_BUILD_WAIT_MS");
    if (!raw || !*raw) return 5000;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(raw, &end, 10);
    if (!end || *end != '\0' || parsed == 0) return 5000;
    if (parsed > 60000UL) return 60000;
    return static_cast<unsigned>(parsed);
}

std::vector<std::string> precompiled_binary_roots() {
    std::vector<std::string> roots;
    const char* env_root = std::getenv("TQ_OPENCL_PRECOMPILED_DIR");
    if (env_root && *env_root) roots.emplace_back(env_root);

    std::string forensics_dir = resolve_forensics_dir();
    if (!forensics_dir.empty()) {
        roots.emplace_back(join_repo_path(forensics_dir, "precompiled-kernel-library"));
    }

    for (const auto& root : runtime_pack_roots()) {
        roots.emplace_back(join_repo_path(root, "layer1-compute/precompiled-kernels"));
    }
    return roots;
}

std::string device_version_string() {
    char buf[256] = {};
    clGetDeviceInfo(get_device(), CL_DRIVER_VERSION, sizeof(buf), buf, nullptr);
    return std::string(buf);
}

std::string device_name_string() {
    char buf[256] = {};
    clGetDeviceInfo(get_device(), CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
    return std::string(buf);
}

std::string source_digest(const std::string& source_path, const std::string& build_opts) {
    KernelBinaryCache cache;
    return cache.device_hash(source_path, build_opts, 0, 0);
}

TqStatus build_program_and_wait(
    cl_program prog,
    cl_device_id dev,
    const char* options,
    const char* stage);

TqStatus build_program_from_source(const std::string& source, const std::string& options, cl_program* out) {
    trace_log("kernel-build", "source_build begin opts=%s source_bytes=%zu", options.c_str(), source.size());
    cl_int err;
    const char* src = source.c_str();
    size_t len = source.size();
    cl_program prog = clCreateProgramWithSource(get_context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) return TqStatus::ERR_BUILD_FAILED;
    cl_device_id dev = get_device();
    if (build_program_and_wait(prog, dev, options.c_str(), "source") != TqStatus::OK) {
        clReleaseProgram(prog);
        return TqStatus::ERR_BUILD_FAILED;
    }
    trace_log("kernel-build", "source_build ok opts=%s", options.c_str());
    *out = prog;
    return TqStatus::OK;
}

struct AsyncBuildState {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    cl_build_status build_status = CL_BUILD_NONE;
};

void CL_CALLBACK async_build_notify(cl_program, void* user_data) {
    auto* state = static_cast<AsyncBuildState*>(user_data);
    if (!state) return;
    {
        std::lock_guard<std::mutex> lock(state->mu);
        state->done = true;
    }
    setenv("TQ_OPENCL_LAST_ASYNC_BUILD_CALLBACK", "1", 1);
    state->cv.notify_all();
}

TqStatus finalize_built_program(cl_program prog, cl_device_id dev, const char* stage) {
    size_t log_size = 0;
    cl_build_status build_status = CL_BUILD_NONE;
    (void)clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_STATUS, sizeof(build_status), &build_status, nullptr);
    if (build_status == CL_BUILD_SUCCESS) {
        const char* async_seen = std::getenv("TQ_OPENCL_LAST_ASYNC_BUILD_CALLBACK");
        runtime_scheduler_note_build_stage(stage, true, async_seen && std::strcmp(async_seen, "1") == 0);
        return TqStatus::OK;
    }
    runtime_scheduler_note_build_stage(stage, false, false);
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    if (log_size > 1) {
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        std::fprintf(stderr, "[OpenCL build error][%s]: %s\n", stage, log.c_str());
    }
    return TqStatus::ERR_BUILD_FAILED;
}

TqStatus build_program_and_wait(
    cl_program prog,
    cl_device_id dev,
    const char* options,
    const char* stage) {
    cl_int err = CL_SUCCESS;
    if (device_has_async_program_compilation()) {
        AsyncBuildState state;
        unsetenv("TQ_OPENCL_LAST_ASYNC_BUILD_CALLBACK");
        err = clBuildProgram(prog, 1, &dev, options, async_build_notify, &state);
        if (err != CL_SUCCESS) return TqStatus::ERR_BUILD_FAILED;

        std::unique_lock<std::mutex> lock(state.mu);
        const auto timeout = std::chrono::milliseconds(async_build_wait_timeout_ms());
        const bool callback_seen = state.cv.wait_for(lock, timeout, [&state] { return state.done; });
        lock.unlock();

        if (!callback_seen) {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            cl_build_status polled_status = CL_BUILD_IN_PROGRESS;
            while (std::chrono::steady_clock::now() < deadline) {
                if (clGetProgramBuildInfo(
                        prog,
                        dev,
                        CL_PROGRAM_BUILD_STATUS,
                        sizeof(polled_status),
                        &polled_status,
                        nullptr) != CL_SUCCESS) {
                    break;
                }
                if (polled_status != CL_BUILD_IN_PROGRESS) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        return finalize_built_program(prog, dev, stage);
    }

    err = clBuildProgram(prog, 1, &dev, options, nullptr, nullptr);
    if (err != CL_SUCCESS) return TqStatus::ERR_BUILD_FAILED;
    return finalize_built_program(prog, dev, stage);
}

TqStatus build_program_from_il(const std::vector<uint8_t>& il, const std::string& options, cl_program* out) {
#if defined(CL_VERSION_2_1)
    trace_log("kernel-build", "il_build begin opts=%s il_bytes=%zu", options.c_str(), il.size());
    cl_int err;
    cl_program prog = clCreateProgramWithIL(get_context(), il.data(), il.size(), &err);
    if (err != CL_SUCCESS || !prog) return TqStatus::ERR_BUILD_FAILED;
    cl_device_id dev = get_device();
    if (build_program_and_wait(prog, dev, options.c_str(), "IL") != TqStatus::OK) {
        clReleaseProgram(prog);
        return TqStatus::ERR_BUILD_FAILED;
    }
    trace_log("kernel-build", "il_build ok opts=%s", options.c_str());
    *out = prog;
    return TqStatus::OK;
#else
    (void)il;
    (void)options;
    (void)out;
    return TqStatus::ERR_BUILD_FAILED;
#endif
}

TqStatus build_program_from_binary(const std::vector<uint8_t>& binary, cl_program* out) {
    trace_log("kernel-build", "binary_build begin binary_bytes=%zu", binary.size());
    cl_int err = CL_SUCCESS;
    cl_int binary_status = CL_SUCCESS;
    const unsigned char* ptr = binary.data();
    size_t len = binary.size();
    cl_device_id dev = get_device();
    cl_program prog = clCreateProgramWithBinary(get_context(), 1, &dev, &len, &ptr, &binary_status, &err);
    if (err != CL_SUCCESS || binary_status != CL_SUCCESS || !prog) return TqStatus::ERR_BUILD_FAILED;
    if (build_program_and_wait(prog, dev, "", "binary") != TqStatus::OK) {
        clReleaseProgram(prog);
        return TqStatus::ERR_BUILD_FAILED;
    }
    trace_log("kernel-build", "binary_build ok binary_bytes=%zu", binary.size());
    *out = prog;
    return TqStatus::OK;
}

void maybe_store_binary(
    const std::string& cache_key,
    const KernelBinaryCache& cache,
    cl_program prog) {
    if (binary_cache_disabled()) return;
    size_t binary_sizes[1] = {};
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(binary_sizes), binary_sizes, nullptr) != CL_SUCCESS) return;
    if (binary_sizes[0] == 0) return;
    std::vector<uint8_t> binary(binary_sizes[0]);
    unsigned char* ptrs[] = { binary.data() };
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(ptrs), ptrs, nullptr) != CL_SUCCESS) return;
    cache.store(cache_key, "program", binary);
    for (const auto& root : precompiled_binary_roots()) {
        KernelBinaryCache export_cache(root);
        export_cache.store(cache_key, "program", binary);
    }
}

bool maybe_load_precompiled_binary(
    const std::string& compile_key,
    std::vector<uint8_t>& binary) {
    for (const auto& root : precompiled_binary_roots()) {
        KernelBinaryCache cache(root);
        if (cache.load(compile_key, "program", binary)) {
            trace_log("kernel-load", "precompiled_binary_hit key=%s root=%s bytes=%zu",
                      compile_key.c_str(), root.c_str(), binary.size());
            return true;
        }
    }
    return false;
}

} // namespace

TqStatus kernel_manager_build_program(const std::string& source, const std::string& options, cl_program* out) {
    if (!is_initialized()) return TqStatus::ERR_NO_PLATFORM;
    return build_program_from_source(source, options, out);
}

std::string infer_spirv_path_from_source(const std::string& source_path) {
    size_t slash = source_path.find_last_of('/');
    std::string file = (slash == std::string::npos) ? source_path : source_path.substr(slash + 1);
    size_t dot = file.rfind('.');
    std::string stem = (dot == std::string::npos) ? file : file.substr(0, dot);
    std::string spirv_dir = resolve_spirv_dir();
    return spirv_dir.empty() ? std::string() : join_repo_path(spirv_dir, (stem + ".spv").c_str());
}

std::string kernel_manager_compile_key(
    const std::string& source_path,
    const std::string& build_opts) {
    KernelBinaryCache key_cache;
    return key_cache.device_hash(
        device_name_string(),
        device_version_string() + "\n" + source_path + "\n" + build_opts + "\n" + source_digest(source_path, build_opts),
        get_compute_units(),
        get_max_wg_size());
}

TqStatus kernel_manager_load_kernel(
    const std::string& kernel_name,
    const std::string& spirv_path,
    const std::string& source_path,
    const std::string& build_opts) {
    if (!is_initialized()) return TqStatus::ERR_NO_PLATFORM;
    if (g_kernels.count(kernel_name)) return TqStatus::OK;

    std::string program_key = source_path + "\n--opts--\n" + build_opts;
    cl_program prog = nullptr;
    if (g_programs.count(program_key)) {
        prog = g_programs[program_key];
        runtime_scheduler_note_program_reuse(kernel_name, program_key);
        trace_log("kernel-load", "program_reuse source=%s kernel=%s", source_path.c_str(), kernel_name.c_str());
    } else {
        KernelBinaryCache binary_cache;
        std::string compile_key = kernel_manager_compile_key(source_path, build_opts);
        runtime_scheduler_note_compile_intent(kernel_name, compile_key);
        const bool has_spirv_il = !spirv_path.empty() && std::filesystem::exists(spirv_path);
        const bool has_source = std::filesystem::exists(source_path);

        std::vector<uint8_t> binary;
        TqStatus st = TqStatus::ERR_BUILD_FAILED;
        bool precompiled_hit = false;
        bool binary_cache_hit = false;
        if (!want_source_only() && !binary_cache_disabled()) {
            precompiled_hit = maybe_load_precompiled_binary(compile_key, binary);
            binary_cache_hit = precompiled_hit || binary_cache.load(compile_key, "program", binary);
        }
        runtime_scheduler_note_compile_candidate(
            kernel_name,
            compile_key,
            precompiled_hit,
            binary_cache_hit && !precompiled_hit,
            has_spirv_il,
            has_source,
            want_source_only());
        if (binary_cache_hit) {
            runtime_scheduler_note_cache_hit(
                precompiled_hit ? "precompiled" : "binary_cache",
                kernel_name,
                compile_key,
                binary.size());
            trace_log("kernel-load", "binary_cache_hit kernel=%s bytes=%zu", kernel_name.c_str(), binary.size());
            st = build_program_from_binary(binary, &prog);
        }

        if ((!prog || st != TqStatus::OK) && !want_source_only() && device_has_il_program()) {
            std::vector<uint8_t> il = read_binary_file(spirv_path);
            if (!il.empty()) {
                trace_log("kernel-load", "spirv_il_attempt kernel=%s path=%s bytes=%zu", kernel_name.c_str(), spirv_path.c_str(), il.size());
                st = build_program_from_il(il, build_opts, &prog);
            }
        }

        if (!prog || st != TqStatus::OK) {
            std::string source = read_file(source_path);
            if (source.empty()) {
                std::fprintf(stderr, "[OpenCL] Cannot read kernel file: %s\n", source_path.c_str());
                return TqStatus::ERR_BUILD_FAILED;
            }
            trace_log("kernel-load", "source_fallback kernel=%s path=%s", kernel_name.c_str(), source_path.c_str());
            st = build_program_from_source(source, build_opts, &prog);
        }

        if (st != TqStatus::OK || !prog) return TqStatus::ERR_BUILD_FAILED;
        if (!binary_cache_disabled()) {
            maybe_store_binary(compile_key, binary_cache, prog);
        }
        g_programs[program_key] = prog;
    }

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(prog, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS || !kernel) {
        std::fprintf(stderr, "[OpenCL] Cannot create kernel '%s': %d\n", kernel_name.c_str(), err);
        return TqStatus::ERR_BUILD_FAILED;
    }
    g_kernels[kernel_name] = kernel;
    KernelMetadata meta = {};
    clGetKernelWorkGroupInfo(kernel, get_device(), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             sizeof(meta.preferred_work_group_size_multiple),
                             &meta.preferred_work_group_size_multiple, nullptr);
    clGetKernelWorkGroupInfo(kernel, get_device(), CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(meta.work_group_size), &meta.work_group_size, nullptr);
    clGetKernelWorkGroupInfo(kernel, get_device(), CL_KERNEL_LOCAL_MEM_SIZE,
                             sizeof(meta.local_mem_size), &meta.local_mem_size, nullptr);
    clGetKernelWorkGroupInfo(kernel, get_device(), CL_KERNEL_PRIVATE_MEM_SIZE,
                             sizeof(meta.private_mem_size), &meta.private_mem_size, nullptr);
    g_kernel_metadata[kernel_name] = meta;
    runtime_scheduler_note_kernel_ready(kernel_name, meta);
    trace_log(
        "kernel-load",
        "kernel_ready name=%s wg_limit=%zu pref_multiple=%zu local_mem=%llu private_mem=%llu",
        kernel_name.c_str(),
        meta.work_group_size,
        meta.preferred_work_group_size_multiple,
        static_cast<unsigned long long>(meta.local_mem_size),
        static_cast<unsigned long long>(meta.private_mem_size));
    return TqStatus::OK;
}

cl_kernel kernel_manager_get_kernel(const std::string& name) {
    auto it = g_kernels.find(name);
    return it == g_kernels.end() ? nullptr : it->second;
}

KernelMetadata kernel_manager_get_kernel_metadata(const std::string& name) {
    auto it = g_kernel_metadata.find(name);
    return it == g_kernel_metadata.end() ? KernelMetadata{} : it->second;
}

void kernel_manager_release_all() {
    for (auto& entry : g_kernels) clReleaseKernel(entry.second);
    for (auto& entry : g_programs) clReleaseProgram(entry.second);
    g_kernels.clear();
    g_kernel_metadata.clear();
    g_programs.clear();
}

} // namespace tq
