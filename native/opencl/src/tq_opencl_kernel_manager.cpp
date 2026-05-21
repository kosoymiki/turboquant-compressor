/**
 * TurboQuant OpenCL — program/kernel loader with binary cache and SPIR-V IL path.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_opencl_kernel_manager.h"
#include "../include/tq_kernel_cache.h"
#include "../include/tq_opencl.h"
#include "../include/tq_repo_paths.h"

#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
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

TqStatus build_program_from_source(const std::string& source, const std::string& options, cl_program* out) {
    cl_int err;
    const char* src = source.c_str();
    size_t len = source.size();
    cl_program prog = clCreateProgramWithSource(get_context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) return TqStatus::ERR_BUILD_FAILED;
    cl_device_id dev = get_device();
    err = clBuildProgram(prog, 1, &dev, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 1) {
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::fprintf(stderr, "[OpenCL build error][source]: %s\n", log.c_str());
        }
        clReleaseProgram(prog);
        return TqStatus::ERR_BUILD_FAILED;
    }
    *out = prog;
    return TqStatus::OK;
}

TqStatus build_program_from_il(const std::vector<uint8_t>& il, const std::string& options, cl_program* out) {
#if defined(CL_VERSION_2_1)
    cl_int err;
    cl_program prog = clCreateProgramWithIL(get_context(), il.data(), il.size(), &err);
    if (err != CL_SUCCESS || !prog) return TqStatus::ERR_BUILD_FAILED;
    cl_device_id dev = get_device();
    err = clBuildProgram(prog, 1, &dev, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 1) {
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::fprintf(stderr, "[OpenCL build error][IL]: %s\n", log.c_str());
        }
        clReleaseProgram(prog);
        return TqStatus::ERR_BUILD_FAILED;
    }
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
    cl_int err = CL_SUCCESS;
    cl_int binary_status = CL_SUCCESS;
    const unsigned char* ptr = binary.data();
    size_t len = binary.size();
    cl_device_id dev = get_device();
    cl_program prog = clCreateProgramWithBinary(get_context(), 1, &dev, &len, &ptr, &binary_status, &err);
    if (err != CL_SUCCESS || binary_status != CL_SUCCESS || !prog) return TqStatus::ERR_BUILD_FAILED;
    err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return TqStatus::ERR_BUILD_FAILED;
    }
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
    } else {
        KernelBinaryCache binary_cache;
        KernelBinaryCache key_cache;
        std::string compile_key = key_cache.device_hash(
            device_name_string(),
            device_version_string() + "\n" + source_path + "\n" + build_opts + "\n" + source_digest(source_path, build_opts),
            get_compute_units(),
            get_max_wg_size());

        std::vector<uint8_t> binary;
        TqStatus st = TqStatus::ERR_BUILD_FAILED;
        if (!want_source_only() && !binary_cache_disabled() && binary_cache.load(compile_key, "program", binary)) {
            st = build_program_from_binary(binary, &prog);
        }

        if ((!prog || st != TqStatus::OK) && !want_source_only() && device_has_il_program()) {
            std::vector<uint8_t> il = read_binary_file(spirv_path);
            if (!il.empty()) {
                st = build_program_from_il(il, build_opts, &prog);
            }
        }

        if (!prog || st != TqStatus::OK) {
            std::string source = read_file(source_path);
            if (source.empty()) {
                std::fprintf(stderr, "[OpenCL] Cannot read kernel file: %s\n", source_path.c_str());
                return TqStatus::ERR_BUILD_FAILED;
            }
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
