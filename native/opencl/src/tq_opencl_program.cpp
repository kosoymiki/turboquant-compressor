/**
 * TurboQuant OpenCL — program/kernel compilation from .cl sources.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl_types.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>

namespace tq {

// Forward declarations from context
extern cl_context get_context();
extern cl_device_id get_device();
extern bool is_initialized();

static std::unordered_map<std::string, cl_program> g_programs;
static std::unordered_map<std::string, cl_kernel> g_kernels;

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

TqStatus build_program(const std::string& source, const std::string& options, cl_program* out) {
    if (!is_initialized()) return TqStatus::ERR_NO_PLATFORM;

    cl_int err;
    const char* src = source.c_str();
    size_t len = source.size();
    cl_program prog = clCreateProgramWithSource(get_context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) return TqStatus::ERR_BUILD_FAILED;

    cl_device_id dev = get_device();
    err = clBuildProgram(prog, 1, &dev, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log for diagnostics
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 1) {
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            fprintf(stderr, "[OpenCL build error]: %s\n", log.c_str());
        }
        clReleaseProgram(prog);
        return TqStatus::ERR_BUILD_FAILED;
    }

    *out = prog;
    return TqStatus::OK;
}

TqStatus load_kernel(const std::string& kernel_file, const std::string& kernel_name, const std::string& build_opts) {
    if (!is_initialized()) return TqStatus::ERR_NO_PLATFORM;

    // Check cache
    if (g_kernels.count(kernel_name)) return TqStatus::OK;

    std::string program_key = kernel_file + "\n--opts--\n" + build_opts;

    // Build program if not cached
    cl_program prog = nullptr;
    if (g_programs.count(program_key)) {
        prog = g_programs[program_key];
    } else {
        std::string source = read_file(kernel_file);
        if (source.empty()) {
            fprintf(stderr, "[OpenCL] Cannot read kernel file: %s\n", kernel_file.c_str());
            return TqStatus::ERR_BUILD_FAILED;
        }
        TqStatus st = build_program(source, build_opts, &prog);
        if (st != TqStatus::OK) return st;
        g_programs[program_key] = prog;
    }

    // Create kernel
    cl_int err;
    cl_kernel k = clCreateKernel(prog, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] Cannot create kernel '%s': %d\n", kernel_name.c_str(), err);
        return TqStatus::ERR_BUILD_FAILED;
    }

    g_kernels[kernel_name] = k;
    return TqStatus::OK;
}

cl_kernel get_kernel(const std::string& name) {
    auto it = g_kernels.find(name);
    return (it != g_kernels.end()) ? it->second : nullptr;
}

void release_all_programs() {
    for (auto& [_, k] : g_kernels) clReleaseKernel(k);
    for (auto& [_, p] : g_programs) clReleaseProgram(p);
    g_kernels.clear();
    g_programs.clear();
}

} // namespace tq
