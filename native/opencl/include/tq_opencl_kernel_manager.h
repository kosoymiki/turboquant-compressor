/**
 * TurboQuant OpenCL — program/kernel loader with binary cache and SPIR-V IL path.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_opencl_errors.h"
#include "tq_opencl.h"
#include <CL/cl.h>
#include <string>

namespace tq {

TqStatus kernel_manager_build_program(const std::string& source, const std::string& options, cl_program* out);
TqStatus kernel_manager_load_kernel(
    const std::string& kernel_name,
    const std::string& spirv_path,
    const std::string& source_path,
    const std::string& build_opts);
cl_kernel kernel_manager_get_kernel(const std::string& name);
KernelMetadata kernel_manager_get_kernel_metadata(const std::string& name);
void kernel_manager_release_all();
std::string infer_spirv_path_from_source(const std::string& source_path);
std::string kernel_manager_compile_key(
    const std::string& source_path,
    const std::string& build_opts);

} // namespace tq
