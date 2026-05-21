/**
 * TurboQuant OpenCL — compatibility wrappers around kernel manager.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "../include/tq_opencl_kernel_manager.h"

namespace tq {

TqStatus build_program(const std::string& source, const std::string& options, cl_program* out) {
    return kernel_manager_build_program(source, options, out);
}

TqStatus load_kernel(const std::string& kernel_file, const std::string& kernel_name, const std::string& build_opts) {
    return kernel_manager_load_kernel(kernel_name, infer_spirv_path_from_source(kernel_file), kernel_file, build_opts);
}

cl_kernel get_kernel(const std::string& name) {
    return kernel_manager_get_kernel(name);
}

KernelMetadata get_kernel_metadata(const std::string& name) {
    return kernel_manager_get_kernel_metadata(name);
}

void release_all_programs() {
    kernel_manager_release_all();
}

} // namespace tq
