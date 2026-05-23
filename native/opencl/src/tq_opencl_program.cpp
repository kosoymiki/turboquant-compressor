/**
 * TurboQuant OpenCL — compatibility wrappers around kernel manager.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "../include/tq_opencl_kernel_manager.h"
#include <string>

namespace tq {

namespace {

std::string compose_kernel_build_opts(const std::string& kernel_name, const std::string& build_opts) {
    KernelTuneParams tune = get_kernel_tune(kernel_name.c_str());
    std::string opts = build_opts;
    opts += " -DTQ_ITEMS_PER_THREAD=" + std::to_string(tune.items_per_thread);
    opts += " -DTQ_VEC_WIDTH=" + std::to_string(tune.vec_width);
    opts += " -DTQ_TILE_M=" + std::to_string(tune.tile_m);
    opts += " -DTQ_TILE_N=" + std::to_string(tune.tile_n);
    opts += " -DTQ_TILE_K=" + std::to_string(tune.tile_k);
    opts += " -DTQ_USE_GMEM_TILING=" + std::to_string(tune.use_gmem_tiling ? 1 : 0);
    opts += " -DTQ_PREFETCH_GLOBAL=" + std::to_string(tune.prefetch_global ? 1 : 0);
    opts += " -DTQ_INPUT_LAYOUT=" + std::to_string(static_cast<unsigned>(tune.input_layout));
    opts += " -DTQ_OUTPUT_LAYOUT=" + std::to_string(static_cast<unsigned>(tune.output_layout));
    return opts;
}

} // namespace

TqStatus build_program(const std::string& source, const std::string& options, cl_program* out) {
    return kernel_manager_build_program(source, options, out);
}

TqStatus load_kernel(const std::string& kernel_file, const std::string& kernel_name, const std::string& build_opts) {
    return kernel_manager_load_kernel(
        kernel_name,
        infer_spirv_path_from_source(kernel_file),
        kernel_file,
        compose_kernel_build_opts(kernel_name, build_opts));
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
