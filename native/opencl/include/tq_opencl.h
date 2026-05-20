/**
 * TurboQuant OpenCL — main header.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_opencl_types.h"
#include "tq_opencl_errors.h"
#include "tq_opencl_probe.h"
#include "tq_opencl_kernels.h"
#include "tq_opencl_memory.h"
#include "tq_gpu_profile.h"
#include "tq_kernel_tune.h"
#include "tq_vk_probe.h"
#include "tq_driver_manifest.h"
#include "tq_cl_vk_interop.h"

#include <CL/cl.h>
#include <string>

namespace tq {

// Context
TqStatus init_context();
void shutdown_context();
cl_platform_id get_platform();
cl_device_id get_device();
cl_context get_context();
cl_command_queue get_queue();
bool is_initialized();
bool is_adreno();
uint32_t get_compute_units();
size_t get_max_wg_size();
bool profiling_enabled();
const GpuProfile& get_gpu_profile();

// Program/kernel compilation
TqStatus build_program(const std::string& source, const std::string& options, cl_program* out);
TqStatus load_kernel(const std::string& kernel_file, const std::string& kernel_name, const std::string& build_opts = "");
cl_kernel get_kernel(const std::string& name);
void release_all_programs();

// CPU reference implementations (always available for parity)
void mse_score_cpu_reference(const MseScoreInput& input, float* scores);
void qjl_score_cpu_reference(const QjlScoreInput& input);
void value_dequant_cpu_reference(const ValueDequantInput& input, float* output);
void fused_attention_cpu_reference(const FusedAttentionInput& input);

} // namespace tq
