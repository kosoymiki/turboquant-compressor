/**
 * TurboQuant OpenCL — error handling.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_opencl_types.h"
#include <string>

namespace tq {

const char* status_string(TqStatus s);
std::string cl_error_string(int cl_err);

} // namespace tq
