/**
 * TurboQuant repo/runtime path resolution helpers.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <string>
#include <vector>

namespace tq {

std::string join_repo_path(const std::string& base, const char* suffix);
std::string detect_repo_root();
std::vector<std::string> repo_root_candidates();
std::vector<std::string> runtime_pack_roots();
std::vector<std::string> mesa_roots();
std::vector<std::string> mesa_build_dirs(const std::string& mesa_root);
std::string resolve_kernel_dir(const std::string& requested_dir = "");

} // namespace tq
