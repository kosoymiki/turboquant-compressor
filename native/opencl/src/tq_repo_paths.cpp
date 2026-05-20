/**
 * TurboQuant repo/runtime path resolution helpers.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_repo_paths.h"

#include <cstdlib>
#include <limits.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace tq {

namespace {

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

bool dir_exists(const std::string& path) {
    struct stat st;
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool file_exists(const std::string& path) {
    struct stat st;
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string parent_dir(const std::string& path) {
    size_t slash = path.rfind('/');
    if (slash == std::string::npos) return "";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

std::string executable_dir() {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = '\0';
    return parent_dir(std::string(buf));
}

std::string current_working_dir() {
    char buf[PATH_MAX];
    if (!getcwd(buf, sizeof(buf))) return "";
    return std::string(buf);
}

bool looks_like_repo_root(const std::string& path) {
    return dir_exists(join_repo_path(path, "native/opencl/kernels")) &&
           file_exists(join_repo_path(path, "package.json"));
}

void append_if_missing(std::vector<std::string>& items, const std::string& value) {
    if (value.empty()) return;
    for (const auto& item : items) {
        if (item == value) return;
    }
    items.push_back(value);
}

void add_ancestor_candidates(std::vector<std::string>& out, const std::string& start) {
    std::string current = start;
    while (!current.empty()) {
        if (looks_like_repo_root(current)) append_if_missing(out, current);
        std::string parent = parent_dir(current);
        if (parent.empty() || parent == current) break;
        current = parent;
    }
}

} // namespace

std::string join_repo_path(const std::string& base, const char* suffix) {
    if (base.empty()) return std::string();
    if (!base.empty() && base.back() == '/') return base + suffix;
    return base + "/" + suffix;
}

std::vector<std::string> repo_root_candidates() {
    std::vector<std::string> roots;
    append_if_missing(roots, env_or_empty("TURBOQUANT_ROOT"));
    add_ancestor_candidates(roots, executable_dir());
    add_ancestor_candidates(roots, current_working_dir());
    return roots;
}

std::string detect_repo_root() {
    auto roots = repo_root_candidates();
    return roots.empty() ? std::string() : roots.front();
}

std::vector<std::string> runtime_pack_roots() {
    std::vector<std::string> roots;
    append_if_missing(roots, env_or_empty("TQ_DRIVER_ROOT"));
    for (const auto& repo_root : repo_root_candidates()) {
        append_if_missing(roots, join_repo_path(repo_root, "native/opencl/runtime-pack"));
        append_if_missing(roots, join_repo_path(repo_root, "native/opencl/driver-pack"));
        append_if_missing(roots, join_repo_path(repo_root, "native/opencl"));
    }
    return roots;
}

std::vector<std::string> mesa_roots() {
    std::vector<std::string> roots;
    append_if_missing(roots, env_or_empty("TQ_MESA_ROOT"));
    for (const auto& repo_root : repo_root_candidates()) {
        append_if_missing(roots, join_repo_path(repo_root, "native/opencl/mesa-source"));
        append_if_missing(roots, join_repo_path(repo_root, "native/opencl"));
    }
    return roots;
}

std::vector<std::string> mesa_build_dirs(const std::string& mesa_root) {
    std::vector<std::string> builds;
    if (mesa_root.empty()) return builds;
    builds.push_back(join_repo_path(mesa_root, "build-tq-zero"));
    builds.push_back(join_repo_path(mesa_root, "build-turboquant-android-aarch64"));
    builds.push_back(join_repo_path(mesa_root, "build"));
    return builds;
}

std::string resolve_kernel_dir(const std::string& requested_dir) {
    if (dir_exists(requested_dir)) return requested_dir;

    std::string env_dir = env_or_empty("TQ_OPENCL_KERNEL_DIR");
    if (dir_exists(env_dir)) return env_dir;

    for (const auto& repo_root : repo_root_candidates()) {
        std::string candidate = join_repo_path(repo_root, "native/opencl/kernels");
        if (dir_exists(candidate)) return candidate;
    }

    std::string exe_dir = executable_dir();
    if (!exe_dir.empty()) {
        std::string sibling_dir = join_repo_path(exe_dir, "../kernels");
        if (dir_exists(sibling_dir)) return sibling_dir;
    }

    return requested_dir;
}

} // namespace tq
