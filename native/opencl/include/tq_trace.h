#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace tq {

inline bool trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char* v = std::getenv("TQ_OPENCL_TRACE");
        enabled = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return enabled == 1;
}

inline void trace_log(const char* scope, const char* fmt, ...) {
    if (!trace_enabled()) return;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    std::fprintf(stderr, "[tq-trace][%s] ", scope ? scope : "unknown");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

} // namespace tq
