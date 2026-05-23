cmake_minimum_required(VERSION 3.18)
project(tq_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Match kernel build: use Termux g++/gcc
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_COMPILER /usr/bin/gcc)

find_path(OPENCL_INCLUDE_DIR CL/cl.h
  PATHS /data/data/com.termux/files/usr/include /usr/include
  NO_DEFAULT_PATH
)

# ── Core library ───────────────────────────────────────────────────────────
add_library(tq_core STATIC)

target_sources(tq_core PRIVATE
    src/tq_compress.cpp
    src/tq_search.cpp
    src/tq_format.cpp
    src/tq_base64.cpp
    src/tq_quality.cpp
    src/tq_limits.cpp
    src/tq_prng.cpp
    src/tq_beta_sphere.cpp
    src/tq_kv_cache.cpp
    src/tq_benchmark.cpp
    src/tq_oracle.cpp
    src/tq_context_pack.cpp
    src/tq_calibration.cpp
    src/tq_lsq_optimizer.cpp
    src/tq_distillation.cpp
    src/tq_qat_training.cpp
)

# Link kernel sources
set(TQ_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../kernel" CACHE PATH "")
if(EXISTS "${TQ_KERNEL_DIR}/src/tq_rotation_kernel.cpp")
    target_sources(tq_core PRIVATE
        "${TQ_KERNEL_DIR}/src/tq_rotation_kernel.cpp"
        "${TQ_KERNEL_DIR}/src/tq_quantizer_kernel.cpp"
        "${TQ_KERNEL_DIR}/src/tq_qjl_kernel.cpp"
    )
    target_include_directories(tq_core PUBLIC "${TQ_KERNEL_DIR}/include")
endif()

target_include_directories(tq_core PUBLIC
    include
    ${OPENCL_INCLUDE_DIR}
)

target_compile_options(tq_core PRIVATE -Wall -Wextra -O2)

# ── CLI executable ──────────────────────────────────────────────────────────
add_executable(turboquant_core src/tq_cli.cpp)
target_link_libraries(turboquant_core PRIVATE tq_core pthread)

# ── Unit tests ─────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_tq_core test/test_tq_core.cpp)
target_link_libraries(test_tq_core PRIVATE tq_core pthread)
add_test(NAME tq_core_test COMMAND test_tq_core)

# ── Full benchmark ─────────────────────────────────────────────────────────
add_executable(test_full_benchmark test/test_full_benchmark.cpp)
target_link_libraries(test_full_benchmark PRIVATE tq_core pthread)
target_include_directories(test_full_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/native/kernel/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/native/kernel/include
)
add_test(NAME tq_full_benchmark COMMAND test_full_benchmark)

# ── QAT Training test ──────────────────────────────────────────────────────
add_executable(test_qat test/test_qat.cpp)
target_link_libraries(test_qat PRIVATE tq_core pthread)
target_include_directories(test_qat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/native/kernel/include
)
add_test(NAME tq_qat_test COMMAND test_qat)