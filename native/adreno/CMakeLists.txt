cmake_minimum_required(VERSION 3.18)
project(turboquant_adreno C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Detect NEON (always available on aarch64)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=armv8-a+simd" HAS_NEON)

add_executable(tq_adreno_cli
  src/tq_adreno_cli.cpp
  src/tq_backend_registry.cpp
  src/tq_cpu_scalar_mse_score.cpp
  src/tq_cpu_scalar_value_dequant.cpp
  src/tq_cpu_scalar_fused_attention.cpp
  src/tq_neon_mse_score.cpp
  src/tq_neon_value_dequant.cpp
  src/tq_neon_fused_attention.cpp
  src/tq_benchmark.cpp
)

target_include_directories(tq_adreno_cli PRIVATE include)

if(HAS_NEON)
  target_compile_options(tq_adreno_cli PRIVATE -march=armv8-a+simd)
  target_compile_definitions(tq_adreno_cli PRIVATE TQ_HAS_NEON=1)
endif()

target_compile_options(tq_adreno_cli PRIVATE -Wall -Wextra -O2)
