cmake_minimum_required(VERSION 3.18)
project(turboquant_opencl C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# OpenCL header path (Termux: pkg install opencl-headers)
find_path(OPENCL_INCLUDE_DIR CL/cl.h
  PATHS
    $ENV{PREFIX}/include
    /usr/include
    /usr/local/include
)

# OpenCL library
find_library(OPENCL_LIB OpenCL
  PATHS
    /system/vendor/lib64
    /vendor/lib64
    /system/lib64
    $ENV{PREFIX}/lib
    $ENV{PREFIX}/opt/vendor/lib
)

if(NOT OPENCL_INCLUDE_DIR)
  message(WARNING "OpenCL headers not found — install: pkg install opencl-headers")
endif()

add_executable(tq_opencl_cli
  src/tq_opencl_cli.cpp
  src/tq_opencl_probe.cpp
  src/tq_opencl_context.cpp
  src/tq_opencl_program.cpp
  src/tq_opencl_memory.cpp
  src/tq_opencl_mse_score.cpp
  src/tq_opencl_qjl_score.cpp
  src/tq_opencl_value_dequant.cpp
  src/tq_opencl_fused_attention.cpp
  src/tq_opencl_fused_attention_tiled.cpp
  src/tq_opencl_benchmark.cpp
  src/tq_driver_loader.cpp
  src/tq_gpu_profile.cpp
  src/tq_kernel_tune.cpp
  src/tq_driver_manifest.cpp
  src/tq_vk_probe.cpp
  src/tq_cl_vk_interop.cpp
)

target_include_directories(tq_opencl_cli PRIVATE include)
if(OPENCL_INCLUDE_DIR)
  target_include_directories(tq_opencl_cli PRIVATE ${OPENCL_INCLUDE_DIR})
endif()

if(OPENCL_LIB)
  target_link_libraries(tq_opencl_cli PRIVATE ${OPENCL_LIB})
else()
  # Dynamic loading fallback — link dl for dlopen
  target_link_libraries(tq_opencl_cli PRIVATE dl)
  target_compile_definitions(tq_opencl_cli PRIVATE TQ_OPENCL_DYNAMIC_LOAD=1)
endif()

target_compile_options(tq_opencl_cli PRIVATE -Wall -Wextra -O2)
