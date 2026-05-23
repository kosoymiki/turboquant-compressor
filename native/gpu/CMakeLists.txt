cmake_minimum_required(VERSION 3.18)
project(turboquant_gpu C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(OPENCL_INCLUDE_DIR CL/cl.h PATHS $ENV{PREFIX}/include /usr/include)

find_library(OPENCL_LIB OpenCL PATHS /vendor/lib64 /system/vendor/lib64 /data/data/com.termux/files/usr/lib NO_DEFAULT_PATH)

add_library(tq_gpu_pipeline STATIC
  src/tq_gpu_pipeline.cpp
)

target_include_directories(tq_gpu_pipeline PUBLIC include)
if(OPENCL_INCLUDE_DIR)
  target_include_directories(tq_gpu_pipeline PRIVATE ${OPENCL_INCLUDE_DIR})
endif()

target_link_libraries(tq_gpu_pipeline PUBLIC dl)
if(OPENCL_LIB)
  target_link_libraries(tq_gpu_pipeline PUBLIC ${OPENCL_LIB})
endif()

target_compile_options(tq_gpu_pipeline PRIVATE -Wall -Wextra -O2)

add_executable(test_gpu_pipeline src/test_gpu_pipeline.cpp)
target_include_directories(test_gpu_pipeline PRIVATE include)
target_link_libraries(test_gpu_pipeline PRIVATE tq_gpu_pipeline dl pthread)
if(OPENCL_LIB)
  target_link_libraries(test_gpu_pipeline PRIVATE ${OPENCL_LIB})
endif()
target_link_libraries(test_gpu_pipeline PRIVATE dl)
