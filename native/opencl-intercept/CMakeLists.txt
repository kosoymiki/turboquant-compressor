cmake_minimum_required(VERSION 3.18)
project(turboquant_intercept C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

find_path(OPENCL_INCLUDE_DIR CL/cl.h
  PATHS $ENV{PREFIX}/include /usr/include /usr/local/include
)

find_library(OPENCL_LIB OpenCL
  PATHS /system/vendor/lib64 /vendor/lib64 /system/lib64
       $ENV{PREFIX}/lib $ENV{PREFIX}/opt/vendor/lib
)

add_library(tq_intercept SHARED
  src/tq_intercept_core.cpp
  src/tq_intercept_api.cpp
  src/tq_intercept_memory.cpp
  src/tq_intercept_kernel.cpp
  src/tq_intercept_profiler.cpp
)

target_include_directories(tq_intercept PRIVATE include)
if(OPENCL_INCLUDE_DIR)
  target_include_directories(tq_intercept PRIVATE ${OPENCL_INCLUDE_DIR})
endif()

target_link_libraries(tq_intercept PRIVATE dl pthread)
if(ANDROID)
  target_link_libraries(tq_intercept PRIVATE log)
endif()

if(OPENCL_LIB)
  target_link_libraries(tq_intercept PRIVATE ${OPENCL_LIB})
endif()

target_compile_options(tq_intercept PRIVATE -Wall -Wextra -O2 -fPIC)

add_executable(intercept_cli src/intercept_cli.cpp)
target_link_libraries(intercept_cli PRIVATE tq_intercept)

add_executable(test_intercept test/test_intercept.cpp)
target_link_libraries(test_intercept PRIVATE tq_intercept)