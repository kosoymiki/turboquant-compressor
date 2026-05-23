cmake_minimum_required(VERSION 3.18)
project(turboquant_kernel C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

find_path(OPENCL_INCLUDE_DIR CL/cl.h
  PATHS $ENV{PREFIX}/include /usr/include /usr/local/include
)

add_library(tq_kernel STATIC
  src/tq_rotation_kernel.cpp
  src/tq_quantizer_kernel.cpp
  src/tq_qjl_kernel.cpp
  src/tq_qjl_hadamard.cpp
)

target_include_directories(tq_kernel PRIVATE include)
if(OPENCL_INCLUDE_DIR)
  target_include_directories(tq_kernel PRIVATE ${OPENCL_INCLUDE_DIR})
endif()

target_compile_options(tq_kernel PRIVATE -Wall -Wextra -O2)

add_executable(test_kernel src/test_kernel.cpp)
target_include_directories(test_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_kernel PRIVATE tq_kernel pthread)

add_executable(kernel_bench src/kernel_bench.cpp)
target_include_directories(kernel_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kernel_bench PRIVATE tq_kernel pthread)

add_executable(test_hadamard_qjl src/test_hadamard_qjl.cpp)
target_include_directories(test_hadamard_qjl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_hadamard_qjl PRIVATE tq_kernel pthread)