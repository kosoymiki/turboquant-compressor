cmake_minimum_required(VERSION 3.18)
add_executable(test_qat test_qat.cpp)
target_link_libraries(test_qat PRIVATE tq_core pthread)
target_include_directories(test_qat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/native/kernel/include
)
