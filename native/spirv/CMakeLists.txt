cmake_minimum_required(VERSION 3.18)
project(turboquant_spirv)

find_program(CLANG_EXECUTABLE clang)
find_program(LLVM_SPIRV_EXECUTABLE llvm-spirv)
find_program(SPIRV_DIS_EXECUTABLE spirv-dis)
find_program(SPIRV_VAL_EXECUTABLE spirv-val)

if(NOT CLANG_EXECUTABLE)
  message(FATAL_ERROR "clang not found — install: pkg install clang")
endif()
if(NOT LLVM_SPIRV_EXECUTABLE)
  message(FATAL_ERROR "llvm-spirv not found — install: pkg install spirv-llvm-translator")
endif()
if(NOT SPIRV_VAL_EXECUTABLE)
  message(WARNING "spirv-val not found — install: pkg install spirv-tools")
endif()

set(KERNEL_DIR "${CMAKE_SOURCE_DIR}/../opencl/kernels")
set(SPIRV_OUTPUT_DIR "${CMAKE_BINARY_DIR}")

set(KERNEL_SOURCES
  tq_mse_score.cl
  tq_qjl_score.cl
  tq_value_dequant.cl
  tq_attention_logits.cl
  tq_attention_apply_values.cl
)

add_custom_target(spirv_all DEPENDS)

foreach(kernel ${KERNEL_SOURCES})
  set_source_files_properties(${kernel} PROPERTIES SKIP_UNITY_BUILD ON)
  get_filename_component(kernel_name ${kernel} NAME_WE)

  set(spv_file "${SPIRV_OUTPUT_DIR}/${kernel_name}.spv")
  set(ir_file "${SPIRV_OUTPUT_DIR}/${kernel_name}.bc")
  set(txt_file "${SPIRV_OUTPUT_DIR}/${kernel_name}.spv.txt")

  # Step 1: clang -> LLVM IR
  add_custom_command(
    OUTPUT ${ir_file}
    COMMAND ${CLANG_EXECUTABLE} -emit-llvm -x cl -O3 -fPIC
      -target spir64-unknown-unknown
      -Xclang "-cl-std=CL2.0"
      -D__OPENCL_VERSION__=200
      -DKEPLER_CORE=0
      -I${KERNEL_DIR}
      ${KERNEL_DIR}/${kernel}
      -o ${ir_file} -c
    DEPENDS ${KERNEL_DIR}/${kernel}
    COMMENT "Clang: ${kernel} -> LLVM IR"
    VERBATIM
  )

  # Step 2: llvm-spirv -> SPIR-V
  add_custom_command(
    OUTPUT ${spv_file}
    COMMAND ${LLVM_SPIRV_EXECUTABLE} -o ${spv_file} ${ir_file}
    DEPENDS ${ir_file}
    COMMENT "llvm-spirv: LLVM IR -> SPIR-V 1.2"
    VERBATIM
  )

  # Step 3: spirv-val (optional validation)
  if(SPIRV_VAL_EXECUTABLE)
    add_custom_command(
      OUTPUT ${spv_file}.validated
      COMMAND ${SPIRV_VAL_EXECUTABLE} ${spv_file}
      DEPENDS ${spv_file}
      COMMENT "spirv-val: validate ${kernel_name}.spv"
      VERBATIM
    )
  endif()

  # Step 4: spirv-dis (text dump)
  if(SPIRV_DIS_EXECUTABLE)
    add_custom_command(
      OUTPUT ${txt_file}
      COMMAND ${SPIRV_DIS_EXECUTABLE} ${spv_file} > ${txt_file} || true
      DEPENDS ${spv_file}
      COMMENT "spirv-dis: ${kernel_name}.spv -> text"
      VERBATIM
    )
  endif()

  add_custom_target(spirv_${kernel_name} ALL
    DEPENDS ${spv_file}
  )
endforeach()

# Convenience targets
add_custom_target(spirv_validate
  COMMAND echo "Validation requires spirv-val"
  COMMENT "Validate all SPIR-V binaries"
)

# Install SPIR-V binaries
install(DIRECTORY ${SPIRV_OUTPUT_DIR}/ DESTINATION share/turboquant/spirv FILES_MATCHING PATTERN "*.spv")

message(STATUS "SPIR-V Compiler Pipeline")
message(STATUS "  CLANG: ${CLANG_EXECUTABLE}")
message(STATUS "  LLVM_SPIRV: ${LLVM_SPIRV_EXECUTABLE}")
message(STATUS "  Kernels: ${KERNEL_SOURCES}")
message(STATUS "  Output: ${SPIRV_OUTPUT_DIR}")