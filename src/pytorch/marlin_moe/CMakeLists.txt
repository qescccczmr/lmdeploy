# Copyright (c) OpenMMLab. All rights reserved.

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
  message(FATAL_ERROR
          "BUILD_MARLIN_MOE_W4A16 requires CUDA toolkit 12.0 or newer")
endif()

set(MARLIN_MOE_W4A16_SOURCE_DIR ""
    CACHE PATH
    "Optional external include root containing gemm/marlin_moe and sgl_kernel")

set(_marlin_vendored_include
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sglang")
if(MARLIN_MOE_W4A16_SOURCE_DIR)
  set(_marlin_include "${MARLIN_MOE_W4A16_SOURCE_DIR}")
else()
  set(_marlin_include "${_marlin_vendored_include}")
endif()

set(_marlin_required_headers
    gemm/marlin/dequant.h
    gemm/marlin/marlin.cuh
    gemm/marlin/marlin_dtypes.cuh
    gemm/marlin_moe/kernel.h
    gemm/marlin_moe/marlin_template.h
    gemm/marlin_moe/moe_wna16_marlin.cuh
    sgl_kernel/scalar_type.hpp
    sgl_kernel/utils.cuh)
foreach(_header IN LISTS _marlin_required_headers)
  if(NOT EXISTS "${_marlin_include}/${_header}")
    message(FATAL_ERROR
            "Missing ${_marlin_include}/${_header}. "
            "Vendor the pinned Apache-2.0 headers documented in "
            "src/pytorch/marlin_moe/THIRD_PARTY.md, or configure "
            "MARLIN_MOE_W4A16_SOURCE_DIR to an equivalent include root.")
  endif()
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    RESULT_VARIABLE _marlin_pybind11_result
    OUTPUT_VARIABLE _marlin_pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_marlin_pybind11_result EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${_marlin_pybind11_cmake_dir}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(FATAL_ERROR
            "pybind11 is required to build _marlin_moe_w4a16")
  endif()
endif()

add_library(marlin_moe_w4a16_kernels OBJECT kernels.cu route_align.cu)
set_target_properties(
  marlin_moe_w4a16_kernels PROPERTIES
  CUDA_ARCHITECTURES 90-real
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  marlin_moe_w4a16_kernels PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${_marlin_include}")
target_compile_options(
  marlin_moe_w4a16_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--prec-div=true;--prec-sqrt=true;--ftz=false>)

pybind11_add_module(
  _marlin_moe_w4a16 MODULE
  bindings.cpp
  $<TARGET_OBJECTS:marlin_moe_w4a16_kernels>)
set_target_properties(
  _marlin_moe_w4a16 PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_include_directories(
  _marlin_moe_w4a16 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(_marlin_moe_w4a16 PRIVATE CUDA::cudart)

if(CALL_FROM_SETUP_PY)
  string(REPLACE "." ";" _cuda_version ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET _cuda_version 0 _cuda_major)
  if(_cuda_major GREATER_EQUAL 13)
    set(_marlin_cuda_rpath "\$ORIGIN;\$ORIGIN/../../nvidia/cu${_cuda_major}/lib/")
  else()
    set(_marlin_cuda_rpath "\$ORIGIN;\$ORIGIN/../../nvidia/cuda_runtime/lib/")
  endif()
  set_target_properties(
    _marlin_moe_w4a16 PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "${_marlin_cuda_rpath}")
endif()
