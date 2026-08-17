// Copyright 2025 SGLang Team. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by OpenMMLab: retain only the CUDA scalar aliases and error helpers
// required by Marlin-MoE; remove the TVM, DLPack, and ROCm dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;
using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
using fp8x2_e5m2_t = __nv_fp8x2_e5m2;
using fp32x4_t = float4;

#define SGLANG_LDG(arg) __ldg(arg)

namespace device {

#define SGL_DEVICE __forceinline__ __device__

template <typename T, typename U>
SGL_DEVICE constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

}  // namespace device

namespace host {

template <typename... Args>
[[noreturn]] inline void Panic(Args&&... args) {
  std::ostringstream os;
  if constexpr (sizeof...(Args) == 0) {
    os << "runtime check failed";
  } else {
    (os << ... << std::forward<Args>(args));
  }
  throw std::runtime_error(os.str());
}

template <typename Cond, typename... Args>
inline void RuntimeCheck(Cond condition, Args&&... args) {
  if (condition) {
    return;
  }
  Panic(std::forward<Args>(args)...);
}

inline void RuntimeDeviceCheck(cudaError_t error) {
  if (error != cudaSuccess) {
    Panic("CUDA error: ", cudaGetErrorString(error));
  }
}

}  // namespace host
