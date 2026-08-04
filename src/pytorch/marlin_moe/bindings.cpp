// Copyright (c) OpenMMLab. All rights reserved.
#include "kernels.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_marlin_moe_w4a16, module) {
  module.doc() =
      "SM90 BF16/uint4b8/group32 Marlin-MoE raw-pointer kernels";
  module.def(
      "launch_safe",
      &launch_marlin_moe_bf16_u4b8_g32,
      py::arg("a_ptr"),
      py::arg("output_bf16_ptr"),
      py::arg("scratch_fp32_ptr"),
      py::arg("packed_weight_ptr"),
      py::arg("scale_ptr"),
      py::arg("workspace_ptr"),
      py::arg("sorted_token_ids_ptr"),
      py::arg("expert_ids_ptr"),
      py::arg("num_tokens_post_padded_ptr"),
      py::arg("moe_block_size"),
      py::arg("top_k"),
      py::arg("size_m"),
      py::arg("size_n"),
      py::arg("size_k"),
      py::arg("device"),
      py::arg("stream_ptr"),
      py::call_guard<py::gil_scoped_release>());
  module.def(
      "align_topk_i64_out",
      &launch_align_topk_i64_out,
      py::arg("topk_ids_ptr"),
      py::arg("sorted_token_ids_ptr"),
      py::arg("expert_ids_ptr"),
      py::arg("num_tokens_post_padded_ptr"),
      py::arg("route_scratch_ptr"),
      py::arg("cumsum_buffer_ptr"),
      py::arg("num_routes"),
      py::arg("num_experts"),
      py::arg("block_size"),
      py::arg("sorted_token_ids_capacity"),
      py::arg("expert_ids_capacity"),
      py::arg("route_scratch_capacity"),
      py::arg("cumsum_buffer_capacity"),
      py::arg("device"),
      py::arg("stream_ptr"),
      py::call_guard<py::gil_scoped_release>());
  module.def(
      "max_aligned_routes",
      &max_aligned_routes,
      py::arg("num_routes"),
      py::arg("num_experts"),
      py::arg("block_size"));
  module.def(
      "deterministic_route_scratch_numel",
      &deterministic_route_scratch_numel,
      py::arg("num_experts"));
  module.def(
      "uses_deterministic_route_alignment",
      &uses_deterministic_route_alignment,
      py::arg("num_routes"),
      py::arg("num_experts"));
  module.attr("ABI_VERSION") = 3;
  module.attr("ROUTE_ALIGN_WORKERS") = 16;
  module.attr("DETERMINISTIC_ROUTE_LIMIT") = 65536;
  module.attr("MAX_DETERMINISTIC_EXPERTS") = 512;
  module.attr("TARGET") = "sm90-bf16-u4b8-g32-marlin-moe";
}
