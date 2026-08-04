# W4A16 Marlin-MoE AOT module

This directory builds the optional `_marlin_moe_w4a16` extension. It is
specialized for CUDA 12+, SM90, BF16 activations, Marlin uint4b8 weights, and
group size 32. It has no PyTorch C++, SGLang, TVM-FFI, or `sgl_kernel`
runtime dependency.

The default LMDeploy build does not enable this target. A source build using
the vendored headers is:

```bash
cmake -S . -B build -GNinja \
  -DBUILD_MARLIN_MOE_W4A16=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90-real
cmake --build build --target _marlin_moe_w4a16
```

The equivalent package-build switch is:

```bash
BUILD_MARLIN_MOE_W4A16=1 pip install -e . --no-build-isolation
```

This uses LMDeploy's CMake extension build, so `DISABLE_TURBOMIND` must not be
set for that installation.

`MARLIN_MOE_W4A16_SOURCE_DIR` is an optional development-only override for
auditing an equivalent adapted include tree.

## ABI

All CUDA buffers and streams are passed as integer addresses. The Python
adapter owns and validates tensor dtype, shape, stride, capacity, device, and
lifetime. Router IDs are trusted to satisfy the already-validated MoE expert
domain contract.

- `launch_safe(...)` accepts route block sizes 8, 16, 32, 48, and 64. It fixes
  Marlin to the non-atomic FP32 split-K reduction and never multiplies router
  weights. No atomic accumulation path is exported.
- `align_topk_i64_out(...)` converts contiguous int64 top-k IDs into Marlin's
  int32 sorted-route, expert-block, and padded-count layout. It explicitly
  receives caller-owned route scratch and cumsum buffers. Up to 65,536 routes
  use two kernels: 16 worker blocks count contiguous route intervals, then a
  512-thread block performs an expert-parallel prefix scan and stable scatter.
  Since each worker owns a contiguous route interval, routes within an expert
  retain their original index order and repeated launches are bitwise stable.
  Requests above that limit fail explicitly so the production backend can
  select its existing stable Triton fallback; this API never silently changes
  its numerical-repeatability contract. It allocates nothing, does not
  synchronize the host, and does not assume the default stream, so it is CUDA
  Graph replay safe.
- `max_aligned_routes(...)` returns the required conservative sorted-route
  capacity. Expert-ID capacity is
  `(capacity + block_size - 1) // block_size`.
- `deterministic_route_scratch_numel(num_experts)` returns the required int32
  route-scratch capacity (`16 * num_experts`). The int32 cumsum capacity is
  `num_experts + 1`.

The Python runtime keeps decode and prefill batches below 4,096 tokens on
block 8. Larger prefill batches select from the supported block sizes using
their average routed-token load per expert. Set
`LMDEPLOY_MARLIN_MOE_PREFILL_BLOCK_SIZE` to `8`, `16`, `32`, `48`, or `64` to
override prefill selection, or leave its default value `auto`; decode always
uses block 8 and ignores the prefill override.
- `uses_deterministic_route_alignment(num_routes, num_experts)` lets the
  backend fail closed to its stable Triton path when routes exceed 65,536 or
  experts exceed 512. The same limits are exported as module attributes.

The production Python adapter uses deterministic AOT alignment through 16,384
routes. Above that measured SM90 crossover it selects LMDeploy's existing
deterministic Triton alignment, even though the AOT ABI remains correct through
65,536 routes.

The AOT GEMM emits unweighted BF16 expert outputs. HF-compatible router
weighting and the top-k sum remain in LMDeploy's existing FP32 `moe_reduce`
path.
