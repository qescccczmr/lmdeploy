# W4A16 Marlin-MoE third-party sources

The optional `_marlin_moe_w4a16` target uses Marlin-MoE CUDA headers
adapted by the SGLang project. The sources are licensed under Apache-2.0 and
must retain their original copyright and license blocks.

Pinned upstream revision:

- repository: `https://github.com/sgl-project/sglang`;
- revision: `7ce2352323b3cb7206143ea2ba9d7805e4c5d0b9`;
- source roots: `python/sglang/jit_kernel/csrc/gemm/marlin` and
  `python/sglang/jit_kernel/csrc/gemm/marlin_moe`;
- support headers: `python/sglang/jit_kernel/include/sgl_kernel`.

The following files are required below `third_party/sglang/`:

```text
gemm/marlin/dequant.h
gemm/marlin/marlin.cuh
gemm/marlin/marlin_dtypes.cuh
gemm/marlin_moe/kernel.h
gemm/marlin_moe/marlin_template.h
gemm/marlin_moe/moe_wna16_marlin.cuh
sgl_kernel/scalar_type.hpp
sgl_kernel/utils.cuh
```

Five compute headers are byte-identical to the pinned revision:
`dequant.h`, `marlin.cuh`, `marlin_dtypes.cuh`, `kernel.h`, and
`marlin_moe/marlin_template.h`. The remaining support adaptations are kept
small and explicit:

- `moe_wna16_marlin.cuh` removes the TVM tensor include and excludes only its
  TVM-FFI wrapper; the raw `marlin_mm` compute implementation is unchanged;
- `scalar_type.hpp` adds its missing direct `<tuple>` include;
- `utils.cuh` is a CUDA-only standalone subset containing the scalar aliases
  and error helpers required by these kernels, without TVM/DLPack/ROCm;
- no atomic-accumulation experiment or compute-template modification is
  carried by the vendored sources.

For auditing, a development build may point `MARLIN_MOE_W4A16_SOURCE_DIR` at
an include root with the same adapted layout. Normal source builds use only
the vendored directory and never search an ambient SGLang installation.

`route_align.cu` follows SGLang's `moe_align_block_size` output contract but
is an LMDeploy raw-pointer specialization for int64 route IDs and
caller-owned buffers. It replaces nondeterministic atomic route placement with
16 deterministic contiguous-route workers. Its source header preserves
SGLang's Apache-2.0 attribution.
