# Copyright (c) OpenMMLab. All rights reserved.
"""Marlin layout helpers for compressed-tensors W4A16 MoE weights.

The layout follows the Apache-2.0 Marlin implementation adapted by Neural
Magic and SGLang.  Unlike the reference conversion, these kernels consume
LMDeploy's checkpoint layout directly, avoiding persistent transposed copies.
"""

import importlib
import importlib.machinery
import importlib.util
from functools import cache
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.language as tl

from .activation import silu_and_mul
from .fused_moe import moe_reduce, moe_reduce_add_fp32

_MARLIN_AOT_MODULE = '_marlin_moe_w4a16'
_MARLIN_AOT_ABI_VERSION = 3
_MARLIN_AOT_TARGET = 'sm90-bf16-u4b8-g32-marlin-moe'
_MARLIN_BLOCK_SIZE = 8
_MARLIN_MAX_THREAD_N = 256
_MARLIN_AOT_ALIGN_MAX_ROUTES = 16384


@triton.jit
def _repack_w4a16_marlin_kernel(
    src,
    dst,
    words_per_expert: tl.constexpr,
    out_features: tl.constexpr,
    packed_k: tl.constexpr,
    n_tiles: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Repack [E, N, K/8] GPTQ words into Marlin's tiled layout."""
    expert_id = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < words_per_expert

    # Each Marlin tile covers K=16 and N=64.  Its 128 output words are
    # ordered as `thread_id * 4 + warp_id` by the reference repack kernel.
    tile_linear = offsets // 128
    word_in_tile = offsets % 128
    k_tile = tile_linear // n_tiles
    n_tile = tile_linear % n_tiles
    thread_id = word_in_tile // 4
    warp_id = word_in_tile % 4

    tc_col = thread_id // 4
    tc_row = (thread_id % 4) * 2
    current_n = n_tile * 64 + warp_id * 16 + tc_col

    packed = tl.zeros((BLOCK, ), dtype=tl.uint32)
    for dst_nibble in tl.static_range(0, 8):
        # The reference first reads vals[0:4] from N, vals[4:8] from
        # N + 8, then packs vals[[0, 2, 4, 6, 1, 3, 5, 7]].
        source_value = (dst_nibble % 4) * 2 + dst_nibble // 4
        source_n = current_n + (source_value // 4) * 8
        source_slot = source_value % 4
        source_k = (k_tile * 16 + tc_row + (source_slot // 2) * 8 +
                    source_slot % 2)
        source_word_k = source_k // 8
        source_shift = (source_k % 8) * 4
        source_offset = (expert_id * out_features * packed_k +
                         source_n * packed_k + source_word_k)
        source_word = tl.load(src + source_offset, mask=mask,
                              other=0).to(tl.uint32)
        code = (source_word >> source_shift) & 0xF
        packed |= code << (dst_nibble * 4)

    tl.store(dst + expert_id * words_per_expert + offsets, packed, mask=mask)


@triton.jit
def _permute_w4a16_marlin_scale_kernel(
    src,
    dst,
    scales_per_expert: tl.constexpr,
    out_features: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Transpose each consecutive 8x8 block of [K/group, N] scales."""
    expert_id = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < scales_per_expert

    block_base = (offsets // 64) * 64
    position = offsets % 64
    source_flat = block_base + position // 8 + (position % 8) * 8
    source_group = source_flat // out_features
    source_n = source_flat % out_features
    source_offset = (expert_id * out_features * num_groups +
                     source_n * num_groups + source_group)
    value = tl.load(src + source_offset, mask=mask, other=0.0)
    tl.store(dst + expert_id * scales_per_expert + offsets, value, mask=mask)


def _normalize_expert_dim(tensor: torch.Tensor,
                          name: str) -> tuple[torch.Tensor, bool]:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0), True
    if tensor.dim() == 3:
        return tensor, False
    raise ValueError(
        f'Expected {name} with shape [N, K] or [E, N, K], got {tensor.shape}')


def _validate_packed_weight(
    weight_packed: torch.Tensor, ) -> tuple[torch.Tensor, bool, int, int, int]:
    weight_packed, squeeze_expert = _normalize_expert_dim(
        weight_packed, 'packed weight')
    if weight_packed.device.type != 'cuda':
        raise ValueError('Marlin repack requires CUDA tensors')
    if weight_packed.dtype != torch.int32:
        raise ValueError(
            f'Marlin repack requires int32 weights, got {weight_packed.dtype}')
    if not weight_packed.is_contiguous():
        raise ValueError('Marlin repack input must be contiguous')

    num_experts, out_features, packed_k = weight_packed.shape
    in_features = packed_k * 8
    if in_features % 16 != 0:
        raise ValueError(
            f'Marlin requires K divisible by 16, got {in_features}')
    if out_features % 64 != 0:
        raise ValueError(
            f'Marlin requires N divisible by 64, got {out_features}')
    return (weight_packed, squeeze_expert, num_experts, out_features,
            in_features)


def _validate_weight_scale(
    weight_scale: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, bool, int, int, int]:
    weight_scale, squeeze_expert = _normalize_expert_dim(
        weight_scale, 'weight scale')
    if weight_scale.device.type != 'cuda':
        raise ValueError('Marlin scale permutation requires CUDA tensors')
    if weight_scale.dtype != torch.bfloat16:
        raise ValueError(
            f'Marlin repack requires bfloat16 scales, got {weight_scale.dtype}'
        )
    if not weight_scale.is_contiguous():
        raise ValueError('Marlin scale permutation input must be contiguous')
    if group_size != 32:
        raise ValueError(
            f'Marlin W4A16 currently requires group_size=32, got {group_size}')

    num_experts, out_features, num_groups = weight_scale.shape
    in_features = num_groups * group_size
    if out_features % 64 != 0:
        raise ValueError(
            f'Marlin requires N divisible by 64, got {out_features}')
    return (weight_scale, squeeze_expert, num_experts, out_features,
            in_features)


def repack_w4a16_weight_for_marlin(
    weight_packed: torch.Tensor, ) -> torch.Tensor:
    """Repack one or more checkpoint-layout INT4 expert projections."""
    (weight_packed, squeeze_expert, num_experts, out_features,
     in_features) = _validate_packed_weight(weight_packed)

    repacked = torch.empty(
        (num_experts, in_features // 16, out_features * 2),
        dtype=torch.int32,
        device=weight_packed.device,
    )
    words_per_expert = weight_packed[0].numel()
    block = 256
    _repack_w4a16_marlin_kernel[(num_experts,
                                 triton.cdiv(words_per_expert, block))](
                                     weight_packed,
                                     repacked,
                                     words_per_expert=words_per_expert,
                                     out_features=out_features,
                                     packed_k=in_features // 8,
                                     n_tiles=out_features // 64,
                                     BLOCK=block)
    return repacked[0] if squeeze_expert else repacked


def permute_w4a16_scale_for_marlin(
    weight_scale: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """Permute one or more checkpoint-layout INT4 scale projections."""
    (weight_scale, squeeze_expert, num_experts, out_features,
     in_features) = _validate_weight_scale(weight_scale, group_size)

    permuted_scale = torch.empty(
        (num_experts, in_features // group_size, out_features),
        dtype=weight_scale.dtype,
        device=weight_scale.device,
    )
    scales_per_expert = weight_scale[0].numel()
    block = 256
    _permute_w4a16_marlin_scale_kernel[(
        num_experts,
        triton.cdiv(scales_per_expert, block),
    )](
        weight_scale,
        permuted_scale,
        scales_per_expert=scales_per_expert,
        out_features=out_features,
        num_groups=in_features // group_size,
        BLOCK=block,
    )
    return permuted_scale[0] if squeeze_expert else permuted_scale


def repack_w4a16_for_marlin(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert LMDeploy checkpoint-layout W4A16 tensors to Marlin layout.

    Args:
        weight_packed: INT32 tensor shaped ``[E, N, K / 8]``.
        weight_scale: BF16 tensor shaped ``[E, N, K / group_size]``.
        group_size: Quantization group size. Only 32 is currently supported.

    Returns:
        Repacked weight ``[E, K / 16, N * 2]`` and permuted scale
        ``[E, K / 32, N]``. The returned tensors have the same respective
        element counts as their inputs.
    """
    (packed, packed_squeeze, packed_experts, packed_n,
     packed_k) = _validate_packed_weight(weight_packed)
    (scale, scale_squeeze, scale_experts, scale_n,
     scale_k) = _validate_weight_scale(weight_scale, group_size)
    if packed.device != scale.device:
        raise ValueError(
            'Packed weights and scales must be on the same device')
    if ((packed_experts, packed_n, packed_k)
            != (scale_experts, scale_n, scale_k)
            or packed_squeeze != scale_squeeze):
        raise ValueError(
            'Packed weights and scales must describe matching expert projections'
        )
    return (
        repack_w4a16_weight_for_marlin(weight_packed),
        permute_w4a16_scale_for_marlin(weight_scale, group_size),
    )


def _validate_marlin_aot_module(module: ModuleType) -> ModuleType:
    abi_version = getattr(module, 'ABI_VERSION', None)
    if abi_version != _MARLIN_AOT_ABI_VERSION:
        raise RuntimeError(f'{_MARLIN_AOT_MODULE} ABI_VERSION must be '
                           f'{_MARLIN_AOT_ABI_VERSION}, got {abi_version!r}')
    target = getattr(module, 'TARGET', None)
    if target != _MARLIN_AOT_TARGET:
        raise RuntimeError(f'{_MARLIN_AOT_MODULE} TARGET must be '
                           f'{_MARLIN_AOT_TARGET!r}, got {target!r}')
    required_exports = (
        'launch_safe',
        'align_topk_i64_out',
        'max_aligned_routes',
        'deterministic_route_scratch_numel',
        'uses_deterministic_route_alignment',
    )
    missing_exports = [
        name for name in required_exports
        if not callable(getattr(module, name, None))
    ]
    if missing_exports:
        raise RuntimeError(
            f'{_MARLIN_AOT_MODULE} does not export '
            f'{", ".join(missing_exports)}')
    return module


def _load_marlin_aot_from_file(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(_MARLIN_AOT_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load Marlin AOT module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def _load_marlin_aot() -> tuple[ModuleType | None, str | None]:
    """Load the standalone Marlin extension without importing SGLang."""
    errors = []
    for module_name in ('lmdeploy.lib.' + _MARLIN_AOT_MODULE,
                        _MARLIN_AOT_MODULE):
        try:
            module = importlib.import_module(module_name)
            return _validate_marlin_aot_module(module), None
        except (ImportError, OSError, RuntimeError) as error:
            errors.append(f'{module_name}: {error}')

    lib_dir = Path(__file__).resolve().parents[3] / 'lib'
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = lib_dir / (_MARLIN_AOT_MODULE + suffix)
        if not candidate.is_file():
            continue
        try:
            module = _load_marlin_aot_from_file(candidate)
            return _validate_marlin_aot_module(module), None
        except (ImportError, OSError, RuntimeError) as error:
            errors.append(f'{candidate}: {error}')
    return None, '; '.join(errors)


def is_marlin_moe_w4a16_available() -> bool:
    """Whether the standalone Marlin-MoE module can run on this device."""
    module, _ = _load_marlin_aot()
    if module is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (9, 0)


@triton.jit
def _align_count_kernel(
    TopKIds,
    TokenCounts,
    NUM_EXPERTS: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    ROUTES_PER_PROGRAM: tl.constexpr,
):
    """Count one contiguous route chunk per program and expert."""
    pid = tl.program_id(0)
    route_start = pid * ROUTES_PER_PROGRAM
    row_offset = (pid + 1) * NUM_EXPERTS
    for route_offset in range(ROUTES_PER_PROGRAM):
        route_id = route_start + route_offset
        if route_id < NUM_ROUTES:
            expert_id = tl.load(TopKIds + route_id)
            count = tl.load(TokenCounts + row_offset + expert_id)
            tl.store(TokenCounts + row_offset + expert_id, count + 1)


@triton.jit
def _align_chunk_prefix_kernel(
    TokenCounts,
    NUM_EXPERTS: tl.constexpr,
):
    """Compute route-count prefixes across chunks for one expert."""
    expert_id = tl.program_id(0)
    count = 0
    for row in range(1, NUM_EXPERTS + 1):
        offset = row * NUM_EXPERTS + expert_id
        count += tl.load(TokenCounts + offset)
        tl.store(TokenCounts + offset, count)


@triton.jit
def _align_expert_prefix_kernel(
    TokenCounts,
    Cumsum,
    NumTokensPostPadded,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the padded exclusive prefix over experts."""
    cumulative = 0
    tl.store(Cumsum, 0)
    last_row_offset = NUM_EXPERTS * NUM_EXPERTS
    for expert_id in range(NUM_EXPERTS):
        count = tl.load(TokenCounts + last_row_offset + expert_id)
        cumulative += tl.cdiv(count, BLOCK_SIZE) * BLOCK_SIZE
        tl.store(Cumsum + expert_id + 1, cumulative)
    tl.store(NumTokensPostPadded, cumulative)


@triton.jit
def _align_scatter_kernel(
    TopKIds,
    SortedTokenIds,
    ExpertIds,
    TokenCounts,
    Cumsum,
    NUM_EXPERTS: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    ROUTES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Write padded expert blocks and scatter the original route ids."""
    pid = tl.program_id(0)
    expert_start = tl.load(Cumsum + pid)
    expert_end = tl.load(Cumsum + pid + 1)
    for block_start in range(expert_start, expert_end, BLOCK_SIZE):
        tl.store(ExpertIds + block_start // BLOCK_SIZE, pid)

    route_start = pid * ROUTES_PER_PROGRAM
    prefix_row_offset = pid * NUM_EXPERTS
    for route_id in range(
            route_start,
            tl.minimum(route_start + ROUTES_PER_PROGRAM, NUM_ROUTES)):
        expert_id = tl.load(TopKIds + route_id)
        count_offset = prefix_row_offset + expert_id
        rank = tl.load(TokenCounts + count_offset)
        sorted_offset = rank + tl.load(Cumsum + expert_id)
        tl.store(SortedTokenIds + sorted_offset, route_id)
        tl.store(TokenCounts + count_offset, rank + 1)


class MarlinMoEWorkspace:
    """Caller-owned buffers for graph-safe Marlin W4A16 MoE execution.

    Routing data is recomputed on every call.  Only capacities and temporary
    tensors are cached, so changing expert ids during CUDA Graph replay is
    supported.
    """

    def __init__(
        self,
        max_tokens: int,
        topk: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device | str,
        block_size: int = _MARLIN_BLOCK_SIZE,
    ) -> None:
        if min(max_tokens, topk, num_experts, hidden_size,
               intermediate_size) <= 0:
            raise ValueError('Marlin workspace dimensions must be positive')
        if block_size != _MARLIN_BLOCK_SIZE:
            raise ValueError(
                f'The current Marlin AOT requires block_size=8, got '
                f'{block_size}')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('Marlin workspace requires a CUDA device')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())

        self.max_tokens = max_tokens
        self.topk = topk
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = device
        self.block_size = block_size
        self.max_routes = max_tokens * topk

        # Include the virtual padding expert used by the alignment contract.
        align_experts = num_experts + 1
        if self.max_routes < align_experts:
            sorted_capacity = self.max_routes * block_size
        else:
            sorted_capacity = (self.max_routes + align_experts *
                               (block_size - 1))
        max_blocks = triton.cdiv(sorted_capacity, block_size)
        self.sorted_token_ids = torch.empty(sorted_capacity,
                                            dtype=torch.int32,
                                            device=device)
        self.expert_ids = torch.empty(max_blocks,
                                      dtype=torch.int32,
                                      device=device)
        self.num_tokens_post_padded = torch.empty(1,
                                                  dtype=torch.int32,
                                                  device=device)
        module, load_error = _load_marlin_aot()
        if module is None:
            raise RuntimeError(
                f'{_MARLIN_AOT_MODULE} is unavailable: '
                f'{load_error or "not found"}')
        route_scratch_numel = module.deterministic_route_scratch_numel(
            num_experts)
        self.route_scratch = torch.empty(route_scratch_numel,
                                         dtype=torch.int32,
                                         device=device)
        self.token_counts = torch.empty((align_experts + 1) * align_experts,
                                        dtype=torch.int32,
                                        device=device)
        self.cumsum = torch.empty(align_experts + 1,
                                  dtype=torch.int32,
                                  device=device)

        properties = torch.cuda.get_device_properties(device)
        sm_count = properties.multi_processor_count
        max_output = max(2 * intermediate_size, hidden_size)
        lock_count = min((max_output // 64) * max_blocks, sm_count * 4)
        self.locks = torch.zeros(max(1, lock_count),
                                 dtype=torch.int32,
                                 device=device)

        def _scratch_numel(out_features: int) -> int:
            numel = min(out_features * sorted_capacity,
                        sm_count * 4 * block_size * _MARLIN_MAX_THREAD_N)
            # The block-8 Marlin specialization launches two reduction
            # groups and therefore requires twice the ordinary scratch.
            return numel * 2

        scratch_numel = max(_scratch_numel(2 * intermediate_size),
                            _scratch_numel(hidden_size))
        self.fp32_scratch = torch.empty(max(1, scratch_numel),
                                        dtype=torch.float32,
                                        device=device)
        self.gate_up = torch.empty((self.max_routes, 2 * intermediate_size),
                                   dtype=torch.bfloat16,
                                   device=device)
        self.activation = torch.empty((self.max_routes, intermediate_size),
                                      dtype=torch.bfloat16,
                                      device=device)
        self.down = torch.empty((self.max_routes, hidden_size),
                                dtype=torch.bfloat16,
                                device=device)

    def workspace_for_tokens(self, tokens: int) -> 'MarlinMoEWorkspace':
        """Return fixed graph buffers or an eager-only temporary workspace.

        The buffers owned by ``self`` may already be referenced by a captured
        CUDA Graph, so they must never be replaced in place. Long prefill
        steps use a temporary workspace that is released after the result has
        been reduced; decode keeps the fixed, graph-replay-safe allocation.
        """
        if tokens <= self.max_tokens:
            return self
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f'Marlin CUDA Graph workspace supports at most '
                f'{self.max_tokens} tokens, got {tokens}. Increase the '
                'configured maximum batch size before graph capture.')
        return type(self)(
            max_tokens=tokens,
            topk=self.topk,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            device=self.device,
            block_size=self.block_size,
        )

    def validate(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor,
                 topk_weights: torch.Tensor) -> None:
        tokens = hidden_states.size(0)
        if tokens > self.max_tokens:
            raise ValueError(
                f'Marlin workspace capacity is {self.max_tokens} tokens, '
                f'got {tokens}')
        if topk_ids.size(1) != self.topk:
            raise ValueError(f'Marlin workspace topk={self.topk}, got '
                             f'{topk_ids.size(1)}')
        if hidden_states.size(1) != self.hidden_size:
            raise ValueError(
                f'Marlin workspace hidden_size={self.hidden_size}, got '
                f'{hidden_states.size(1)}')
        tensors = (hidden_states, topk_ids, topk_weights)
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError(
                'Marlin inputs and workspace must be on the same device')


def _align_topk_out(module: ModuleType, topk_ids: torch.Tensor,
                    workspace: MarlinMoEWorkspace) -> None:
    """Align routes into caller-owned padded buffers without a host sync."""
    num_routes = topk_ids.numel()
    # The AOT path retains original route order within every expert. Decode
    # and ordinary small/medium prefills use it; beyond the measured crossover
    # the existing deterministic Triton path is faster. No nondeterministic
    # route-scatter implementation is exposed to production.
    if (topk_ids.dtype == torch.int64
            and num_routes <= _MARLIN_AOT_ALIGN_MAX_ROUTES
            and module.uses_deterministic_route_alignment(
                num_routes, workspace.num_experts)):
        stream = torch.cuda.current_stream(topk_ids.device).cuda_stream
        module.align_topk_i64_out(
            topk_ids.data_ptr(),
            workspace.sorted_token_ids.data_ptr(),
            workspace.expert_ids.data_ptr(),
            workspace.num_tokens_post_padded.data_ptr(),
            workspace.route_scratch.data_ptr(),
            workspace.cumsum.data_ptr(),
            num_routes,
            workspace.num_experts,
            workspace.block_size,
            workspace.sorted_token_ids.numel(),
            workspace.expert_ids.numel(),
            workspace.route_scratch.numel(),
            workspace.cumsum.numel(),
            topk_ids.device.index,
            stream,
        )
        return

    # Also covers int32 callers without casting or allocating another tensor.
    align_experts = workspace.num_experts + 1
    routes_per_program = triton.cdiv(num_routes, align_experts)
    workspace.sorted_token_ids.fill_(num_routes)
    workspace.token_counts.zero_()
    workspace.cumsum.zero_()
    _align_count_kernel[(align_experts, )](
        topk_ids,
        workspace.token_counts,
        NUM_EXPERTS=align_experts,
        NUM_ROUTES=num_routes,
        ROUTES_PER_PROGRAM=routes_per_program,
    )
    _align_chunk_prefix_kernel[(align_experts, )](
        workspace.token_counts,
        NUM_EXPERTS=align_experts,
    )
    _align_expert_prefix_kernel[(1, )](
        workspace.token_counts,
        workspace.cumsum,
        workspace.num_tokens_post_padded,
        NUM_EXPERTS=align_experts,
        BLOCK_SIZE=workspace.block_size,
    )
    _align_scatter_kernel[(align_experts, )](
        topk_ids,
        workspace.sorted_token_ids,
        workspace.expert_ids,
        workspace.token_counts,
        workspace.cumsum,
        NUM_EXPERTS=align_experts,
        NUM_ROUTES=num_routes,
        ROUTES_PER_PROGRAM=routes_per_program,
        BLOCK_SIZE=workspace.block_size,
    )


def _validate_marlin_projection(
    packed: torch.Tensor,
    scale: torch.Tensor,
    name: str,
) -> tuple[int, int, int]:
    if packed.dim() != 3 or scale.dim() != 3:
        raise ValueError(f'{name} Marlin weight and scale must both be 3D')
    if packed.device.type != 'cuda' or packed.dtype != torch.int32:
        raise ValueError(f'{name} Marlin weight must be a CUDA int32 tensor')
    if scale.device.type != 'cuda' or scale.dtype != torch.bfloat16:
        raise ValueError(f'{name} Marlin scale must be a CUDA bfloat16 tensor')
    if not packed.is_contiguous() or not scale.is_contiguous():
        raise ValueError(f'{name} Marlin tensors must be contiguous')
    num_experts, packed_k, packed_n = packed.shape
    if packed_n % 2 != 0:
        raise ValueError(f'{name} Marlin packed N dimension must be even')
    in_features = packed_k * 16
    out_features = packed_n // 2
    expected_scale = (num_experts, in_features // 32, out_features)
    if scale.shape != expected_scale:
        raise ValueError(
            f'{name} Marlin scale must have shape {expected_scale}, got '
            f'{tuple(scale.shape)}')
    if in_features % 32 != 0 or out_features % 64 != 0:
        raise ValueError(f'{name} Marlin requires K%32=0 and N%64=0, got '
                         f'K={in_features}, N={out_features}')
    return num_experts, in_features, out_features


def _validate_marlin_moe_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    workspace: MarlinMoEWorkspace,
) -> tuple[int, int]:
    if hidden_states.dim() != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError('Marlin hidden_states must be a 2D bfloat16 tensor')
    if hidden_states.device.type != 'cuda' or not hidden_states.is_contiguous(
    ):
        raise ValueError('Marlin hidden_states must be contiguous on CUDA')
    if topk_ids.dim() != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Marlin topk_ids must be a 2D integer tensor')
    if topk_weights.shape != topk_ids.shape or topk_weights.dtype != torch.float32:
        raise ValueError(
            'Marlin topk_weights must be contiguous FP32 with the same '
            'shape as topk_ids')
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError('Marlin routing tensors must be contiguous')
    if topk_ids.size(0) != hidden_states.size(0):
        raise ValueError('Marlin routing token count must match activations')

    gate_experts, gate_k, gate_n = _validate_marlin_projection(
        gate_up_packed, gate_up_scale, 'gate-up')
    down_experts, down_k, down_n = _validate_marlin_projection(
        down_packed, down_scale, 'down')
    if gate_experts != down_experts or gate_experts != workspace.num_experts:
        raise ValueError(
            'Marlin projections and workspace expert counts differ')
    if topk_ids.size(1) > gate_experts:
        raise ValueError('Marlin topk cannot exceed the expert count')
    if gate_k != hidden_states.size(1) or gate_n % 2 != 0:
        raise ValueError('Marlin gate-up dimensions do not match activations')
    intermediate_size = gate_n // 2
    if (down_k, down_n) != (intermediate_size, hidden_states.size(1)):
        raise ValueError('Marlin down dimensions do not match gate-up output')
    if (workspace.intermediate_size,
            workspace.hidden_size) != (intermediate_size,
                                       hidden_states.size(1)):
        raise ValueError('Marlin projection dimensions do not match workspace')

    device = hidden_states.device
    weight_tensors = (gate_up_packed, gate_up_scale, down_packed, down_scale)
    if any(tensor.device != device for tensor in weight_tensors):
        raise ValueError('All Marlin tensors must be on the same CUDA device')
    workspace.validate(hidden_states, topk_ids, topk_weights)
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise ValueError(
            f'The current Marlin AOT requires SM90, got sm{capability[0]}'
            f'{capability[1]}')
    return gate_n, intermediate_size


def _launch_marlin_projection(
    module: ModuleType,
    activation: torch.Tensor,
    output: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    workspace: MarlinMoEWorkspace,
    size_m: int,
    size_n: int,
    size_k: int,
    topk: int,
) -> None:
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    module.launch_safe(
        activation.data_ptr(),
        output.data_ptr(),
        workspace.fp32_scratch.data_ptr(),
        packed.data_ptr(),
        scale.data_ptr(),
        workspace.locks.data_ptr(),
        workspace.sorted_token_ids.data_ptr(),
        workspace.expert_ids.data_ptr(),
        workspace.num_tokens_post_padded.data_ptr(),
        workspace.block_size,
        topk,
        size_m,
        size_n,
        size_k,
        activation.device.index,
        stream,
    )


def marlin_moe_w4a16(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    workspace: MarlinMoEWorkspace,
    shared_addend: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the accuracy-preserving Marlin W4A16 routed-MoE chain.

    Both Marlin projections use deterministic non-atomic FP32 split-K
    reduction.  W2 deliberately leaves router weights unapplied; LMDeploy's
    FP32 combine below matches the HF reference accumulation contract.
    """
    if not isinstance(workspace, MarlinMoEWorkspace):
        raise TypeError('workspace must be a MarlinMoEWorkspace')
    module, load_error = _load_marlin_aot()
    if module is None:
        raise RuntimeError(
            f'{_MARLIN_AOT_MODULE} is unavailable: {load_error or "not found"}'
        )
    if shared_addend is not None:
        if (shared_addend.shape != hidden_states.shape
                or shared_addend.dtype != torch.bfloat16
                or shared_addend.device != hidden_states.device
                or not shared_addend.is_contiguous()):
            raise ValueError(
                'Marlin fused shared addend must be contiguous BF16 and '
                'match hidden_states')
    workspace = workspace.workspace_for_tokens(hidden_states.size(0))
    gate_n, intermediate_size = _validate_marlin_moe_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        gate_up_packed,
        gate_up_scale,
        down_packed,
        down_scale,
        workspace,
    )
    num_tokens = hidden_states.size(0)
    topk = topk_ids.size(1)
    num_routes = num_tokens * topk
    if num_tokens == 0:
        if shared_addend is not None:
            return hidden_states.new_empty(
                (0, hidden_states.size(1)), dtype=torch.float32)
        return hidden_states.new_empty((0, hidden_states.size(1)))

    gate_up = workspace.gate_up[:num_routes]
    activated = workspace.activation[:num_routes]
    expert_output = workspace.down[:num_routes]
    with torch.cuda.device(hidden_states.device):
        _align_topk_out(module, topk_ids, workspace)
        _launch_marlin_projection(
            module,
            hidden_states,
            gate_up,
            gate_up_packed,
            gate_up_scale,
            workspace,
            size_m=num_tokens,
            size_n=gate_n,
            size_k=hidden_states.size(1),
            topk=topk,
        )
        silu_and_mul(gate_up, out=activated)
        _launch_marlin_projection(
            module,
            activated,
            expert_output,
            down_packed,
            down_scale,
            workspace,
            size_m=num_routes,
            size_n=hidden_states.size(1),
            size_k=intermediate_size,
            topk=1,
        )
        expert_output = expert_output.view(
            num_tokens, topk, hidden_states.size(1))
        if shared_addend is not None:
            return moe_reduce_add_fp32(
                expert_output,
                topk_weights,
                shared_addend,
            )
        return moe_reduce(expert_output, topk_weights, fp32_acc=True)


__all__ = [
    'MarlinMoEWorkspace',
    'is_marlin_moe_w4a16_available',
    'marlin_moe_w4a16',
    'permute_w4a16_scale_for_marlin',
    'repack_w4a16_for_marlin',
    'repack_w4a16_weight_for_marlin',
]
