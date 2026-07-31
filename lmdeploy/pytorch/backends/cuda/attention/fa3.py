# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl, TritonAttentionMetadata, _cdiv

logger = get_logger('lmdeploy')

FA3_MLA_MAX_BATCH_SIZE = 8


class FA3AbsorbedMLAImpl(TritonAttentionImpl):
    """FA3 decoding for absorbed multi-head latent attention.

    The absorbed MLA query contains the latent query followed by its RoPE
    component, while the shared key cache contains the latent KV followed by
    the RoPE key. FA3 exposes this layout through its ``qv`` interface and can
    avoid reading the 512-dimensional latent cache twice during decoding.

    Prefill intentionally remains on :class:`TritonAttentionImpl`; this class
    only replaces the strictly compatible, single-token decoding path.
    """

    _LATENT_HEAD_SIZE = 512
    _ROPE_HEAD_SIZE = 64
    _HEAD_SIZE = _LATENT_HEAD_SIZE + _ROPE_HEAD_SIZE
    _MAX_DECODE_BATCH_SIZE = FA3_MLA_MAX_BATCH_SIZE

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: tuple = None,
        logit_softcapping: float = 0.0,
        causal: bool = True,
        **kwargs,
    ):
        if head_size != self._HEAD_SIZE:
            raise ValueError(f'absorbed MLA FA3 requires head_size={self._HEAD_SIZE}, got {head_size}')
        if v_head_size != self._LATENT_HEAD_SIZE:
            raise ValueError(
                f'absorbed MLA FA3 requires v_head_size={self._LATENT_HEAD_SIZE}, got {v_head_size}')
        if num_kv_heads != 1:
            raise ValueError(f'absorbed MLA FA3 requires num_kv_heads=1, got {num_kv_heads}')
        if alibi:
            raise ValueError('absorbed MLA FA3 does not support alibi')

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            causal=causal,
            **kwargs,
        )
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_with_kvcache
        self.flash_attn_with_kvcache_v3 = flash_attn_with_kvcache

    def _can_run_absorbed_mla(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
    ) -> bool:
        """Check the runtime contract before entering the specialized kernel."""
        if max_q_seqlen != 1 or attn_metadata.quant_policy != QuantPolicy.NONE:
            return False
        if k_scales_zeros is not None or v_scales_zeros is not None or learnable_sink is not None:
            return False
        if query.ndim != 3 or query.size(-1) != self._HEAD_SIZE:
            return False
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            return False
        if k_cache.size(-2) != 1 or v_cache.size(-2) != 1:
            return False
        if k_cache.size(-1) != self._HEAD_SIZE or v_cache.size(-1) != self._LATENT_HEAD_SIZE:
            return False
        if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
            return False
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False
        if query.dtype != k_cache.dtype or query.dtype != v_cache.dtype:
            return False
        batch_size = attn_metadata.kv_seqlens.size(0)
        return batch_size <= self._MAX_DECODE_BATCH_SIZE and query.size(0) == batch_size

    def _forward_decoding(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run single-token absorbed MLA decoding with FA3."""
        if not self._can_run_absorbed_mla(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros,
                v_scales_zeros,
                learnable_sink,
        ):
            return super()._forward_decoding(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                learnable_sink=learnable_sink,
            )

        batch_size = attn_metadata.kv_seqlens.size(0)
        q_nope, q_rope = query.split(
            [self._LATENT_HEAD_SIZE, self._ROPE_HEAD_SIZE],
            dim=-1,
        )
        q_nope = q_nope.unflatten(0, (batch_size, max_q_seqlen))
        q_rope = q_rope.unflatten(0, (batch_size, max_q_seqlen))
        k_rope_cache = k_cache[..., self._LATENT_HEAD_SIZE:]

        output = self.flash_attn_with_kvcache_v3(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=v_cache,
            qv=q_nope,
            cache_seqlens=attn_metadata.kv_seqlens.to(torch.int32),
            page_table=attn_metadata.block_offsets,
            max_seqlen_q=max_q_seqlen,
            softmax_scale=self.scale,
            causal=self.causal,
            window_size=(-1, -1),
            softcap=max(self.logit_softcapping, 0.0),
            scheduler_metadata=attn_metadata.scheduler_metadata,
            num_splits=0,
        )
        return output.flatten(0, 1)


class FA3Impl(TritonAttentionImpl):
    """Flash Attention 3 implementation.

    This implementation leverages Flash Attention 3's optimized kernels for both
    prefill and decoding stages. FA3 provides significant performance improvements
    on Ampere and above (SM80+) with CUDA >= 12.3.

    Key features:
    - Optimized prefill using flash_attn_varlen_func
    - Speculative decoding support with multi-token queries
    - Standard single-token decoding with paged attention
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: tuple = None,
        logit_softcapping: float = 0.0,
        causal: bool = True,
        **kwargs,
    ):
        assert alibi is False, 'alibi not supported for FA3'
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            causal=causal,
            **kwargs,
        )
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
        self.flash_attn_varlen_func_v3 = flash_attn_varlen_func
        self.flash_attn_with_kvcache_v3 = flash_attn_with_kvcache

    def _get_max_q_seqlen(
        self,
        query: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> int:
        """Get max q seqlen."""
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        if attn_metadata.is_decoding:
            batch_size = attn_metadata.q_seqlens.size(0)
            max_q_seqlen = max_q_seqlen // batch_size
        return max_q_seqlen

    def _normalize_sliding_window(self, sliding_window):
        """Normalize sliding window to tuple format.

        Args:
            sliding_window: Sliding window size (None, int, or tuple).

        Returns:
            Tuple of (left_window, right_window) or (-1, -1) if None.
        """
        if sliding_window is None:
            return (-1, -1)
        if isinstance(sliding_window, int):
            return (sliding_window, sliding_window)
        return sliding_window

    def _decoding_speculative(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
    ) -> torch.Tensor:
        """Speculative decoding with multi-token queries.

        This path handles speculative decoding where multiple tokens are generated
        in parallel (max_q_seqlen > 1). Uses FA3's flash_attn_with_kvcache for
        efficient batched computation.

        Args:
            query: Query tensor to unflatten.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length (> 1).

        Returns:
            Attention output tensor.
        """
        quant_policy = attn_metadata.quant_policy

        # TurboQuant stores packed uint8 data in cache, which FA3's native
        # flash_attn_with_kvcache cannot dequantize directly.
        if quant_policy == QuantPolicy.TURBO_QUANT:
            raise NotImplementedError(
                'quant_policy=QuantPolicy.TURBO_QUANT is not supported with '
                'FA3 speculative decoding (max_q_seqlen > 1). '
                'FA3 speculative decoding accesses raw KV cache directly '
                'and cannot dequantize TurboQuant packed data. '
                'Use standard decoding (max_q_seqlen=1).'
            )

        block_offsets = attn_metadata.block_offsets
        sliding_window = self._normalize_sliding_window(self.sliding_window)

        # Reshape query for batched processing
        query = query.unflatten(0, (-1, max_q_seqlen))

        attn_output = self.flash_attn_with_kvcache_v3(
            query,
            k_cache,
            v_cache,
            cache_seqlens=attn_metadata.kv_seqlens.to(torch.int32),
            max_seqlen_q=max_q_seqlen,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            page_table=block_offsets,
            softmax_scale=self.scale,
            causal=self.causal,
            window_size=sliding_window,
            softcap=self.logit_softcapping,
        )
        return attn_output

    def _decoding_standard(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ) -> torch.Tensor:
        """Standard single-token decoding.

        This path handles standard decoding where only one token is generated
        per request (max_q_seqlen = 1). Uses paged attention for memory efficiency.

        Args:
            query: Query tensor (single token per request).
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length (= 1).
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.

        Returns:
            Attention output tensor.
        """
        block_offsets = attn_metadata.block_offsets
        quant_policy = attn_metadata.quant_policy

        attn_output = self.paged_attention_fwd(
            query,
            k_cache,
            v_cache,
            cache_seqlens=attn_metadata.kv_seqlens,
            page_table=block_offsets,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            max_seqlen_q=max_q_seqlen,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            softmax_scale=self.scale,
            causal=self.causal,
            softcap=self.logit_softcapping,
            window_size=self.sliding_window,
            # custom args
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            quant_policy=quant_policy,
        )
        return attn_output

    def _forward_decoding(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for decoding stage.

        Supports two decoding modes:
        1. Speculative decoding: Multiple tokens (max_q_seqlen > 1)
        2. Standard decoding: Single token (max_q_seqlen = 1)

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.

        Returns:
            Attention output tensor.
        """
        if max_q_seqlen > 1:
            return self._decoding_speculative(query, k_cache, v_cache, attn_metadata, max_q_seqlen)
        return self._decoding_standard(
            query,
            k_cache,
            v_cache,
            attn_metadata,
            max_q_seqlen,
            k_scales_zeros,
            v_scales_zeros,
        )

    def _forward_prefill(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for prefill stage.

        Uses FA3's flash_attn_varlen_func for efficient variable-length attention
        computation during the prefill phase.

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.

        Returns:
            Attention output tensor.
        """
        block_offsets = attn_metadata.block_offsets
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy

        # Flatten KV cache for varlen attention
        block_size = k_cache.size(1)
        out_size = _cdiv(kv_flatten_size, block_size) * block_size + block_size
        flatten_k, flatten_v = self.flatten_kv_cache(
            k_cache,
            v_cache,
            kv_seqlens,
            block_offsets,
            start_loc=kv_start_loc,
            out_size=out_size,
            out_dtype=query.dtype,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            quant_policy=quant_policy,
            flatten_kv_layout='shd',
        )

        sliding_window = self._normalize_sliding_window(self.sliding_window)

        # For TurboQuant, flattened K/V are in rotated domain.
        # Rotate Q to match, and inverse-rotate output afterwards.
        if quant_policy == QuantPolicy.TURBO_QUANT:
            from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
                hadamard_rotate,
                hadamard_rotate_inv,
            )
            query = hadamard_rotate(query)

        attn_output = self.flash_attn_varlen_func_v3(
            q=query,
            k=flatten_k,
            v=flatten_v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=max_q_seqlen,
            max_seqlen_k=attn_metadata.max_kv_seqlen,
            softmax_scale=self.scale,
            causal=self.causal,
            window_size=sliding_window,
            softcap=self.logit_softcapping,
        )

        # Inverse-rotate output back to original domain
        if quant_policy == QuantPolicy.TURBO_QUANT:
            attn_output = hadamard_rotate_inv(attn_output)

        return attn_output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        """Forward pass for FA3 attention computation.

        This method handles both prefill and decoding stages by:
        1. Computing max query sequence length
        2. Filling KV cache if new key/value are provided
        3. Dispatching to appropriate stage-specific method

        Architecture:
        - Decoding: Supports both speculative (multi-token) and standard (single-token)
        - Prefill: Uses flash_attn_varlen_func for efficient varlen attention

        Args:
            query: Query tensor.
            key: Key tensor (None for decoding-only).
            value: Value tensor (None for decoding-only).
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata containing stage info and indices.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.
            learnable_sink: Learnable sink tokens (unused in FA3).
            inplace: Whether to modify query inplace (unused, kept for compatibility).

        Returns:
            Attention output tensor.
        """
        # Shared preparation
        max_q_seqlen = self._get_max_q_seqlen(query, attn_metadata)

        # Fill KV cache with new key/value if provided
        if key is not None and value is not None:
            self._fill_kv_cache_impl(
                key,
                value,
                k_cache=k_cache,
                v_cache=v_cache,
                attn_metadata=attn_metadata,
                max_q_seqlen=max_q_seqlen,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
            )

        # Dispatch to stage-specific forward method
        if attn_metadata.is_decoding:
            return self._forward_decoding(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros,
                v_scales_zeros,
            )
        else:
            return self._forward_prefill(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros,
                v_scales_zeros,
            )
