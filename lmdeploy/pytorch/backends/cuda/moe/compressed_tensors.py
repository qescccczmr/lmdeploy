# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist

from lmdeploy.pytorch.backends.cuda.token_dispatcher import (
    DeepEPBuffer,
    DeepEPTokenDispatcherNormal,
    use_deepep,
)
from lmdeploy.pytorch.backends.deepep_state import get_deepep_state
from lmdeploy.pytorch.backends.moe import FusedMoEW4A16Builder, FusedMoEW4A16Impl
from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16
from lmdeploy.pytorch.kernels.cuda.fused_moe import _renormalize

from .ep_utils import gather_outputs_by_attn_tp, split_inputs_by_attn_tp


class TritonFusedMoEW4A16Impl(FusedMoEW4A16Impl):
    """Eager direct-packed W4A16 routed-expert implementation."""

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        renormalize: bool,
        num_bits: int,
        group_size: int,
    ):
        if top_k < 1 or num_experts < 1 or top_k > num_experts:
            raise ValueError(
                f'Expected 1 <= top_k <= num_experts, got {top_k} and {num_experts}'
            )
        if num_bits != 4 or group_size != 32:
            raise ValueError(
                f'Only INT4 group-size 32 is supported, got bits={num_bits}, group_size={group_size}'
            )
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.num_bits = num_bits
        self.group_size = group_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        gate_up_packed: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down_packed: torch.Tensor,
        down_scale: torch.Tensor,
    ):
        """Run the CUDA eager kernel."""
        if gate_up_packed.shape[0] != self.num_experts:
            raise ValueError(
                f'Expected {self.num_experts} experts, got {gate_up_packed.shape[0]}'
            )
        return fused_moe_w4a16(
            hidden_states,
            gate_up_packed,
            gate_up_scale,
            down_packed,
            down_scale,
            topk_weights,
            topk_ids,
            topk=self.top_k,
            renormalize=self.renormalize,
            num_bits=self.num_bits,
            group_size=self.group_size,
        )


class DeepEPFusedMoEW4A16Impl(FusedMoEW4A16Impl):
    """Synchronous DeepEP normal-mode W4A16 routed experts."""

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        hidden_dim: int,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        renormalize: bool,
        num_bits: int,
        group_size: int,
        out_dtype: torch.dtype,
        num_max_dispatch_tokens_per_rank: int,
        layer_idx: int,
    ):
        if top_k < 1 or num_experts < 1 or top_k > num_experts:
            raise ValueError(
                f'Expected 1 <= top_k <= num_experts, got {top_k} and {num_experts}'
            )
        if ep_size <= 1:
            raise ValueError(
                f'DeepEP W4A16 requires ep_size > 1, got {ep_size}')
        if num_experts % ep_size != 0:
            raise ValueError(
                f'num_experts={num_experts} must be divisible by ep_size={ep_size}'
            )
        if ep_group is None:
            raise ValueError('DeepEP W4A16 requires an EP process group.')
        if dist.is_initialized():
            group_world_size = dist.get_world_size(ep_group)
            if group_world_size != ep_size:
                raise ValueError(
                    f'EP process group size {group_world_size} does not match ep_size={ep_size}'
                )
        if num_bits != 4 or group_size != 32:
            raise ValueError(
                f'Only INT4 group-size 32 is supported, got bits={num_bits}, group_size={group_size}'
            )
        if out_dtype != torch.bfloat16:
            raise ValueError(
                f'DeepEP W4A16 requires bfloat16 activations, got {out_dtype}'
            )
        if not use_deepep:
            raise ImportError(
                'DeepEP is required for DeepEP W4A16. Please install '
                'https://github.com/deepseek-ai/DeepEP.')

        self.top_k = top_k
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.hidden_dim = hidden_dim
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.renormalize = renormalize
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_dtype = out_dtype
        self.num_max_dispatch_tokens_per_rank = (
            num_max_dispatch_tokens_per_rank)
        self.layer_idx = layer_idx

        get_deepep_state().enable()
        if hasattr(DeepEPBuffer, 'set_explicitly_destroy'):
            DeepEPBuffer.set_explicitly_destroy()

        # The packed W4 kernel sorts routes itself and does not require the
        # grouped-GEMM expert alignment used by DeepGEMM.
        self.token_dispatcher = DeepEPTokenDispatcherNormal(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            expert_alignment=1,
        )

    def _validate_local_weights(
        self,
        gate_up_packed: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down_packed: torch.Tensor,
        down_scale: torch.Tensor,
    ):
        expert_counts = {
            'gate_up_packed': gate_up_packed.shape[0],
            'gate_up_scale': gate_up_scale.shape[0],
            'down_packed': down_packed.shape[0],
            'down_scale': down_scale.shape[0],
        }
        mismatched = {
            name: count
            for name, count in expert_counts.items()
            if count != self.num_local_experts
        }
        if mismatched:
            raise ValueError(
                f'Expected {self.num_local_experts} local experts on each W4A16 tensor, '
                f'got {mismatched}')

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        gate_up_packed: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down_packed: torch.Tensor,
        down_scale: torch.Tensor,
    ):
        """Dispatch, run local packed experts, and synchronously combine."""
        self._validate_local_weights(gate_up_packed, gate_up_scale,
                                     down_packed, down_scale)
        hidden_states, topk_weights, topk_ids, split_size = (
            split_inputs_by_attn_tp(hidden_states, topk_weights, topk_ids))

        # Renormalization belongs to the global Top-K domain.  Applying it to
        # DeepEP's rank-local routes would independently rescale each rank.
        topk_weights = _renormalize(topk_weights, self.renormalize)

        try:
            (recv_hidden_states, recv_topk_ids, recv_topk_weights,
             _) = self.token_dispatcher.dispatch(
                 hidden_states,
                 topk_ids,
                 topk_weights,
             )

            # DeepEP normal mode already maps global expert IDs to local
            # [0, E_local) IDs and marks non-local slots as -1/zero.  The W4
            # kernel must filter those invalid routes instead of remapping
            # them to a real expert.
            out_states = fused_moe_w4a16(
                recv_hidden_states,
                gate_up_packed,
                gate_up_scale,
                down_packed,
                down_scale,
                recv_topk_weights,
                recv_topk_ids,
                topk=self.top_k,
                renormalize=False,
                num_bits=self.num_bits,
                group_size=self.group_size,
                allow_invalid_routes=True,
            )
            out_states = self.token_dispatcher.combine(out_states)
        finally:
            self.token_dispatcher.release()

        return gather_outputs_by_attn_tp(out_states, split_size)


class TritonFusedMoEW4A16Builder(FusedMoEW4A16Builder):
    """Build the CUDA eager compressed-tensors MoE implementation."""

    @staticmethod
    def build(
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        num_bits: int = 4,
        group_size: int = 32,
        hidden_dim: int = 1,
        ep_size: int = 1,
        ep_group: dist.ProcessGroup = None,
        out_dtype: torch.dtype = torch.bfloat16,
        num_max_dispatch_tokens_per_rank: int = 128,
        layer_idx: int = 0,
    ):
        if ep_size > 1:
            return DeepEPFusedMoEW4A16Impl(
                top_k=top_k,
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                ep_size=ep_size,
                ep_group=ep_group,
                renormalize=renormalize,
                num_bits=num_bits,
                group_size=group_size,
                out_dtype=out_dtype,
                num_max_dispatch_tokens_per_rank=
                num_max_dispatch_tokens_per_rank,
                layer_idx=layer_idx,
            )
        return TritonFusedMoEW4A16Impl(
            top_k=top_k,
            num_experts=num_experts,
            renormalize=renormalize,
            num_bits=num_bits,
            group_size=group_size,
        )
