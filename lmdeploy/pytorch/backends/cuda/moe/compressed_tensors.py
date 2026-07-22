# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.moe import FusedMoEW4A16Builder, FusedMoEW4A16Impl
from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16


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


class TritonFusedMoEW4A16Builder(FusedMoEW4A16Builder):
    """Build the CUDA eager compressed-tensors MoE implementation."""

    @staticmethod
    def build(
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        num_bits: int = 4,
        group_size: int = 32,
    ):
        return TritonFusedMoEW4A16Impl(
            top_k=top_k,
            num_experts=num_experts,
            renormalize=renormalize,
            num_bits=num_bits,
            group_size=group_size,
        )
