"""Megatron-style distributed cross-entropy with log-sum-exp trick.

Implements vocabulary-parallel cross-entropy loss that never materializes the
full logit tensor.  Numerically stable at 500K+ vocab sizes via the log-sum-exp
trick with a global max reduction.

Complexity
----------
- Forward : O(V_local × B) compute  +  2 AllReduce(B) collectives
- Backward: O(V_local × B) compute  +  1 AllReduce(B) collective
- Memory  : O(V_local × B) — no full-vocab tensor ever materialized

Where V_local = vocab_size / world_size, B = batch × seq_len.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

__all__ = ["VocabParallelCrossEntropy", "vocab_parallel_cross_entropy"]

logger = logging.getLogger(__name__)


class VocabParallelCrossEntropy(torch.autograd.Function):
    """Distributed cross-entropy that shards the vocabulary across ranks.

    Each rank holds ``logits_parallel`` for its slice ``[vocab_start, vocab_end)``
    of the full vocabulary.  Two AllReduce ops (MAX then SUM) produce the exact
    global log-sum-exp denominator without ever gathering the full logit row.

    Complexity (forward)
    --------------------
    Time : O(V_local × B)  +  2 × AllReduce(B)
    Space: O(V_local × B)
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits_parallel: torch.Tensor,
        targets: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Compute distributed cross-entropy loss (forward pass).

        Parameters
        ----------
        logits_parallel : Tensor[..., V_local]
            Local shard of logits for vocabulary range ``[vocab_start, vocab_end)``.
        targets : Tensor[...]
            Ground-truth token ids (global vocabulary indices).
        vocab_start_index : int
            Inclusive start of this rank's vocabulary slice.
        vocab_end_index : int
            Exclusive end of this rank's vocabulary slice.
        process_group : ProcessGroup | None
            Torch distributed process group.  ``None`` → default group.

        Returns
        -------
        Tensor[...]
            Per-token cross-entropy loss (not reduced).

        Complexity
        ----------
        Time : O(V_local × B) + 2 AllReduce(B)
        Space: O(V_local × B)
        """
        # ---- 1. Global max for numerical stability ----
        local_max: torch.Tensor = logits_parallel.max(dim=-1).values  # [...,]
        global_max = local_max.clone()
        if dist.is_initialized():
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=process_group)

        # ---- 2. Shift logits ----
        logits_shifted = logits_parallel - global_max.unsqueeze(-1)

        # ---- 3. Global exp-sum ----
        local_exp_sum = logits_shifted.exp().sum(dim=-1)  # [...,]
        global_exp_sum = local_exp_sum.clone()
        if dist.is_initialized():
            dist.all_reduce(global_exp_sum, op=dist.ReduceOp.SUM, group=process_group)

        log_sum = global_exp_sum.log()  # [...,]

        # ---- 4. Extract target logit from local shard ----
        # Mask: which targets fall inside this rank's slice
        target_local = targets - vocab_start_index
        in_range = (targets >= vocab_start_index) & (targets < vocab_end_index)

        # Clamp to valid indices (out-of-range will be zeroed via mask)
        target_local_clamped = target_local.clamp(0, logits_parallel.size(-1) - 1)

        # Gather target logit (shifted) from local shard
        target_logit_shifted = logits_shifted.gather(
            dim=-1, index=target_local_clamped.unsqueeze(-1)
        ).squeeze(-1)

        # Zero out contributions from targets not in this shard
        target_logit_shifted = target_logit_shifted * in_range.float()

        # AllReduce SUM to collect the target logit from whichever rank owns it
        if dist.is_initialized():
            dist.all_reduce(
                target_logit_shifted, op=dist.ReduceOp.SUM, group=process_group
            )

        # ---- 5. Loss = log_sum - target_logit ----
        loss = log_sum - target_logit_shifted

        # ---- Save for backward ----
        # softmax_logits = exp(shifted) / global_exp_sum  → needed for grad
        softmax_logits = logits_shifted.exp() / global_exp_sum.unsqueeze(-1)
        ctx.save_for_backward(softmax_logits, target_local_clamped, in_range)
        ctx.vocab_start_index = vocab_start_index
        ctx.vocab_end_index = vocab_end_index
        ctx.process_group = process_group

        return loss

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None]:
        """Backward pass: gradient w.r.t. ``logits_parallel``.

        The gradient is ``softmax(logits) - one_hot(target)`` (for the local
        shard only), scaled by ``grad_output``.  All-reduce gradient volume
        scales with ``d_model`` only, NOT ``vocab_size``.

        Complexity
        ----------
        Time : O(V_local × B)
        Space: O(V_local × B)
        """
        softmax_logits, target_local_clamped, in_range = ctx.saved_tensors

        # grad = softmax - one_hot(target)
        grad_logits = softmax_logits

        # Subtract 1 at the target position (only if target is in this shard)
        one_hot = torch.zeros_like(grad_logits)
        in_range_expanded = in_range.unsqueeze(-1).float()
        one_hot.scatter_(
            dim=-1, index=target_local_clamped.unsqueeze(-1), src=in_range_expanded
        )
        grad_logits = grad_logits - one_hot

        # Scale by upstream gradient
        grad_logits = grad_logits * grad_output.unsqueeze(-1)

        return grad_logits, None, None, None, None


def vocab_parallel_cross_entropy(
    logits_parallel: torch.Tensor,
    targets: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute vocabulary-parallel cross-entropy loss.

    Wrapper around :class:`VocabParallelCrossEntropy` autograd function.

    Parameters
    ----------
    logits_parallel : Tensor[..., V_local]
        Local shard of logits for vocabulary range ``[vocab_start, vocab_end)``.
    targets : Tensor[...]
        Ground-truth token ids (global vocabulary indices).
    vocab_start_index : int
        Inclusive start of this rank's vocabulary slice.
    vocab_end_index : int
        Exclusive end of this rank's vocabulary slice.
    process_group : ProcessGroup | None
        Torch distributed process group.

    Returns
    -------
    Tensor[...]
        Per-token cross-entropy loss.

    Complexity
    ----------
    Time : O(V_local × B) + 2 AllReduce(B)
    Space: O(V_local × B)
    """
    return VocabParallelCrossEntropy.apply(
        logits_parallel, targets, vocab_start_index, vocab_end_index, process_group
    )
