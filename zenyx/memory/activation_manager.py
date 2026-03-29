"""Selective activation management for memory efficiency.

Strategy:
- Attention activations: Compress via torch.utils.checkpoint
- FFN activations: Recompute via gradient checkpointing
- Never compress all tensors — numerical instability.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class ActivationManager:
    """Manages selective activation checkpointing and recomputation.

    This manager implements a selective strategy:
    - Attention layers: Use gradient checkpointing (recompute on backward)
    - Feed-forward layers: Use gradient checkpointing (recompute on backward)
    - Never compress tensors (prevents numerical instability)

    Usage:
        >>> manager = ActivationManager()
        >>> wrapped_attn = manager.checkpoint_attention(attention_layer)
        >>> wrapped_ffn = manager.checkpoint_ffn(ffn_layer)
    """

    def __init__(self, use_checkpoint: bool = True):
        """Initialize the activation manager.

        Parameters
        ----------
        use_checkpoint : bool
            Whether to use gradient checkpointing. Default: True.
        """
        self.use_checkpoint = use_checkpoint
        logger.info("ActivationManager(use_checkpoint=%s)", use_checkpoint)

    def checkpoint_attention(
        self,
        module: nn.Module,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Wrap an attention module with gradient checkpointing.

        Checkpointing recomputes activations during backward, saving memory.

        Parameters
        ----------
        module : nn.Module
            Attention module (e.g., MultiHeadAttention).

        Returns
        -------
        Callable
            A function that runs the module with checkpointing.
        """
        if not self.use_checkpoint:
            return module.forward

        def forward_with_checkpoint(*args: Any, **kwargs: Any) -> torch.Tensor:
            """Forward pass with gradient checkpointing."""
            return checkpoint(
                module.forward,
                *args,
                use_reentrant=False,  # More stable
                **kwargs,
            )

        return forward_with_checkpoint

    def checkpoint_ffn(
        self,
        module: nn.Module,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Wrap a feed-forward module with gradient checkpointing.

        Checkpointing recomputes activations during backward, saving memory.

        Parameters
        ----------
        module : nn.Module
            Feed-forward module (e.g., MLP).

        Returns
        -------
        Callable
            A function that runs the module with checkpointing.
        """
        if not self.use_checkpoint:
            return module.forward

        def forward_with_checkpoint(x: torch.Tensor) -> torch.Tensor:
            """Forward pass with gradient checkpointing."""
            return checkpoint(
                module.forward,
                x,
                use_reentrant=False,
                **{},
            )

        return forward_with_checkpoint

    @staticmethod
    def hook_into_model(
        model: nn.Module,
        use_checkpoint: bool = True,
    ) -> None:
        """Install checkpointing hooks into all attention/FFN layers.

        Parameters
        ----------
        model : nn.Module
            The model to instrument.
        use_checkpoint : bool
            Whether to enable checkpointing.
        """
        if not use_checkpoint:
            return

        manager = ActivationManager(use_checkpoint=True)

        for name, module in model.named_modules():
            # Detect attention layers
            if "attention" in name.lower() or "attn" in name.lower():
                original_forward = module.forward

                def make_checkpoint_forward(
                    orig_forward: Callable[..., Any],
                ) -> Callable[..., Any]:
                    def checkpoint_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                        return checkpoint(
                            orig_forward,
                            *args,
                            use_reentrant=False,
                            **kwargs,
                        )

                    return checkpoint_forward

                module.forward = make_checkpoint_forward(original_forward)

            # Detect FFN layers
            elif "mlp" in name.lower() or "ffn" in name.lower() or "feed_forward" in name.lower():
                original_forward = module.forward

                def make_checkpoint_forward(
                    orig_forward: Callable[..., Any],
                ) -> Callable[..., Any]:
                    def checkpoint_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                        return checkpoint(
                            orig_forward,
                            *args,
                            use_reentrant=False,
                            **kwargs,
                        )

                    return checkpoint_forward

                module.forward = make_checkpoint_forward(original_forward)

        logger.info("Installed checkpointing hooks into model")

    @staticmethod
    def estimate_memory_saving(
        num_layers: int,
        hidden_dim: int,
        seq_len: int,
        batch_size: int,
    ) -> dict[str, float]:
        """Estimate memory savings from checkpointing.

        Parameters
        ----------
        num_layers : int
            Number of transformer layers.
        hidden_dim : int
            Hidden dimension size.
        seq_len : int
            Sequence length.
        batch_size : int
            Batch size.

        Returns
        -------
        dict[str, float]
            Memory savings estimate in MB.
        """
        # Rough estimate: ~4 bytes per float32, 2 bytes per float16
        bytes_per_param = 2  # Assuming float16

        # Attention activations
        attn_size = batch_size * seq_len * hidden_dim * bytes_per_param
        attn_save = attn_size * num_layers / 1e6  # Convert to MB

        # FFN activations
        ffn_size = batch_size * seq_len * (hidden_dim * 4) * bytes_per_param  # Assuming 4x expansion
        ffn_save = ffn_size * num_layers / 1e6

        total_save = (attn_save + ffn_save) / 1e3  # Convert to GB

        return {
            "attention_memory_mb": attn_save,
            "ffn_memory_mb": ffn_save,
            "total_memory_gb": total_save,
        }
