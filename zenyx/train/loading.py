"""Chunked state-dict loading with non-blocking CPU->GPU transfer."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def load_state_dict_chunked(model: nn.Module, state_dict: dict[str, torch.Tensor], chunk_size: int = 64) -> None:
    keys = list(state_dict.keys())
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    for i in range(0, len(keys), chunk_size):
        for key in keys[i : i + chunk_size]:
            src = state_dict[key]
            if key in params:
                dst = params[key]
            elif key in buffers:
                dst = buffers[key]
            else:
                continue

            if dst.device.type == "cuda":
                src = src.pin_memory() if src.device.type == "cpu" else src
                dst.data.copy_(src.to(dst.device, non_blocking=True))
            else:
                dst.data.copy_(src)


__all__ = ["load_state_dict_chunked"]
