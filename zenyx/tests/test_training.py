import os

import pytest
import torch
import torch.nn as nn

from zenyx.train import Trainer, TrainerConfig


def test_single_process_training_smoke():
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    trainer = Trainer(model, TrainerConfig(lr=1e-2, overlap=False))
    criterion = nn.MSELoss()

    batches = [(torch.randn(8, 8), torch.randn(8, 4)) for _ in range(3)]
    losses = trainer.fit(batches, criterion, steps=3)
    assert len(losses) == 3


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_two_gpu_training_smoke():
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    trainer = Trainer(model, TrainerConfig(lr=1e-2, overlap=True))

    criterion = nn.MSELoss()

    batches = []
    for _ in range(4):
        x = torch.randn(8, 8)
        y = torch.randn(8, 4)
        batches.append((x, y))

    losses = trainer.fit(batches, criterion, steps=4)

    assert len(losses) == 4
    assert losses[-1] <= losses[0] or abs(losses[-1] - losses[0]) < 1e-3
