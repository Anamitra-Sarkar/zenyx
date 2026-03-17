import torch

from zenyx.train.lr_schedule import CosineWithWarmup


def _make_sched():
    p = torch.nn.Parameter(torch.tensor([1.0], device="cpu"))
    opt = torch.optim.SGD([p], lr=0.0)
    return CosineWithWarmup(opt, peak_lr=1e-3, warmup_steps=10, total_steps=100, min_lr_ratio=0.1), opt


def test_warmup_and_peak_lr():
    sched, _ = _make_sched()
    assert sched.get_lr() == 0.0
    for _ in range(10):
        lr = sched.step()
    assert abs(lr - 1e-3) < 1e-12


def test_cosine_reaches_min_lr_at_total_steps():
    sched, _ = _make_sched()
    for _ in range(100):
        lr = sched.step()
    assert abs(lr - 1e-4) < 1e-12


def test_state_dict_roundtrip():
    sched, opt = _make_sched()
    for _ in range(17):
        sched.step()
    state = sched.state_dict()

    sched2 = CosineWithWarmup(opt, peak_lr=1.0, warmup_steps=1, total_steps=2)
    sched2.load_state_dict(state)
    assert sched2.current_step == 17
    assert abs(sched2.get_lr() - sched.get_lr()) < 1e-12


def test_post_total_steps_clamps_to_min_lr():
    sched, _ = _make_sched()
    for _ in range(1100):
        lr = sched.step()
    assert abs(lr - 1e-4) < 1e-12


def test_lr_never_negative():
    sched, _ = _make_sched()
    lrs = []
    for _ in range(1100):
        lrs.append(sched.step())
    assert all(lr >= 0.0 for lr in lrs)


def test_state_dict_has_required_keys():
    sched, _ = _make_sched()
    state = sched.state_dict()
    for key in ("peak_lr", "warmup_steps", "total_steps", "min_lr", "step"):
        assert key in state


def test_repr_contains_class_name():
    sched, _ = _make_sched()
    assert "CosineWithWarmup" in repr(sched)
