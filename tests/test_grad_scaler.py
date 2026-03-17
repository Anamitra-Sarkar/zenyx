import torch
import torch.nn as nn

from zenyx.train.grad_scaler import ZenyxGradScaler


class _TrackingSGD(torch.optim.SGD):
    def __init__(self, params, lr=1e-2):
        super().__init__(params, lr=lr)
        self.step_called = 0

    def step(self, closure=None):
        self.step_called += 1
        return super().step(closure=closure)


def test_grad_scaler_disabled_is_noop_step():
    p = torch.nn.Parameter(torch.tensor([1.0], device="cpu"))
    p.grad = torch.tensor([0.5], device="cpu")
    opt = _TrackingSGD([p], lr=0.1)
    scaler = ZenyxGradScaler(enabled=False)

    before = p.detach().clone()
    scaler.step(opt)
    assert opt.step_called == 1
    assert not torch.equal(before, p.detach())


def test_manual_path_overflow_skips_step_and_unscales_all_grads_once():
    p1 = torch.nn.Parameter(torch.tensor([1.0], device="cpu"))
    p2 = torch.nn.Parameter(torch.tensor([2.0], device="cpu"))
    opt = _TrackingSGD([p1, p2], lr=0.1)

    scaler = ZenyxGradScaler(init_scale=8.0, enabled=True)
    scaler._scaler = None

    p1.grad = torch.tensor([16.0], device="cpu")
    p2.grad = torch.tensor([float("inf")], device="cpu")

    scaler.step(opt)

    assert opt.step_called == 0
    assert torch.allclose(p1.grad, torch.tensor([2.0], device="cpu"))
    assert torch.isinf(p2.grad).all()


def test_grad_scaler_state_dict_roundtrip_manual_path():
    scaler = ZenyxGradScaler(
        init_scale=32.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=3,
    )
    scaler._scaler = None
    scaler._growth_tracker = 2
    scaler._found_inf = True
    state = scaler.state_dict()

    restored = ZenyxGradScaler(init_scale=1.0)
    restored._scaler = None
    restored.load_state_dict(state)

    assert restored.state_dict()["scale"] == 32.0
    assert restored.state_dict()["growth_tracker"] == 2
    assert restored.state_dict()["found_inf"] is True


def test_enabled_backward_pass_has_gradients():
    model = nn.Linear(4, 2, device=torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = ZenyxGradScaler(enabled=True)
    scaler._scaler = None

    x = torch.randn(3, 4, device=torch.device("cpu"))
    y = torch.randn(3, 2, device=torch.device("cpu"))
    loss = ((model(x) - y) ** 2).mean()

    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(optimizer)

    for p in model.parameters():
        assert p.grad is not None


def test_disabled_scale_is_identity():
    scaler = ZenyxGradScaler(enabled=False)
    out = scaler.scale(torch.tensor(3.14, device=torch.device("cpu")))
    assert torch.allclose(out, torch.tensor(3.14, device=torch.device("cpu")))


def test_update_does_not_raise_on_nan_gradient():
    model = nn.Linear(4, 2, device=torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = ZenyxGradScaler(init_scale=32.0, enabled=True)
    scaler._scaler = None
    initial_scale = scaler._scale

    x = torch.randn(2, 4, device=torch.device("cpu"))
    y = torch.randn(2, 2, device=torch.device("cpu"))
    loss = ((model(x) - y) ** 2).mean()
    scaler.scale(loss).backward()

    for p in model.parameters():
        if p.grad is not None:
            p.grad.fill_(float("nan"))

    scaler.step(optimizer)
    scaler.update()
    assert scaler._scale < initial_scale


def test_update_does_not_raise_on_inf_gradient():
    model = nn.Linear(4, 2, device=torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = ZenyxGradScaler(init_scale=32.0, enabled=True)
    scaler._scaler = None
    initial_scale = scaler._scale

    x = torch.randn(2, 4, device=torch.device("cpu"))
    y = torch.randn(2, 2, device=torch.device("cpu"))
    loss = ((model(x) - y) ** 2).mean()
    scaler.scale(loss).backward()

    for p in model.parameters():
        if p.grad is not None:
            p.grad.fill_(float("inf"))

    scaler.step(optimizer)
    scaler.update()
    assert scaler._scale < initial_scale
