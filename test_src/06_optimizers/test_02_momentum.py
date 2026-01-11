import numpy as np
import torch
import torch.optim as optim

def momentum_step(param: np.ndarray, grad: np.ndarray, velocity: np.ndarray, lr: float, momentum: float = 0.9) -> tuple:
    return None, None

def test_momentum_step():
    np.random.seed(42)
    param_np = np.random.randn(5).astype(np.float32)
    grad_np = np.random.randn(5).astype(np.float32)
    velocity_np = np.zeros(5, dtype=np.float32)
    lr = 0.01
    momentum = 0.9
    
    new_param_np, new_velocity_np = momentum_step(param_np.copy(), grad_np, velocity_np.copy(), lr, momentum)
    
    assert new_param_np is not None, "momentum_step が実装されていません"
    assert new_velocity_np is not None, "velocity が実装されていません"
    
    param_torch = torch.tensor(param_np.copy(), requires_grad=True)
    param_torch.grad = torch.tensor(grad_np)
    optimizer = optim.SGD([param_torch], lr=lr, momentum=momentum)
    optimizer.step()
    new_param_torch = param_torch.detach().numpy()
    
    assert np.allclose(new_param_np, new_param_torch, atol=1e-5), "Momentumの更新が一致しません"
