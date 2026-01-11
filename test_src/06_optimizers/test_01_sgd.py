import numpy as np
import torch
import torch.optim as optim

def sgd_step(param: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
    return None

def test_sgd_step():
    np.random.seed(42)
    param_np = np.random.randn(5).astype(np.float32)
    grad_np = np.random.randn(5).astype(np.float32)
    lr = 0.01
    
    new_param_np = sgd_step(param_np.copy(), grad_np, lr)
    
    assert new_param_np is not None, "sgd_step が実装されていません"
    
    param_torch = torch.tensor(param_np.copy(), requires_grad=True)
    param_torch.grad = torch.tensor(grad_np)
    optimizer = optim.SGD([param_torch], lr=lr)
    optimizer.step()
    new_param_torch = param_torch.detach().numpy()
    
    assert np.allclose(new_param_np, new_param_torch, atol=1e-6), "SGDの更新が一致しません"
