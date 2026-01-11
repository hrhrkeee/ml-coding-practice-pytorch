# project_root/test_src/02_loss_function/test_01_mse.py

import numpy as np
import torch
import torch.nn as nn

# ノートブック実行時には、notebook_runner.py によってこれらの関数は動的に上書きされます。
def mse_forward(pred: np.ndarray, target: np.ndarray) -> float:
    return None

def mse_backward(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return None

# ---------------------------------------
# Forward の値一致確認
# ---------------------------------------
def test_forward_mse():
    pred_np = np.random.randn(10, 5).astype(float)
    target_np = np.random.randn(10, 5).astype(float)
    
    loss_np = mse_forward(pred_np, target_np)

    assert loss_np is not None, "mse_forward が実装されていません (None が返されました)"

    pred_torch = torch.tensor(pred_np, dtype=torch.float32, requires_grad=True)
    target_torch = torch.tensor(target_np, dtype=torch.float32)
    loss_fn = nn.MSELoss()
    loss_torch = loss_fn(pred_torch, target_torch).item()

    assert np.isclose(loss_np, loss_torch, atol=1e-6), \
        f"mse_forward の出力が PyTorch と一致しません: {loss_np} vs {loss_torch}"

# ---------------------------------------
# Backward の勾配一致確認
# ---------------------------------------
def test_backward_mse():
    pred_np = np.random.randn(10, 5).astype(float)
    target_np = np.random.randn(10, 5).astype(float)

    grad_np = mse_backward(pred_np, target_np)

    assert grad_np is not None, "mse_backward が実装されていません (None が返されました)"

    pred_torch = torch.tensor(pred_np, dtype=torch.float32, requires_grad=True)
    target_torch = torch.tensor(target_np, dtype=torch.float32)
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred_torch, target_torch)
    loss.backward()
    grad_torch = pred_torch.grad.numpy()

    assert np.allclose(grad_np, grad_torch, atol=1e-6), \
        "mse_backward の勾配が PyTorch と一致しません"
