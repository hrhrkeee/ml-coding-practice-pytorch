# project_root/test_src/02_loss_function/test_02_cross_entropy.py

import numpy as np
import torch
import torch.nn as nn

# ノートブック実行時には、notebook_runner.py によってこれらの関数は動的に上書きされます。
def cross_entropy_forward(logits: np.ndarray, targets: np.ndarray) -> float:
    return None

def cross_entropy_backward(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return None

# ---------------------------------------
# Forward の値一致確認
# ---------------------------------------
def test_forward_cross_entropy():
    np.random.seed(42)
    logits_np = np.random.randn(10, 5).astype(float)
    targets_np = np.random.randint(0, 5, size=(10,)).astype(int)
    
    loss_np = cross_entropy_forward(logits_np, targets_np)

    assert loss_np is not None, "cross_entropy_forward が実装されていません (None が返されました)"

    logits_torch = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    targets_torch = torch.tensor(targets_np, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    loss_torch = loss_fn(logits_torch, targets_torch).item()

    assert np.isclose(loss_np, loss_torch, atol=1e-6), \
        f"cross_entropy_forward の出力が PyTorch と一致しません: {loss_np} vs {loss_torch}"

# ---------------------------------------
# Backward の勾配一致確認
# ---------------------------------------
def test_backward_cross_entropy():
    np.random.seed(42)
    logits_np = np.random.randn(10, 5).astype(float)
    targets_np = np.random.randint(0, 5, size=(10,)).astype(int)

    grad_np = cross_entropy_backward(logits_np, targets_np)

    assert grad_np is not None, "cross_entropy_backward が実装されていません (None が返されました)"

    logits_torch = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    targets_torch = torch.tensor(targets_np, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits_torch, targets_torch)
    loss.backward()
    grad_torch = logits_torch.grad.numpy()

    assert np.allclose(grad_np, grad_torch, atol=1e-6), \
        "cross_entropy_backward の勾配が PyTorch と一致しません"
