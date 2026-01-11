import sys
sys.path.insert(0, "test_src")

import numpy as np
import torch
import torch.nn as nn

print("=" * 60)
print("損失関数の実装テスト")
print("=" * 60)

# Test 1: MSE
print("\n1. MSE (Mean Squared Error)")
print("-" * 60)
pred = np.array([[1.0, 2.0], [3.0, 4.0]])
target = np.array([[1.5, 2.5], [3.5, 4.5]])

loss_np = np.mean((pred - target) ** 2)
grad_np = 2.0 * (pred - target) / pred.size

pred_torch = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
target_torch = torch.tensor(target, dtype=torch.float32)
loss_fn = nn.MSELoss()
loss_torch = loss_fn(pred_torch, target_torch)
loss_torch.backward()
grad_torch = pred_torch.grad.numpy()

print(f"Forward - NumPy: {loss_np:.6f}, PyTorch: {loss_torch.item():.6f}")
print(f"Forward Match: {np.isclose(loss_np, loss_torch.item(), atol=1e-6)}")
print(f"Backward Match: {np.allclose(grad_np, grad_torch, atol=1e-6)}")

# Test 2: Cross Entropy
print("\n2. Cross Entropy Loss")
print("-" * 60)
np.random.seed(42)
logits = np.random.randn(3, 4).astype(float)
targets = np.array([0, 2, 1])

N = logits.shape[0]
logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
exp_logits = np.exp(logits_shifted)
sum_exp = np.sum(exp_logits, axis=1)
log_sum_exp = np.log(sum_exp)
correct_logits = logits_shifted[np.arange(N), targets.astype(int)]
loss_np = -np.mean(correct_logits - log_sum_exp)

softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
grad_np = softmax.copy()
grad_np[np.arange(N), targets.astype(int)] -= 1
grad_np /= N

logits_torch = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
targets_torch = torch.tensor(targets, dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()
loss_torch = loss_fn(logits_torch, targets_torch)
loss_torch.backward()
grad_torch = logits_torch.grad.numpy()

print(f"Forward - NumPy: {loss_np:.6f}, PyTorch: {loss_torch.item():.6f}")
print(f"Forward Match: {np.isclose(loss_np, loss_torch.item(), atol=1e-6)}")
print(f"Backward Match: {np.allclose(grad_np, grad_torch, atol=1e-6)}")

# Test 3: Binary Cross Entropy
print("\n3. Binary Cross Entropy Loss")
print("-" * 60)
np.random.seed(42)
pred = np.array([[0.7, 0.3], [0.4, 0.9]])
target = np.array([[1.0, 0.0], [0.0, 1.0]])

eps = 1e-7
pred_clipped = np.clip(pred, eps, 1 - eps)
loss_np = -np.mean(target * np.log(pred_clipped) + (1 - target) * np.log(1 - pred_clipped))
grad_np = -(target / pred_clipped - (1 - target) / (1 - pred_clipped)) / pred.size

pred_torch = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
target_torch = torch.tensor(target, dtype=torch.float32)
loss_fn = nn.BCELoss()
loss_torch = loss_fn(pred_torch, target_torch)
loss_torch.backward()
grad_torch = pred_torch.grad.numpy()

print(f"Forward - NumPy: {loss_np:.6f}, PyTorch: {loss_torch.item():.6f}")
print(f"Forward Match: {np.isclose(loss_np, loss_torch.item(), atol=1e-6)}")
print(f"Backward Match: {np.allclose(grad_np, grad_torch, atol=1e-6)}")

print("\n" + "=" * 60)
print("すべてのテストが完了しました！")
print("=" * 60)
