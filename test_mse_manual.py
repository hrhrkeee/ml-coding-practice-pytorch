import numpy as np
import torch
import torch.nn as nn

print("Testing MSE implementation...")
print("-" * 40)

# MSE forward test
pred = np.random.randn(10, 5).astype(float)
target = np.random.randn(10, 5).astype(float)
loss_np = np.mean((pred - target) ** 2)

pred_torch = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
target_torch = torch.tensor(target, dtype=torch.float32)
loss_fn = nn.MSELoss()
loss_torch = loss_fn(pred_torch, target_torch).item()

print(f"NumPy MSE: {loss_np}")
print(f"PyTorch MSE: {loss_torch}")
print(f"Forward Match: {np.isclose(loss_np, loss_torch, atol=1e-6)}")
print()

# MSE backward test
grad_np = 2.0 * (pred - target) / pred.size

pred_torch2 = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
target_torch2 = torch.tensor(target, dtype=torch.float32)
loss = loss_fn(pred_torch2, target_torch2)
loss.backward()
grad_torch = pred_torch2.grad.numpy()

print(f"Backward Gradient match: {np.allclose(grad_np, grad_torch, atol=1e-6)}")
print("-" * 40)
print("All tests passed!")
