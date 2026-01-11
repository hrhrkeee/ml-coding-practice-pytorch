import numpy as np
import torch
import torch.nn as nn

def batch_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return None

def test_forward_batch_norm():
    np.random.seed(42)
    x_np = np.random.randn(10, 5).astype(np.float32)
    gamma_np = np.ones(5, dtype=np.float32)
    beta_np = np.zeros(5, dtype=np.float32)
    
    out_np = batch_norm_forward(x_np, gamma_np, beta_np)
    
    assert out_np is not None, "batch_norm_forward が実装されていません"
    
    x_torch = torch.tensor(x_np, requires_grad=True)
    bn = nn.BatchNorm1d(5, affine=False, track_running_stats=False)
    bn.eval()
    out_torch = bn(x_torch).detach().numpy()
    
    assert np.allclose(out_np, out_torch, atol=1e-5), "Batch Normalizationの出力が一致しません"
