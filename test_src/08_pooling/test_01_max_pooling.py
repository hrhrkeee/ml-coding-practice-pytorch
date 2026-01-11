import numpy as np
import torch
import torch.nn.functional as F

def max_pooling_forward(x: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    return None

def test_max_pooling_forward():
    np.random.seed(42)
    x_np = np.random.randn(4, 4).astype(np.float32)
    
    out_np = max_pooling_forward(x_np, pool_size=2, stride=2)
    
    assert out_np is not None, "max_pooling_forward が実装されていません"
    
    x_torch = torch.tensor(x_np).unsqueeze(0).unsqueeze(0)
    out_torch = F.max_pool2d(x_torch, kernel_size=2, stride=2).squeeze().numpy()
    
    assert np.allclose(out_np, out_torch, atol=1e-5), "Max Poolingの出力が一致しません"
