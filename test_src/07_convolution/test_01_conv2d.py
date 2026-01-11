import numpy as np
import torch
import torch.nn.functional as F

def conv2d_forward(x: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    return None

def test_conv2d_forward():
    np.random.seed(42)
    x_np = np.random.randn(5, 5).astype(np.float32)
    kernel_np = np.random.randn(3, 3).astype(np.float32)
    
    out_np = conv2d_forward(x_np, kernel_np, stride=1, padding=0)
    
    assert out_np is not None, "conv2d_forward が実装されていません"
    
    x_torch = torch.tensor(x_np).unsqueeze(0).unsqueeze(0)
    kernel_torch = torch.tensor(kernel_np).unsqueeze(0).unsqueeze(0)
    out_torch = F.conv2d(x_torch, kernel_torch, stride=1, padding=0).squeeze().numpy()
    
    assert np.allclose(out_np, out_torch, atol=1e-5), "畳み込みの出力が一致しません"
