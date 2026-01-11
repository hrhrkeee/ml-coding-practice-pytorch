import numpy as np
import torch

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return None

def test_matrix_multiply():
    np.random.seed(42)
    A_np = np.random.randn(3, 4).astype(np.float32)
    B_np = np.random.randn(4, 5).astype(np.float32)
    
    C_np = matrix_multiply(A_np, B_np)
    
    assert C_np is not None, "matrix_multiply が実装されていません"
    
    A_torch = torch.tensor(A_np)
    B_torch = torch.tensor(B_np)
    C_torch = torch.matmul(A_torch, B_torch).numpy()
    
    assert np.allclose(C_np, C_torch, atol=1e-5), "行列積が一致しません"
