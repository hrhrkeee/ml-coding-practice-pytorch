import numpy as np
import torch

def reshape_tensor(x: np.ndarray, new_shape: tuple) -> np.ndarray:
    return None

def flatten_tensor(x: np.ndarray) -> np.ndarray:
    return None

def test_reshape_tensor():
    np.random.seed(42)
    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    new_shape = (6, 4)
    
    reshaped_np = reshape_tensor(x_np, new_shape)
    
    assert reshaped_np is not None, "reshape_tensor が実装されていません"
    assert reshaped_np.shape == new_shape, f"形状が一致しません: {reshaped_np.shape} vs {new_shape}"
    
    x_torch = torch.tensor(x_np)
    reshaped_torch = x_torch.reshape(new_shape).numpy()
    
    assert np.allclose(reshaped_np, reshaped_torch, atol=1e-6), "Reshapeの結果が一致しません"

def test_flatten_tensor():
    np.random.seed(42)
    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    
    flattened_np = flatten_tensor(x_np)
    
    assert flattened_np is not None, "flatten_tensor が実装されていません"
    assert flattened_np.ndim == 1, f"1次元になっていません: {flattened_np.ndim}D"
    
    x_torch = torch.tensor(x_np)
    flattened_torch = x_torch.flatten().numpy()
    
    assert np.allclose(flattened_np, flattened_torch, atol=1e-6), "Flattenの結果が一致しません"
