import numpy as np
import torch

def create_tensor(shape: tuple) -> np.ndarray:
    return None

def concat_tensors(tensors: list, axis: int = 0) -> np.ndarray:
    return None

def test_create_tensor():
    shape = (3, 4)
    tensor_np = create_tensor(shape)
    
    assert tensor_np is not None, "create_tensor が実装されていません"
    assert tensor_np.shape == shape, f"形状が一致しません: {tensor_np.shape} vs {shape}"
    assert tensor_np.dtype == np.float32, f"データ型が一致しません: {tensor_np.dtype}"

def test_concat_tensors():
    np.random.seed(42)
    t1 = np.random.randn(2, 3).astype(np.float32)
    t2 = np.random.randn(2, 3).astype(np.float32)
    
    result_np = concat_tensors([t1, t2], axis=0)
    
    assert result_np is not None, "concat_tensors が実装されていません"
    
    t1_torch = torch.tensor(t1)
    t2_torch = torch.tensor(t2)
    result_torch = torch.cat([t1_torch, t2_torch], dim=0).numpy()
    
    assert np.allclose(result_np, result_torch, atol=1e-6), "結合結果が一致しません"
