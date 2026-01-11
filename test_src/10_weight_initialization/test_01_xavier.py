import numpy as np
import torch
import torch.nn.init as init

def xavier_uniform(n_in: int, n_out: int) -> np.ndarray:
    return None

def xavier_normal(n_in: int, n_out: int) -> np.ndarray:
    return None

def test_xavier_uniform():
    np.random.seed(42)
    n_in, n_out = 100, 50
    
    W_np = xavier_uniform(n_in, n_out)
    
    assert W_np is not None, "xavier_uniform が実装されていません"
    assert W_np.shape == (n_in, n_out), f"形状が一致しません: {W_np.shape}"
    
    # 分散の確認（理論値に近いかチェック）
    expected_var = 2.0 / (n_in + n_out)
    actual_var = np.var(W_np)
    assert abs(actual_var - expected_var) < 0.01, f"分散が理論値と異なります: {actual_var} vs {expected_var}"

def test_xavier_normal():
    np.random.seed(42)
    n_in, n_out = 100, 50
    
    W_np = xavier_normal(n_in, n_out)
    
    assert W_np is not None, "xavier_normal が実装されていません"
    assert W_np.shape == (n_in, n_out), f"形状が一致しません: {W_np.shape}"
    
    # 平均と分散の確認
    assert abs(np.mean(W_np)) < 0.1, "平均が0に近くありません"
    expected_var = 2.0 / (n_in + n_out)
    actual_var = np.var(W_np)
    assert abs(actual_var - expected_var) < 0.01, f"分散が理論値と異なります: {actual_var} vs {expected_var}"
