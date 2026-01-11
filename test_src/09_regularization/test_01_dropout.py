import numpy as np
import torch
import torch.nn.functional as F

def dropout_forward(x: np.ndarray, drop_rate: float = 0.5, training: bool = True) -> np.ndarray:
    return None

def test_dropout_forward():
    np.random.seed(42)
    torch.manual_seed(42)
    x_np = np.random.randn(10, 5).astype(np.float32)
    
    # 推論モードのテスト
    out_np = dropout_forward(x_np.copy(), drop_rate=0.5, training=False)
    
    assert out_np is not None, "dropout_forward が実装されていません"
    assert np.allclose(out_np, x_np, atol=1e-6), "推論時は入力をそのまま返すべきです"
    
    # 訓練モードのテスト（形状とスケールの確認）
    out_train_np = dropout_forward(x_np.copy(), drop_rate=0.5, training=True)
    assert out_train_np.shape == x_np.shape, "形状が一致しません"
