import numpy as np
import torch
import torch.optim as optim

def adam_update(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    return None, None, None

def test_adam_update():
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 初期パラメータと勾配
    params_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    grads_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    m_np = np.zeros(3, dtype=np.float32)
    v_np = np.zeros(3, dtype=np.float32)
    
    # PyTorchでの実装
    params_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer = optim.Adam([params_torch], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    
    # 複数ステップ実行してテスト
    for t in range(1, 4):
        # NumPy実装
        params_np, m_np, v_np = adam_update(params_np, grads_np, m_np, v_np, t)
        
        assert params_np is not None, "adam_update が実装されていません"
        
        # PyTorch実装
        optimizer.zero_grad()
        params_torch.grad = torch.tensor([0.1, 0.2, 0.3])
        optimizer.step()
        
        # 比較
        np.testing.assert_allclose(params_np, params_torch.detach().numpy(), rtol=1e-5, atol=1e-6)
