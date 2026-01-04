# project_root/test_src/test_tanh.py

import numpy as np
import torch
import importlib.util
from pathlib import Path

# 数字で始まるディレクトリ名によるインポートエラーを避けるため、importlib を使用して動的に読み込む
current_file = Path(__file__).resolve()
target_path = current_file.parents[2] / "01_question" / "01_activation_function" / "03_tanh.py"

spec = importlib.util.spec_from_file_location("tanh_module", target_path)
tanh_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tanh_module)

tanh_forward = tanh_module.tanh_forward
tanh_backward = tanh_module.tanh_backward

# ---------------------------------------
# Forward の値一致確認
# ---------------------------------------
def test_tanh_forward_value():
    """
    PyTorch の Tanh と値が一致するか確認
    """
    x_np = np.random.randn(10, 10).astype(np.float32)
    
    # PyTorch
    x_torch = torch.tensor(x_np, requires_grad=False)
    y_torch = torch.tanh(x_torch)
    
    # 実装
    y_custom = tanh_forward(x_np)
    
    if y_custom is None:
        assert False, "tanh_forward が実装されていません"

    assert np.allclose(y_custom, y_torch.numpy(), atol=1e-6), \
        "Forward の出力が PyTorch と一致しません"

# ---------------------------------------
# Backward の値一致確認
# ---------------------------------------
def test_tanh_backward_value():
    """
    PyTorch の Tanh の勾配と一致するか確認
    """
    x_np = np.random.randn(10, 10).astype(np.float32)
    grad_output_np = np.random.randn(10, 10).astype(np.float32)

    # PyTorch
    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tanh(x_torch)
    y_torch.backward(torch.tensor(grad_output_np))
    grad_torch = x_torch.grad.numpy()

    # 実装
    grad_custom = tanh_backward(x_np, grad_output_np)

    if grad_custom is None:
        assert False, "tanh_backward が実装されていません"

    assert np.allclose(grad_custom, grad_torch, atol=1e-6), \
        "Backward の勾配が PyTorch と一致しません"
