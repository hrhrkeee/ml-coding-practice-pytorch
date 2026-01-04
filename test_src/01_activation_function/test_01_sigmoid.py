# project_root/test_src/test_sigmoid.py

import numpy as np
import torch
import importlib.util
from pathlib import Path

# 数字で始まるディレクトリ名によるインポートエラーを避けるため、importlib を使用して動的に読み込む
current_file = Path(__file__).resolve()
target_path = current_file.parents[2] / "01_question" / "01_activation_function" / "01_sigmoid.py"

spec = importlib.util.spec_from_file_location("sigmoid_module", target_path)
sigmoid_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sigmoid_module)

sigmoid_forward = sigmoid_module.sigmoid_forward
sigmoid_backward = sigmoid_module.sigmoid_backward

# ---------------------------------------
# Forward の値一致確認
# ---------------------------------------
def test_forward_sigmoid():
    x_np = np.linspace(-6, 6, 13).reshape(-1,1).astype(float)
    forward_np = sigmoid_forward(x_np)

    assert forward_np is not None, "sigmoid_forward が実装されていません (None が返されました)"

    forward_torch = torch.sigmoid(torch.tensor(x_np, dtype=torch.float32)).numpy()

    assert np.allclose(forward_np, forward_torch, atol=1e-6), \
        "sigmoid_forward の出力が PyTorch と一致しません"

# ---------------------------------------
# Backward の勾配一致確認
# ---------------------------------------
def test_backward_sigmoid():
    x_np = np.linspace(-6, 6, 13).reshape(-1,1).astype(float)
    grad_np = np.ones_like(x_np, dtype=float)

    backward_np = sigmoid_backward(x_np, grad_np)

    assert backward_np is not None, "sigmoid_backward が実装されていません (None が返されました)"

    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    y = torch.sigmoid(x_torch)
    y.sum().backward()
    grad_torch = x_torch.grad.numpy()

    assert np.allclose(backward_np, grad_torch, atol=1e-6), \
        "sigmoid_backward の勾配が PyTorch と一致しません"
