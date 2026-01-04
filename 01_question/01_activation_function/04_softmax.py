# project_root/01_question/01_activation_function/04_softmax.py

import numpy as np

def softmax_forward(x: np.ndarray) -> np.ndarray:
    """
    NumPy による Softmax のフォワード計算 (axis=-1)：
        y_i = exp(x_i) / sum(exp(x_j))

    Parameters
    ----------
    x : np.ndarray
        入力配列 (形状: (batch_size, num_classes) など)

    Returns
    -------
    np.ndarray
        Softmax 適用済み出力
    """
    result = None

    # ここにコードを記述
    # Hint: オーバーフロー対策として、各サンプルの最大値を引いてから exp を計算すると良い
    # x_max = np.max(x, axis=-1, keepdims=True)
    # exp_x = np.exp(x - x_max)

    return result


def softmax_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """
    Softmax のバックワード勾配：
        ∂L/∂x_i = y_i * (grad_output_i - sum(grad_output_k * y_k))
        ここで y = softmax(x)

    Parameters
    ----------
    x : np.ndarray
        フォワード入力
    grad_output : np.ndarray
        上流勾配

    Returns
    -------
    np.ndarray
        入力に対する勾配
    """
    result = None

    # ここにコードを記述

    return result


# --------------------------------------------------
# 直接実行時のテスト起動
# --------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    p = Path(__file__).resolve()
    sys.path[:0] = [str(p.parents[2]), str(p.parents[2] / "test_src")]
    from test_runner import run_tests
    run_tests(".".join(list(p.relative_to(p.parents[1]).parent.parts) + [p.stem]))
