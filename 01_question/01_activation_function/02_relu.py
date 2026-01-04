# project_root/01_question/01_activation_function/02_relu.py

import numpy as np

def relu_forward(x: np.ndarray) -> np.ndarray:
    """
    NumPy による ReLU (Rectified Linear Unit) のフォワード計算：
        f(x) = max(0, x)

    Parameters
    ----------
    x : np.ndarray
        入力配列

    Returns
    -------
    np.ndarray
        ReLU 適用済み出力
    """
    result = None

    # ここにコードを記述

    return result


def relu_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """
    ReLU のバックワード勾配：
        ∂f/∂x = 1 (if x > 0)
                0 (if x <= 0)

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
