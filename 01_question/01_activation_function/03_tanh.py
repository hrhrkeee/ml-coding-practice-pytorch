# project_root/01_question/01_activation_function/03_tanh.py

import numpy as np

def tanh_forward(x: np.ndarray) -> np.ndarray:
    """
    NumPy による Tanh (Hyperbolic Tangent) のフォワード計算：
        f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Parameters
    ----------
    x : np.ndarray
        入力配列

    Returns
    -------
    np.ndarray
        Tanh 適用済み出力
    """
    result = None

    # ここにコードを記述

    return result


def tanh_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """
    Tanh のバックワード勾配：
        ∂f/∂x = 1 - tanh(x)^2

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
