# project_root/01_question/01_activation_function/01_sigmoid.py

import numpy as np

def sigmoid_forward(x: np.ndarray) -> np.ndarray:
    """
    NumPy による Sigmoid のフォワード計算：
        σ(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    x : np.ndarray
        入力配列

    Returns
    -------
    np.ndarray
        Sigmoid 適用済み出力
    """
    result = None

    # ここにコードを記述
    # result = 1.0 / (1.0 + np.exp(-x))

    return result


def sigmoid_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """
    Sigmoid のバックワード勾配：
        ∂σ/∂x = σ(x) * (1 - σ(x))

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
