import sys
from pathlib import Path

# 現在のファイルの場所から test_src へのパスを解決して sys.path に追加
# 想定配置: test_exam/01_question_notebook/01_activation_function/notebook_setup.py
# 目標: test_exam/test_src

current_file = Path(__file__).resolve()
# parents[0]: 01_activation_function
# parents[1]: 01_question_notebook
# parents[2]: test_exam
test_src_path = current_file.parents[2] / "test_src"

if str(test_src_path) not in sys.path:
    sys.path.append(str(test_src_path))

# notebook_runner がインポートできるか確認し、run_test を公開
try:
    from notebook_runner import run_test
except ImportError as e:
    print(f"Setup Error: {e}")
    print(f"Failed to import 'notebook_runner' from {test_src_path}")
    
    # ダミー関数を定義してエラーを防ぐ
    def run_test(*args, **kwargs):
        print("Error: run_test is not available because setup failed.")


def test(test_module_name: str, filter_name: str = None):
    """
    関数をテストするためのデコレータ
    
    Parameters
    ----------
    test_module_name : str
        テストモジュール名（例: "01_activation_function.test_01_sigmoid"）
    filter_name : str, optional
        実行するテスト関数名を絞り込むための文字列
    
    Examples
    --------
    @test("01_activation_function.test_01_sigmoid", filter_name="forward")
    def sigmoid_forward(x: np.ndarray) -> np.ndarray:
        ...
    """
    def decorator(func):
        # テストを実行
        run_test(test_module_name, filter_name=filter_name, **{func.__name__: func})
        # 元の関数をそのまま返す
        return func
    return decorator
