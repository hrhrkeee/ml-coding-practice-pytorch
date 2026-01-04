import sys
import importlib
import inspect
from pathlib import Path

# Try to import colored_text from test_runner in the same directory
try:
    from .test_runner import colored_text
except ImportError:
    # If running as a script or path issues, try absolute import or define fallback
    try:
        from test_src.test_runner import colored_text
    except ImportError:
        def colored_text(text, color):
            colors = {'green': '\033[92m', 'red': '\033[91m', 'reset': '\033[0m'}
            return f"{colors.get(color, '')}{text}{colors['reset']}"

def run_test(test_module_name: str, filter_name: str = None, **kwargs):
    """
    ノートブック上で定義した関数をテストモジュールに注入してテストを実行する。
    
    Parameters
    ----------
    test_module_name : str
        テストモジュール名（例: "01_activation_function.test_01_sigmoid"）
        test_src 以下のパスを指定する。
    filter_name : str, optional
        実行するテスト関数名を絞り込むための文字列。
        指定した場合、この文字列を含むテスト関数のみ実行される。
    **kwargs : dict
        テストモジュール内のグローバル変数（関数など）を上書きするためのキーワード引数。
        例: sigmoid_forward=my_sigmoid_forward
    """
    
    # test_src がパスに含まれているか確認し、なければ追加を試みる
    # ノートブックの場所から相対的に test_src を探すロジックを入れると親切
    # ただし、呼び出し元のパスが不明な場合もあるため、基本は sys.path 依存とする
    
    try:
        test_mod = importlib.import_module(test_module_name)
    except ImportError:
        # test_src. をつけてみる
        try:
            test_mod = importlib.import_module(f"test_src.{test_module_name}")
        except ImportError as e:
            print(f"ERROR: Cannot import test module '{test_module_name}': {e}")
            print("Make sure 'test_src' directory is in your sys.path.")
            return

    # 関数を注入（オーバーライド）
    for name, func in kwargs.items():
        if hasattr(test_mod, name):
            setattr(test_mod, name, func)
        else:
            # テストモジュールにその名前がない場合でも、テストコードが動的に参照している可能性や
            # 意図的に注入したい場合があるためセットするが、警告は出す
            print(f"Warning: Module '{test_module_name}' does not have attribute '{name}'. Injection might fail or be useless.")
            setattr(test_mod, name, func)

    # テスト関数の収集
    test_funcs = [
        getattr(test_mod, name)
        for name in dir(test_mod)
        if name.startswith("test_") and inspect.isfunction(getattr(test_mod, name))
    ]

    # フィルタリング
    if filter_name:
        test_funcs = [f for f in test_funcs if filter_name in f.__name__]

    if not test_funcs:
        print(f"No test functions found in '{test_module_name}'" + (f" matching '{filter_name}'" if filter_name else "") + ".")
        return

    print("-" * 40)
    print(f"Running tests for: {test_module_name}")
    print("-" * 40)

    success = True

    for func in test_funcs:
        try:
            # テスト実行
            func()
            print(f"  {func.__name__} {colored_text('PASS', 'green')}")
        except AssertionError as e:
            print(f"  {func.__name__} {colored_text('FAIL', 'red')}: {e}")
            success = False
        except Exception as e:
            print(f"  {func.__name__} {colored_text('ERROR', 'red')}: {e}")
            success = False

    result = 'PASS' if success else 'FAIL'
    print(f"\n => Result: {colored_text(result, 'green' if success else 'red')}")
    print("-" * 40)
