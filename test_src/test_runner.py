# project_root/test_src/test_runner.py

import importlib
import inspect

def run_tests(module_name: str):
    """
    test_<module_name>.py 内の test_ で始まる関数を順次実行

    Parameters
    ----------
    module_name : str
        実装ファイル名（例: "01_sigmoid"）
    """

    print("-" * 40)
    print(f"Running tests for: {module_name}")
    print("-" * 40)
    print()

    # ディレクトリ構造（ドット区切り）を考慮してテストモジュール名を構築
    if "." in module_name:
        parts = module_name.split(".")
        parts[-1] = f"test_{parts[-1]}"
        test_mod_name = ".".join(parts)
    else:
        test_mod_name = f"test_{module_name}"

    try:
        test_mod = importlib.import_module(test_mod_name)
    except ImportError as e:
        print(f"ERROR: Cannot import test module '{test_mod_name}': {e}")
        return False

    test_funcs = [
        getattr(test_mod, name)
        for name in dir(test_mod)
        if name.startswith("test_") and inspect.isfunction(getattr(test_mod, name))
    ]

    if not test_funcs:
        print(f"No test functions found in '{test_mod_name}'.")
        return False

    success = True

    for func in test_funcs:
        try:
            func()
            print(f"  {func.__name__} {colored_text('PASS', 'green')}")
        except AssertionError as e:
            print(f"  {func.__name__} {colored_text('FAIL', 'red')}: {e}")
            success = False
        except Exception as e:
            print(f"  {func.__name__} {colored_text('ERROR', 'red')}: {e}")
            success = False

    result = 'PASS' if success else 'FAIL'
    print(f"\n => Result for {module_name}: {colored_text(result, 'green' if success else 'red')}")
    print("-" * 40)
    return success


def colored_text(text: str, color: str) -> str:
    """
    テキストに色を付ける

    Parameters
    ----------
    text : str
        色を付けるテキスト
    color : str
        色の種類（'green' または 'red'）

    Returns
    -------
    str
        ANSI カラーコード付きテキスト
    """
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'reset': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"