# 損失関数 (Loss Functions)

このディレクトリには、機械学習でよく使われる損失関数の実装と、それをテストするためのコードが含まれています。

## 実装されている損失関数

### 01. MSE (Mean Squared Error / 平均二乗誤差)
- **用途**: 回帰問題
- **説明**: 予測値と実際の値の差の二乗の平均を計算します
- **ファイル**: 
  - 問題: `01_question/02_loss_function/01_mse.ipynb`
  - 回答: `02_answer/02_loss_function/01_mse.ipynb`
  - テスト: `test_src/02_loss_function/test_01_mse.py`

### 02. Cross Entropy Loss (クロスエントロピー損失)
- **用途**: 多クラス分類問題
- **説明**: LogSoftmax + NLLLoss の組み合わせと等価で、多クラス分類の損失を計算します
- **ファイル**:
  - 問題: `01_question/02_loss_function/02_cross_entropy.ipynb`
  - 回答: `02_answer/02_loss_function/02_cross_entropy.ipynb`
  - テスト: `test_src/02_loss_function/test_02_cross_entropy.py`

### 03. Binary Cross Entropy Loss (二値クロスエントロピー損失)
- **用途**: 二値分類問題
- **説明**: 二値分類における損失を計算します
- **ファイル**:
  - 問題: `01_question/02_loss_function/03_binary_cross_entropy.ipynb`
  - 回答: `02_answer/02_loss_function/03_binary_cross_entropy.ipynb`
  - テスト: `test_src/02_loss_function/test_03_binary_cross_entropy.py`

## ノートブックの使い方

1. `01_question/02_loss_function/` または `02_answer/02_loss_function/` ディレクトリ内のノートブックを開きます
2. 各セルを順番に実行します
3. `@test` デコレータを使用して、実装が正しいかどうか自動的にテストされます
4. テストが成功すると、PyTorchの実装と一致していることが確認できます

## テストの実行方法

### ノートブックから実行
ノートブック内のセルを実行すると、自動的にテストが実行されます。

### コマンドラインから実行
```bash
# MSEのテスト
pytest test_src/02_loss_function/test_01_mse.py -v

# Cross Entropyのテスト
pytest test_src/02_loss_function/test_02_cross_entropy.py -v

# Binary Cross Entropyのテスト
pytest test_src/02_loss_function/test_03_binary_cross_entropy.py -v

# すべての損失関数のテスト
pytest test_src/02_loss_function/ -v
```

## 実装の詳細

各損失関数について、以下の2つの関数を実装しています:

1. **Forward (順伝播)**: 損失値を計算する関数
2. **Backward (逆伝播)**: 勾配を計算する関数

すべての実装はPyTorchの実装と数値的に一致することが検証されています。

## 注意事項

- 実装はNumPyを使用しています
- テストではPyTorchの実装と比較して正確性を検証しています
- 数値安定性のための工夫（log-sum-expトリックやクリッピングなど）も含まれています
