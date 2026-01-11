# ml-coding-practice-pytorch
機械学習の勉強用リポジトリ

このリポジトリは、PyTorchとNumPyを使用して機械学習の基礎的な概念を実装し、理解を深めるための教育用リポジトリです。

##  実装されている内容

### 01. 活性化関数 (Activation Functions)
- Sigmoid
- ReLU
- Tanh
- Softmax

### 02. 損失関数 (Loss Functions)
- MSE (Mean Squared Error / 平均二乗誤差)
- Cross Entropy Loss (クロスエントロピー損失)
- Binary Cross Entropy Loss (二値クロスエントロピー損失)

### 03. Tensor操作 (Tensor Operations)
- Tensor作成
- Tensor結合

### 04. 正規化 (Normalization)
- Batch Normalization

### 05. 行列計算 (Matrix Operations)
- 行列積 (Matrix Multiplication)

### 06. 最適化アルゴリズム (Optimizers)
- SGD (Stochastic Gradient Descent)
- Momentum

### 07. 畳み込み (Convolution)
- 2D Convolution (2次元畳み込み)

### 08. プーリング (Pooling)
- Max Pooling

### 09. 正則化 (Regularization)
- Dropout

### 10. 重み初期化 (Weight Initialization)
- Xavier初期化 (Glorot初期化)
- He初期化

### 11. 高度な最適化アルゴリズム (Advanced Optimizers)
- Adam (Adaptive Moment Estimation)
- RMSProp (Root Mean Square Propagation)

各実装について:
- **Forward (順伝播)**: 値を計算する関数
- **Backward (逆伝播)**: 勾配を計算する関数（該当する場合）

すべての実装はPyTorchの実装と数値的に一致することが自動テストで検証されます。

##  実行方法

### 1. 環境のセットアップ

リポジトリをクローンした後、プロジェクトフォルダに移動し、以下のコマンドを実行:

```bash
uv sync
```

### 2. ノートブックの実行

VS Codeで以下のディレクトリ内のノートブックを開いて実行します:

- `01_question/` - 問題形式（回答も含まれています）
- `02_answer/` - 回答例

各ノートブックのセルを実行すると、自動的にテストが実行され、実装の正確性が検証されます。

### 3. テストの実行

コマンドラインからもテストを実行できます:

```bash
# 活性化関数のテスト
pytest test_src/01_activation_function/ -v

# 損失関数のテスト
pytest test_src/02_loss_function/ -v

# Tensor操作のテスト
pytest test_src/03_tensor_operations/ -v

# 正規化のテスト
pytest test_src/04_normalization/ -v

# 行列計算のテスト
pytest test_src/05_matrix_operations/ -v

# 最適化アルゴリズムのテスト
pytest test_src/06_optimizers/ -v

# 畳み込みのテスト
pytest test_src/07_convolution/ -v

# プーリングのテスト
pytest test_src/08_pooling/ -v

# 正則化のテスト
pytest test_src/09_regularization/ -v

# 重み初期化のテスト
pytest test_src/10_weight_initialization/ -v

# 高度な最適化アルゴリズムのテスト
pytest test_src/11_advanced_optimizers/ -v

# すべてのテスト
pytest test_src/ -v
```

##  ディレクトリ構造

```
ml-coding-practice-pytorch/
├─ 01_question/              # 問題ノートブック（回答付き）
│   ├─ 01_activation_function/
│   ├─ 02_loss_function/
│   ├─ 03_tensor_operations/
│   ├─ 04_normalization/
│   ├─ 05_matrix_operations/
│   ├─ 06_optimizers/
│   ├─ 07_convolution/
│   ├─ 08_pooling/
│   ├─ 09_regularization/
│   ├─ 10_weight_initialization/
│   └─ 11_advanced_optimizers/
├─ 02_answer/                # 回答ノートブック
│   ├─ 01_activation_function/
│   ├─ 02_loss_function/
│   ├─ 03_tensor_operations/
│   ├─ 04_normalization/
│   ├─ 05_matrix_operations/
│   ├─ 06_optimizers/
│   ├─ 07_convolution/
│   ├─ 08_pooling/
│   ├─ 09_regularization/
│   ├─ 10_weight_initialization/
│   └─ 11_advanced_optimizers/
└─ test_src/                 # テストコード
    ├─ 01_activation_function/
    ├─ 02_loss_function/
    ├─ 03_tensor_operations/
    ├─ 04_normalization/
    ├─ 05_matrix_operations/
    ├─ 06_optimizers/
    ├─ 07_convolution/
    ├─ 08_pooling/
    ├─ 09_regularization/
    ├─ 10_weight_initialization/
    └─ 11_advanced_optimizers/
```

##  テストシステム

各ノートブックには `@test` デコレータが付いており、セルを実行すると自動的に:

1. NumPyでの実装を実行
2. PyTorchの実装と比較
3. 結果が一致するか検証
4. 結果を表示

##  注意事項

- 実装はNumPyを使用しています
- テストではPyTorchの実装と比較して正確性を検証しています
- 数値安定性のための工夫（log-sum-expトリック、クリッピングなど）も含まれています

##  学習のポイント

1. **順伝播と逆伝播の理解**: 各関数について、値の計算と勾配の計算の両方を実装
2. **数値安定性**: 実際の実装で必要な数値計算上の工夫を学習
3. **テスト駆動**: PyTorchとの比較により、実装の正確性を確認
4. **行列演算**: 線形代数の基礎となる行列操作を実装
5. **最適化**: 勾配降下法の仕組みを理解
6. **CNN**: 畳み込みとプーリングによる画像処理
7. **正則化**: 過学習を防ぐ技術
8. **重み初期化**: ネットワークの学習安定性を確保
9. **適応的学習率**: 効率的な学習のための最適化手法

##  E資格対策

トピック07-11は、日本ディープラーニング協会のE資格（JDLA Deep Learning for ENGINEER）試験対策に特化した内容を含んでいます：

- **CNNの基礎**: 畳み込み、プーリング
- **正則化手法**: Dropout
- **重み初期化**: Xavier/He初期化
- **最適化アルゴリズム**: Adam、RMSProp

これらの実装を通じて、ディープラーニングの理論と実践の両方を学ぶことができます。
