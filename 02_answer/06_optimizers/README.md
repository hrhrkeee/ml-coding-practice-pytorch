# 最適化アルゴリズム (Optimizers)

勾配降下法に基づく最適化アルゴリズムを実装します。

## 実装されている内容

### 01. SGD (Stochastic Gradient Descent)
確率的勾配降下法 - 最もシンプルな最適化手法

### 02. Momentum SGD
運動量を使った確率的勾配降下法 - SGDの改良版

## 学習のポイント

- 勾配降下法の基本原理
- 学習率の役割
- Momentumによる最適化の加速
- パラメータ更新の仕組み

## 使い方

各ノートブックを開いてセルを実行すると、自動的にPyTorchの実装と比較してテストされます。

## テストの実行

```bash
pytest test_src/06_optimizers/ -v
```
