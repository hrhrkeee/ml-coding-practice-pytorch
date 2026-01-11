# Tensor操作 (Tensor Operations)

PyTorchのTensorを使った基本的な操作を実装します。

## 実装されている内容

### 01. Tensor Basics
- **Tensor作成**: 指定された形状のランダムなTensorを作成
- **Tensor結合**: 複数のTensorを指定された軸で結合（concatenate）

### 02. Tensor Reshape
- **Reshape**: Tensorの形状を変更
- **Flatten**: 多次元Tensorを1次元に変換

## 学習のポイント

- Tensorの基本的な生成方法
- 軸（axis/dim）の概念と操作
- メモリレイアウトと形状変換

## 使い方

各ノートブックを開いてセルを実行すると、自動的にPyTorchの実装と比較してテストされます。

## テストの実行

```bash
pytest test_src/03_tensor_operations/ -v
```
