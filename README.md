# DEMOS_SOLVE_MUL_bsp

基于 **Batch Spectral Penalization (BSP)** 的跨域语音情感识别（Speech Emotion Recognition, SER）代码，数据集为 DEMOS 情感语音库。

## 方法概述

- **特征**：预提取的 wav2vec 特征（`.npz`，由 `data_loader.py` 读取），可经 `Feature_Extraction.py` 用 CNN-LSTM-Attention 模型进一步提取 embedding。
- **域适应**：对抗训练（梯度反转层 GRL + 域判别器）+ BSP 损失（对源域 / 目标域特征矩阵的奇异值平方做惩罚，抑制域迁移时可迁移性的退化）。
- **模型**：
  - `BSP`：本文提出的方法；
  - `CNN_LSTM_ATT_SINGLE`：DANN / GRL 对抗域适应基线。
- **微调基线**：`wav2vec.py` / `huBERT.py` 端到端微调预训练模型（wav2vec2 / WavLM）直接做 SER 分类。

## 目录结构

```
├── train.py               # 训练入口（argparse 配置）
├── model_trainer.py       # 训练器（含 BSP 损失 batch_spectral_penalization）
├── data_loader.py         # 数据加载（含声学特征集定义）
├── Feature_Extraction.py  # 特征提取：CNN-LSTM-Attention 模型提 embedding 存 .npz
├── utils.py               # 参数初始化、混淆矩阵绘制等
├── params.yaml            # 微调相关配置
├── wav2vec.py             # 微调 wav2vec2 做 SER 分类（基线）
├── huBERT.py              # 微调 WavLM 做 SER 分类（基线，文件名与实际模型不符）
└── models/
    ├── BSP.py                  # BSP 域适应模型（核心方法）
    ├── CNN_LSTM_ATT_SINGLE.py  # DANN / GRL 对抗基线
    └── CNN_ATT_LSTM_SINGLE.py  # 24 层特征提取模型（Feature_Extraction.py 用）
```

## 环境依赖

见 `requirements.txt`。主要依赖：PyTorch、transformers、datasets、evaluate、scikit-learn 等。

## 数据准备

代码中的绝对路径已用占位符 `USERNAME` 替换，运行前请把以下文件里含 `USERNAME` 的路径改成你自己的数据路径：

- `train.py`（argparse 默认路径）
- `data_loader.py`（特征 `.npz` 路径）
- `Feature_Extraction.py`（输入 wav2vec 特征 / 输出 embedding 路径）
- `utils.py`（结果保存路径）
- `wav2vec.py`、`huBERT.py`（标签 / wav / 结果路径）

## 训练

```bash
python train.py --model BSP --lr 0.0005 --epoch 30
```

- `--model`：`BSP` 或 `CNN_LSTM_ATT_SINGLE`
- 其余路径参数见 `train.py` 的 argparse 默认值。

微调分类（基线）：

```bash
python wav2vec.py   # 或 python huBERT.py
```

## 声学特征集说明

`data_loader.py` 中保留了 `FEATURES_LENGTH` 与 `load_feature()`，支持以下特征集：

- `deepspectrum`：DeepSpectrum，1920 维
- `opensmile`：openSMILE 特征，6374 维（对应 ComParE 一类 openSMILE 特征集）
- `audeep`：auDeep，4096 维
- `xbow`：bag-of-audio-words，2000 维

注意：当前训练主流程硬编码为 wav2vec 的 `.npz` 特征，上述特征集的读取入口存在，但尚未接入 `__getitem__` 的实际数据加载。

## 引用

TODO: 补充论文引用信息。
