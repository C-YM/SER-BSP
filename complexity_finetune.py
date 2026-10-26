# -*- coding: utf-8 -*-
"""
微调基线（wav2vec2-large / hubert-large / wavlm-large）的复杂度统计。

运行前注意：
    1. 需要联网，会自动下载预训练权重（每个约 1.2GB）。
    2. 建议在 GPU 上跑；batch=1 计时用 2 秒音频（与 wav2vec.py 里
       max_length = 2 * target_sampling_rate = 32000 对齐）。
    3. 这三个模型主干参数量是公开已知的（~316M / ~317M），
       如果不想下载，直接引文献值即可；本脚本只用于精确复核 + 计时。

用法：
    python complexity_finetune.py
"""
import time

import torch
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2ForSequenceClassification,
    WavLMForSequenceClassification,
)

AUDIO_LEN = 32000          # 2 秒 @ 16kHz
BATCH = 1                  # 微调模型较大，计时用 batch=1
N_WARMUP = 3
N_ITER = 10

# (名称, checkpoint, 模型类)
BASELINES = [
    ("wav2vec2-large", "jonatasgrosman/wav2vec2-large-xlsr-53-english", Wav2Vec2ForSequenceClassification),
    ("hubert-large",   "facebook/hubert-large-ls960-ft",           HubertForSequenceClassification),
    ("wavlm-large",    "microsoft/wavlm-large",                    WavLMForSequenceClassification),
]


def benchmark(model, x, device, n_warmup=N_WARMUP, n_iter=N_ITER):
    model.to(device)
    model.train()  # 测前向+反向（训练开销），须用 train 模式
    x = x.to(device)
    for _ in range(n_warmup):
        loss = model(x).logits.sum()
        loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        model.zero_grad()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(x).logits.sum()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sum(times) / len(times)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}, audio_len = {AUDIO_LEN} (2s @16kHz), batch = {BATCH}\n")

    rows = []
    for name, ckpt, cls in BASELINES:
        print(f"=== {name} ===  (loading {ckpt} ...)")
        model = cls.from_pretrained(ckpt, num_labels=8, ignore_mismatched_sizes=True)

        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 与你代码里 freezeExtractor=true 对齐：冻结 conv 特征提取器后统计可训练参数
        if hasattr(model, "freeze_feature_extractor"):
            model.freeze_feature_extractor()
        train_frozen_ext = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  params: total={total:,} ({total/1e6:.2f}M)")
        print(f"          trainable(全训练)  = {train:,} ({train/1e6:.2f}M)")
        print(f"          trainable(freezeExtractor=true) = {train_frozen_ext:,} ({train_frozen_ext/1e6:.2f}M)")

        x = torch.randn(BATCH, AUDIO_LEN)
        try:
            ms = benchmark(model, x, device)
            print(f"  fwd+bwd: {ms:.1f} ms/sample (batch={BATCH})")
        except Exception as e:
            print(f"  [计时失败: {e}]")
            ms = float('nan')
        print()
        rows.append((name, total, train_frozen_ext, ms))

    print("=" * 72)
    print(f"{'model':<16}{'total(M)':>11}{'trainable(M)':>13}{'ms/sample':>12}")
    for name, total, train, ms in rows:
        print(f"{name:<16}{total/1e6:>11.2f}{train/1e6:>13.2f}{ms:>12.1f}")


if __name__ == "__main__":
    main()
