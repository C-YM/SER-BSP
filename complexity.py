# -*- coding: utf-8 -*-
"""
复杂度分析脚本：统计各模型的参数量 / MACs / FLOPs / 前向+反向时间。

在能正常 import torch 的机器上，于项目根目录运行：
    python complexity.py

- 参数量：torch 精确统计（total / trainable）
- MACs / FLOPs：手工解析式估算（保证可跑，公式见各 *_macs 函数）
                 + 若安装了 fvcore 则额外打印 fvcore 实测值作为交叉验证
- 时间：单 batch 前向+反向的平均耗时（batch_size 用训练默认 16）
"""
import time

import torch

from models.BSP import BSP
from models.CNN_LSTM_ATT_SINGLE import CNN_LSTM_ATT_SINGLE
from models.CNN_ATT_LSTM_SINGLE import CNN_ATT_LSTM_SINGLE

# 序列长度 / 特征维度：与 data_loader 里 wav2vec 特征对齐
T = 150
D = 1024
BATCH = 16          # 计时用的 batch_size（train.py 默认 16）
N_WARMUP = 3        # 计时预热次数
N_ITER = 20         # 计时测量次数


# ---------------------------------------------------------------------------
# 手工解析式 MACs 估算（MACs = 乘加次数；FLOPs 按惯例 = 2 * MACs）
# ---------------------------------------------------------------------------
def lstm_macs(input_size, hidden_size, num_layers, bidirectional, seq_len, batch=1):
    """单层/多层 LSTM 的 MACs，忽略 bias（量级可忽略）。"""
    macs = 0
    directions = 2 if bidirectional else 1
    cur_input = input_size
    for _ in range(num_layers):
        per_step = 4 * hidden_size * (cur_input + hidden_size)  # 4 个门
        macs += directions * per_step * seq_len * batch
        cur_input = hidden_size * directions
    return macs


def mha_macs(embed_dim, seq_len, batch=1):
    """MultiheadAttention(self-attention) 的 MACs。"""
    qkv = 3 * seq_len * embed_dim * embed_dim        # Q/K/V 投影
    score = seq_len * seq_len * embed_dim            # Q K^T
    attn = seq_len * seq_len * embed_dim             # softmax(A) V
    out = seq_len * embed_dim * embed_dim            # 输出投影
    return batch * (qkv + score + attn + out)


def conv2d_macs(in_ch, out_ch, h, w, kernel=1, batch=1):
    return batch * out_ch * in_ch * h * w * kernel * kernel


def linear_macs(in_f, out_f, batch=1):
    return batch * in_f * out_f


def manual_complexity(name):
    """按已知结构累加各模型的 MACs（batch=1，即单样本）。"""
    if name == 'BSP':
        macs = (
            lstm_macs(1024, 512, 1, True, T)
            + linear_macs(1024, 64) + linear_macs(64, 2)      # domain_classifier
            + linear_macs(1024, 256) + linear_macs(256, 8)    # classificationHead
        )
    elif name == 'CNN_LSTM_ATT_SINGLE':
        macs = (
            conv2d_macs(2, 64, T, D) + conv2d_macs(64, 1, T, D)   # feature
            + lstm_macs(1024, 512, 1, True, T)
            + mha_macs(1024, T)
            + linear_macs(1024, 64) + linear_macs(64, 2)
            + linear_macs(1024, 256) + linear_macs(256, 8)
        )
    elif name == 'CNN_ATT_LSTM_SINGLE':
        macs = (
            conv2d_macs(24, 128, T, D) + conv2d_macs(128, 1, T, D)  # conv1 + conv2
            + mha_macs(1024, T)
            + lstm_macs(1024, 512, 1, True, T)
            + linear_macs(1024, 256)                            # fc
            + linear_macs(256, 8)                               # classificationHead
        )
    else:
        raise ValueError(name)
    return macs


def fvcore_flops(model, inputs):
    """尝试用 fvcore 测 FLOPs，失败则返回 None。"""
    try:
        from fvcore.nn import FlopCountAnalysis
        return FlopCountAnalysis(model, inputs).total()
    except Exception as e:  # 未安装 / 不支持某算子
        print(f"    [fvcore 不可用或失败: {e}]")
        return None


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------
def benchmark(model, make_forward, device, batch=BATCH, n_warmup=N_WARMUP, n_iter=N_ITER):
    model.to(device)
    model.train()  # 测前向+反向（训练开销），须用 train 模式；cuDNN LSTM 反向只支持 train
    # 先预热
    for _ in range(n_warmup):
        loss = make_forward(model, batch)
        loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        model.zero_grad()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = make_forward(model, batch)
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    mean = sum(times) / len(times)
    return mean, mean / batch


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}\n")

    # (名称, 类, 输入形状, 前向函数)
    models = {
        'BSP': (BSP, (1, T, D), lambda m, x: m(x, 1.0)),
        'CNN_LSTM_ATT_SINGLE': (CNN_LSTM_ATT_SINGLE, (2, T, D), lambda m, x: m(x, 1.0)),
        'CNN_ATT_LSTM_SINGLE': (CNN_ATT_LSTM_SINGLE, (24, T, D), lambda m, x: m(x)),
    }

    rows = []
    for name, (cls, shape, fwd) in models.items():
        print(f"=== {name} ===")
        model = cls()

        # 1) 参数量
        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params: total={total:,} ({total/1e6:.3f}M), trainable={train:,}")

        # 2) MACs / FLOPs（手工）
        macs = manual_complexity(name)
        print(f"  MACs/sample (manual)  = {macs/1e9:.4f} G")
        print(f"  FLOPs/sample (manual) = {2*macs/1e9:.4f} G  (= 2 x MACs)")

        # 3) fvcore 交叉验证（可选）
        x = torch.randn(shape).unsqueeze(0).to(device)  # batch=1
        inputs = (x, 1.0) if name != 'CNN_ATT_LSTM_SINGLE' else (x,)
        flops = fvcore_flops(model, inputs)
        if flops is not None:
            print(f"  FLOPs/sample (fvcore) = {flops/1e9:.4f} G")

        # 4) 前向+反向时间
        def make_forward(m, b):
            return fwd(m, torch.randn(b, *shape).to(device))[0].sum()

        ms, ms_per_sample = benchmark(model, make_forward, device)
        print(f"  fwd+bwd: {ms:.2f} ms/batch (batch={BATCH}) = {ms_per_sample:.2f} ms/sample")
        print()

        rows.append((name, train, macs, 2 * macs, ms, ms_per_sample))

    # 汇总表
    print("=" * 96)
    print(f"{'model':<22}{'params(M)':>10}{'MACs(G)':>10}{'FLOPs(G)':>10}"
          f"{'ms/batch':>11}{'ms/sample':>11}")
    for name, train, macs, flops, ms, msps in rows:
        print(f"{name:<22}{train/1e6:>10.3f}{macs/1e9:>10.4f}{flops/1e9:>10.4f}"
              f"{ms:>11.2f}{msps:>11.2f}")


if __name__ == "__main__":
    main()
