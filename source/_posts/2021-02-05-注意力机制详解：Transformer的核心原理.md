---
title: 注意力机制详解：Transformer的核心原理
comments: true
categories: 人工智能
tags: [注意力机制, Transformer, 深度学习]
date: 2021-02-05 10:00:00
photos:
  - "https://images.unsplash.com/photo-1558618666-fcd25c85cd64"
description: 深入理解注意力机制的工作原理
---

## 前言

注意力机制（Attention Mechanism）是深度学习领域的重大突破之一，尤其在Transformer架构中发挥着核心作用。本文将详细解析注意力机制的数学原理和实现细节。

## 注意力机制的起源

注意力机制最早在视觉领域提出，后来被引入到序列模型中。2017年的《Attention Is All You Need》将其发扬光大，成为现代深度学习的基石。

## 注意力机制的数学原理

### Scaled Dot-Product Attention

标准的注意力机制定义为：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- Q（Query）：查询向量
- K（Key）：键向量  
- V（Value）：值向量
- d_k：键向量的维度

### 为什么需要缩放因子？

除以√d_k是为了防止点积过大导致softmax进入饱和区域：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力
    
    Args:
        Q: (batch_size, num_heads, seq_len, d_k)
        K: (batch_size, num_heads, seq_len, d_k)
        V: (batch_size, num_heads, seq_len, d_v)
        mask: (batch_size, num_heads, seq_len, seq_len)
    
    Returns:
        output: (batch_size, num_heads, seq_len, d_v)
        attention_weights: (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # 计算点积
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# 示例
batch_size = 2
num_heads = 8
seq_len = 10
d_k = 64

Q = torch.randn(batch_size, num_heads, seq_len, d_k)
K = torch.randn(batch_size, num_heads, seq_len, d_k)
V = torch.randn(batch_size, num_heads, seq_len, d_k)

output, attention = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention.shape}")
```

## 多头注意力机制

多头注意力允许模型同时关注不同位置的不同子空间：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x, batch_size):
        """将隐藏维度分割成多个头"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq, d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 分割头
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # 计算注意力
        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并头
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(attn_output)
        
        return output, attention_weights
```

## 掩码机制

### 填充掩码（Padding Mask）

处理不同长度的序列：

```python
def create_padding_mask(seq, pad_idx=0):
    """
    创建填充掩码
    返回的掩码中，1表示有效位置，0表示填充位置
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # (batch, 1, 1, seq_len)

# 示例
seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
padding_mask = create_padding_mask(seq)
print(padding_mask)
# tensor([[[[True, True, True, False, False]]],
#         [[[True, True, False, False, False]]]])
```

### 因果掩码（Causal Mask）

防止看到未来信息：

```python
def create_causal_mask(seq_len):
    """
    创建因果掩码（上三角为负无穷）
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len), diagonal=1
    ).type(torch.bool)
    return mask

# 或在attention中直接实现
def causal_mask_attention(Q, K, V):
    """带因果掩码的注意力"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 创建因果掩码
    seq_len = Q.size(2)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=Q.device), diagonal=1
    ).bool()
    scores.masked_fill_(causal_mask, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

## 注意力可视化

```python
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens, save_path='attention.png'):
    """可视化注意力权重"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    attention = attention_weights[0].cpu().numpy()  # 取第一个样本
    
    im = ax.imshow(attention, cmap='Blues')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 示例tokens
tokens = ['[CLS]', '今天', '天气', '很', '好', '[SEP]']
attention_weights = attention[0]  # 假设的注意力权重
visualize_attention(attention_weights.unsqueeze(0), tokens)
```

## 注意力机制的类型

### 1. 自注意力（Self-Attention）

Query、Key、Value来自同一输入：

```python
# 自注意力
self_attn_output, _ = multi_head_attn(x, x, x, mask)
```

### 2. 编码器-解码器注意力（Cross Attention）

Query来自解码器，Key/Value来自编码器：

```python
# 交叉注意力
cross_attn_output, _ = multi_head_attn(
    query=decoder_output,  # 解码器
    key=encoder_output,    # 编码器
    value=encoder_output,   # 编码器
    mask=None
)
```

### 3. 双向注意力

BERT使用的双向注意力：

```python
# 双向注意力（无因果掩码）
bidirectional_attn, _ = scaled_dot_product_attention(Q, K, V)
```

## 实际应用

注意力机制广泛应用于：
- **机器翻译**：对齐源语言和目标语言
- **图像描述**：关注图像的相关区域
- **语音识别**：对齐音频和文本
- **推荐系统**：建模用户-物品交互

## 总结

注意力机制通过动态加权聚合信息，解决了长距离依赖问题，是Transformer成功的关键。其变体（多头注意力、交叉注意力等）为各种深度学习任务提供了强大的建模能力。

<!-- more -->

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
