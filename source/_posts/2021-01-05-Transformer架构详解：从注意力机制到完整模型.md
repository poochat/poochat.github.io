---
title: Transformer架构详解：从注意力机制到完整模型
comments: true
categories: 人工智能
tags: [Transformer, 深度学习, NLP]
date: 2021-01-05 10:00:00
photos:
  - "https://images.unsplash.com/photo-1633356122544-f60d17f03e20"
description: 深入解析Transformer架构的原理与实现
---

## 前言

2017年，Google在论文《Attention Is All You Need》中提出了Transformer架构，彻底改变了自然语言处理领域的发展方向。本文将深入解析Transformer的核心组件和工作原理。

## 注意力机制（Attention Mechanism）

注意力机制是Transformer的核心创新。传统的序列模型（如LSTM）需要按顺序处理序列，而注意力机制允许模型同时关注序列中的所有位置。

### 自注意力机制（Self-Attention）

自注意力机制通过三个可学习的权重矩阵将输入转换为Query、Key和Value：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "embed_size must be divisible by heads"
        
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = F.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        return self.fc_out(out)
```

### 多头注意力（Multi-Head Attention）

多头注意力并行运行多个注意力机制，让模型能够同时关注不同位置的不同表示子空间：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

## 位置编码（Positional Encoding）

由于Transformer没有循环结构，需要添加位置编码来注入序列位置信息：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

## Transformer编码器结构

编码器由多个相同的层堆叠而成，每层包含两个子层：

1. **多头自注意力层**：使用自注意力机制处理输入
2. **前馈神经网络**：两层全连接网络

每个子层都使用残差连接和层归一化：

```
LayerNorm(x + Sublayer(x))
```

## 实际应用

Transformer架构已成为现代NLP的基础：

- **BERT**：用于各种NLP任务的预训练模型
- **GPT系列**：生成式预训练Transformer
- **机器翻译**：Google Translate等系统
- **文本摘要**：自动生成文章摘要

## 总结

Transformer通过注意力机制实现了并行计算，大大提高了训练效率。其核心思想——通过自注意力建模序列关系——已成为深度学习领域的重要范式。

<!-- more -->

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
