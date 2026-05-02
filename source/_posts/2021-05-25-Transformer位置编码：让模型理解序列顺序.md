---
title: Transformer位置编码：让模型理解序列顺序
comments: true
categories: 人工智能
tags: [位置编码, Transformer, 深度学习]
date: 2021-05-25 10:00:00
photos:
  - "https://images.unsplash.com/photo-1526379095098-d400fd0bf935"
description: Transformer位置编码原理与实现
---

## 前言

Transformer架构中没有循环结构，需要通过位置编码（Positional Encoding）来注入序列位置信息。本文深入解析各种位置编码方法的原理和实现。

## 位置编码概述

```
mermaid
graph LR
A[输入Token] --> B[词嵌入]
A --> C[位置编码]
B --> D[加和]
C --> D
D --> E[Transformer层]
```

## 绝对位置编码

### 正弦/余弦位置编码

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """正弦/余弦位置编码（原始Transformer）"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 频率因子
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 数学公式
def get_sinusoid_encoding_table(n_position, d_hid):
    """
    获取正弦位置编码表
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def get_angle_dim(position, dim_idx):
        return position / (10000 ** (2 * dim_idx / d_hid))
    
    # 计算所有位置和维度的编码
    table = torch.zeros(n_position, d_hid)
    for pos in range(n_position):
        for dim_idx in range(d_hid):
            angle = get_angle_dim(pos, dim_idx)
            if dim_idx % 2 == 0:
                table[pos, dim_idx] = math.sin(angle)
            else:
                table[pos, dim_idx] = math.cos(angle)
    
    return table
```

### 可学习的位置编码

```python
class LearnedPositionalEncoding(nn.Module):
    """可学习的位置编码"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 可学习的参数
        self.position_embeddings = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 创建位置索引
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 获取位置嵌入
        position_emb = self.position_embeddings(position_ids)
        
        # 相加
        x = x + position_emb
        return self.dropout(x)
```

## 相对位置编码

相对位置编码关注token之间的相对距离：

```python
class RelativePositionalEncoding(nn.Module):
    """相对位置编码（Shaw等人提出）"""
    
    def __init__(self, d_model, max_relative_position=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 相对位置表示
        self.relative_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, d_model)
        )
    
    def forward(self, length):
        """
        生成相对位置编码
        """
        # 生成相对位置索引
        range_vec = torch.arange(length)
        relative_indices = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        relative_indices = relative_indices.clamp(-self.max_relative_position, 
                                                   self.max_relative_position)
        relative_indices = relative_indices + self.max_relative_position
        
        # 获取编码
        return self.relative_embeddings[relative_indices]

class ShawAttention(nn.Module):
    """带相对位置编码的注意力"""
    
    def __init__(self, d_model, num_heads, max_relative_position=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_relative_position = max_relative_position
        
        # QKV投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 相对位置编码
        self.relative_embeddings = RelativePositionalEncoding(
            d_model, max_relative_position
        )
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # QKV变换
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 添加相对位置编码
        length = query.size(1)
        relative_encoding = self.relative_embeddings(length)
        relative_scores = torch.matmul(Q.transpose(1, 2), relative_encoding.transpose(0, 1))
        scores = scores + relative_scores
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)
```

## Rotary Position Embedding (RoPE)

RoPE是一种旋转式位置编码：

```python
class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码（RoPE）"""
    
    def __init__(self, dim, max_position=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position = max_position
        
        # 预计算旋转角度
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 缓存
        self._cache = {}
    
    def forward(self, seq_len, device):
        if seq_len in self._cache:
            return self._cache[seq_len].to(device)
        
        # 计算位置
        position = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        positions = torch.einsum('i,j->ij', position, self.inv_freq)
        
        # 复数形式
        embeddings = torch.cat([positions, positions], dim=-1)
        embeddings = torch.polar(
            torch.ones_like(embeddings), embeddings
        )  # (seq_len, dim)
        
        self._cache[seq_len] = embeddings
        return embeddings
    
    def rotate_half(self, x):
        """将x的后半部分旋转"""
        x1 = x[..., :x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """应用旋转位置编码"""
        # (seq_len, dim) -> (1, seq_len, 1, dim)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed

def apply_rope(x, freqs_cis):
    """应用RoPE"""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)
```

## ALiBi位置编码

ALiBi（Attention with Linear Biases）不需要位置嵌入：

```python
class ALiBiAttention(nn.Module):
    """ALiBi注意力"""
    
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
        # 预定义斜率
        self.register_buffer(
            'slopes',
            torch.tensor(self._get_slopes(num_heads))
        )
    
    def _get_slopes(self, num_heads):
        """获取斜率"""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2) +
                self._get_slopes(closest_power_of_2 * 2)[0::2][:num_heads - closest_power_of_2]
            )
        
        return slopes
    
    def forward(self, q, k, v, mask=None):
        # 计算距离矩阵
        b, h, l, d = q.shape
        mk = torch.arange(l, device=q.device).unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
        qk = torch.arange(l, device=q.device).unsqueeze(0).unsqueeze(-1).expand(b, h, -1, -1)
        
        distance = mk - qk  # (b, h, l, l)
        
        # 应用线性偏置
        distance = distance.abs().neg()
        distance = distance.unsqueeze(1) * self.slopes.view(1, -1, 1, 1)
        
        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores + distance
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        return F.softmax(scores, dim=-1) @ v
```

## 位置编码对比

| 类型 | 优点 | 缺点 | 应用 |
|------|------|------|------|
| Sin/Cos | 可外推 | 表达能力有限 | 原始Transformer |
| Learned | 灵活 | 可能过拟合 | BERT |
| Relative | 更好建模相对关系 | 计算复杂 | XLNet |
| RoPE | 高效、无需额外参数 | 实现复杂 | LLaMA, GPT-NeoX |
| ALiBi | 无需训练、可外推 | 与attention耦合 | BLOOM |

## 总结

位置编码是Transformer理解序列顺序的关键组件。从最初的固定正弦编码到现在的旋转编码和ALiBi，位置编码技术不断演进，为大语言模型的发展提供了重要支撑。

<!-- more -->

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [ALiBi: TRAIN SHORT, TEST LONG](https://arxiv.org/abs/2108.12409)
