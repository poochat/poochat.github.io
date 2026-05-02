---
title: ViT：Vision Transformer详解
comments: true
categories: 人工智能
tags: [ViT, Vision Transformer, 计算机视觉]
date: 2021-06-05 10:00:00
photos:
  - "https://images.unsplash.com/photo-1633356122544-f60d17f03e20"
description: Vision Transformer原理与PyTorch实现
---

## 前言

Vision Transformer（ViT）将NLP领域的Transformer架构成功应用于计算机视觉，在ImageNet上取得了超越CNN的成绩。本文深入解析ViT的原理和实现。

## ViT核心架构

```
mermaid
graph TB
A[输入图像 224x224] --> B[Patch Embedding]
B --> C[线性投影+位置编码]
C --> D[CLS Token]
D --> E[Transformer编码器]
E --> F[分类头]
F --> G[预测类别]
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """图像分块嵌入"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积实现分块+嵌入
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # flatten后维度
        self.flatten_dim = in_channels * patch_size * patch_size
        
        # 可选的线性投影
        self.linear_proj = nn.Linear(self.flatten_dim, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        # 卷积分块
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        
        # 展平
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        
        # 维度变换
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # 线性投影（可选）
        x = self.linear_proj(x)  # (B, num_patches, embed_dim)
        
        return x

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        self.scale = self.d_k ** -0.5
    
    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # QKV投影
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 注意力
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, return_attention=False):
        # 自注意力 + 残差
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MLP + 残差
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x
```

## 完整ViT模型

```python
class VisionTransformer(nn.Module):
    """完整的Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # 分块嵌入
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        
        # 可学习类别token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, attn_dropout
            )
            for _ in range(depth)
        ])
        
        # 输出头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, return_attention=False):
        B = x.shape[0]
        
        # 分块嵌入
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 通过Transformer块
        attentions = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x)
        
        x = self.norm(x)
        
        # 使用CLS token做分类
        cls_output = x[:, 0]
        
        # 分类
        logits = self.head(cls_output)
        
        if return_attention:
            return logits, attentions
        return logits
```

## ViT变体配置

```python
configs = {
    'ViT-Base': {
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'params': 86M
    },
    'ViT-Large': {
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'params': 307M
    },
    'ViT-Huge': {
        'patch_size': 14,
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'params': 632M
    },
    'DeiT-Small': {
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'params': 22M
    }
}

def create_vit(variant='ViT-Base', num_classes=1000):
    config = configs[variant]
    return VisionTransformer(
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        num_classes=num_classes
    )
```

## ViT训练

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class ViTTrainer:
    """ViT训练器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total

def get_pretrained_vit(model_name='ViT-B_16', num_classes=1000, pretrained=True):
    """加载预训练ViT"""
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    
    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
    else:
        model = vit_b_16(weights=None)
    
    # 修改分类头
    if num_classes != 1000:
        model.head = nn.Linear(model.hidden_dim, num_classes)
    
    return model
```

## ViT vs CNN

| 特性 | ViT | CNN |
|------|-----|-----|
| 归纳偏置 | 无，需要更多数据 | 局部性、平移不变性 |
| 计算复杂度 | O(n²d) | O(k²dn) |
| 长距离依赖 | 自然建模 | 需要更多层 |
| 数据效率 | 需大数据集 | 高 |
| 可解释性 | Attention可视化 | Filter可视化 |

## ViT应用场景

- **图像分类**：ImageNet、CIFAR
- **目标检测**：DETR、Deformable DETR
- **语义分割**：SegFormer、SETR
- **图像生成**：DiT（DALL-E 2）

## 总结

ViT证明了Transformer可以直接应用于图像任务，其在大规模数据训练下展现出超越CNN的潜力。但在小数据集上，CNN由于其归纳偏置仍然具有优势。

<!-- more -->

## 参考资源

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877)
