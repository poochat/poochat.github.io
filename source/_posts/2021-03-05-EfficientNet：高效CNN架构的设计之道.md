---
title: EfficientNet：高效CNN架构的设计之道
comments: true
categories: 人工智能
tags: [EfficientNet, 计算机视觉, 模型优化]
date: 2021-03-05 10:00:00
photos:
  - "https://images.unsplash.com/photo-1563207153-f403bf289096"
description: EfficientNet架构详解与高效模型设计
---

## 前言

EfficientNet是Google在2019年提出的高效CNN架构，通过复合缩放策略在效率和准确率之间取得了出色的平衡。本文将深入解析EfficientNet的设计理念和实现细节。

## 复合缩放策略

EfficientNet的核心思想是**复合缩放**——同时缩放网络的深度、宽度和分辨率：

```
depth: d = α^φ
width: w = β^φ  
resolution: r = γ^φ

s.t. α × β² × γ² ≈ 2
α ≥ 1, β ≥ 1, γ ≥ 1
```

```python
import torch
import torch.nn as nn
import math

class EfficientNetConfig:
    """EfficientNet配置参数"""
    def __init__(self, width_coef, depth_coef, resolution, dropout_rate, drop_connect_rate):
        self.width_coef = width_coef
        self.depth_coef = depth_coef
        self.resolution = resolution
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate

# 不同规模的EfficientNet配置
efficientnet_b0 = EfficientNetConfig(1.0, 1.0, 224, 0.2, 0.2)
efficientnet_b1 = EfficientNetConfig(1.0, 1.1, 240, 0.2, 0.2)
efficientnet_b2 = EfficientNetConfig(1.1, 1.2, 260, 0.3, 0.2)
efficientnet_b3 = EfficientNetConfig(1.2, 1.4, 300, 0.3, 0.2)
efficientnet_b4 = EfficientNetConfig(1.4, 1.8, 380, 0.4, 0.2)
efficientnet_b5 = EfficientNetConfig(1.6, 2.2, 456, 0.4, 0.2)
efficientnet_b6 = EfficientNetConfig(1.8, 2.6, 528, 0.5, 0.2)
efficientnet_b7 = EfficientNetConfig(2.0, 3.1, 600, 0.5, 0.2)
```

## 移动逆 Bottleneck 模块（MBConv）

EfficientNet使用深度可分离卷积和倒残差块：

```python
class SwishActivation(nn.Module):
    """Swish激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    """SE模块：通道注意力"""
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            SwishActivation(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    """移动逆Bottleneck卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, expand_ratio=1, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 扩展因子
        expanded_channels = in_channels * expand_ratio
        
        if expand_ratio != 1:
            # 逐点卷积扩展
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels, eps=0.001, momentum=0.01),
                SwishActivation()
            )
        else:
            self.expand_conv = nn.Identity()
        
        # 深度可分离卷积
        pad = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride, pad, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels, eps=0.001, momentum=0.01),
            SwishActivation()
        )
        
        # SE模块
        if se_ratio > 0:
            reduced_dim = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(expanded_channels, reduced_dim)
        else:
            self.se = nn.Identity()
        
        # 投影层
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        )
        
        # 残差连接
        self.use_residual = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x):
        identity = x
        
        # 扩展
        out = self.expand_conv(x)
        
        # 深度卷积
        out = self.depthwise_conv(out)
        
        # SE注意力
        out = self.se(out)
        
        # 投影
        out = self.project_conv(out)
        
        # DropConnect
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                keep_prob = 1 - self.drop_connect_rate
                mask = torch.empty(out.shape[0], 1, 1, 1). Bernoulli_(keep_prob).to(out.device)
                out = out / keep_prob * mask
            out = out + identity
        
        return out
```

## EfficientNet架构实现

```python
class EfficientNet(nn.Module):
    """EfficientNet主网络"""
    
    def __init__(self, config, num_classes=1000, include_top=True):
        super().__init__()
        
        self.config = config
        
        # Stem
        out_channels = self._round_filters(32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            SwishActivation()
        )
        
        # Blocks
        blocks = self._make_blocks()
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        in_channels = blocks[-1].out_channels
        out_channels = self._round_filters(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            SwishActivation(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        if include_top:
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Linear(out_channels, num_classes)
            )
        
        self._initialize_weights()
    
    def _round_filters(self, filters):
        """根据宽度系数缩放滤波器数"""
        filters *= self.config.width_coef
        new_filters = max(1, int(filters + self.config.width_coef / 2))
        new_filters = int(new_filters / 8) * 8
        return new_filters
    
    def _make_blocks(self):
        """构建所有MBConv块"""
        blocks = []
        in_channels = self._round_filters(32)
        
        # 预定义的block配置
        block_configs = [
            # (expand_ratio, channels, repeats, stride, kernel_size)
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]
        
        for expand_ratio, out_ch, num_repeat, stride, kernel_size in block_configs:
            out_channels = self._round_filters(out_ch)
            strides = [stride] + [1] * (num_repeat - 1)
            
            for s in strides:
                blocks.append(MBConvBlock(
                    in_channels, out_channels, kernel_size,
                    s, expand_ratio
                ))
                in_channels = out_channels
        
        return blocks
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        if hasattr(self, 'classifier'):
            x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
```

## 训练配置

```python
def train_efficientnet(model, train_loader, num_epochs=350):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=0.256,
        momentum=0.9,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.001
    )
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
```

## EfficientNet-V2

EfficientNet-V2做了进一步优化：
- 使用Fused-MBConv替换早期阶段的MBConv
- 训练时使用更大的图像尺寸
- 使用更激进的正则化

```python
class EfficientNetV2(nn.Module):
    """EfficientNet-V2简化实现"""
    
    def __init__(self, model_cnf, num_classes=1000, dropout_rate=0.3):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        blocks = []
        for expand_ratio, out_ch, num_repeat, stride, kernel_size, se_ratio in model_cnf:
            for i in range(num_repeat):
                s = stride if i == 0 else 1
                blocks.append(
                    MBConvBlockV2(
                        32 if i == 0 else out_ch,
                        out_ch, kernel_size, s, expand_ratio, se_ratio
                    )
                )
        
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return self.classifier(torch.flatten(x, 1))
```

## 实际应用

EfficientNet广泛应用于：
- **图像分类**：ImageNet、CIFAR-10
- **目标检测**：YOLO、Faster R-CNN的骨干网络
- **语义分割**：DeepLabV3+的骨干网络
- **移动端部署**：资源受限环境的高效推理

## 总结

EfficientNet通过复合缩放策略和高效的MBConv模块，实现了出色的准确率-效率权衡，为移动端和边缘设备上的深度学习应用提供了优秀的基础架构。

<!-- more -->

## 参考资源

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
