---
title: MobileNetV3：轻量级网络的极致优化
comments: true
categories: 人工智能
tags: [MobileNet, 模型优化, 边缘计算]
date: 2021-07-15 10:00:00
photos:
  - "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb"
description: MobileNetV3架构与移动端部署
---

## 前言

MobileNetV3是Google在2019年发布的轻量级神经网络，通过NAS（神经架构搜索）和多种优化技术，在保持高精度的同时实现了极致的效率和速度。

## MobileNetV3核心组件

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HardSwish(nn.Module):
    """HardSwish激活函数"""
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class HardSigmoid(nn.Module):
    """HardSigmoid"""
    def forward(self, x):
        return F.relu6(x + 3) / 6

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation模块"""
    
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            HardSwish(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            HardSigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class Bneck(nn.Module):
    """MobileNetV3的基本模块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, expanded_dim=16, se=True, nl='RE'):
        super().__init__()
        
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # 非线性激活函数
        if nl == 'RE':
            self.nl = nn.ReLU(inplace=True)
        elif nl == 'HS':
            self.nl = HardSwish()
        
        # 逐点卷积扩展
        self.expanded_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_dim, 1, bias=False),
            nn.BatchNorm2d(expanded_dim),
            self.nl
        )
        
        # 深度可分离卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_dim, expanded_dim, kernel_size,
                stride, (kernel_size - 1) // 2, groups=expanded_dim, bias=False
            ),
            nn.BatchNorm2d(expanded_dim),
            self.nl
        )
        
        # SE模块
        if se:
            reduced_dim = max(1, expanded_dim // 4)
            self.se = SqueezeExcitation(expanded_dim, reduced_dim)
        else:
            self.se = nn.Identity()
        
        # 投影层
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # 扩展
        x = self.expanded_conv(x)
        
        # 深度卷积
        x = self.depthwise_conv(x)
        
        # SE注意力
        x = self.se(x)
        
        # 投影
        x = self.project_conv(x)
        
        # 残差连接
        if self.use_residual:
            return x
        else:
            return x
```

## MobileNetV3整体架构

```python
class MobileNetV3Large(nn.Module):
    """MobileNetV3-Large"""
    
    def __init__(self, num_classes=1000, dropout=0.0005):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        
        # 配置: (out_channels, kernel_size, stride, expanded_dim, se, nl)
        config = [
            # Stage 1
            (16, 3, 1, 16, False, 'RE'),
            # Stage 2
            (24, 3, 2, 64, False, 'RE'),
            (24, 3, 1, 72, False, 'RE'),
            # Stage 3
            (40, 5, 2, 72, True, 'RE'),
            (40, 5, 1, 120, True, 'RE'),
            (40, 5, 1, 120, True, 'RE'),
            # Stage 4
            (80, 3, 2, 240, False, 'HS'),
            (80, 3, 1, 200, False, 'HS'),
            (80, 3, 1, 184, False, 'HS'),
            (80, 3, 1, 184, False, 'HS'),
            # Stage 5
            (112, 3, 1, 480, True, 'HS'),
            (112, 3, 1, 672, True, 'HS'),
            # Stage 6
            (160, 5, 2, 672, True, 'HS'),
            (160, 5, 1, 960, True, 'HS'),
            (160, 5, 1, 960, True, 'HS'),
        ]
        
        # 构建bottleneck层
        layers = []
        for out_ch, kernel, stride, exp_ch, se, nl in config:
            layers.append(Bneck(
                in_channels=16 if len(layers) == 0 else config[len(layers)-1][0],
                out_channels=out_ch,
                kernel_size=kernel,
                stride=stride,
                expanded_dim=exp_ch,
                se=se,
                nl=nl
            ))
        
        self.bottlenecks = nn.Sequential(*layers)
        
        # 最后的卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish()
        )
        
        # 全局池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(960, 1280),
            HardSwish(),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.stem(x)
        x = self.bottlenecks(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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

class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small（更轻量）"""
    
    def __init__(self, num_classes=1000, dropout=0.0005):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        
        config = [
            (16, 3, 2, 16, True, 'RE'),
            (24, 3, 2, 72, False, 'RE'),
            (24, 3, 1, 88, False, 'RE'),
            (40, 5, 2, 96, True, 'HS'),
            (40, 5, 1, 240, True, 'HS'),
            (40, 5, 1, 240, True, 'HS'),
            (48, 5, 1, 120, True, 'HS'),
            (48, 5, 1, 144, True, 'HS'),
            (96, 5, 2, 288, True, 'HS'),
            (96, 5, 1, 576, True, 'HS'),
            (96, 5, 1, 576, True, 'HS'),
        ]
        
        layers = []
        for out_ch, kernel, stride, exp_ch, se, nl in config:
            layers.append(Bneck(
                in_channels=16 if len(layers) == 0 else config[len(layers)-1][0],
                out_channels=out_ch,
                kernel_size=kernel,
                stride=stride,
                expanded_dim=exp_ch,
                se=se,
                nl=nl
            ))
        
        self.bottlenecks = nn.Sequential(*layers)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HardSwish()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(576, 1024),
            HardSwish(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
```

## 训练与评估

```python
def train_mobilenetv3(model, train_loader, criterion, optimizer, device, epochs=300):
    """MobileNetV3训练"""
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%")

def evaluate_mobilenetv3(model, test_loader, device):
    """MobileNetV3评估"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total
```

## MobileNetV3特性对比

| 特性 | MobileNetV1 | MobileNetV2 | MobileNetV3 |
|------|-------------|-------------|-------------|
| 深度可分离卷积 | ✓ | ✓ | ✓ |
| 线性瓶颈 | - | ✓ | ✓ |
| 倒残差 | - | ✓ | ✓ |
| SE注意力 | - | - | ✓ |
| Hard-Swish | - | - | ✓ |
| NAS搜索 | - | - | ✓ |

## 模型量化和部署

```python
def quantize_mobilenetv3(model, dataloader):
    """量化MobileNetV3"""
    model.eval()
    
    # 动态量化
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized_model

def export_to_tflite(model, input_shape=(1, 3, 224, 224)):
    """导出为TensorFlow Lite格式"""
    # 1. 导出为ONNX
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model, dummy_input, "mobilenetv3.onnx",
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    # 2. 使用ONNX TensorFlow转换
    # onnx_tf.backend.prepare(onnx_model)
    # tf_model = onnx_tf.backend.tfgraph_def_from_onnx_model(onnx_model)
    
    # 3. 转换为TensorFlow Lite
    # converter = tf.lite.TFLiteConverter.from_session(tf_session,...)
    # tflite_model = converter.convert()
```

## MobileNetV3应用场景

- **移动端图像分类**：手机APP中的实时识别
- **边缘设备**：IoT设备上的AI推理
- **自动驾驶**：车载系统的目标检测
- **视频监控**：低功耗设备的视频分析

## 总结

MobileNetV3通过NAS搜索、SE注意力、Hard-Swish激活等多种优化技术，实现了轻量级网络在精度和速度上的最佳平衡，为移动端和边缘设备的深度学习应用提供了优秀的基础模型。

<!-- more -->

## 参考资源

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [MobileNetV3 Official Implementation](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
