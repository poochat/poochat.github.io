---
title: Transformer可视化与可解释性技术
date: 2024-09-10 09:00:00
categories: 可解释AI
tags: [Transformer, 可视化, 可解释性, 注意力机制, XAI]
photos:
  - "https://images.unsplash.com/photo-1620712943543-bcc4688e7485"
---

# Transformer可视化与可解释性技术

## 引言

Transformer模型因其卓越性能被广泛应用，但其"黑盒"特性带来了可解释性的挑战。本文介绍Transformer可视化与可解释性的核心技术。

## 注意力可视化

### 注意力权重可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    """注意力可视化工具"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def visualize_attention(self, text, layer=6, head=0):
        inputs = self.tokenizer(text, return_tensors='pt')
        
        # 获取注意力权重
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions[layer][0, head].numpy()
        
        # 可视化
        plt.figure(figsize=(12, 10))
        sns.heatmap(attentions, 
                    xticklabels=tokens, 
                    yticklabels=tokens,
                    cmap='viridis')
        plt.title(f'Layer {layer}, Head {head} Attention')
        plt.savefig('attention_heatmap.png')
        
    def visualize_token_importance(self, text):
        """词元重要性分析"""
        tokens = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(text, return_tensors='pt')
        
        # Integrated Gradients
        ig = IntegratedGradients(self.model)
        importances = ig.attribute(inputs['input_ids'])
        
        return list(zip(tokens, importances))
```

## 可解释性方法

### Integrated Gradients

```python
class IntegratedGradients:
    """积分梯度方法"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
    def attribute(self, inputs, baseline=None, n_steps=50):
        if baseline is None:
            baseline = torch.zeros_like(inputs)
            
        # 线性插值
        scaled_inputs = [
            baseline + (float(i) / n_steps) * (inputs - baseline)
            for i in range(n_steps + 1)
        ]
        
        # 计算梯度
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)
            output = self.model(scaled_input)
            grad = torch.autograd.grad(output.sum(), scaled_input)[0]
            gradients.append(grad)
            
        # 平均梯度
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # 积分
        attribution = (inputs - baseline) * avg_gradients
        
        return attribution
```

### Layer-wise Relevance Propagation

```python
class LRPTransformer:
    """LRP用于Transformer"""
    
    def __init__(self, model):
        self.model = model
        
    def decompose(self, inputs):
        """分解预测到输入"""
        relevance = self.forward_pass(inputs)
        relevance = self.backward_pass(relevance)
        return relevance
    
    def forward_pass(self, x):
        # 获取各层输出
        layer_outputs = []
        for layer in self.model.transformer.h:
            x = layer(x)
            layer_outputs.append(x)
        return layer_outputs
    
    def backward_pass(self, layer_outputs):
        # 反向传播相关性
        relevance = torch.ones_like(layer_outputs[-1])
        
        for i, layer in enumerate(reversed(self.model.transformer.h)):
            relevance = self.compute_layer_relevance(
                layer_outputs[-(i+1)],
                relevance,
                layer
            )
            
        return relevance
```

## 特征可视化

### 激活图可视化

```python
class ActivationVisualizer:
    """激活图可视化"""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
    def register_hooks(self, layer):
        def hook_fn(module, input, output):
            self.activations.append(output)
            
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        
    def visualize_layer_activations(self, image, layer_idx):
        self.activations = []
        self.model(image)
        
        # 获取激活
        activation = self.activations[layer_idx][0]
        
        # 绘制通道
        n_channels = min(16, activation.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        for i, ax in enumerate(axes.flat):
            if i < n_channels:
                ax.imshow(activation[i].cpu().detach(), cmap='viridis')
                ax.set_title(f'Channel {i}')
                ax.axis('off')
                
        plt.savefig(f'layer_{layer_idx}_activations.png')
```

## 总结

可解释性技术帮助我们理解Transformer的工作原理，对于调试、改进和建立信任至关重要。

---

*推荐阅读：《A Survey of Transformers》可视化章节*
