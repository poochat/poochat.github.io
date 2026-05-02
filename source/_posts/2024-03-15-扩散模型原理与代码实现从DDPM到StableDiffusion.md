---
title: 扩散模型原理与代码实现：从DDPM到Stable Diffusion
date: 2024-03-15 10:30:00
categories: AI生成
tags: [扩散模型, DDPM, Stable Diffusion, 生成模型, 深度学习]
photos:
  - "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e"
---

# 扩散模型原理与代码实现：从DDPM到Stable Diffusion

## 引言

扩散模型（Diffusion Models）是当前最强大的生成模型之一，在图像生成、音频合成、分子设计等领域取得了突破性成果。本文将从原理出发，详细讲解扩散模型的核心机制，并提供完整的代码实现。

## 扩散模型概述

### 核心思想

扩散模型的核心思想是：通过逐步添加噪声破坏数据，然后学习逆向去噪过程来生成新数据。

```
前向过程（破坏）：x₀ → x₁ → x₂ → ... → xₜ → ... → xₜ
逆向过程（生成）：xₜ → xₜ₋₁ → xₜ₋₂ → ... → x₀
```

### 与其他生成模型对比

| 模型 | 训练目标 | 采样方式 | 优缺点 |
|------|----------|----------|--------|
| GAN | 对抗训练 | 单步 | 训练不稳定 |
| VAE | 重建+KL散度 | 单步 | 模糊 |
| Flow | 精确对数似然 | 单步 | 需可逆网络 |
| Diffusion | 去噪 | 多步 | 训练稳定，质量高 |

## 数学基础

### 前向扩散过程

前向过程 $q(\mathbf{x}_{t}|\mathbf{x}_{t-1})$ 是在数据 $\mathbf{x}_0$ 上逐步添加高斯噪声：

```python
def q_sample(self, x0, t, noise=None):
    """
    前向扩散过程：添加噪声
    
    q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t)x_{t-1}, β_t I)
    
    Args:
        x0: 原始图像
        t: 时间步
        noise: 高斯噪声
    """
    if noise is None:
        noise = torch.randn_like(x0)
    
    # 计算累积系数
    sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t]
    sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
    
    # 公式：x_t = sqrt(ᾱ_t) * x₀ + sqrt(1 - ᾱ_t) * ε
    return (
        sqrt_alphas_cumprod * x0 + 
        sqrt_one_minus_alphas_cumprod * noise
    ), noise
```

### 关键参数

```python
class DDPMScheduler:
    """DDPM调度器"""
    
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod[:-1]) / 
            (1.0 - self.alphas_cumprod[1:])
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod[:-1]) /
            (1.0 - self.alphas_cumprod[1:])
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod[:-1]) * torch.sqrt(self.alphas[1:]) /
            (1.0 - self.alphas_cumprod[1:])
        )
```

### 闭式解

由于前向过程是可学习的马尔可夫链，可以直接采样任意时间步 $t$ 的噪声：

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}
$$

```python
def get_variance(self, t):
    """获取闭式解"""
    return self.posterior_variance.to(t.device)
```

## UNet骨干网络

### 时间步嵌入

```python
class SinusoidalPositionEmbeddings(nn.Module):
    """Transformer中的位置编码用于时间步"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """基础残差块"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, 
                 groups=8, scale_factor=2):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
        
        # 时间步投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
    def forward(self, x, time_emb):
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        return h
```

### UNet架构

```python
class UNet(nn.Module):
    """
    用于去噪的UNet网络
    预测噪声 εθ(x_t, t)
    """
    
    def __init__(self, channels=3, base_channels=128, 
                 channel_multipliers=[1, 2, 4, 8],
                 num_res_blocks=2):
        super().__init__()
        
        self.channels = channels
        self.time_channels = base_channels * 4
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, self.time_channels),
            nn.GELU(),
            nn.Linear(self.time_channels, self.time_channels)
        )
        
        # 输入投影
        self.input_conv = nn.Conv2d(channels, base_channels, 3, padding=1)
        
        # 编码器（下采样）
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        in_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(in_ch, out_ch, self.time_channels)
                )
                in_ch = out_ch
                
            if i < len(channel_multipliers) - 1:
                self.encoder_downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
        
        # 中间层
        self.mid_block = ResidualBlock(in_ch, in_ch, self.time_channels)
        
        # 解码器（上采样）
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks + 1):
                self.decoder_blocks.append(
                    ResidualBlock(in_ch + out_ch, out_ch, self.time_channels)
                )
                in_ch = out_ch
                
            if i > 0:
                self.decoder_ups.append(nn.ConvTranspose2d(
                    in_ch, in_ch, 4, stride=2, padding=1
                ))
        
        # 输出投影
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, 3, padding=1)
        )
        
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 输入
        h = self.input_conv(x)
        
        # 编码器
        hs = []
        for block in self.encoder_blocks:
            h = block(h, t_emb)
            hs.append(h)
        h = self.encoder_downs[-1](h) if self.encoder_downs else h
        
        # 中间
        h = self.mid_block(h, t_emb)
        
        # 解码器
        for block in self.decoder_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, t_emb)
        
        # 输出
        return self.output_conv(h)
```

## 训练过程

```python
class DDPM:
    """DDPM扩散模型"""
    
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        
    def train_step(self, x0):
        """单步训练"""
        batch_size = x0.shape[0]
        device = x0.device
        
        # 随机采样时间步
        t = torch.randint(0, self.scheduler.num_timesteps, 
                         (batch_size,), device=device)
        
        # 采样噪声
        noise = torch.randn_like(x0)
        
        # 前向加噪
        noisy_x, noise_added = self.scheduler.q_sample(x0, t, noise)
        
        # 预测噪声
        noise_pred = self.model(noisy_x, t)
        
        # MSE损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def train(self, dataloader, epochs):
        """完整训练流程"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                loss = self.train_step(batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
```

## 采样过程

### DDPM采样

```python
    @torch.no_grad()
    def ddpm_sample(self, shape, device):
        """DDPM反向过程：从噪声生成图像"""
        # 从纯噪声开始
        x_t = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in reversed(range(self.scheduler.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.model(x_t, t_batch)
            
            # 计算均值和方差
            model_mean, model_var, model_log_variance = (
                self.scheduler.condition_mean(x_t, noise_pred, t_batch)
            )
            
            # 采样
            x_t = model_mean + torch.exp(0.5 * model_log_variance) * torch.randn_like(x_t)
            
        return x_t
```

### DDIM加速采样

```python
class DDIMScheduler:
    """DDIM加速采样"""
    
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50,
                 beta_schedule='linear', clip_sample=True):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            num_train_timesteps - 1, 0, num_inference_steps
        ).long()
        self.clip_sample = clip_sample
        
    @torch.no_grad()
    def ddim_sample(self, model, shape, device, eta=0.0):
        """DDIM采样"""
        # 初始化
        x_t = torch.randn(shape, device=device)
        
        timesteps = self.timesteps
        
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            # 预测噪声
            noise_pred = model(x_t, t.unsqueeze(0).repeat(shape[0]))
            
            # 计算x_{t-1}
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[timesteps[i-1]] if i > 0 else 1.0
            
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # DDIM步骤
            pred_x0 = (x_t - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            if self.clip_sample:
                pred_x0 = torch.clamp(pred_x0, -1, 1)
                
            direction_pointing = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
            
            x_prev = alpha_prod_t_prev ** 0.5 * pred_x0 + direction_pointing
            
            x_t = x_prev
            
        return x_t
```

## 条件生成：Classifier-Free Guidance

### 引导技术

```python
class ConditionalUNet(UNet):
    """条件UNet"""
    
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, self.time_channels)
        
    def forward(self, x, t, y=None):
        """
        Args:
            x: 噪声图像
            t: 时间步
            y: 类别标签（可选）
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 类别条件
        if y is not None:
            t_emb = t_emb + self.class_embedding(y)
            
        # UNet前向传播
        return super().forward(x, t_emb)


def classifier_free_guidance(model, x, t, y, cfg_scale=7.5):
    """
    Classifier-Free Guidance采样
    
    εθ(x, ∅) + s * (εθ(x, y) - εθ(x, ∅))
    """
    # 无条件预测
    noise_pred_uncond = model(x, t, y=None)
    
    # 有条件预测
    noise_pred_cond = model(x, t, y=y)
    
    # CFG组合
    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
    
    return noise_pred
```

## Stable Diffusion架构

### 潜在空间扩散

Stable Diffusion的关键创新是在VAE的潜在空间中进行扩散：

```python
class StableDiffusion(nn.Module):
    """
    Stable Diffusion完整架构
    """
    
    def __init__(self, config):
        super().__init__()
        
        # VAE：图像压缩到潜在空间
        self.vae = AutoencoderKL.from_pretrained(
            config.vae_path
        )
        
        # UNet：潜在空间去噪
        self.unet = UNet(
            channels=4,  # 潜在空间通道
            base_channels=320,
            channel_multipliers=[1, 2, 4, 4],
            num_res_blocks=2
        )
        
        # 文本编码器
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.text_encoder_path
        )
        
        # 调度器
        self.scheduler = DDIMScheduler(
            num_inference_steps=50
        )
        
    def encode_prompt(self, prompt, device):
        """编码文本提示"""
        text_inputs = self.text_encoder(
            tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        )
        return text_inputs.last_hidden_state
    
    @torch.no_grad()
    def generate(self, prompt, height=512, width=512, 
                 num_inference_steps=50, guidance_scale=7.5):
        """文本生成图像"""
        device = self.text_encoder.device
        
        # 编码文本
        text_embeddings = self.encode_prompt(prompt, device)
        
        # 无条件文本嵌入（用于CFG）
        uncond_embeddings = self.encode_prompt("", device)
        
        # 组合
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # 初始化潜在空间
        latents = torch.randn(
            (1, 4, height // 8, width // 8),
            device=device
        )
        
        # 扩散采样
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            # 扩展latents以处理CFG
            latent_model_input = torch.cat([latents] * 2)
            
            # 预测噪声
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            # 去噪步骤
            latents = self.scheduler.step(noise_pred, t, latents)
            
        # 潜在空间解码为图像
        image = self.vae.decode(latents / 0.18215)
        
        return image
```

## 实践指南

### 完整训练脚本

```python
def train_stable_diffusion():
    """Stable Diffusion训练流程"""
    
    # 模型初始化
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
    unet = UNet(...)
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
    
    # 数据加载
    dataset = TextImageDataset(
        image_dir='./data/images',
        captions_file='./data/captions.json'
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 优化器
    params = list(unet.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    
    # 训练循环
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            captions = batch['caption']
            
            # 编码图像到潜在空间
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            
            # 编码文本
            text_embeddings = text_encoder(
                tokenizer(captions, padding=True, return_tensors='pt').input_ids.to(device)
            ).last_hidden_state
            
            # 采样噪声和时间步
            noise = torch.randn_like(latents)
            t = torch.randint(0, 1000, (batch_size,), device=device)
            
            # 加噪
            noisy_latents = scheduler.q_sample(latents, t, noise)
            
            # 预测噪声
            noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings)
            
            # 损失
            loss = F.mse_loss(noise_pred, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## 总结

扩散模型从DDPM到Stable Diffusion经历了快速发展。其核心优势在于训练稳定性和生成质量。随着采样加速技术的进步（如DDIM、LCM），扩散模型在实际应用中的潜力正在不断释放。

---

*推荐阅读：*
- *Ho et al. "Denoising Diffusion Probabilistic Models"*
- *Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"*
- *Song et al. "Denoising Diffusion Implicit Models"*
