---
title: Physics-Informed Neural Networks (PINN)：深度学习与物理方程融合
date: 2023-10-12 10:00:00
categories: AI+科研
tags: [PINN, 物理信息神经网络, 科学计算, 深度学习]
photos:
  - "https://images.unsplash.com/photo-1551288049-bebda4e38f71"
---

## 引言

Physics-Informed Neural Networks (PINN) 是近年来科学计算与深度学习交叉领域的重要突破。PINN通过将物理定律（偏微分方程）嵌入到神经网络的损失函数中，实现了数据驱动与物理约束的有机结合。本文将深入解析PINN的理论基础、架构设计以及在流体力学、热传导等领域的应用实践。

## PINN理论基础

### 1. 问题定义

PINN的核心思想是用神经网络近似偏微分方程的解，同时满足物理方程的约束。

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Callable

class PINN(nn.Module):
    """
    Physics-Informed Neural Network
    物理信息神经网络
    """
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_layers: list = [64, 64, 64, 64]):
        super().__init__()
        
        # 神经网络架构
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 自动微分计算
        self.d = {}  # 存储导数
        
    def forward(self, x):
        """前向传播"""
        return self.network(x)
    
    def compute_derivatives(self, x, u):
        """
        使用自动微分计算导数
        """
        # 一阶导数 du/dx
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        # 二阶导数 d²u/dx²
        d2u_dx2 = torch.autograd.grad(
            du_dx, x, grad_outputs=torch.ones_like(du_dx),
            create_graph=True, retain_graph=True
        )[0]
        
        return du_dx, d2u_dx2
```

### 2. 损失函数设计

PINN的损失函数由三部分组成：

```python
class PINNLoss:
    """
    PINN损失函数
    L = L_data + L_physics + L_boundary
    """
    def __init__(self, pde_fn: Callable, bc_fn: Callable, ic_fn: Callable = None):
        self.pde_fn = pde_fn      # PDE残差
        self.bc_fn = bc_fn         # 边界条件
        self.ic_fn = ic_fn         # 初始条件
    
    def compute_loss(self, model, collocation_points, 
                     boundary_points, data_points=None, ic_points=None):
        """
        计算总损失
        """
        # 1. PDE残差损失
        loss_physics = self.compute_pde_loss(model, collocation_points)
        
        # 2. 边界条件损失
        loss_bc = self.compute_boundary_loss(model, boundary_points)
        
        # 3. 数据损失（可选）
        loss_data = torch.tensor(0.0)
        if data_points is not None:
            loss_data = self.compute_data_loss(model, data_points)
        
        # 4. 初始条件损失（可选）
        loss_ic = torch.tensor(0.0)
        if ic_points is not None and self.ic_fn is not None:
            loss_ic = self.compute_ic_loss(model, ic_points)
        
        # 总损失（可加权）
        total_loss = (
            1.0 * loss_physics + 
            1.0 * loss_bc + 
            1.0 * loss_data + 
            1.0 * loss_ic
        )
        
        return total_loss, {
            'physics': loss_physics.item(),
            'boundary': loss_bc.item(),
            'data': loss_data.item(),
            'ic': loss_ic.item()
        }
    
    def compute_pde_loss(self, model, points):
        """计算PDE残差"""
        points.requires_grad = True
        
        u_pred = model(points)
        
        # 计算导数
        if points.shape[1] == 1:  # 一维
            u_x = torch.autograd.grad(
                u_pred, points, 
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True
            )[0]
            u_xx = torch.autograd.grad(
                u_x, points,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True
            )[0]
            
            # Burgers方程: u_t + u*u_x - ν*u_xx = 0
            residual = u_x + model(points) * u_x - 0.01 * u_xx
            
        else:  # 二维
            # 计算偏导数
            u = u_pred
            x = points[:, 0:1]
            y = points[:, 1:2]
            
            # 简化处理
            residual = self.pde_fn(model, points)
        
        return torch.mean(residual ** 2)
    
    def compute_boundary_loss(self, model, boundary_points):
        """边界条件损失"""
        u_pred = model(boundary_points)
        u_bc = self.bc_fn(boundary_points)
        return torch.mean((u_pred - u_bc) ** 2)
    
    def compute_data_loss(self, model, data_points):
        """数据损失"""
        u_pred = model(data_points[:, :-1])  # 输入特征
        u_true = data_points[:, -1:]          # 真实值
        return torch.mean((u_pred - u_true) ** 2)
    
    def compute_ic_loss(self, model, ic_points):
        """初始条件损失"""
        u_pred = model(ic_points)
        u_ic = self.ic_fn(ic_points)
        return torch.mean((u_pred - u_ic) ** 2)
```

## PINN应用案例

### 1. 一维热传导方程

```python
class HeatEquationPINN(PINN):
    """
    一维热传导方程：
    ∂u/∂t = α * ∂²u/∂x²
    
    边界条件：u(0,t) = u(1,t) = 0
    初始条件：u(x,0) = sin(πx)
    """
    def __init__(self, alpha=0.01):
        super().__init__(input_dim=2, output_dim=1)  # (x, t) -> u
        self.alpha = alpha
        
    def pde_residual(self, x, t, u):
        """
        PDE残差: ∂u/∂t - α * ∂²u/∂x² = 0
        """
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        
        return u_t - self.alpha * u_xx
    
    def train(self, num_epochs=10000, lr=1e-3):
        """训练PINN"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=500, factor=0.5
        )
        
        # 生成训练点
        x_coll = torch.rand(1000, 1) * 1.0  # 空间点 [0, 1]
        t_coll = torch.rand(1000, 1) * 1.0  # 时间点 [0, 1]
        
        x_bc = torch.tensor([[0.0], [1.0]]).repeat(100, 1)
        t_bc = torch.rand(200, 1)
        
        x_ic = torch.rand(100, 1)
        t_ic = torch.zeros(100, 1)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # 合并点
            x = torch.cat([x_coll, x_bc, x_ic])
            t = torch.cat([t_coll, t_bc, t_ic])
            
            # 创建输入
            inputs = torch.cat([x, t], dim=1)
            inputs.requires_grad = True
            
            # 预测
            u = self.forward(inputs)
            
            # 计算PDE残差
            residual = self.pde_residual(x, t, u)
            loss_pde = torch.mean(residual ** 2)
            
            # 边界条件
            u_bc_pred = self.forward(torch.cat([x_bc, t_bc], dim=1))
            loss_bc = torch.mean(u_bc_pred ** 2)
            
            # 初始条件 u(x,0) = sin(πx)
            u_ic_pred = self.forward(torch.cat([x_ic, t_ic], dim=1))
            u_ic_true = torch.sin(np.pi * x_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
            
            # 总损失
            loss = loss_pde + loss_bc + loss_ic
            loss.backward()
            optimizer.step()
            
            scheduler.step(loss)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        return self
```

### 2. 流体力学：Navier-Stokes方程

```python
class NavierStokesPINN(PINN):
    """
    2D Navier-Stokes方程（不可压缩流体）
    
    连续性方程: ∂u/∂x + ∂v/∂y = 0
    动量方程: 
        ∂u/∂t + u*∂u/∂x + v*∂u/∂y = -1/ρ * ∂p/∂x + ν * ∇²u
        ∂v/∂t + u*∂v/∂x + v*∂v/∂y = -1/ρ * ∂p/∂y + ν * ∇²v
    """
    def __init__(self, Re=100, nu=0.01):
        super().__init__(input_dim=3, output_dim=3)  # (x, y, t) -> (u, v, p)
        self.Re = Re
        self.nu = nu
        
    def continuity_residual(self, u_x, v_y):
        """连续性方程残差"""
        return u_x + v_y
    
    def momentum_residual(self, x, y, t, u, v, p):
        """
        动量方程残差
        """
        # 计算导数
        u_t = self.jacobian(u, t)
        u_x = self.jacobian(u, x)
        u_y = self.jacobian(u, y)
        u_xx = self.hessian(u, x)
        u_yy = self.hessian(u, y)
        
        v_t = self.jacobian(v, t)
        v_x = self.jacobian(v, x)
        v_y = self.jacobian(v, y)
        v_xx = self.hessian(v, x)
        v_yy = self.hessian(v, y)
        
        p_x = self.jacobian(p, x)
        p_y = self.jacobian(p, y)
        
        # x方向动量
        residual_x = u_t + u*u_x + v*u_y + p_x - (u_xx + u_yy)/self.Re
        
        # y方向动量
        residual_y = v_t + u*v_x + v*v_y + p_y - (v_xx + v_yy)/self.Re
        
        return residual_x, residual_y
    
    def jacobian(self, y, x):
        """计算一阶导数"""
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
    
    def hessian(self, y, x):
        """计算二阶导数"""
        return torch.autograd.grad(
            self.jacobian(y, x), x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
```

## PINN训练技巧

### 1. 自适应采样

```python
class AdaptiveSampling:
    """
    自适应采样策略
    重点关注残差较大的区域
    """
    def __init__(self, model, residual_threshold=0.1):
        self.model = model
        self.residual_threshold = residual_threshold
        
    def sample_adaptive(self, domain, num_points=1000, num_iterations=5):
        """
        自适应采样
        """
        all_points = []
        
        for _ in range(num_iterations):
            # 均匀采样
            x_rand = torch.rand(num_points, 1) * (domain[0][1] - domain[0][0]) + domain[0][0]
            t_rand = torch.rand(num_points, 1) * (domain[1][1] - domain[1][0]) + domain[1][0]
            
            points = torch.cat([x_rand, t_rand], dim=1)
            points.requires_grad = True
            
            # 计算残差
            u = self.model(points)
            residual = self.compute_residual(points, u)
            
            # 选择残差大的点
            mask = residual > self.residual_threshold * residual.mean()
            high_residual_points = points[mask.flatten()]
            
            all_points.append(high_residual_points)
            
            # 减少均匀采样点
            num_points = num_points // 2
        
        return torch.cat(all_points)
    
    def compute_residual(self, points, u):
        """计算残差"""
        # 具体实现根据问题而定
        return torch.ones(len(points))
```

### 2. 课程学习

```python
class CurriculumLearning:
    """
    课程学习训练策略
    从简单到复杂逐渐增加难度
    """
    def __init__(self, model):
        self.model = model
        
    def train_with_curriculum(self, epochs=10000):
        """
        分阶段训练
        """
        stages = [
            {"t_max": 0.1, "lr": 1e-3, "epochs": 3000},   # 短期
            {"t_max": 0.5, "lr": 5e-4, "epochs": 3000},   # 中期
            {"t_max": 1.0, "lr": 1e-4, "epochs": 4000}    # 完整时域
        ]
        
        for i, stage in enumerate(stages):
            print(f"Stage {i+1}: t_max={stage['t_max']}")
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=stage['lr'])
            
            for epoch in range(stage['epochs']):
                # 生成当前阶段的训练点
                x = torch.rand(1000, 1)
                t = torch.rand(1000, 1) * stage['t_max']
                
                # 训练...
                
                if epoch % 1000 == 0:
                    loss = self.compute_loss(x, t)
                    print(f"  Epoch {epoch}, Loss: {loss:.6f}")
```

## DeepXDE框架

```python
# 使用DeepXDE简化PINN实现
"""
安装: pip install deepxde
"""

import deepxde as dde
import numpy as np

def solve_heat_equation():
    """
    使用DeepXDE求解热传导方程
    """
    # 定义几何域
    geom = dde.geometry.TimeGeometry1D(xmin=0, xmax=1)
    
    # 定义PDE
    def pde(x, y):
        dy_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_dt - 0.01 * dy_xx
    
    # 边界条件
    def boundary(x, on_boundary):
        return on_boundary
    
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)
    
    # 初始条件
    def initial_condition(x):
        return np.sin(np.pi * x[:, 0:1])
    
    ic = dde.icbc.IC(geom, initial_condition, lambda _, on_initial: on_initial)
    
    # 数据点
    data = dde.data.TimePDE(
        geom,
        pde,
        [bc, ic],
        num_domain=1000,
        num_boundary=100,
        num_initial=100
    )
    
    # 神经网络
    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot uniform")
    
    # 模型
    model = dde.Model(data, net)
    
    # 训练
    model.compile("adam", lr=1e-3)
    model.train(iterations=10000)
    
    return model
```

## 总结

PINN将深度学习的表达力与物理方程的先验知识相结合，为科学计算提供了新的范式。通过将物理约束嵌入损失函数，PINN可以在稀疏数据条件下学习偏微分方程的解。本文介绍了PINN的理论基础、损失函数设计、训练技巧以及在热传导和流体力学中的应用。随着技术的不断发展，PINN将在更多科学领域发挥重要作用。

## 参考资源

- [PINN原始论文](https://arxiv.org/abs/1711.10561)
- [DeepXDE库](https://github.com/lululxvi/deepxde)
- [SciML开源项目](https://sciml.ai/)
