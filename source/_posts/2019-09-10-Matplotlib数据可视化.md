---
title: Matplotlib数据可视化从入门到精通
date: 2019-09-10 09:00:00
categories: Python编程
tags: [Matplotlib, 数据可视化, Python]
photos:
  - "https://images.unsplash.com/photo-1451187580459-43490279c0fa"
---

## Matplotlib数据可视化从入门到精通

数据可视化是数据分析的重要环节，Matplotlib是Python最基础也最灵活的绘图库。

### 基础绘图

```python
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 折线图
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sin and Cos Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 常见图表类型

#### 柱状图

```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 33]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 垂直柱状图
axes[0].bar(categories, values, color='steelblue', alpha=0.8)
axes[0].set_title('Bar Chart')
axes[0].set_ylabel('Values')

# 水平柱状图
axes[1].barh(categories, values, color='coral', alpha=0.8)
axes[1].set_title('Horizontal Bar Chart')
axes[1].set_xlabel('Values')

plt.tight_layout()
plt.show()
```

#### 散点图

```python
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5
colors = np.random.rand(100)
sizes = 100 * np.random.rand(100)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Color Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()
```

#### 直方图

```python
data = np.random.randn(1000)

plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, density=True, alpha=0.7,
         color='steelblue', edgecolor='white')

# 叠加正态分布曲线
from scipy.stats import norm
mu, std = norm.fit(data)
x_range = np.linspace(data.min(), data.max(), 100)
plt.plot(x_range, norm.pdf(x_range, mu, std), 'r-', linewidth=2)

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram with Normal Distribution Fit')
plt.show()
```

#### 饼图

```python
labels = ['Python', 'Java', 'C++', 'JavaScript', 'Go']
sizes = [35, 25, 15, 20, 5]
explode = (0.1, 0, 0, 0, 0)

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=plt.cm.Set3.colors[:5])
plt.title('Programming Language Popularity')
plt.show()
```

#### 箱线图

```python
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red', linewidth=2))
plt.ylabel('Value')
plt.title('Box Plot')
plt.show()
```

### 子图布局

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(0, 2 * np.pi, 100)

axes[0, 0].plot(x, np.sin(x), 'b-')
axes[0, 0].set_title('Sin(x)')

axes[0, 1].plot(x, np.cos(x), 'r--')
axes[0, 1].set_title('Cos(x)')

axes[1, 0].plot(x, np.tan(x), 'g-.')
axes[1, 0].set_ylim(-5, 5)
axes[1, 0].set_title('Tan(x)')

axes[1, 1].plot(x, np.exp(-x), 'm:')
axes[1, 1].set_title('Exp(-x)')

plt.tight_layout()
plt.show()
```

### 热力图

```python
data = np.random.randn(10, 10)
corr = np.corrcoef(data)

plt.figure(figsize=(10, 8))
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation')
plt.xticks(range(10), [f'F{i}' for i in range(10)])
plt.yticks(range(10), [f'F{i}' for i in range(10)])
plt.title('Correlation Heatmap')

# 添加数值标注
for i in range(10):
    for j in range(10):
        plt.text(j, i, f'{corr[i, j]:.2f}',
                ha='center', va='center', fontsize=8)

plt.show()
```

### 3D绘图

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot')
plt.show()
```

### 样式定制

```python
# 使用内置样式
plt.style.use('seaborn-v0_8-darkgrid')

# 自定义样式
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})
```

### 保存图表

```python
plt.savefig('plot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')
plt.savefig('plot.svg', format='svg', bbox_inches='tight')
```

### 总结

Matplotlib是Python数据可视化的基础库，支持几乎所有常见图表类型。掌握折线图、柱状图、散点图、直方图等基础图表，以及子图布局和样式定制，足以应对大部分可视化需求。对于更美观的统计图表，可以结合Seaborn库使用。
