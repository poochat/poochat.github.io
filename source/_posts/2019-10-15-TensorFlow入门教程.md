---
title: TensorFlow入门教程：从张量到模型训练
date: 2019-10-15 10:00:00
categories: 深度学习框架
tags: [TensorFlow, 深度学习, Keras]
photos:
  - "https://images.unsplash.com/photo-1518770660439-4636190af475"
---

## TensorFlow入门教程：从张量到模型训练

TensorFlow是Google开源的深度学习框架，以其强大的计算能力和灵活的架构被广泛应用。

### 张量（Tensor）

张量是TensorFlow的核心数据结构：

```python
import tensorflow as tf

# 创建张量
a = tf.constant([1, 2, 3])                    # 一维张量
b = tf.constant([[1, 2], [3, 4]])             # 二维张量
c = tf.zeros((3, 4))                          # 全零张量
d = tf.ones((2, 3))                           # 全一张量
e = tf.random.normal((3, 3), mean=0, stddev=1) # 正态分布
f = tf.range(0, 10, delta=2)                  # 等差数列

print(f"张量形状: {b.shape}")
print(f"数据类型: {b.dtype}")
print(f"设备位置: {b.device}")

# 张量运算
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0, 6.0])

print(tf.add(x, y))        # 加法
print(tf.multiply(x, y))   # 逐元素乘法
print(tf.matmul(x[:2, tf.newaxis], y[tf.newaxis, :2]))  # 矩阵乘法
print(tf.reduce_mean(x))   # 均值
print(tf.reduce_sum(x))    # 求和
```

### 自动微分（Autograd）

```python
# 自动求导
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1

grad = tape.gradient(y, x)
print(f"dy/dx at x=3: {grad}")  # 2*3 + 2 = 8

# 多变量梯度
w = tf.Variable(tf.random.normal((3, 2)))
b = tf.Variable(tf.zeros(2))
x = tf.random.normal((1, 3))

with tf.GradientTape() as tape:
    y = tf.matmul(x, w) + b
    loss = tf.reduce_mean(y ** 2)

grads = tape.gradient(loss, [w, b])
print(f"w梯度形状: {grads[0].shape}")
print(f"b梯度形状: {grads[1].shape}")
```

### Keras Sequential API

```python
from tensorflow import keras
from tensorflow.keras import layers

# 构建顺序模型
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.summary()

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Keras Functional API

对于更复杂的模型结构，使用Functional API：

```python
# 函数式API
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### 训练MNIST分类器

```python
# 加载数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 预处理
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# 训练
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
)

# 评估
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### 训练可视化

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 自定义训练循环

```python
# 自定义训练步骤
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy()
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc_metric.update_state(y, logits)
    return loss_value

# 训练循环
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(128)

for epoch in range(20):
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)

    acc = train_acc_metric.result()
    print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.4f}")
    train_acc_metric.reset_states()
```

### 模型保存与加载

```python
# 保存完整模型
model.save('my_model.h5')

# 加载模型
loaded_model = keras.models.load_model('my_model.h5')

# 保存权重
model.save_weights('my_weights.h5')

# SavedModel格式
model.save('my_saved_model')

# 仅保存架构
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)
```

### 总结

TensorFlow 2.x通过Keras高级API大大降低了使用门槛。Sequential API适合简单的线性模型，Functional API适合复杂的网络结构。自动微分、回调机制和自定义训练循环提供了灵活性和控制力。掌握TensorFlow是从实践角度理解深度学习的最佳途径之一。
