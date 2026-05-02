---
title: InstructGPT与人类对齐技术
date: 2022-02-20 14:00:00
categories: 大模型
tags: [InstructGPT, RLHF, 人类对齐, NLP]
photos:
  - "https://images.unsplash.com/photo-1485827404703-89b55fcc595e"
---

# InstructGPT与人类对齐技术

在ChatGPT发布之前，OpenAI于2022年初发表了InstructGPT论文，系统性地提出了通过人类反馈来对齐语言模型的方法。这篇论文是理解现代大模型训练范式的关键。

## 1. 研究背景与动机

GPT-3虽然展现了强大的语言能力，但存在严重的"对齐问题"：

- 用户期望模型遵循指令，但模型倾向于"续写"而非"回答"
- 模型可能输出有毒、偏见或不真实的内容
- 在安全关键场景下，模型行为不可预测

InstructGPT的核心洞察：**问题不在于模型能力不足，而在于模型目标与人类期望不一致**。

## 2. 数据收集流程

### 2.1 演示数据（Demonstration Data）

用于SFT阶段，标注员编写高质量的prompt-response对：

```
Prompt: 请用简单的语言解释量子纠缠
Response: 量子纠缠是量子力学中的一种现象。当两个粒子"纠缠"在一起时，
无论它们相隔多远，测量其中一个粒子的状态会立即影响另一个粒子的状态。
这就像是有一对魔法骰子，无论相隔多远，它们总是显示相同的数字。
```

### 2.2 比较数据（Comparison Data）

用于RM训练阶段，标注员对多个模型输出进行排序：

```
Prompt: 写一首关于春天的诗

Output A: 春风拂面花自开...  → 排名: 1
Output B: 春天来了天气好...  → 排名: 3  
Output C: 桃花流水春风暖... → 排名: 2
```

## 3. 奖励模型设计

奖励模型是RLHF的核心组件，其训练目标是将人类的偏好量化：

```python
class InstructRMRewardModel(nn.Module):
    """
    基于GPT-3的奖励模型
    输入: prompt + response
    输出: 标量奖励值
    """
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT3Model(config)
        # 移除语言模型头，添加奖励头
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # 获取transformer输出
        outputs = self.transformer(input_ids, attention_mask)
        # 取最后一个token的表示
        last_hidden = outputs.last_hidden_state[:, -1, :]
        # 映射到奖励值
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)
```

### 比较损失函数

```python
def bradley_terry_loss(rewards, rankings):
    """
    Bradley-Terry模型用于学习偏好排序
    """
    loss = 0
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            # 排名靠前的应该有更高的奖励
            preferred = rankings[i]
            rejected = rankings[j]
            loss -= torch.log(
                torch.sigmoid(rewards[preferred] - rewards[rejected])
            )
    return loss / len(rankings)
```

## 4. PPO优化细节

InstructGPT使用PPO算法进行策略优化，以下是关键实现细节：

```python
class PPOTrainer:
    def __init__(self, policy, ref_policy, reward_model, 
                 value_fn, kl_coef=0.2):
        self.policy = policy          # 待优化的策略模型
        self.ref_policy = ref_policy  # 参考模型（SFT模型）
        self.reward_model = reward_model
        self.value_fn = value_fn
        self.kl_coef = kl_coef       # KL惩罚系数
    
    def compute_reward(self, query, response):
        """计算实际奖励 = RM奖励 - KL惩罚"""
        with torch.no_grad():
            rm_reward = self.reward_model(query, response)
            kl_div = self.compute_kl(query, response)
        
        # KL惩罚防止策略偏离参考模型太远
        actual_reward = rm_reward - self.kl_coef * kl_div
        return actual_reward
    
    def compute_kl(self, query, response):
        """计算策略模型与参考模型的KL散度"""
        policy_logprobs = self.policy.logprob(query, response)
        ref_logprobs = self.ref_policy.logprob(query, response)
        kl = (policy_logprobs - ref_logprobs).mean()
        return kl
    
    def train_step(self, batch):
        queries = batch["query"]
        responses = self.policy.generate(queries)
        rewards = self.compute_reward(queries, responses)
        
        # 计算GAE优势估计
        values = self.value_fn(queries, responses)
        advantages = self.compute_gae(rewards, values)
        
        # PPO裁剪目标
        old_logprobs = self.policy.logprob(queries, responses).detach()
        new_logprobs = self.policy.logprob(queries, responses)
        
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, rewards)
        
        return policy_loss + 0.5 * value_loss
```

## 5. 实验结果

InstructGPT在多个维度上显著优于GPT-3：

| 评估维度 | GPT-3 | InstructGPT (1.3B) | InstructGPT (175B) |
|---------|-------|---------------------|---------------------|
| 指令遵循 | 28% | 73% | 85% |
| 真实性 | 21% | 52% | 67% |
| 无毒性 | 83% | 93% | 97% |
| 有用性 | 35% | 68% | 81% |

值得注意的是，1.3B参数的InstructGPT在人类评估中甚至优于175B的GPT-3，说明**对齐比规模更重要**。

## 6. 对后续工作的影响

InstructGPT确立的RLHF范式对整个AI行业产生了深远影响：

1. **Anthropic**：提出了Constitutional AI（CAI），用AI反馈替代部分人类反馈
2. **Google**：在Sparrow和LaMDA中采用了类似的RLHF方法
3. **开源社区**：OpenAssistant、Alpaca等项目将RLHF引入开源模型训练

## 总结

InstructGPT的核心贡献在于证明了：**通过相对少量的人类反馈数据，可以将大规模语言模型有效对齐到人类偏好**。这一发现彻底改变了大模型的训练范式，RLHF已成为行业标准。
