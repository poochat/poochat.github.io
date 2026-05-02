---
title: 提示工程Prompt Engineering高级技巧
date: 2025-09-20 10:00:00
categories: LLM应用
tags: [提示工程, Prompt Engineering, LLM优化, CoT, Few-Shot]
photos:
  - "https://images.unsplash.com/photo-1593720219276-0b1eacd0aef4"
---

## 概述

提示工程是发挥大模型能力的关键技术，本文介绍从基础到高级的提示技巧。

## 提示工程核心技巧

### 零样本 vs 少样本

```mermaid
flowchart TB
    subgraph Zero-Shot
        ZS[零样本提示]
        ZS --> QUERY[直接提问]
    end
    
    subgraph Few-Shot
        FS[少样本提示]
        FS --> EX1[示例1]
        FS --> EX2[示例2]
        FS --> EX3[示例3]
        EX1 --> QUERY2[最终问题]
    end
```

### 思维链提示

```python
class ChainOfThought:
    """思维链提示"""
    
    def zero_shot_cot(self, question):
        """零样本思维链"""
        prompt = f"""
问题: {question}

请逐步思考，然后给出答案。
"""
        return self.llm.generate(prompt)
    
    def few_shot_cot(self, question, examples):
        """少样本思维链"""
        prompt = "请逐步推理：\n\n"
        for ex in examples:
            prompt += f"问题: {ex['q']}\n思考: {ex['thought']}\n答案: {ex['a']}\n\n"
        prompt += f"问题: {question}\n思考:"
        return self.llm.generate(prompt)
```

## 高级提示模式

| 模式 | 适用场景 | 效果提升 |
|------|----------|----------|
| CoT | 推理任务 | +30% |
| Few-Shot | 格式要求 | +50% |
| ReAct | 工具使用 | +100% |
| Tree-of-Thought | 复杂决策 | +40% |

## 总结

```mermaid
mindmap
  root((提示工程))
    基础技巧
      清晰指令
      格式指定
      角色设定
    进阶技巧
      思维链
      少样本学习
      分解问题
    高级技巧
      ReAct
      Tree-of-Thought
      自动提示优化
```

掌握提示工程能显著提升大模型的使用效率和输出质量。
