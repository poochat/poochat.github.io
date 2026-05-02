---
title: Jetson嵌入式部署大模型：TensorRT-LLM实战
date: 2023-11-05 14:00:00
categories: AI嵌入式
tags: [Jetson, TensorRT-LLM, 嵌入式AI, 模型部署]
photos:
  - "https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d"
---

## 引言

在边缘设备上部署大语言模型是2023年的重要研究方向。NVIDIA Jetson系列平台凭借强大的GPU算力，成为嵌入式AI的理想选择。本文将详细介绍如何在Jetson Nano、Xavier NX等设备上使用TensorRT-LLM部署和优化大模型。

## Jetson平台概述

### 1. Jetson系列对比

| 型号 | GPU | AI算力 | 内存 | 功耗 | 适用场景 |
|-----|-----|-------|-----|-----|---------|
| Jetson Nano | 128-core Maxwell | 0.5 TFLOPS | 4GB | 5-10W | 轻量推理 |
| Jetson Xavier NX | 384-core Volta | 21 TFLOPS | 8/16GB | 10-15W | 中量推理 |
| Jetson AGX Orin | 2048-core Ampere | 275 TOPS | 32/64GB | 15-60W | 大模型部署 |

### 2. 环境准备

```bash
# 刷写JetPack
# 下载SDK Manager: https://developer.nvidia.com/embedded/jetpack

# 基础环境设置
sudo apt-get update
sudo apt-get install -y python3-pip
sudo apt-get install -y libopenmpi-dev

# 安装PyTorch (jetson版本)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/torch-stable.html

# 安装TensorRT-LLM
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout tags/v0.5.0
```

## TensorRT-LLM核心概念

### 1. 模型转换流程

```python
"""
TensorRT-LLM模型转换
将HuggingFace模型转换为TensorRT格式
"""

import torch
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm._utils import pad_vocab_size
from transformers import LlamaForCausalLM, LlamaTokenizer
import tensorrt as trt

class ModelConverter:
    """
    模型转换器
    """
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        
    def convert_llama(self, model_size: int = 7):
        """
        转换LLaMA模型
        """
        print(f"Loading LLaMA-{model_size}B from {self.model_path}")
        
        # 加载HuggingFace模型
        hf_model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        )
        tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        
        # 构建TensorRT-LLM模型
        if model_size == 7:
            num_layers = 32
            hidden_size = 4096
            num_heads = 32
            vocab_size = 32000
        elif model_size == 13:
            num_layers = 40
            hidden_size = 5120
            num_heads = 40
            vocab_size = 32000
        
        # 创建模型配置
        trt_llm = LLaMAForCausalLM(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            vocab_size=pad_vocab_size(vocab_size, 64),
            max_position_embeddings=4096
        )
        
        # 复制权重
        trt_llm.load_from_hf_model(hf_model)
        
        # 保存
        self.save_model(trt_llm, tokenizer)
        
        return trt_llm
    
    def save_model(self, model, tokenizer):
        """保存转换后的模型"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存模型
        model.save(self.output_dir)
        
        # 保存tokenizer
        tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to {self.output_dir}")
```

### 2. TensorRT引擎构建

```bash
#!/bin/bash
# build_engine.sh
# 构建TensorRT引擎

MODEL_DIR="./models/llama-7b"
ENGINE_DIR="./engines/llama-7b"
TP_SIZE=1  # Tensor Parallel size
MAX_BATCH_SIZE=8
MAX_INPUT_LEN=4096
MAX_OUTPUT_LEN=512

python3 tensorrt_llm/examples/llama/build.py \
    --model_dir ${MODEL_DIR} \
    --tp_size ${TP_SIZE} \
    --pp_size 1 \
    --dtype float16 \
    --max_batch_size ${MAX_BATCH_SIZE} \
    --max_input_len ${MAX_INPUT_LEN} \
    --max_output_len ${MAX_OUTPUT_LEN} \
    --output_dir ${ENGINE_DIR} \
    --use_gpt_attention_plugin \
    --use_rms_norm_plugin \
    --use_gemm_plugin \
    --enable_context_fmha
```

### 3. Python推理接口

```python
import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

class TensorRTLLMRunner:
    """
    TensorRT-LLM运行时封装
    """
    def __init__(self, engine_dir: str, tokenizer_dir: str):
        self.runtime = trt.Runtime(trt.Logger(trt.WARNING))
        
        # 加载引擎
        with open(f"{engine_dir}/rank0.engine", "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.runner = ModelRunner.from_engine(self.engine)
        
        # 加载tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        
    def generate(self, prompt: str, max_new_tokens: int = 256, 
                 temperature: float = 0.7, top_p: float = 0.9):
        """
        生成文本
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_len = input_ids.shape[1]
        
        # 准备batch
        batch_input_ids = input_ids.cuda()
        
        # 采样参数
        sampling_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens
        }
        
        # 运行推理
        outputs = self.runner.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_words_list=None,
            pad_id=0
        )
        
        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][0][input_len:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def batch_generate(self, prompts: list, max_new_tokens: int = 256):
        """
        批量生成
        """
        # Tokenize
        input_ids_list = [
            self.tokenizer.encode(p, return_tensors="pt") 
            for p in prompts
        ]
        
        # Padding
        max_len = max(ids.shape[1] for ids in input_ids_list)
        padded = [
            torch.cat([
                ids, 
                torch.zeros(1, max_len - ids.shape[1], dtype=torch.long)
            ], dim=1).cuda()
            for ids in input_ids_list
        ]
        
        batch_input_ids = torch.cat(padded, dim=0)
        
        # 推理
        outputs = self.runner.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens
        )
        
        # 解码
        results = []
        for i, ids in enumerate(input_ids_list):
            input_len = ids.shape[1]
            text = self.tokenizer.decode(
                outputs[i][0][input_len:],
                skip_special_tokens=True
            )
            results.append(text)
        
        return results
```

## Jetson部署实战

### 1. 完整部署脚本

```python
#!/usr/bin/env python3
"""
Jetson部署脚本
"""
import torch
import gc
import time
import argparse
from pathlib import Path

class JetsonDeployer:
    """
    Jetson部署器
    """
    def __init__(self, model_path: str, quantize_mode: str = "fp16"):
        self.model_path = model_path
        self.quantize_mode = quantize_mode
        self.model = None
        
        # 检查GPU
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    def load_model(self):
        """加载模型"""
        print(f"Loading model from {self.model_path}")
        
        if "llama" in self.model_path.lower():
            from transformers import LlamaForCausalLM, LlamaTokenizer
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            
        elif "chatglm" in self.model_path.lower():
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).half().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
        
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Model loaded successfully")
        
    def quantize_model(self, mode: str = "int8"):
        """模型量化"""
        print(f"Quantizing model to {mode}")
        
        if mode == "int8":
            from transformers import BitsAndBytesConfig
            
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            
            # 重新加载量化模型
            if "llama" in self.model_path.lower():
                from transformers import LlamaForCausalLM
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quant_config,
                    device_map="auto"
                )
        
        gc.collect()
        print("Quantization completed")
    
    def benchmark(self, prompt: str = "介绍自己", num_runs: int = 10):
        """性能测试"""
        print("\n=== Benchmark ===")
        
        # Warm up
        _ = self.generate(prompt, max_new_tokens=32)
        
        # 计时
        latencies = []
        tokens_per_second = []
        
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            
            output = self.generate(prompt, max_new_tokens=128)
            
            torch.cuda.synchronize()
            end = time.time()
            
            latency = end - start
            tokens = len(self.tokenizer.encode(output))
            tps = tokens / latency
            
            latencies.append(latency)
            tokens_per_second.append(tps)
            
            print(f"Run {i+1}: {latency:.2f}s, {tps:.2f} tokens/s")
        
        print(f"\n=== Results ===")
        print(f"Average latency: {sum(latencies)/len(latencies):.2f}s")
        print(f"Average throughput: {sum(tokens_per_second)/len(tokens_per_second):.2f} tokens/s")
    
    def generate(self, prompt: str, max_new_tokens: int = 256):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--quantize", type=str, default="fp16", 
                        choices=["fp16", "int8", "int4"])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--prompt", type=str, default="介绍人工智能的发展历史")
    args = parser.parse_args()
    
    # 部署
    deployer = JetsonDeployer(args.model_path, args.quantize)
    deployer.load_model()
    
    if args.quantize != "fp16":
        deployer.quantize_model(args.quantize)
    
    # 测试
    output = deployer.generate(args.prompt)
    print(f"\nOutput:\n{output}")
    
    if args.benchmark:
        deployer.benchmark(args.prompt)
```

### 2. TensorRT-LLM推理服务

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="LLM Inference API")

class InferenceService:
    """
    推理服务
    """
    def __init__(self, engine_dir: str):
        self.runner = TensorRTLLMRunner(engine_dir, tokenizer_dir)
        
    def infer(self, prompt: str, **kwargs) -> str:
        return self.runner.generate(prompt, **kwargs)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@app.post("/generate")
async def generate(request: GenerateRequest):
    """生成接口"""
    try:
        result = inference_service.infer(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


# 启动
if __name__ == "__main__":
    import sys
    engine_dir = sys.argv[1] if len(sys.argv) > 1 else "./engines"
    inference_service = InferenceService(engine_dir)
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## 性能优化技巧

### 1. 内存优化

```python
class MemoryOptimizer:
    """
    内存优化工具
    """
    @staticmethod
    def enable_gradient_checkpointing(model):
        """梯度检查点，减少显存占用"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        return model
    
    @staticmethod
    def clear_cache():
        """清理GPU缓存"""
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    
    @staticmethod
    def print_memory_stats():
        """打印显存使用情况"""
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"Memory allocated: {allocated:.2f} GB")
            print(f"Memory reserved: {reserved:.2f} GB")
```

### 2. 批处理优化

```python
class DynamicBatching:
    """
    动态批处理
    合并多个请求提高吞吐量
    """
    def __init__(self, runner, max_batch_size=8, max_wait_ms=100):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
        
    async def add_request(self, prompt: str):
        """添加请求"""
        future = asyncio.Future()
        self.pending_requests.append((prompt, future))
        
        # 检查是否需要处理
        if len(self.pending_requests) >= self.max_batch_size:
            await self.process_batch()
        
        return await future
    
    async def process_batch(self):
        """处理批次"""
        if not self.pending_requests:
            return
        
        # 取出请求
        batch_size = min(len(self.pending_requests), self.max_batch_size)
        batch = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]
        
        prompts = [p for p, _ in batch]
        
        # 批量推理
        outputs = self.runner.batch_generate(prompts)
        
        # 设置结果
        for (_, future), output in zip(batch, outputs):
            future.set_result(output)
```

## 总结

在Jetson嵌入式平台上部署大模型需要综合考虑模型大小、推理速度和内存限制。通过TensorRT-LLM优化、模型量化和批处理等技术，可以在边缘设备上实现大模型的高效推理。本文介绍了完整的部署流程和优化技巧，为开发者提供了实用的参考方案。

## 参考资源

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [Jetson官方文档](https://docs.nvidia.com/jetson/)
- [Deep Learning Inference](https://developer.nvidia.com/deep-learning-inference)
