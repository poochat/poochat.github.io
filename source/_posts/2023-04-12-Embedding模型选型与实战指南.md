---
title: Embedding模型选型与实战指南
date: 2023-04-12 10:00:00
categories: RAG系统
tags: [Embedding, 向量表示, 模型选型, 语义搜索]
photos:
  - "https://images.unsplash.com/photo-1504868584819-f8e0b037a5e4"
---

## 引言

Embedding模型是现代AI系统的基础组件，广泛应用于语义搜索、推荐系统、RAG等领域。选择合适的Embedding模型对系统性能至关重要。本文将全面对比分析主流Embedding模型，并提供实战选型建议。

## Embedding模型分类

### 1. 模型架构分类

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class EmbeddingModel(ABC):
    """
    Embedding模型基类
    """
    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """将文本编码为向量"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass

class BiEncoder(EmbeddingModel):
    """
    Bi-Encoder: 独立编码查询和文档
    优点：高效，适合大规模检索
    缺点：查询和文档无交互
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model()
        self.dim = self._get_embedding_dim()
        
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        # 独立编码
        embeddings = self.model.encode(texts, **kwargs)
        return embeddings
    
    def get_dimension(self) -> int:
        return self.dim

class CrossEncoder(EmbeddingModel):
    """
    Cross-Encoder: 联合编码查询和文档
    优点：精度高，适合重排序
    缺点：计算量大，不适合大规模检索
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model()
        self.dim = 1  # 输出相似度分数
        
    def encode(self, texts: List[Tuple[str, str]], **kwargs) -> np.ndarray:
        """
        编码(query, document)对
        """
        scores = self.model.predict(texts, **kwargs)
        return np.array(scores).reshape(-1, 1)
    
    def get_dimension(self) -> int:
        return 1
    
    def compute_score(self, query: str, documents: List[str]) -> List[float]:
        """计算查询与每个文档的相似度"""
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        return scores.tolist()
```

### 2. 主流模型对比

```python
EMBEDDING_MODELS = {
    # OpenAI Embeddings
    "text-embedding-ada-002": {
        "provider": "openai",
        "dimension": 1536,
        "max_tokens": 8191,
        "description": "通用最强模型",
        "cost_per_1k": 0.0001,
        "strengths": ["通用能力强", "稳定性好"],
        "weaknesses": ["需要API调用", "有延迟"]
    },
    
    # HuggingFace Models
    "sentence-transformers/all-MiniLM-L6-v2": {
        "provider": "huggingface",
        "dimension": 384,
        "max_tokens": 256,
        "description": "轻量高效模型",
        "cost_per_1k": 0,
        "strengths": ["速度快", "本地部署", "中文还行"],
        "weaknesses": ["精度略低"]
    },
    
    "sentence-transformers/all-mpnet-base-v2": {
        "provider": "huggingface",
        "dimension": 768,
        "max_tokens": 384,
        "description": "精度最高模型",
        "cost_per_1k": 0,
        "strengths": ["精度最高", "支持多语言"],
        "weaknesses": ["相对较慢"]
    },
    
    # 中文优化模型
    "moka-ai/m3e-base": {
        "provider": "huggingface",
        "dimension": 768,
        "max_tokens": 512,
        "description": "中文优化模型",
        "cost_per_1k": 0,
        "strengths": ["中文优秀", "性价比高"],
        "weaknesses": ["英文较弱"]
    },
    
    "BAAI/bge-large-zh": {
        "provider": "huggingface",
        "dimension": 1024,
        "max_tokens": 512,
        "description": "中文BGE大模型",
        "cost_per_1k": 0,
        "strengths": ["中文最强", "精度高"],
        "weaknesses": ["占用资源大"]
    },
    
    # E5 Embeddings
    "intfloat/e5-base-v2": {
        "provider": "huggingface",
        "dimension": 768,
        "max_tokens": 512,
        "description": "E5通用模型",
        "cost_per_1k": 0,
        "strengths": ["通用性好", "支持多语言"],
        "weaknesses": ["需要前缀"]
    }
}
```

## 模型实现

### 1. OpenAI Embedding

```python
import openai
import numpy as np
from typing import List

class OpenAIEmbedding:
    """
    OpenAI Embedding封装
    """
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        self.model = model
        if api_key:
            openai.api_key = api_key
        
    def encode(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        批量编码
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = openai.Embedding.create(
                model=self.model,
                input=batch
            )
            
            embeddings = [item['embedding'] for item in response['data']]
            all_embeddings.extend(embeddings)
            
        return np.array(all_embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """单文本编码"""
        response = openai.Embedding.create(
            model=self.model,
            input=text
        )
        return np.array(response['data'][0]['embedding'])
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        
        # 余弦相似度
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

### 2. HuggingFace Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
import torch

class HuggingFaceEmbedding:
    """
    HuggingFace Sentence Transformers封装
    """
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = SentenceTransformer(model_name)
        self.device = device
        self.model.to(device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def encode(self, texts: List[str], batch_size: int = 32, 
               normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        """
        批量编码
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_on_device(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """在GPU上编码，返回Tensor"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True
        )
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """计算余弦相似度"""
        emb1 = self.encode([text1])[0]
        emb2 = self.encode([text2])[0]
        return float(np.dot(emb1, emb2))
    
    def batch_similarity(self, query: str, documents: List[str]) -> List[float]:
        """批量计算相似度"""
        query_emb = self.encode([query])
        doc_embs = self.encode(documents)
        
        # 余弦相似度
        similarities = np.dot(query_emb, doc_embs.T)[0]
        return similarities.tolist()
    
    def semantic_search(self, query: str, documents: List[str], 
                       top_k: int = 5) -> List[dict]:
        """语义搜索"""
        query_emb = self.encode([query])
        doc_embs = self.encode(documents)
        
        # 计算相似度
        scores = np.dot(query_emb, doc_embs.T)[0]
        
        # 排序
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            {"index": int(idx), "score": float(scores[idx]), "text": documents[idx]}
            for idx in sorted_indices
        ]
```

### 3. 多语言Embedding

```python
class MultilingualEmbedding:
    """
    多语言Embedding模型
    """
    MODELS = {
        "xlm-roberta-base": {
            "dim": 768,
            "languages": "100+",
            "description": "XLM-RoBERTa基础版"
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "dim": 384,
            "languages": "50+",
            "description": "多语言MiniLM"
        },
        "distiluse-base-multilingual-cased-v2": {
            "dim": 512,
            "languages": "15",
            "description": "支持15种语言"
        }
    }
    
    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """多语言编码"""
        return self.model.encode(texts, **kwargs)
    
    def cross_lingual_search(self, query: str, documents: List[str]) -> List[dict]:
        """
        跨语言搜索
        例如：英文查询，中文文档
        """
        return self.model semantic_search(query, documents)
```

## 模型选型指南

### 1. 选型决策树

```python
class EmbeddingSelector:
    """
    Embedding模型选择器
    """
    @staticmethod
    def select_model(use_case: str, language: str = "en", 
                   budget: str = "medium") -> str:
        """
        根据场景选择模型
        
        Args:
            use_case: 使用场景
            language: 语言
            budget: 预算 (low/medium/high)
        """
        # RAG/检索场景
        if use_case == "rag" or use_case == "retrieval":
            if language == "zh":
                if budget == "low":
                    return "moka-ai/m3e-base"
                else:
                    return "BAAI/bge-large-zh"
            else:
                if budget == "low":
                    return "sentence-transformers/all-MiniLM-L6-v2"
                elif budget == "medium":
                    return "intfloat/e5-base-v2"
                else:
                    return "text-embedding-ada-002"
        
        # 语义相似度
        elif use_case == "similarity":
            if language == "zh":
                return "moka-ai/m3e-base"
            else:
                return "sentence-transformers/all-mpnet-base-v2"
        
        # 重排序
        elif use_case == "reranking":
            return "BAAI/bge-reranker-base"
        
        # 多语言
        elif use_case == "multilingual":
            return "xlm-roberta-base"
        
        return "sentence-transformers/all-MiniLM-L6-v2"
    
    @staticmethod
    def get_recommended_config(use_case: str) -> dict:
        """
        获取推荐配置
        """
        configs = {
            "rag": {
                "normalize": True,
                "batch_size": 32,
                "show_progress": True
            },
            "similarity": {
                "normalize": True,
                "batch_size": 64
            },
            "reranking": {
                "normalize": False,
                "batch_size": 8
            }
        }
        return configs.get(use_case, {})
```

### 2. Benchmark实现

```python
class EmbeddingBenchmark:
    """
    Embedding模型性能基准测试
    """
    def __init__(self, test_dataset: dict):
        """
        test_dataset: {
            "queries": [...],
            "documents": [...],
            "relevant": {...}  # query_id -> [doc_ids]
        }
        """
        self.dataset = test_dataset
        
    def evaluate(self, model: HuggingFaceEmbedding, k_values=[1, 5, 10, 20]) -> dict:
        """
        评估模型性能
        """
        results = {
            "recall@k": {},
            "mrr": [],
            "ndcg": []
        }
        
        for query_id, query in enumerate(self.dataset["queries"]):
            # 编码查询和文档
            query_emb = model.encode([query])
            doc_embs = model.encode(self.dataset["documents"])
            
            # 计算相似度
            scores = np.dot(query_emb, doc_embs.T)[0]
            
            # 排序
            ranked_indices = np.argsort(scores)[::-1]
            ranked_docs = [self.dataset["documents"][i] for i in ranked_indices]
            
            # 计算指标
            relevant_docs = set(self.dataset["relevant"][query_id])
            
            for k in k_values:
                top_k_docs = set(ranked_indices[:k])
                recall = len(top_k_docs & relevant_docs) / len(relevant_docs)
                
                if k not in results["recall@k"]:
                    results["recall@k"][k] = []
                results["recall@k"][k].append(recall)
            
            # MRR
            for i, doc_idx in enumerate(ranked_indices):
                if doc_idx in relevant_docs:
                    results["mrr"].append(1.0 / (i + 1))
                    break
        
        # 平均
        for k in k_values:
            results["recall@k"][k] = np.mean(results["recall@k"][k])
        results["mrr"] = np.mean(results["mrr"]) if results["mrr"] else 0
        
        return results
    
    def compare_models(self, models: dict) -> pd.DataFrame:
        """
        对比多个模型
        """
        comparison = []
        
        for model_name, model in models.items():
            metrics = self.evaluate(model)
            
            comparison.append({
                "model": model_name,
                "recall@1": metrics["recall@k"].get(1, 0),
                "recall@5": metrics["recall@k"].get(5, 0),
                "recall@10": metrics["recall@k"].get(10, 0),
                "mrr": metrics["mrr"]
            })
        
        return pd.DataFrame(comparison)
```

## Embedding优化

### 1. Matryoshka表示学习

```python
class MatryoshkaEmbedding:
    """
    Matryoshka表示学习
    一模型多维度输出，节省存储
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str], dimensions: List[int] = [768, 512, 256, 128]):
        """
        输出多维度表示
        """
        # 获取完整表示
        full_embeddings = self.model.encode(texts)
        
        # 截取不同维度
        matryoshka_embeddings = []
        for dim in dimensions:
            truncated = full_embeddings[:, :dim]
            # 归一化
            truncated = truncated / np.linalg.norm(truncated, axis=1, keepdims=True)
            matryoshka_embeddings.append(truncated)
        
        return matryoshka_embeddings
    
    def encode_with_dim(self, text: str, dim: int) -> np.ndarray:
        """编码为指定维度"""
        emb = self.model.encode([text])
        truncated = emb[0, :dim]
        return truncated / np.linalg.norm(truncated)
```

### 2. 在线量化

```python
class EmbeddingQuantizer:
    """
    Embedding向量量化
    减少存储空间
    """
    def __init__(self, n_bits: int = 8):
        self.n_bits = n_bits
        self.codebook = None
        
    def fit(self, embeddings: np.ndarray):
        """
        训练量化器
        """
        from sklearn.cluster import KMeans
        
        n_clusters = 2 ** self.n_bits
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        
        self.codebook = kmeans.cluster_centers_
        
    def quantize(self, embeddings: np.ndarray) -> tuple:
        """
        量化
        """
        # 计算到各中心的距离
        distances = np.linalg.norm(
            embeddings[:, np.newaxis, :] - self.codebook[np.newaxis, :, :],
            axis=2
        )
        
        # 最近邻
        codes = np.argmin(distances, axis=1)
        
        # 量化值
        quantized = self.codebook[codes]
        
        return codes, quantized
    
    def compression_ratio(self) -> float:
        """压缩比"""
        original_bits = 32
        compressed_bits = self.n_bits + np.log2(len(self.codebook))
        return original_bits / compressed_bits
```

## 总结

选择合适的Embedding模型需要综合考虑语言特性、精度要求、推理延迟和成本等因素。本文详细对比了主流Embedding模型的特性和适用场景，并提供了完整的实现代码和选型指南。建议根据具体业务需求进行benchmark测试，选择最适合的模型。

## 参考资源

- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
