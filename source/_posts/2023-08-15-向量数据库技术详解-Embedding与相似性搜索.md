---
title: 向量数据库技术详解：Embedding与相似性搜索
date: 2023-08-15 09:00:00
categories: RAG系统
tags: [向量数据库, Embedding, Milvus, Pinecone, FAISS]
photos:
  - "https://images.unsplash.com/photo-1518186285589-2f7649de83e0"
---

## 引言

向量数据库是2023年AI基础设施领域最热门的技术之一。随着大语言模型和RAG系统的快速发展，高效的向量存储和检索成为关键能力。本文将全面介绍向量数据库的核心技术、主流产品对比以及工程实践。

## 向量Embedding基础

### 1. 什么是向量Embedding

向量Embedding是将高维稀疏数据（如文本、图像）映射到低维稠密向量的技术。相似的对象在向量空间中距离更近。

```python
import numpy as np
from typing import List, Union

class EmbeddingModel:
    """
    Embedding模型封装
    支持多种Embedding服务
    """
    def __init__(self, provider="openai", model="text-embedding-ada-002"):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            import openai
            self.client = openai.Embedding
        elif provider == "huggingface":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)
    
    def embed(self, text: str) -> np.ndarray:
        """
        单文本嵌入
        """
        if self.provider == "openai":
            response = self.client.create(
                input=text,
                model=self.model
            )
            return np.array(response['data'][0]['embedding'])
        elif self.provider == "huggingface":
            return self.model.encode(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量嵌入
        """
        if self.provider == "openai":
            response = self.client.create(
                input=texts,
                model=self.model
            )
            embeddings = [item['embedding'] for item in response['data']]
            return np.array(embeddings)
        elif self.provider == "huggingface":
            return self.model.encode(texts)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个向量的相似度
        """
        # 余弦相似度
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

### 2. 常见Embedding模型

```python
class EmbeddingModels:
    """
    常用Embedding模型对比
    """
    MODELS = {
        # OpenAI
        "text-embedding-ada-002": {
            "dims": 1536,
            "provider": "openai",
            "context_length": 8191,
            "description": "最强通用模型"
        },
        # HuggingFace
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dims": 384,
            "provider": "huggingface",
            "description": "轻量高效，适合中文"
        },
        "moka-ai/m3e-base": {
            "dims": 768,
            "provider": "huggingface",
            "description": "中文优化，M3E"
        },
        # BGE
        "BAAI/bge-large-zh": {
            "dims": 1024,
            "provider": "huggingface",
            "description": "中文BGE，大型模型"
        },
        # Cohere
        "embed-english-v3.0": {
            "dims": 1024,
            "provider": "cohere",
            "description": "英文专用"
        }
    }
```

## 相似性搜索算法

### 1. 精确搜索 vs 近似搜索

```python
class ExactSearch:
    """
    精确最近邻搜索
    适用于小规模数据集
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = []
        self.metadata = []
    
    def add(self, vectors: np.ndarray, metadata: List[dict]):
        """
        添加向量
        """
        self.vectors.append(vectors)
        self.metadata.extend(metadata)
        self.vectors = np.vstack(self.vectors) if len(self.vectors) > 1 else self.vectors[0]
    
    def search(self, query: np.ndarray, k: int = 5) -> List[dict]:
        """
        精确搜索
        O(N) 时间复杂度
        """
        # 计算所有距离
        distances = np.linalg.norm(self.vectors - query, axis=1)
        
        # 获取top-k
        top_k_idx = np.argsort(distances)[:k]
        
        return [
            {
                "id": idx,
                "distance": float(distances[idx]),
                "metadata": self.metadata[idx]
            }
            for idx in top_k_idx
        ]
```

### 2. FAISS索引详解

```python
import faiss
import numpy as np

class FAISSIndex:
    """
    FAISS向量索引封装
    """
    def __init__(self, dim: int, index_type: str = "IVF"):
        self.dim = dim
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self._build_index()
    
    def _build_index(self):
        """
        根据类型构建索引
        """
        if self.index_type == "Flat":
            # IndexFlatIP: 内积相似度（余弦相似度需归一化）
            # IndexFlatL2: 欧氏距离
            self.index = faiss.IndexFlatIP(self.dim)
            
        elif self.index_type == "IVF":
            # 倒排文件索引
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.dim, 
                nlist=100  # 聚类中心数
            )
            
        elif self.index_type == "HNSW":
            # 分层可导航小世界图
            self.index = faiss.IndexHNSWFlat(self.dim, m=32)
            
        elif self.index_type == "PQ":
            # 产品量化
            self.index = faiss.IndexPQ(self.dim, m=16, nbits=8)
            
        elif self.index_type == "IVF_PQ":
            # 组合IVF和PQ
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, 
                self.dim, 
                nlist=100,
                m=16,
                nbits=8
            )
    
    def train(self, vectors: np.ndarray):
        """
        训练索引（某些索引需要训练）
        """
        if not self.index.is_trained:
            # 归一化用于余弦相似度
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            self.index.train(vectors)
    
    def add(self, vectors: np.ndarray, metadata: List[dict]):
        """
        添加向量
        """
        # 归一化
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.index.add(vectors)
        self.metadata.extend(metadata)
    
    def search(self, query: np.ndarray, k: int = 5) -> List[dict]:
        """
        近似最近邻搜索
        """
        # 归一化
        query = query / np.linalg.norm(query)
        
        # 搜索
        distances, indices = self.index.search(query.reshape(1, -1), k)
        
        # 构建结果
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "id": int(idx),
                    "score": float(dist),
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def save(self, path: str):
        """保存索引"""
        faiss.write_index(self.index, path)
    
    def load(self, path: str):
        """加载索引"""
        self.index = faiss.read_index(path)
```

### 3. HNSW算法原理

```python
class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) 算法实现
    
    核心思想：
    1. 构建多层图，上层稀疏、下层稠密
    2. 搜索时从上层开始，快速定位到大致区域
    3. 在下层精细搜索
    """
    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200):
        self.dim = dim
        self.m = m  # 每个节点的最大连接数
        self.ef_construction = ef_construction  # 构建时的搜索范围
        self.ef_search = 100  # 搜索时的搜索范围
        
        # FAISS提供了HNSW实现
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.index.hnsw.efConstruction = ef_construction
        
    def set_search_params(self, ef: int):
        """设置搜索参数"""
        self.ef_search = ef
        self.index.hnsw.efSearch = ef
    
    def add(self, vectors: np.ndarray):
        """添加向量"""
        self.index.add(vectors)
    
    def search(self, query: np.ndarray, k: int = 5) -> tuple:
        """搜索"""
        return self.index.search(query, k)
```

## 主流向量数据库对比

### 1. Milvus部署与使用

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

class MilvusDB:
    """
    Milvus向量数据库操作封装
    """
    def __init__(self, host="localhost", port="19530"):
        connections.connect(host=host, port=port)
        self.collection = None
        
    def create_collection(self, name: str, dim: int, metric_type="IP"):
        """
        创建集合
        """
        if utility.has_collection(name):
            utility.drop_collection(name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields=fields, description=f"Collection: {name}")
        self.collection = Collection(name=name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        self.collection.load()
        
    def insert(self, embeddings: np.ndarray, texts: List[str], metadata: List[dict]):
        """
        插入数据
        """
        import json
        
        entities = [
            embeddings.tolist(),
            texts,
            [json.dumps(m) for m in metadata]
        ]
        
        self.collection.insert(entities)
        self.collection.flush()
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        相似性搜索
        """
        import json
        
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata"]
        )
        
        return [
            {
                "id": hit.id,
                "distance": hit.distance,
                "text": hit.entity.get("text"),
                "metadata": json.loads(hit.entity.get("metadata"))
            }
            for hit in results[0]
        ]
    
    def delete_by_ids(self, ids: List[int]):
        """删除数据"""
        self.collection.delete(f"id in {ids}")
        
    def drop(self):
        """删除集合"""
        if self.collection:
            self.collection.drop()
```

### 2. Chroma向量数据库

```python
import chromadb
from chromadb.config import Settings

class ChromaDB:
    """
    Chroma向量数据库
    轻量级，易于使用
    """
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # 持久化版本
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def get_or_create_collection(self, name: str):
        """获取或创建集合"""
        return self.client.get_or_create_collection(name=name)
    
    def add_documents(self, collection_name: str, documents: List[str], 
                      embeddings: np.ndarray, metadatas: List[dict] = None, 
                      ids: List[str] = None):
        """
        添加文档
        """
        collection = self.get_or_create_collection(collection_name)
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, collection_name: str, query_embedding: np.ndarray, 
              n_results: int = 5, where: dict = None):
        """
        查询
        """
        collection = self.get_or_create_collection(collection_name)
        
        return collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where
        )
    
    def delete_collection(self, name: str):
        """删除集合"""
        self.client.delete_collection(name)
```

### 3. 向量数据库对比

| 特性 | Milvus | Chroma | Pinecone | Weaviate |
|-----|--------|--------|----------|----------|
| 部署方式 | 自托管/云 | 本地/嵌入式 | 云服务 | 自托管/云 |
| 规模 | 十亿级 | 百万级 | 亿级 | 亿级 |
| 延迟 | 低 | 低 | 低 | 低 |
| 过滤 | 支持 | 支持 | 支持 | 支持 |
| 成本 | 开源免费 | 开源免费 | 按量付费 | 开源免费 |

## 工程最佳实践

### 1. 向量索引配置

```python
class VectorIndexConfig:
    """
    向量索引配置指南
    """
    @staticmethod
    def get_config(dataset_size: int, dim: int, use_case: str) -> dict:
        """
        根据场景选择索引配置
        """
        if use_case == "recall_first":
            # 优先召回率
            return {
                "index_type": "HNSW",
                "params": {
                    "m": 32,
                    "efConstruction": 200
                },
                "ef_search": 500
            }
        
        elif use_case == "balance":
            # 均衡模式
            return {
                "index_type": "IVF_PQ",
                "params": {
                    "nlist": max(100, dataset_size // 10),
                    "m": 16,
                    "nbits": 8
                },
                "nprobe": 20
            }
        
        elif use_case == "memory_optimized":
            # 内存优化
            return {
                "index_type": "PQ",
                "params": {
                    "m": 16,
                    "nbits": 8
                }
            }
        
        return {"index_type": "Flat"}
    
    @staticmethod
    def estimate_memory(dim: int, num_vectors: int, index_type: str) -> int:
        """
        估算内存占用
        """
        bytes_per_float = 4
        
        if index_type == "Flat":
            return dim * num_vectors * bytes_per_float
        
        elif index_type == "HNSW":
            # 图结构开销
            m = 32
            return dim * num_vectors * bytes_per_float * 2
        
        elif index_type == "PQ":
            # 量化压缩
            m = 16
            return num_vectors * m * bytes_per_float
        
        return dim * num_vectors * bytes_per_float
```

### 2. 混合搜索实现

```python
class HybridSearch:
    """
    混合搜索：向量检索 + 标量过滤
    """
    def __init__(self, vector_db, mysql_db):
        self.vector_db = vector_db
        self.mysql_db = mysql_db
    
    def search_with_filter(self, query: str, embedding_model, 
                           filter_conditions: dict, top_k: int = 10):
        """
        带过滤的混合搜索
        """
        # 1. 获取向量
        query_emb = embedding_model.embed(query)
        
        # 2. 向量检索
        vector_results = self.vector_db.search(query_emb, top_k * 2)
        
        # 3. 标量过滤
        filtered_results = []
        for result in vector_results:
            metadata = result['metadata']
            
            # 检查过滤条件
            match = True
            for key, value in filter_conditions.items():
                if metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
```

## 总结

向量数据库是现代AI应用的核心基础设施。通过合理选择索引类型、配置搜索参数，可以实现高效的向量存储和检索。本文介绍的FAISS、Milvus、Chroma等工具各有特点，开发者应根据具体场景选择合适的方案。随着AI应用的持续发展，向量数据库将在知识管理、推荐系统、语义搜索等领域发挥越来越重要的作用。

## 参考资源

- [FAISS官方文档](https://faiss.ai/)
- [Milvus官方文档](https://milvus.io/)
- [Chroma文档](https://docs.trychroma.com/)
- [向量数据库对比](https://superlinked.com/vector-db-comparison)
