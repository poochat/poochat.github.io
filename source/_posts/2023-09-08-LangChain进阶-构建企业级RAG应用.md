---
title: LangChain进阶：构建企业级RAG应用
date: 2023-09-08 11:00:00
categories: RAG系统
tags: [LangChain, RAG, LLM应用, 向量检索]
photos:
  - "https://images.unsplash.com/photo-1558494949-ef010cbdcc31"
---

## 引言

LangChain是构建LLM应用的主流框架，提供了丰富的组件和工具链。本文将深入探讨如何使用LangChain构建生产级别的RAG应用，包括文档处理、检索优化、结果重排序等高级特性。

## LangChain核心组件

### 1. 文档加载与分割

```python
from langchain.document_loaders import (
    PyPDFLoader, UnstructuredHTMLLoader, 
    CSVLoader, JSONLoader, Docx2txtLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain.schema import Document

class DocumentPipeline:
    """
    文档处理流水线
    """
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitters = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""]
            ),
            "markdown": MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ),
            "code": PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        }
    
    def load_pdf(self, file_path: str) -> list[Document]:
        """加载PDF"""
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        return pages
    
    def load_html(self, file_path: str) -> list[Document]:
        """加载HTML"""
        loader = UnstructuredHTMLLoader(file_path)
        docs = loader.load()
        return docs
    
    def load_docx(self, file_path: str) -> list[Document]:
        """加载Word文档"""
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        return docs
    
    def load_json(self, file_path: str, jq_schema: str = ".[]") -> list[Document]:
        """加载JSON"""
        loader = JSONLoader(file_path, jq_schema=jq_schema)
        docs = loader.load()
        return docs
    
    def split_documents(self, documents: list[Document], 
                        splitter_type: str = "recursive") -> list[Document]:
        """
        分割文档
        """
        splitter = self.text_splitters.get(splitter_type, self.text_splitters["recursive"])
        return splitter.split_documents(documents)
    
    def process_url(self, url: str) -> list[Document]:
        """从URL加载内容"""
        from langchain.document_loaders import WebBaseLoader
        
        loader = WebBaseLoader(url)
        docs = loader.load()
        return self.split_documents(docs)
```

### 2. Embedding与向量化

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS, Milvus
from langchain.embeddings import CacheBackedEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingManager:
    """
    Embedding管理器
    """
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.embeddings = self._init_embeddings()
        
    def _init_embeddings(self):
        if self.provider == "openai":
            return OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key="your-api-key"
            )
        elif self.provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif self.provider == "m3e":
            return HuggingFaceEmbeddings(
                model_name="moka-ai/m3e-base",
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def create_vectorstore(self, documents: list[Document], 
                          db_type: str = "chroma",
                          persist_directory: str = None) -> any:
        """
        创建向量数据库
        """
        if db_type == "chroma":
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
        elif db_type == "faiss":
            return FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        elif db_type == "milvus":
            return Milvus.from_documents(
                documents=documents,
                embedding=self.embeddings,
                connection_args={"host": "localhost", "port": "19530"}
            )
```

## 高级RAG模式

### 1. 父子文档检索

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

class ParentDocumentSearch:
    """
    父子文档检索器
    父文档提供完整上下文，子文档用于精确匹配
    """
    def __init__(self, vectorstore, embeddings):
        # 存储父文档
        self.store = InMemoryStore()
        
        self.retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=self.store,
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=40
            ),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            ),
            child_metadata={"source": "child"},
            parent_metadata={"source": "parent"}
        )
    
    def add_documents(self, documents: list[Document]):
        """添加文档"""
        self.retriever.add_documents(documents)
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        """获取相关文档"""
        return self.retriever.get_relevant_documents(query)
```

### 2. 自查询检索器

```python
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

class SelfQuerySearch:
    """
    自查询检索器
    自动从自然语言中提取过滤条件
    """
    def __init__(self, vectorstore, llm, document_contents: str, metadata_field_info: list):
        self.metadata_field_info = [
            AttributeInfo(
                name="source",
                description="The source of the document",
                type="string"
            ),
            AttributeInfo(
                name="date",
                description="The date of the document",
                type="datetime"
            ),
            AttributeInfo(
                name="category",
                description="The category of the document",
                type="string"
            )
        ]
        
        self.retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents=document_contents,
            metadata_field_info=self.metadata_field_info
        )
    
    def search(self, query: str) -> list[Document]:
        """
        支持自然语言过滤的检索
        例如: "查找2023年3月关于AI的技术文档"
        """
        return self.retriever.get_relevant_documents(query)
```

### 3. 集成检索器

```python
from langchain.retrievers import EnsembleRetriever

class EnsembleSearch:
    """
    集成检索器
    结合多种检索方法
    """
    def __init__(self, vectorstore, bm25_retriever, keyword_weights: list = None):
        # 向量检索
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 10}
        )
        
        # BM25检索
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=keyword_weights or [0.5, 0.5]  # 默认各50%
        )
    
    def search(self, query: str) -> list[Document]:
        """集成检索"""
        return self.retriever.get_relevant_documents(query)
```

### 4. 上下文压缩检索

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter

class ContextualCompressionSearch:
    """
    上下文压缩检索
    压缩检索结果，提取最相关的部分
    """
    def __init__(self, base_retriever, llm):
        # 使用LLM提取相关内容
        compressor = LLMChainExtractor.from_llm(llm)
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    def search(self, query: str) -> list[Document]:
        """
        返回压缩后的相关文档
        """
        return self.retriever.get_relevant_documents(query)
```

## Chain构建

### 1. 基础QA Chain

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class QABot:
    """
    问答机器人
    """
    def __init__(self, vectorstore, llm_model="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.vectorstore = vectorstore
        
    def build_chain(self, prompt_template: str = None):
        """
        构建问答链
        """
        if prompt_template is None:
            prompt_template = """使用以下背景信息回答问题。如果你不知道答案，请明确说明，不要编造答案。

背景信息：
{context}

问题：{question}

回答："""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={
                "prompt": PROMPT,
                "document_variable_name": "context"
            },
            return_source_documents=True
        )
    
    def ask(self, question: str) -> dict:
        """
        提问
        """
        chain = self.build_chain()
        result = chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
```

### 2. 对话检索链

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class ConversationalBot:
    """
    对话式问答机器人
    支持多轮对话
    """
    def __init__(self, vectorstore, llm_model="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.vectorstore = vectorstore
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
    def build_chain(self):
        """
        构建对话链
        """
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            memory=self.memory,
            condense_question_prompt=self._get_condense_prompt(),
            combine_docs_chain_kwargs={"prompt": self._get_doc_prompt()}
        )
    
    def _get_condense_prompt(self):
        """
        问题改写prompt
        将对话历史+新问题改写为独立问题
        """
        template = """给定以下对话历史和一个后续问题，将后续问题重新表述为一个独立的问题。

对话历史：
{chat_history}

后续问题：{question}

独立问题："""
        return PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
    
    def _get_doc_prompt(self):
        """
        文档问答prompt
        """
        template = """使用以下文档回答问题。

文档：
{context}

问题：{question}

回答："""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def chat(self, question: str) -> dict:
        """
        对话
        """
        chain = self.build_chain()
        result = chain({"question": question})
        
        return {
            "answer": result["answer"],
            "chat_history": result.get("chat_history", [])
        }
```

### 3. 带来源引用的RAG

```python
class SourceCitationChain:
    """
    带来源引用的RAG链
    """
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        
    def build_chain(self):
        """
        构建带引用的chain
        """
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        # 检索+生成链
        prompt = PromptTemplate(
            template="""根据以下文档回答问题，并在回答中引用文档来源。

文档：
{context}

问题：{question}

请在回答中用[1]、[2]等形式标注引用来源。
回答：""",
            input_variables=["context", "question"]
        )
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        from langchain.chains import RetrievalQAWithSourcesChain
        
        return RetrievalQAWithSourcesChain(
            combine_documents_chain=llm_chain,
            retriever=self.vectorstore.as_retriever()
        )
    
    def ask_with_sources(self, question: str) -> dict:
        """
        提问并返回来源
        """
        chain = self.build_chain()
        result = chain({"question": question})
        
        return {
            "answer": result["answer"],
            "sources": result.get("sources", ""),
            "source_documents": result.get("source_documents", [])
        }
```

## 生产级部署

### 1. 完整RAG服务

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="RAG API", version="1.0")

class RAGService:
    """
    RAG服务
    """
    def __init__(self, model_path: str, vectorstore_path: str):
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings
        
        # 加载向量数据库
        embeddings = HuggingFaceEmbeddings(
            model_name="moka-ai/m3e-base"
        )
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embeddings
        )
        
        # 初始化LLM
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        # 构建chain
        self.qa_chain = self._build_qa_chain()
    
    def _build_qa_chain(self):
        from langchain.chains import RetrievalQA
        
        return RetrievalQA.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
        )
    
    def query(self, question: str) -> dict:
        """查询"""
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
    
    def add_documents(self, documents: List[str], metadatas: List[dict]):
        """添加文档"""
        from langchain.schema import Document
        
        docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(documents, metadatas)
        ]
        self.vectorstore.add_documents(docs)


# API端点
class QueryRequest(BaseModel):
    question: str
    use_citation: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


# 全局RAG服务实例
rag_service = None

@app.on_event("startup")
async def startup():
    global rag_service
    rag_service = RAGService(
        model_path="gpt-3.5-turbo",
        vectorstore_path="./vectorstore"
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询接口"""
    if rag_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    result = rag_service.query(request.question)
    
    return QueryResponse(
        answer=result["answer"],
        sources=[
            {
                "content": doc.page_content[:200],
                "metadata": doc.metadata
            }
            for doc in result.get("sources", [])
        ]
    )

@app.post("/documents")
async def add_documents(documents: List[str], metadatas: List[dict]):
    """添加文档"""
    if rag_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    rag_service.add_documents(documents, metadatas)
    return {"status": "success", "count": len(documents)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 性能优化

### 1. 缓存Embedding

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from functools import partial

class CachedEmbedding:
    """
    带缓存的Embedding
    避免重复计算
    """
    def __init__(self, underlying_embeddings):
        # 本地文件系统缓存
        store = LocalFileStore("./cache/embeddings")
        
        self.embeddings = CacheBackedEmbeddings(
            underlying_embeddings=underlying_embeddings,
            document_embedding_cache=store
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """缓存的文档嵌入"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """缓存的查询嵌入"""
        return self.embeddings.embed_query(query)
```

### 2. 批量处理

```python
class BatchProcessor:
    """
    批量文档处理
    """
    def __init__(self, embedding_manager, vectorstore):
        self.embedding_manager = embedding_manager
        self.vectorstore = vectorstore
    
    def process_large_dataset(self, documents: List[Document], 
                              batch_size: int = 100):
        """
        分批处理大量文档
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}")
            
            # 添加到向量数据库
            self.vectorstore.add_documents(batch)
    
    def incremental_indexing(self, new_documents: List[Document]):
        """
        增量索引
        """
        # 只索引新文档
        self.vectorstore.add_documents(new_documents)
```

## 总结

LangChain为构建RAG应用提供了完整的基础设施。通过合理使用文档加载器、分割器、检索器和Chain组件，可以构建功能强大的企业级问答系统。本文介绍的高级特性如父子文档检索、自查询检索、上下文压缩等，可以进一步提升系统的准确性和用户体验。在实际生产环境中，还需要考虑性能优化、监控告警、多租户隔离等问题，以确保系统的稳定运行。

## 参考资源

- [LangChain官方文档](https://python.langchain.com/)
- [RAG教程](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Hub](https://smith.langchain.com/hub)
