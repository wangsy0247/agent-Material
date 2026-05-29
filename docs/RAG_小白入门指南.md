# 企业落地 RAG 小白入门指南（LangGraph 实战版）

> **目标读者**：想在企业中落地 RAG 系统、但不知道从哪下手的开发者和架构师。
>
> **学习目标**：理解 RAG 的完整数据流（离线 + 在线），掌握各种切分/检索/重排策略，能根据业务场景选择传统 RAG / GraphRAG / AgenticRAG，并能在 LangGraph 中写出生产级代码。

---

## 目录

- [企业落地 RAG 小白入门指南（LangGraph 实战版）](#企业落地-rag-小白入门指南langgraph-实战版)
  - [目录](#目录)
  - [一、RAG 是什么？一句话理解](#一rag-是什么一句话理解)
  - [二、离线模式：数据准备 Pipeline](#二离线模式数据准备-pipeline)
    - [2.1 收集文档](#21-收集文档)
    - [2.2 加载文档（LangGraph / LangChain 中如何加载）](#22-加载文档langgraph--langchain-中如何加载)
    - [2.3 数据清洗 ⭐](#23-数据清洗-)
      - [2.3.1 为什么要数据清洗？](#231-为什么要数据清洗)
      - [2.3.2 常见脏数据类型](#232-常见脏数据类型)
      - [2.3.3 文本层清洗](#233-文本层清洗)
      - [2.3.4 文档层清洗](#234-文档层清洗)
      - [2.3.5 结构化内容保留](#235-结构化内容保留)
      - [2.3.6 元数据清洗](#236-元数据清洗)
      - [2.3.7 完整清洗 Pipeline 代码](#237-完整清洗-pipeline-代码)
      - [2.3.8 清洗质量检查](#238-清洗质量检查)
    - [2.4 文档切分](#24-文档切分)
      - [2.4.1 为什么要切分？](#241-为什么要切分)
      - [2.4.2 常见切分策略详解](#242-常见切分策略详解)
      - [2.4.3 切分参数怎么调？（企业实战建议）](#243-切分参数怎么调企业实战建议)
      - [2.4.4 不同切分方式对检索质量的影响](#244-不同切分方式对检索质量的影响)
    - [2.5 向量化](#25-向量化)
      - [2.5.1 什么是向量化？](#251-什么是向量化)
      - [2.5.2 如何选择 Embedding 模型？](#252-如何选择-embedding-模型)
      - [2.5.3 相似度计算](#253-相似度计算)
      - [2.5.4 ANN 近似最近邻算法](#254-ann-近似最近邻算法)
      - [2.5.5 批量向量化（Batch Embedding）](#255-批量向量化batch-embedding)
    - [2.6 向量入库](#26-向量入库)
      - [2.6.1 常见向量数据库对比](#261-常见向量数据库对比)
      - [2.6.2 LangGraph 中如何入库](#262-langgraph-中如何入库)
  - [三、在线模式：查询 Pipeline](#三在线模式查询-pipeline)
    - [3.1 用户提出问题](#31-用户提出问题)
    - [3.2 Query 重写](#32-query-重写)
    - [3.3 检索](#33-检索)
    - [3.4 重排序（Rerank）](#34-重排序rerank)
    - [3.5 Prompt 拼接与大模型生成](#35-prompt-拼接与大模型生成)
  - [四、RAG 架构选型](#四rag-架构选型)
  - [五、企业落地：成本与稳定性控制](#五企业落地成本与稳定性控制)
  - [六、完整 LangGraph 实战代码](#六完整-langgraph-实战代码)
  - [附录：RAG 流程全景图](#附录rag-流程全景图)

---

## 一、RAG 是什么？一句话理解

**RAG（Retrieval-Augmented Generation，检索增强生成）** = **给大模型配一个"外接大脑"**。

大模型本身的知识是"死"的（训练完就固定了），而 RAG 让它能够：

1. **查资料**（检索）：从你企业的私有文档中找出相关内容
2. **写答案**（生成）：基于查到的资料，组织语言回答用户

```
用户提问: "我们公司今年的年假政策是什么？"
           │
           ▼
┌─────────────────┐
│   检索模块      │  ◄── 从企业知识库中搜索相关文档片段
│  (Retriever)    │
└────────┬────────┘
         │ [相关片段1] 根据《员工手册》第3章，年假为5-15天...
         │ [相关片段2] 入职满1年享受5天，满10年享受10天...
         ▼
┌─────────────────┐
│   生成模块      │  ◄── 把检索结果塞给大模型
│  (Generator)    │      "请根据以下资料回答问题：..."
└────────┬────────┘
         │
         ▼
回答: "根据《员工手册》，入职满1年享受5天年假，满10年享受10天..."
```

**没有 RAG**：大模型要么说"我不知道"，要么瞎编（幻觉）。
**有了 RAG**：大模型基于你的真实文档回答，准确、可追溯、可更新。

---

## 二、离线模式：数据准备 Pipeline

离线模式是 RAG 的"基建工程"——把企业里乱七八糟的文档，整理成 AI 能查的向量知识库。

```
收集文档 ──► 加载文档 ──► 数据清洗 ──► 文档切分 ──► 向量化 ──► 向量入库
(各种格式)    (解析内容)    (去脏去噪)    (切成块)     (转成向量)   (存进数据库)
```

### 2.1 收集文档

企业里的文档五花八门：

| 文档类型 | 常见格式 | 处理难点 |
|----------|----------|----------|
| 规章制度 | PDF、Word | 扫描版 PDF 需要 OCR |
| 产品手册 | PDF、Markdown | 图片、表格混排 |
| 技术文档 | Markdown、HTML | 代码块、链接 |
| 会议纪要 | Word、TXT | 口语化、格式混乱 |
| 客服对话 | JSON、CSV | 多轮对话结构 |
| 数据库文档 | SQL、Excel | 结构化数据 |

**企业实战建议：**

- 先梳理清楚有哪些数据源，不要一上来就全量导入
- 区分"高频查询文档"和"低频归档文档"，优先处理高频的
- 脏数据（乱码、空文档、重复文档）先清洗，否则会影响检索质量

### 2.2 加载文档（LangGraph / LangChain 中如何加载）

LangChain 提供了丰富的 **Document Loader**，LangGraph 中可以直接复用。

```python
from langchain_community.document_loaders import (
    PyPDFLoader,           # PDF
    UnstructuredWordDocumentLoader,  # Word
    TextLoader,            # TXT
    CSVLoader,             # CSV
    JSONLoader,            # JSON
    WebBaseLoader,         # 网页
    DirectoryLoader,       # 整个目录
)

# 加载单个 PDF
loader = PyPDFLoader("/path/to/员工手册.pdf")
docs = loader.load()  # 返回 List[Document]

# 加载整个目录（递归加载所有 PDF）
loader = DirectoryLoader(
    "/path/to/knowledge_base/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
docs = loader.load()

# 加载网页
docs = WebBaseLoader("https://company.com/docs").load()

# 加载 Markdown（带元数据）
from langchain_community.document_loaders import UnstructuredMarkdownLoader
loader = UnstructuredMarkdownLoader("README.md", mode="elements")
docs = loader.load()
```

**`Document` 对象长什么样？**

```python
from langchain_core.documents import Document

doc = Document(
    page_content="这里是文档的文本内容...",
    metadata={
        "source": "员工手册.pdf",      # 来源文件
        "page": 3,                    # 第几页
        "title": "年假政策",           # 标题
        "department": "人力资源部",     # 自定义元数据
    }
)
```

> **metadata 非常重要！** 它决定了检索后能不能溯源（"这个答案出自哪份文档的第几页"）。

---

### 2.3 数据清洗 ⭐

加载出来的文档，**千万别直接切分**！ raw 文档里充满了各种"噪音"——乱码、HTML 标签、多余空格、页眉页脚、重复内容……这些脏数据会严重污染向量库，导致检索质量直线下降。

> **业界经验：RAG 系统中 70% 的问题都出在数据质量上，而不是模型不够强。**

#### 2.3.1 为什么要数据清洗？

先看一组"脏数据"的真实案例：

```
【加载后的原始文本】

\n\n\n\n                 员工手册\n\n                 第 3 页\n\n\n\n年假政策\n\n入职满1年\n\n的员工可享受5天年假。\n\n\n\n                 第 4 页\n\n\n\n满10年可享受10天。\n\n\n\nCopyright © 2024 Company Inc. All rights reserved.\n\n\n\n【问题】
这段文本被切分后，会产生大量只有"\n\n"、页码、版权信息的 chunk，
它们会被向量化存入数据库。当用户问"年假"时，这些垃圾 chunk 
也会参与相似度计算，挤占真正有用的文档的位置。
```

**脏数据对 RAG 的破坏**：

| 脏数据类型 | 对 RAG 的影响 |
|-----------|--------------|
| 乱码 / 编码错误 | 无法检索，embedding 产生无意义向量 |
| HTML/XML 标签 | 标签文字混入内容，干扰语义理解 |
| 多余空白/换行 | 产生空 chunk 或纯 whitespace chunk |
| 页眉页脚/页码 | 每页重复的垃圾信息，污染向量库 |
| 重复内容 | 同一信息多次召回，浪费上下文窗口 |
| 特殊字符 / Unicode 艺术字 | 破坏切分逻辑，embedding 质量下降 |
| 口语化填充词 | "嗯""啊""那个"增加噪声 |
| 版权 / 免责声明 | 与业务无关，却占用向量空间 |

#### 2.3.2 常见脏数据类型

| 类别 | 具体表现 | 例子 |
|------|----------|------|
| **编码问题** | UTF-8/BOM/GBK 混用导致乱码 | `�` `å‘˜å·¥æ‰‹å†Œ` |
| **格式残留** | HTML/XML/Markdown 标签未清除 | `<div>内容</div>` `**加粗**` |
| **版式噪音** | 页眉页脚、页码、分隔线 | `--- Page 5 ---` `Copyright ©` |
| **空白异常** | 连续换行、制表符、全角空格 | `\n\n\n\n` `\t\t\t` |
| **重复内容** | 同一文档多版本、模板重复 | 免责声明每页都一样 |
| **低质内容** | 太短（<50字）、无意义、纯数字 | `1.\n2.\n3.` `2024-01-01` |
| **结构化破坏** | 表格变成纯文本、代码缩进丢失 | 表格列对齐被破坏 |
| **OCR 错误** | 扫描件识别错误 | `年假` → `年假`(错别字) |

#### 2.3.3 文本层清洗

文本层清洗针对 `page_content` 的内容本身。

```python
import re
import unicodedata
from typing import List
from langchain_core.documents import Document

def clean_text_content(text: str) -> str:
    """
    通用文本清洗函数
    输入：加载后的原始文本
    输出：清洗后的干净文本
    """
    # Step 1: 统一编码，处理乱码
    # 确保是标准 Unicode，去除控制字符
    text = unicodedata.normalize("NFKC", text)
    
    # Step 2: 去除零宽字符（Zero-width characters）
    # 这些字符肉眼看不见，但会影响切分和 embedding
    zero_width_chars = '\u200b\u200c\u200d\ufeff\u2060\u180e'
    for char in zero_width_chars:
        text = text.replace(char, '')
    
    # Step 3: 去除 HTML/XML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # Step 4: 去除 URL（可选：保留还是替换为 [URL]）
    text = re.sub(r'https?://\S+', '[URL]', text)
    
    # Step 5: 去除邮箱（可选）
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    
    # Step 6: 去除页眉页脚常见模式
    # 页码："- 5 -" "Page 5 of 10" "第 5 页"
    text = re.sub(r'\n?\s*[-–—]\s*\d+\s*[-–—]\s*\n?', '\n', text)
    text = re.sub(r'\n?\s*Page\s+\d+\s+(of|/|\u5f85)\s*\d+\s*\n?', '\n', text)
    text = re.sub(r'\n?\s*\u7b2c\s*\d+\s*\u9875\s*\n?', '\n', text)
    
    # Step 7: 去除版权 / 免责声明（常见固定句式）
    copyright_patterns = [
        r'Copyright\s+©.*?\.?\s*\n?',
        r'All\s+rights\s+reserved.*?\n?',
        r'\u672c\u6587\u4ef6\u4ec5\u4f9b\u5185\u90e8\u4f7f\u7528.*?\n?',  # 本文仅供内部使用
        r'\u7981\u6b62\u5916\u4f20.*?\n?',  # 禁止外传
    ]
    for pattern in copyright_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 8: 合并连续空白（包括换行、空格、制表符）
    # 把 3 个以上连续换行合并为 2 个（保留段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 把连续空格合并为 1 个
    text = re.sub(r' {2,}', ' ', text)
    # 去除行首行尾空白
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # Step 9: 去除孤立标点（行首行尾的零散标点）
    text = re.sub(r'\n[\s\-\*\•\·\◦]+\n', '\n', text)
    
    # Step 10: 去除纯数字行（通常是页码或编号残留）
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    return text.strip()


# ========== 应用清洗到所有文档 ==========
def clean_documents(docs: List[Document]) -> List[Document]:
    """批量清洗文档"""
    cleaned = []
    for doc in docs:
        cleaned_text = clean_text_content(doc.page_content)
        
        # 清洗后如果变成空字符串，跳过
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            continue
        
        cleaned.append(Document(
            page_content=cleaned_text,
            metadata=doc.metadata
        ))
    
    print(f"清洗前: {len(docs)} 篇, 清洗后: {len(cleaned)} 篇")
    return cleaned
```

**清洗前后对比**：

```
【清洗前】



                 员工手册

                 第 3 页



年假政策

入职满1年

的员工可享受5天年假。



                 第 4 页



满10年可享受10天。



Copyright © 2024 Company Inc. All rights reserved.



【清洗后】
员工手册

年假政策

入职满1年的员工可享受5天年假。

满10年可享受10天。
```

#### 2.3.4 文档层清洗

文档层清洗针对整个 Document 对象，处理**文档级别**的问题。

```python
from collections import Counter
import hashlib

def deduplicate_documents(docs: List[Document], 
                          similarity_threshold: float = 0.9) -> List[Document]:
    """
    文档去重
    策略1: 精确去重（内容完全一样）
    策略2: 近似去重（内容高度相似，基于哈希或 MinHash）
    """
    # === 策略1: 精确去重 ===
    seen_hashes = set()
    unique_docs = []
    
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    print(f"精确去重: {len(docs)} -> {len(unique_docs)}")
    
    # === 策略2: 近似去重（基于 SimHash/MinHash，可选）===
    # 对于大规模数据，可以用 datasketch 库做 MinHash LSH
    # 这里展示一个简化版：基于 n-gram 重叠度判断
    def ngram_similarity(text1: str, text2: str, n: int = 5) -> float:
        """计算两个文本的 n-gram Jaccard 相似度"""
        def get_ngrams(text, n):
            text = text.lower().replace(' ', '')
            return set(text[i:i+n] for i in range(len(text)-n+1))
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0.0
    
    # 两两比较去重（O(N^2)，适合 <10K 文档；大数据量需用 LSH）
    final_docs = []
    for doc in unique_docs:
        is_duplicate = False
        for kept_doc in final_docs:
            if ngram_similarity(doc.page_content, kept_doc.page_content) > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            final_docs.append(doc)
    
    print(f"近似去重后: {len(unique_docs)} -> {len(final_docs)}")
    return final_docs


def filter_low_quality_documents(docs: List[Document],
                                  min_length: int = 50,
                                  max_length: int = 50000) -> List[Document]:
    """
    过滤低质量文档
    """
    filtered = []
    for doc in docs:
        text = doc.page_content.strip()
        length = len(text)
        
        # 规则1: 长度过滤
        if length < min_length:
            continue
        if length > max_length:
            # 超长的可能是整个文件没切分好，这里做标记而不是丢弃
            doc.metadata["oversized"] = True
        
        # 规则2: 中文字符比例检查（防止全是乱码/符号）
        # 适用于中文为主的文档库
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if len(text) > 100 and chinese_chars / len(text) < 0.1:
            # 中文字符 < 10%，可能是纯英文文档或乱码
            # 如果是预期内的英文文档，可以放宽这个检查
            pass  # 根据业务决定是否过滤
        
        # 规则3: 有效信息密度检查
        # 如果一行平均 < 3 个字符，可能是列表/目录碎片
        lines = [l for l in text.split('\n') if l.strip()]
        if lines and sum(len(l) for l in lines) / len(lines) < 3:
            continue
        
        # 规则4: 去除纯模板文档（所有内容都一样）
        # 已经在去重阶段处理
        
        filtered.append(doc)
    
    print(f"质量过滤: {len(docs)} -> {len(filtered)}")
    return filtered
```

#### 2.3.5 结构化内容保留

清洗时**不要破坏结构化内容的语义**——表格、代码块、列表的格式对理解很重要。

```python
def preserve_structured_content(text: str) -> str:
    """
    清洗的同时保留结构化内容的可读性
    """
    # 表格：把 Markdown 表格的空格对齐规范化
    # 原始：|  年假天数  |  入职年限  |
    # 清洗后保持表格格式
    lines = text.split('\n')
    cleaned_lines = []
    
    in_code_block = False
    for line in lines:
        # 保留代码块
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            continue
        
        if in_code_block:
            cleaned_lines.append(line)
            continue
        
        # 保留 Markdown 表格行
        if line.strip().startswith('|') and line.strip().endswith('|'):
            cleaned_lines.append(line)
            continue
        
        # 保留缩进列表
        if re.match(r'^[\s]*[-\*\+•·]\s', line):
            cleaned_lines.append(line)
            continue
        
        # 保留编号列表
        if re.match(r'^[\s]*\d+[\.\)]\s', line):
            cleaned_lines.append(line)
            continue
        
        # 普通文本做基础清洗
        cleaned_line = line.strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)


# 表格清洗示例
def clean_table_content(text: str) -> str:
    """
    把混乱的表格文本规范化
    适合处理从 PDF 提取后格式混乱的表格
    """
    # 把多个空格分隔的列对齐，转换为 Markdown 表格
    lines = text.split('\n')
    table_lines = []
    
    for line in lines:
        # 检测可能是表格的行（有多个连续空格或制表符分隔）
        if re.search(r'\S+\s{2,}\S+', line) or '\t' in line:
            # 把多个空格/制表符统一为 | 分隔
            cells = re.split(r'\s{2,}|\t', line.strip())
            cells = [c.strip() for c in cells if c.strip()]
            if len(cells) >= 2:
                table_lines.append('| ' + ' | '.join(cells) + ' |')
                continue
        
        table_lines.append(line)
    
    return '\n'.join(table_lines)
```

#### 2.3.6 元数据清洗

metadata 也需要清洗，否则会影响过滤和溯源。

```python
def clean_metadata(docs: List[Document]) -> List[Document]:
    """清洗和规范化元数据"""
    for doc in docs:
        meta = doc.metadata
        
        # 1. 标准化 source 路径
        if 'source' in meta:
            source = meta['source']
            # 统一为正斜杠
            source = source.replace('\\', '/')
            # 去除冗余前缀
            source = re.sub(r'^.*?/knowledge_base/', '', source)
            meta['source'] = source
        
        # 2. 页码规范化
        if 'page' in meta:
            try:
                meta['page'] = int(meta['page'])
            except (ValueError, TypeError):
                meta['page'] = 0
        
        # 3. 去除空值和无效元数据
        meta = {k: v for k, v in meta.items() if v is not None and str(v).strip()}
        
        # 4. 添加清洗时间戳（方便追踪）
        from datetime import datetime
        meta['cleaned_at'] = datetime.now().isoformat()
        
        doc.metadata = meta
    
    return docs
```

#### 2.3.7 完整清洗 Pipeline 代码

```python
# cleaning_pipeline.py - 完整的数据清洗流水线
import re
import hashlib
import unicodedata
from typing import List, Callable
from datetime import datetime
from langchain_core.documents import Document

class DocumentCleaner:
    """可配置的企业级文档清洗器"""
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_copyright: bool = True,
        remove_page_numbers: bool = True,
        normalize_whitespace: bool = True,
        min_doc_length: int = 50,
        max_doc_length: int = 50000,
        dedup_exact: bool = True,
        dedup_approx: bool = False,
        dedup_threshold: float = 0.9,
        preserve_tables: bool = True,
        preserve_code_blocks: bool = True,
    ):
        self.config = {
            'remove_html': remove_html,
            'remove_urls': remove_urls,
            'remove_emails': remove_emails,
            'remove_copyright': remove_copyright,
            'remove_page_numbers': remove_page_numbers,
            'normalize_whitespace': normalize_whitespace,
            'min_doc_length': min_doc_length,
            'max_doc_length': max_doc_length,
            'dedup_exact': dedup_exact,
            'dedup_approx': dedup_approx,
            'dedup_threshold': dedup_threshold,
            'preserve_tables': preserve_tables,
            'preserve_code_blocks': preserve_code_blocks,
        }
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'removed_empty': 0,
            'removed_short': 0,
            'removed_duplicate': 0,
            'removed_oversized': 0,
        }
    
    def clean_text(self, text: str) -> str:
        """清洗单个文本"""
        cfg = self.config
        
        # 1. 编码规范化
        text = unicodedata.normalize("NFKC", text)
        
        # 2. 去除零宽字符
        zero_width = '\u200b\u200c\u200d\ufeff\u2060\u180e'
        for c in zero_width:
            text = text.replace(c, '')
        
        # 3. 去除 HTML 标签
        if cfg['remove_html']:
            text = re.sub(r'<[^>]+>', '', text)
        
        # 4. 去除 URL
        if cfg['remove_urls']:
            text = re.sub(r'https?://\S+', '[URL]', text)
        
        # 5. 去除邮箱
        if cfg['remove_emails']:
            text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        
        # 6. 去除页码
        if cfg['remove_page_numbers']:
            text = re.sub(r'\n?\s*[-–—]\s*\d+\s*[-–—]\s*\n?', '\n', text)
            text = re.sub(r'\n?\s*Page\s+\d+.*?\n?', '\n', text, flags=re.I)
            text = re.sub(r'\n?\s*第\s*\d+\s*页\s*\n?', '\n', text)
        
        # 7. 去除版权
        if cfg['remove_copyright']:
            patterns = [
                r'Copyright\s+©.*?\.?\s*\n?',
                r'All\s+rights\s+reserved.*?\n?',
                r'本文[件]?仅供[内部]?使用.*?\n?',
                r'禁止[外传传播].*?\n?',
                r'Confidential.*?\n?',
            ]
            for p in patterns:
                text = re.sub(p, '', text, flags=re.IGNORECASE)
        
        # 8. 规范化空白
        if cfg['normalize_whitespace']:
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # 9. 去除纯数字行
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()
    
    def deduplicate(self, docs: List[Document]) -> List[Document]:
        """文档去重"""
        cfg = self.config
        
        # 精确去重
        if cfg['dedup_exact']:
            seen = set()
            unique = []
            for d in docs:
                h = hashlib.md5(d.page_content.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    unique.append(d)
                else:
                    self.stats['removed_duplicate'] += 1
            docs = unique
        
        return docs
    
    def filter_quality(self, docs: List[Document]) -> List[Document]:
        """质量过滤"""
        cfg = self.config
        filtered = []
        
        for d in docs:
            text = d.page_content.strip()
            length = len(text)
            
            # 空文档
            if not text:
                self.stats['removed_empty'] += 1
                continue
            
            # 太短
            if length < cfg['min_doc_length']:
                self.stats['removed_short'] += 1
                continue
            
            # 太长（标记但不删除，让切分处理）
            if length > cfg['max_doc_length']:
                self.stats['removed_oversized'] += 1
                d.metadata['oversized'] = True
            
            filtered.append(d)
        
        return filtered
    
    def clean(self, docs: List[Document]) -> List[Document]:
        """执行完整清洗流程"""
        self.stats['input_count'] = len(docs)
        print(f"[DocumentCleaner] 输入: {len(docs)} 篇文档")
        
        # Step 1: 文本清洗
        cleaned = []
        for d in docs:
            new_text = self.clean_text(d.page_content)
            if new_text:
                cleaned.append(Document(
                    page_content=new_text,
                    metadata=d.metadata.copy()
                ))
        print(f"  → 文本清洗后: {len(cleaned)} 篇")
        
        # Step 2: 去重
        cleaned = self.deduplicate(cleaned)
        print(f"  → 去重后: {len(cleaned)} 篇")
        
        # Step 3: 质量过滤
        cleaned = self.filter_quality(cleaned)
        print(f"  → 质量过滤后: {len(cleaned)} 篇")
        
        # Step 4: 元数据清洗
        for d in cleaned:
            d.metadata['cleaned_at'] = datetime.now().isoformat()
            if 'source' in d.metadata:
                d.metadata['source'] = d.metadata['source'].replace('\\', '/')
        
        self.stats['output_count'] = len(cleaned)
        self._print_stats()
        
        return cleaned
    
    def _print_stats(self):
        s = self.stats
        print(f"\n[清洗统计]")
        print(f"  输入: {s['input_count']}")
        print(f"  输出: {s['output_count']}")
        print(f"  删除空文档: {s['removed_empty']}")
        print(f"  删除太短: {s['removed_short']}")
        print(f"  删除重复: {s['removed_duplicate']}")
        print(f"  超大标记: {s['removed_oversized']}")
        print(f"  保留率: {s['output_count']/max(s['input_count'],1)*100:.1f}%")


# ========== 使用示例 ==========
if __name__ == "__main__":
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    
    # 加载
    loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    
    # 清洗
    cleaner = DocumentCleaner(
        min_doc_length=30,
        remove_copyright=True,
        dedup_exact=True,
    )
    clean_docs = cleaner.clean(raw_docs)
    
    # 清洗后的文档继续走切分 → 向量化流程
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # chunks = splitter.split_documents(clean_docs)
```

#### 2.3.8 清洗质量检查

清洗完后，需要验证清洗效果，确保没有"误杀"重要信息。

```python
def inspect_cleaning_quality(raw_docs: List[Document], 
                              clean_docs: List[Document],
                              sample_size: int = 5):
    """
    抽查清洗前后的对比，人工检查质量
    """
    import random
    
    print("=" * 60)
    print("清洗质量抽查")
    print("=" * 60)
    
    # 找出被删除的文档
    raw_texts = {d.page_content[:100]: d for d in raw_docs}
    clean_texts = {d.page_content[:100]: d for d in clean_docs}
    
    removed = [d for h, d in raw_texts.items() if h not in clean_texts]
    print(f"\n被移除的文档数: {len(removed)}")
    
    # 抽查保留的文档
    samples = random.sample(clean_docs, min(sample_size, len(clean_docs)))
    
    for i, doc in enumerate(samples, 1):
        print(f"\n--- 样本 {i} ---")
        print(f"来源: {doc.metadata.get('source', 'N/A')}")
        print(f"长度: {len(doc.page_content)} 字符")
        print(f"内容预览:\n{doc.page_content[:300]}...")
    
    # 检查指标
    avg_raw_len = sum(len(d.page_content) for d in raw_docs) / len(raw_docs)
    avg_clean_len = sum(len(d.page_content) for d in clean_docs) / len(clean_docs)
    
    print(f"\n[统计指标]")
    print(f"平均长度变化: {avg_raw_len:.0f} -> {avg_clean_len:.0f} ({avg_clean_len/avg_raw_len*100:.1f}%)")
    print(f"文档数变化: {len(raw_docs)} -> {len(clean_docs)}")


# 自动化质量检测
def auto_quality_check(docs: List[Document]) -> dict:
    """自动化检测清洗后文档的潜在问题"""
    issues = {
        'empty_content': 0,
        'too_short': 0,
        'encoding_issues': 0,
        'excessive_newlines': 0,
        'suspicious_patterns': 0,
    }
    
    for d in docs:
        text = d.page_content
        
        # 空内容
        if not text.strip():
            issues['empty_content'] += 1
        
        # 太短
        if len(text) < 20:
            issues['too_short'] += 1
        
        # 编码问题（还有乱码残留）
        if '�' in text or '\ufffd' in text:
            issues['encoding_issues'] += 1
        
        # 过多换行（可能格式被破坏）
        if text.count('\n') > len(text) / 5:
            issues['excessive_newlines'] += 1
        
        # 可疑模式（URL/邮箱没被清除）
        if re.search(r'https?://', text) or re.search(r'\S+@\S+\.\S+', text):
            issues['suspicious_patterns'] += 1
    
    return issues
```

**清洗在企业 Pipeline 中的位置**：

```
收集文档
   │
   ▼
加载文档（Loader）
   │
   ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  数据清洗    │────►│  清洗质量检查   │────►│  不通过 → 人工复核  │
│ （去脏去重） │     │ （自动+抽查）   │     │  通过 → 继续        │
└──────────────┘     └─────────────────┘     └──────────────┘
   │
   ▼
文档切分
   │
   ▼
向量化
```

> **企业最佳实践**：
> 1. 清洗规则要先在小批量数据上测试，确认不误杀后再全量跑
> 2. 保留原始数据副本，清洗出问题可以回溯
> 3. 清洗规则要版本化管理（规则变了可以重新清洗）
> 4. 定期抽查清洗效果（特别是新增数据源时）

---

### 2.4 文档切分

#### 2.4.1 为什么要切分？

直接把整个 PDF 丢给大模型？不行，原因有三：

1. **上下文长度限制**：大模型一次能处理的 Token 有限（4K~128K），一本 100 页的手册塞不进去
2. **检索精度问题**：整本手册变成一个向量，用户问"年假"，向量可能和"薪资"更相似，导致检索不准
3. **信息密度问题**：无关内容会淹没关键信息（"噪声"太大）

**切分的核心目标**：让每个 chunk（块）都包含一个**独立、完整、语义内聚**的信息单元。

#### 2.4.2 常见切分策略详解

LangChain 提供了多种 **Text Splitter**，每种适合不同场景：

##### ① 固定大小切分（CharacterTextSplitter）

最简单的切法：每隔 N 个字符切一刀， optionally 重叠 M 个字符。

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,      # 每个块 500 字符
    chunk_overlap=50,    # 相邻块重叠 50 字符
    separator="\n",      # 优先在换行处切
)

chunks = splitter.split_documents(docs)
```

**优点**：简单、快、可预测
**缺点**：可能把一句话切两半（"年假政策"切成了"年"和"假政策"），破坏语义

**适用场景**：对切分质量要求不高的快速原型、格式混乱的纯文本

---

##### ② 递归字符切分（RecursiveCharacterTextSplitter）⭐ 最常用

**LangChain 默认推荐**。它不是一刀切，而是按优先级尝试多种分隔符：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 目标块大小
    chunk_overlap=100,     # 重叠大小
    separators=[           # 切分优先级（从高到低）
        "\n\n",   # 优先在段落之间切
        "\n",     # 其次在换行处切
        "。",      # 再次在句号处切
        " ",      # 最后在空中切
        "",       # 实在不行硬切
    ]
)

chunks = splitter.split_documents(docs)
```

**递归逻辑**：

```
目标：切成 ~500 字符的块

第1轮：尝试用 "\n\n" 切
  └─ 如果切出来的块 > 500，进入第2轮
     
第2轮：用 "\n" 切大块
  └─ 如果还是 > 500，进入第3轮
     
第3轮：用 "。" 切
  └─ 如果还是 > 500，进入第4轮
     
第4轮：用 " " 切
  └─ 如果还是 > 500，硬切（第5轮）
```

**优点**：尽量保持段落/句子完整，语义损失最小
**缺点**：对代码、表格等结构化内容不够智能

**适用场景**：**通用文档首选**（规章制度、产品手册、技术文档）

---

##### ③ 语义切分（SemanticChunker）

不是按字符切，而是按**语义边界**切。原理：

1. 先按句子切
2. 把相邻句子分别向量化
3. 计算相邻句子的**语义相似度**
4. 如果相似度骤降（说明话题变了），就在这儿切开

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),              # 需要 embedding 模型
    breakpoint_threshold_type="percentile",  # 阈值类型
    breakpoint_threshold_amount=95,  # 95 分位数阈值
)

chunks = splitter.split_documents(docs)
```

**优点**：切分边界天然是语义边界，检索精度最高
**缺点**：慢（每个句子都要 embedding）、成本高（调用 embedding API）

**适用场景**：对检索质量要求极高的场景（法律合同、医学文献）

---

##### ④ 父子文档检索（ParentDocumentRetriever）⭐ 企业高级用法

**核心思想**：

- **子文档（Child）**：小块，用于检索（精度高）
- **父文档（Parent）**：大块，用于生成（上下文完整）

```
原始文档:
"第一章 总则... 第二章 年假政策... 第三章 薪资福利..."
         │
         ▼ 切成大块（父文档，比如 2000 字符）
父块1: "第一章 总则... 第二章 年假政策..."
父块2: "第三章 薪资福利..."
         │
         ▼ 再切成小块（子文档，比如 200 字符）
子块1: "第一章 总则"
子块2: "第二章 年假政策"
子块3: "第三章 薪资福利"

检索时：用子块找到"年假政策" → 返回对应的父块给大模型
```

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 子文档的切分器（小块，用于检索）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# 父文档的切分器（大块，用于生成）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# 向量库（存子文档的向量）
vectorstore = PGVector(
    collection_name="parent_docs",
    embedding_function=OpenAIEmbeddings(),
)

# 文档存储（存父文档的原文）
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,      # 子文档向量库
    docstore=store,               # 父文档存储
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加文档
retriever.add_documents(docs)

# 检索时：先用子块找相似，再返回对应父块
results = retriever.get_relevant_documents("年假有多少天？")
# 返回的是父块（2000 字符），信息更完整
```

**优点**：兼顾检索精度和生成质量，企业级方案
**缺点**：实现复杂、存储翻倍（子+父都要存）

**适用场景**：长文档问答（如整本手册、长篇报告）

---

##### ⑤ 其他切分策略

| 切分器 | 特点 | 适用场景 |
|--------|------|----------|
| `MarkdownHeaderTextSplitter` | 按 Markdown 标题层级切 | Markdown 技术文档 |
| `HTMLHeaderTextSplitter` | 按 HTML 标签（h1/h2/h3）切 | 网页内容 |
| `TokenTextSplitter` | 按 Token 数切（不是字符） | 对 Token 敏感的场景 |
| `NLTKTextSplitter` | 用 NLP 句子边界检测切 | 英文文档 |
| `SpacyTextSplitter` | 用 Spacy 语言学模型切 | 多语言文档 |

#### 2.4.3 切分参数怎么调？（企业实战建议）

**`chunk_size`（块大小）**：

| 场景 | 推荐 chunk_size | 理由 |
|------|----------------|------|
| FAQ / 短问答 | 200-500 | 每个问题答案独立 |
| 产品手册 / 技术文档 | 500-1000 | 需要一定上下文 |
| 法律合同 / 论文 | 1000-2000 | 逻辑链条长，需要大上下文 |
| 代码文档 | 300-600 | 函数级别 |

**`chunk_overlap`（重叠大小）**：

```
块1: [AAAAAAAAAAAAAA]
          ↑↑↑↑↑↑↑↑
          overlap
          ↓↓↓↓↓↓↓↓
块2:      [BBBBBBBBBBBBBB]
```

- **推荐值**：chunk_size 的 **10%~20%**
- **作用**：防止关键信息被切分线斩断（"年假政策"的"年假"在块1、"政策"在块2）
- **overlap 太大**：重复存储，浪费空间，检索结果冗余
- **overlap 太小**：边界信息丢失

**企业调参经验公式**：

```python
chunk_size = 你的大模型上下文窗口的 10%~20%
# 比如模型支持 8K context，chunk_size 取 800~1600

chunk_overlap = chunk_size * 0.15  # 15% 重叠
```

#### 2.4.4 不同切分方式对检索质量的影响

假设原文：

> "公司员工年假政策：入职满1年享受5天年假。满10年享受10天。年假需提前申请。"

| 切分方式 | 切分结果 | 用户问"年假"时检索结果 | 问题 |
|----------|----------|----------------------|------|
| **硬切 20 字符** | `"公司员工年假政策：入职"` / `"满1年享受5天年假。"` / `"满10年享受10天。"` / `"年假需提前申请。"` | 可能只召回 `"满1年享受5天年假。"`，丢失"满10年10天" | 信息碎片化 |
| **递归字符切分** | `"公司员工年假政策：入职满1年享受5天年假。"` / `"满10年享受10天。年假需提前申请。"` | 召回第一条，包含"5天"但丢失"10天" | 同一段信息被切开 |
| **语义切分** | `"公司员工年假政策：入职满1年享受5天年假。满10年享受10天。"` / `"年假需提前申请。"` | 召回第一条，完整包含 5 天和 10 天 | ✅ 最佳 |
| **父子文档** | 子块：`"年假政策"` → 父块：`"公司员工年假政策：入职满1年享受5天年假。满10年享受10天。年假需提前申请。"` | 子块召回 → 返回完整父块 | ✅ 最佳（兼顾精度+完整度） |

**结论**：

- 快速验证 → 递归字符切分
- 生产环境 → 父子文档检索 或 语义切分
- 结构化文档（Markdown/HTML）→ 按标题层级切分

### 2.5 向量化

#### 2.5.1 什么是向量化？

向量化 = **把文字变成数字向量**，让计算机能"计算"文字之间的关系。

```
"年假政策"  ──►  Embedding Model  ──►  [0.12, -0.05, 0.88, ..., 0.33]
                                            ↑
                                       一个 768 维或 1536 维的向量
                                       （一串浮点数）
```

**核心特性**：语义相近的文本，向量距离也近。

```
"年假政策"  ──►  [0.12, -0.05, 0.88, ...]
"带薪休假"  ──►  [0.15, -0.03, 0.85, ...]   ← 距离很近 ✅
"薪资福利"  ──►  [0.10,  0.20, 0.30, ...]   ← 距离较远
```

这样当用户问"怎么请假"时，系统就能找到向量空间里离这个问题最近的文档片段。

#### 2.5.2 如何选择 Embedding 模型？

| 模型 | 维度 | 语言 | 特点 | 适用场景 |
|------|------|------|------|----------|
| **text-embedding-3-large** (OpenAI) | 3072 | 多语言 | 效果顶级，贵 | 预算充足、质量优先 |
| **text-embedding-3-small** (OpenAI) | 1536 | 多语言 | 性价比高 | 大多数企业的首选 |
| **text-embedding-ada-002** (OpenAI) | 1536 | 多语言 | 老版本，已不推荐 | 历史项目维护 |
| **bge-large-zh-v1.5** (智源) | 1024 | 中文强 | 开源免费，中文 SOTA | 中文文档、成本敏感 |
| **bge-m3** (智源) | 1024 | 多语言 | 支持 100+ 语言 | 多语言混合文档 |
| **m3e-base** (Moka) | 768 | 中文 | 轻量、推理快 | 高并发场景 |
| **e5-large-v2** (Microsoft) | 1024 | 英文 | 英文效果极好 | 英文为主的企业 |
| **Jina-Embeddings-v2** | 768 | 多语言 | 支持 8K 长文本 | 长文档切分后向量化 |

**企业选型建议**：

1. **中文文档为主 + 成本敏感** → `bge-large-zh-v1.5`（本地部署）
2. **多语言 + 追求效果** → `text-embedding-3-large`（OpenAI API）
3. **高并发 + 低延迟** → `m3e-base` 或 `bge-small`
4. **有 GPU 资源** → 本地部署开源模型，长期成本最低
5. **没 GPU / 快速启动** → OpenAI / 智谱 API

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# 方式1：OpenAI API（简单，贵）
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 方式2：本地开源模型（免费，需要算力）
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cuda"},  # 或 "cpu"
    encode_kwargs={"normalize_embeddings": True}  # bge 需要归一化
)
```

#### 2.5.3 相似度计算

拿到两个向量后，怎么算它们有多"像"？

| 相似度算法 | 公式直觉 | 特点 | 适用场景 |
|-----------|---------|------|----------|
| **余弦相似度（Cosine）** | 两个向量夹角越小越相似 | 不受向量长度影响，最常用 | **通用首选** |
| **欧氏距离（Euclidean）** | 向量终点之间的直线距离 | 受向量长度影响 | 向量已归一化时等同于余弦 |
| **点积（Dot Product）** | 简单相乘求和 | 计算最快，需归一化向量 | 大规模检索、性能优先 |

```python
import numpy as np

vec_a = np.array([0.1, 0.2, 0.3])
vec_b = np.array([0.15, 0.25, 0.35])

# 余弦相似度（范围 -1~1，越接近 1 越相似）
cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

# 欧氏距离（越小越相似）
euclidean_dist = np.linalg.norm(vec_a - vec_b)
```

**企业建议**：无脑用 **余弦相似度**，99% 的向量数据库都默认支持。

#### 2.5.4 ANN 近似最近邻算法

如果知识库有 **100 万条**文档片段，用户提问时，要把问题向量跟 100 万个文档向量逐一比较吗？

** brute force（暴力扫描）**：100 万次计算，太慢！

**ANN（Approximate Nearest Neighbor，近似最近邻）**：用空间换时间，先建索引，检索时只查一小部分，**速度快 100~1000 倍，精度损失很小**。

| ANN 算法 | 原理 | 特点 | 代表数据库 |
|----------|------|------|-----------|
| **HNSW** (Hierarchical Navigable Small World) | 构建多层图结构，上层粗筛、下层精筛 | 速度极快、精度高、内存占用大 | Milvus、Weaviate、pgvector |
| **IVF** (Inverted File Index) | 把空间分成 N 个区域，先找区域再找点 | 平衡速度和精度，可调参数多 | Milvus、Faiss |
| **Flat** | 暴力扫描，不建索引 | 最准、最慢 | 小规模数据 (<10K) |
| **PQ** (Product Quantization) | 向量压缩，减少存储和计算量 | 省内存、略有精度损失 | Faiss、Milvus |

**企业建议**：

- 数据量 < 10 万 → `Flat` 或 `HNSW`（图省事直接用 HNSW）
- 数据量 10 万~1000 万 → `HNSW`（速度和精度的最佳平衡）
- 数据量 > 1000 万 → `IVF_PQ`（IVF + PQ 组合，省内存）
- 对精度要求极高 → `Flat`（慢但最准）

```python
# 以 pgvector 为例，HNSW 索引通过 SQL 创建
from langchain_community.vectorstores import PGVector

vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embedding,
    connection_string="postgresql+psycopg://user:password@localhost:5432/rag_db",
    collection_name="company_knowledge",
)

# pgvector 创建 HNSW 索引（在 PostgreSQL 中执行）
# CREATE INDEX ON company_knowledge USING hnsw (embedding vector_cosine_ops);

```

#### 2.5.5 批量向量化（Batch Embedding）

##### 为什么需要批处理？

在离线模式下，一个知识库可能有数千甚至数万个 chunk。如果不使用批处理，逐条向量化会带来三个严重问题：

| 问题 | 不批处理的后果 | 示例 |
|------|---------------|------|
| **网络延迟累积** | 每个 chunk 发一次 HTTP 请求，RTT 累加非常可怕 | 1000 个 chunk × 200ms = **200 秒**；批处理后 10 次请求 ≈ **2 秒** |
| **触发 API 限流** | 大多数 Embedding API（OpenAI、DashScope、智谱）有 QPS 限制 | 逐条调用大概率收到 `429 Too Many Requests` |
| **GPU 利用率低** | 本地部署的开源模型（如 BGE）一次处理一条，无法发挥 GPU 并行计算优势 | A100 处理 1 条和处理 128 条，耗时可能只差 20% |

> **一句话总结**：批处理 = 减少网络往返 + 规避限流 + 提升 GPU 利用率 = **更快、更稳定、更省钱**。

##### 批处理的实现方式

**方式1：LangChain 自动批处理（推荐，最简单）**

LangChain 的 `embed_documents()` 方法内部通常已经做了批处理。当你调用 `PGVector.from_documents()` 或 `vectorstore.add_documents()` 时，LangChain 会自动把所有 texts 传给 `embed_documents()`，而具体的 Embedding 类（如 `OpenAIEmbeddings`、`DashScopeEmbeddings`、`HuggingFaceEmbeddings`）会在内部按模型/服务商要求分批发送。

```python
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

# LangChain 内部自动批处理
vectorstore = PGVector.from_documents(
    documents=chunks,           # 可能有 5000 个 chunk
    embedding=OpenAIEmbeddings(),
)
```

**方式2：手动控制批次（大规模数据、需要进度监控）**

当 chunk 数量极大（如 >10000），或需要进度条、异常隔离时，建议手动分批：

```python
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

embedding = OpenAIEmbeddings()
vectorstore = PGVector(
    collection_name="company_knowledge",
    embedding_function=embedding,
    connection_string="postgresql+psycopg://user:password@localhost:5432/rag_db",
)

batch_size = 100  # 每批 100 条，根据 API 限制调整

for i in tqdm(range(0, len(chunks), batch_size), desc="向量化中"):
    batch = chunks[i : i + batch_size]
    try:
        vectorstore.add_documents(batch)
    except Exception as e:
        print(f"批次 {i//batch_size + 1} 失败: {e}")
        # 可以选择：跳过、重试、或记录到失败队列
```

##### 不同场景的 batch_size 建议

| 场景 | 推荐 batch_size | 理由 |
|------|----------------|------|
| OpenAI API | 100~500 | 官方支持一次最多 2048 条，但建议 100~500 避免超时 |
| DashScope（通义） | 100~200 | 服务端有最大文本长度限制，太大容易报错 |
| 本地 BGE（GPU） | 32~128 | 取决于 GPU 显存，A100 可以到 128，T4 建议 32 |
| 本地 BGE（CPU） | 8~16 | CPU 并行度有限，太大反而慢 |
| HuggingFace Inference API | 32~64 | 免费 tier 有速率限制 |

##### 异常隔离：为什么很重要？

如果不分批，5000 个 chunk 一次性传给 API，中间任何一条出错（如包含非法字符、长度超限），整个请求都会失败，前功尽弃。

**分批后的优势**：
- 某一批失败，其他批次已成功入库
- 可以单独重试失败批次
- 可以定位到具体是哪条文本有问题

```python
# 带异常隔离和重试的批处理
import time

def batch_embed_with_retry(chunks, vectorstore, batch_size=100, max_retries=3):
    failed_batches = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        for attempt in range(max_retries):
            try:
                vectorstore.add_documents(batch)
                print(f"✅ 批次 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} 成功")
                break
            except Exception as e:
                print(f"⚠️ 批次 {i//batch_size + 1} 第 {attempt+1} 次重试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    failed_batches.append((i, batch, str(e)))
    
    if failed_batches:
        print(f"\n❌ 最终失败批次: {len(failed_batches)} 个")
        # 保存失败批次，后续人工排查
        import json
        with open("failed_batches.json", "w", encoding="utf-8") as f:
            json.dump([{"index": i, "error": err} for i, _, err in failed_batches], f, ensure_ascii=False)
    
    return failed_batches
```

> **企业最佳实践**：
> 1. 小批量场景（<1000 chunk）：直接用 LangChain 的自动批处理，代码最简洁
> 2. 中大规模场景（>10000 chunk）：手动分批 + 进度条 + 异常隔离，确保可控
> 3. 批大小先从 100 开始测试，根据实际延迟和错误率逐步调大
> 4. 永远保留失败批次的记录，方便回溯和修复

### 2.6 向量入库

#### 2.6.1 常见向量数据库对比

| 数据库 | 特点 | 部署方式 | 适用场景 |
|--------|------|----------|----------|
| **pgvector** | PostgreSQL 插件，关系+向量统一，成熟稳定 | PostgreSQL | **企业首选**，已有 PG 基础设施或从零开始都适用 |
| **Milvus/Zilliz** | 功能最全、性能最强、云原生 | 分布式/云托管 | **超大规模企业级** |
| **Qdrant** | Rust 写的高性能、过滤查询强 | 容器/云 | 高并发、需要复杂过滤条件 |
| **Pinecone** | 全托管、零运维 | SaaS | 不想运维基础设施的团队 |
| **Weaviate** | 模块化、GraphQL 接口、内置向量化 | 容器/云 | 需要复杂查询的企业 |
| **Elasticsearch** | 全文搜索+向量搜索混合 | 分布式 | 已有 ES 集群的团队 |
| **Chroma** | 轻量、易用、Python 原生 | 本地/嵌入式 | 快速原型、个人项目 |
| **Faiss** | Meta 开源、纯 ANN 库 | 嵌入式 | 需要自己封装上层业务 |

**企业选型建议**：

- **绝大多数企业场景** → **pgvector**（与 PostgreSQL 一体，关系+向量统一，生态成熟，运维简单）
- **已有 PostgreSQL** → pgvector（不用新学数据库，直接装插件）
- **超大规模（>1000万文档）** → Milvus 或 Qdrant
- **不想运维** → Pinecone / Zilliz Cloud（全托管）
- **需要关键词+向量混合检索** → pgvector（搭配 PostgreSQL 全文检索）或 Elasticsearch

#### 2.6.2 LangGraph 中如何入库

```python
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载
loader = DirectoryLoader("./docs", glob="**/*.pdf")
docs = loader.load()

# 2. 清洗（新增！）
from cleaning_pipeline import DocumentCleaner
cleaner = DocumentCleaner(min_doc_length=30, dedup_exact=True)
clean_docs = cleaner.clean(docs)

# 3. 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(clean_docs)

# 4. 向量化 + 入库（一步完成）
vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    connection_string="postgresql+psycopg://user:password@localhost:5432/rag_db",  # PostgreSQL 连接串
    collection_name="company_knowledge",
)

# 5. 后续使用时加载已有库
vectorstore = PGVector(
    connection_string="postgresql+psycopg://user:password@localhost:5432/rag_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="company_knowledge",
)
```


---

## 三、在线模式：查询 Pipeline

离线阶段建好了知识库，在线阶段就是回答用户问题的"流水线"：

```
用户提问 ──► Query 重写 ──► 检索 ──► 重排序 ──► Prompt 拼接 ──► 大模型生成
```

### 3.1 用户提出问题

这是 Pipeline 的起点。用户的提问可能有各种问题：

| 用户提问问题 | 例子 | 影响 |
|-------------|------|------|
| **表述模糊** | "那个政策" | 不知道"那个"指什么 |
| **口语化** | "咋请假啊" | 和文档正式表述差异大 |
| **多意图** | "年假怎么请？加班费怎么算？" | 需要拆成两个问题 |
| **缺少上下文** | "第三条说什么" | 不知道"第三条"属于哪个文件 |
| **外语提问** | "How many annual leave days?" | 和中文文档不匹配 |

这些问题会直接影响检索质量。后面讲的 **Query 重写** 就是为了解决它们。

### 3.2 Query 重写

#### 3.2.1 什么是 Query 重写？

Query 重写 = **把用户的原始问题，改写成更适合检索的"标准问题"**。

```
用户原话: "咋请假啊"
          │
          ▼ 重写
重写后: "公司员工如何申请年假？请假流程是什么？"
          │
          ▼ 检索
召回: "年假申请流程：需提前3天在OA系统提交..."
```

#### 3.2.2 为什么要重写？

| 原因 | 解释 | 例子 |
|------|------|------|
| **语义鸿沟** | 用户口语 ≠ 文档书面语 | "咋请假" vs "请假申请流程" |
| **指代消解** | 多轮对话中的"它/那个"需要补全 | "那个政策是什么"→"年假政策是什么" |
| **意图扩展** | 把一个问题扩展成多个相关表述 | "年假"→"年假、带薪休假、annual leave" |
| **去噪** | 去掉无关寒暄，提取核心问题 | "你好，我想问一下，那个..."→"年假天数" |
| **多语言对齐** | 中英文提问都能检索中文文档 | "annual leave"→"年假" |

#### 3.2.3 常见重写策略与代码

##### ① 指代消解 + 上下文补全（多轮对话场景）

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 维护对话历史
chat_history = []

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个查询重写专家。请根据对话历史，把用户的最新问题改写成
一个自包含、明确的问题（不需要上下文就能理解）。

要求：
1. 把代词（它/那个/这个）替换为具体实体
2. 补全省略的主语和宾语
3. 保持问题原意
4. 只输出改写后的问题，不要解释"""),
    ("human", "对话历史：\n{history}\n\n用户最新问题：{question}\n\n改写后的问题：")
])

llm = ChatOpenAI(model="gpt-4o-mini")
rewriter = rewrite_prompt | llm

# 示例
history = """用户: 公司年假有多少天？
AI: 入职满1年享受5天年假。
用户: 那怎么申请呢？"""  # "那"指的是"年假"

result = rewriter.invoke({
    "history": history,
    "question": "那怎么申请呢？"
})
print(result.content)  # 输出: "如何申请公司年假？"
```

##### ② HyDE（Hypothetical Document Embedding）假设文档嵌入

**核心思想**：让大模型先"编"一个理想的答案，然后用这个答案去向量库检索，而不是直接用用户的问题检索。

```
用户提问: "公司年假政策"
          │
          ▼ 让 LLM 生成一个假设答案
假设答案: "根据公司员工手册，入职满1年的员工可享受5天带薪年假..."
          │
          ▼ 把假设答案向量化后检索
召回结果: [真实文档片段1] [真实文档片段2]...
```

为什么有效？因为**用户问题很短，包含的信息量低**；而假设答案包含了问题的意图和可能的答案关键词，向量匹配更准。

```python
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "请根据用户问题，写一段简短但信息丰富的回答（假设你能访问所有公司文档）。"),
    ("human", "问题：{question}\n\n假设答案：")
])

hyde_chain = hyde_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 生成假设答案
hypothetical_doc = hyde_chain.invoke({"question": "公司年假怎么申请？"}).content

# 用假设答案检索（而不是用原始问题）
results = vectorstore.similarity_search(hypothetical_doc, k=5)
```

**优点**：检索精度显著提升，尤其是用户问题很短时
**缺点**：多一次 LLM 调用，增加成本和延迟

##### ③ 查询扩展（Query Expansion）

把一个问题扩展成多个同义/相关查询，分别检索后合并结果。

```python
expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", """请为用户问题生成 3 个不同表述的等价问题，帮助更全面检索。
用换行分隔，不要编号。"""),
    ("human", "问题：{question}")
])

expansion_chain = expansion_prompt | ChatOpenAI(model="gpt-4o-mini")

# 示例
question = "年假政策"
expanded = expansion_chain.invoke({"question": question}).content
# 输出可能：
# 公司员工年假规定是什么？
# 带薪休假天数和申请流程？
# annual leave policy 怎么规定的？

queries = [question] + expanded.strip().split("\n")

# 每个查询分别检索，合并去重
all_results = []
for q in queries:
    all_results.extend(vectorstore.similarity_search(q, k=3))
# 去重后取 Top-K
```

##### ④ Step-back Prompting（退一步提问）

把具体问题抽象成更通用的问题，先检索通用知识，再检索具体知识。

```
具体问题: "GPT-4 在医疗诊断中的准确率是多少？"
          │
          ▼ 退一步
抽象问题: "大语言模型在医疗领域的应用和局限性"
          │
          ▼ 分别检索
召回: [通用医疗AI综述] + [GPT-4具体评测论文]
```

```python
stepback_prompt = ChatPromptTemplate.from_messages([
    ("system", """请把用户问题改写成更通用、更抽象的问题。
这样可以帮助检索到更多背景知识。"""),
    ("human", "问题：{question}\n\n更通用的问题：")
])

stepback_chain = stepback_prompt | ChatOpenAI(model="gpt-4o-mini")

# 原始问题 + 抽象问题，分别检索
```

##### ⑤ 重写策略对比

| 重写策略 | 适用场景 | 成本 | 延迟 | 效果提升 |
|----------|----------|------|------|----------|
| **指代消解** | 多轮对话 | 低 | 低 | 🟡 中等 |
| **HyDE** | 问题短、意图模糊 | 中 | 中 | 🟢 显著 |
| **查询扩展** | 专业术语多义词 | 中 | 中 | 🟢 显著 |
| **Step-back** | 需要背景知识的复杂问题 | 中 | 中 | 🟡 中等 |

**企业建议**：

- 快速启动 → 只做指代消解
- 追求质量 → HyDE + 查询扩展组合
- 成本敏感 → 用轻量模型（`gpt-4o-mini` / `qwen-turbo`）做重写

### 3.3 检索

检索是 RAG 的**核心引擎**。常见策略有三种：关键词检索、语义检索、混合检索。

#### 3.3.1 关键词检索：BM25

BM25 是搜索引擎的经典算法（Elasticsearch 默认使用）。它基于**词频**和**逆文档频率**计算相关性。

```
BM25 的核心思想：
- 词出现次数越多，越相关（但不是线性增长，有饱和）
- 罕见词（如"年假"）比常见词（如"公司"）权重更高
- 文档越短，词的权重越高
```

**优点**：
- 精确匹配专有名词、产品名、ID（如"iPhone 15"、"员工编号 HR001"）
- 速度快，无需 GPU
- 对英文分词成熟

**缺点**：
- 无法理解语义（"请假"和"休假"被认为是不同的词）
- 对中文需要好的分词器
- 同义词、口语化表达检索不到

```python
from langchain_community.retrievers import BM25Retriever

# 基于已有 chunks 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(chunks, k=5)

# 检索
results = bm25_retriever.get_relevant_documents("年假怎么申请")
```

#### 3.3.2 语义相似度检索

基于 embedding 向量空间的距离检索，前面已经讲过。

```python
# 向量检索（默认余弦相似度）
results = vectorstore.similarity_search(
    query="年假怎么申请",
    k=5,                    # 返回 Top 5
    filter={"department": "人力资源部"}  # 可选：按元数据过滤
)
```

**优点**：
- 理解语义（"请假"和"休假"能匹配）
- 支持多语言（中文提问匹配英文文档）
- 对口语化、模糊表述友好

**缺点**：
- 精确匹配不如 BM25（比如产品型号"iPhone 15 Pro Max"可能匹配到"iPhone 15"）
- 需要 embedding 推理，有计算成本
- 对罕见词、专有名词可能不准

#### 3.3.3 混合检索（Hybrid Search）⭐ 企业推荐

**核心思想**：BM25 + 向量检索 各取所长，结果融合。

```
用户提问: "公司年假申请流程"
          │
          ├──► BM25 检索 ──► [结果A: 年假申请表.doc] [结果B: ...]
          │
          ├──► 向量检索 ──► [结果X: 年假政策第3章] [结果Y: ...]
          │
          ▼ 融合排序
最终: [结果A] [结果X] [结果B] [结果Y] ...
```

**融合方法**：

| 融合方法 | 原理 | 适用场景 |
|----------|------|----------|
| **RRF（Reciprocal Rank Fusion）** | 按排名加权求和，不用原始分数 | 最常用，BM25 和向量分数不可比时 |
| **线性加权** | `score = w1 * bm25_score + w2 * vector_score` | 分数可标准化时 |

```python
from langchain.retrievers import EnsembleRetriever

# 创建两个检索器
bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 组合（RRF 融合）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],  # 两个检索器的权重
)

# 检索
results = ensemble_retriever.get_relevant_documents("年假怎么申请")
```

**企业建议**：

- 绝大多数场景 → **混合检索**（RRF 融合）
- 纯技术文档（大量代码、ID）→ 偏重 BM25（权重 0.7 : 0.3）
- 纯规章制度（自然语言为主）→ 偏重向量（权重 0.3 : 0.7）

#### 3.3.4 ANN 算法在检索中的应用

在实际检索时，向量数据库内部的 ANN 索引决定了检索速度和精度：

```python
# 检索时影响 ANN 性能的参数
results = vectorstore.similarity_search_with_score(
    query="年假政策",
    k=10,
    # 以下参数取决于数据库实现
    # HNSW: ef_search（越大越准越慢）
    # IVF: nprobe（查询的区域数）
)
```

| 参数 | 作用 | 调大效果 | 调小效果 |
|------|------|----------|----------|
| **ef_search** (HNSW) | 搜索时访问的节点数 | 更准、更慢 | 更快、可能漏召回 |
| **nprobe** (IVF) | 查询的聚类中心数 | 更准、更慢 | 更快、可能漏召回 |
| **k** (返回数量) | 返回 Top-K 结果 | 更多候选、更全 | 更快、可能漏召回 |

**企业调参经验**：

- `ef_search = 64~128`（默认 64，精度要求高用 128）
- `k = 10~20`（给重排序留足够候选）
- 数据量 < 10 万 → 直接用 HNSW Flat（不用压缩）

### 3.4 重排序（Rerank）

#### 3.4.1 为什么要重排序？

检索器（BM25 或向量检索）返回的 Top-K 结果，**排序质量往往不够好**：

```
向量检索 Top-5：
1. [文档A] 公司成立于2010年...              ← 和"年假"关系不大，但向量碰巧近
2. [文档B] 年假政策：入职满1年享受5天...      ← ✅ 真正相关的排第2
3. [文档C] 员工福利包括医疗保险...             ← 弱相关
4. [文档D] 年假申请需提前3天...               ← ✅ 也相关，排第4
5. [文档E] 公司地址在北京市...                ← 完全不相关

问题：向量检索只算"语义相似"，不理解"问题-答案"的匹配关系
```

**重排序（Rerank）** 用专门的模型，计算"**问题与文档的相关性**"，把真正相关的文档排到前面。

```
原始检索 Top-5 ──► Rerank 模型打分 ──► 重新排序 Top-5
  [A, B, C, D, E]      B: 0.95        [B, D, C, A, E]
                       D: 0.88
                       C: 0.45
                       A: 0.12
                       E: 0.01
```

#### 3.4.2 重排序模型与方法

| 重排序模型 | 特点 | 精度 | 速度 | 成本 |
|-----------|------|------|------|------|
| **Cohere Rerank** | API 服务，效果顶级 | 🟢 极高 | 快 | 收费 |
| **bge-reranker-v2-m3** | 开源，中文好 | 🟢 高 | 中等 | 免费（需 GPU） |
| **bge-reranker-large** | 开源，效果好 | 🟢 高 | 较慢 | 免费（需 GPU） |
| **Cross-Encoder** (通用) | 本地部署，问题+文档一起编码 | 🟡 中高 | 较慢 | 免费 |
| **LLM-as-Reranker** | 让大模型打分 | 🟡 中高 | 很慢 | 贵 |

```python
# 方式1：Cohere Rerank API（简单，效果顶级）
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})  # 先召回20个
)

# 检索：先召回20个 → 重排序取Top 5
results = compression_retriever.get_relevant_documents("年假怎么申请")


# 方式2：开源 bge-reranker（本地，免费）
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=5)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})
)

results = compression_retriever.get_relevant_documents("年假怎么申请")
```

**重排序的经典 Pipeline**：

```
召回阶段（Recall）:    从 100 万文档中快速捞出 ~100 个候选
                      │ 用 ANN（速度快，允许一定误差）
                      ▼
粗排阶段（Ranking）:   从 ~100 个候选中选出 ~20 个
                      │ 用 BM25 + 向量混合检索
                      ▼
精排阶段（Rerank）:    从 ~20 个候选中选出 Top 5
                      │ 用 Rerank 模型（慢但准）
                      ▼
生成阶段（Generate）:   把 Top 5 文档塞给大模型生成答案
```

**企业建议**：

- 数据量 < 10 万 → 检索 20 个 + Rerank 取 Top 5
- 数据量 > 100 万 → 需要多级索引（粗排 + 精排 + Rerank）
- 成本敏感 → 用开源 `bge-reranker-v2-m3` 本地部署
- 效果优先 → Cohere Rerank API

### 3.5 Prompt 拼接与大模型生成

#### 3.5.1 Prompt 模板设计

检索到的文档片段需要拼成一个 Prompt 给大模型。Prompt 设计直接影响回答质量。

**基础模板（最简版）**：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个企业知识助手。请根据以下参考文档回答用户问题。
如果文档中没有相关信息，请明确说"根据现有资料，我无法回答这个问题"。
不要编造信息。

参考文档：
{context}"""),
    ("human", "{question}")
])
```

**进阶模板（带引用溯源）**：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的企业知识助手。请严格根据参考文档回答问题。

规则：
1. 只使用参考文档中的信息，不要编造
2. 如果文档中没有答案，回答"根据现有资料，我无法回答这个问题"
3. 回答中引用来源，格式：[来源: 文件名.pdf 第X页]
4. 保持回答简洁、准确

参考文档：
{context}"""),
    ("human", "{question}")
])

# context 拼接时带上来源信息
def format_docs(docs):
    return "\n\n".join([
        f"[来源: {doc.metadata.get('source', '未知')} 第{doc.metadata.get('page', '?')}页]\n{doc.page_content}"
        for doc in docs
    ])
```

**Prompt 拼接的注意事项**：

| 注意点 | 说明 | 后果 |
|--------|------|------|
| **文档顺序** | 把最相关的文档放前面 | 大模型注意力衰减，后面的文档容易被忽略 |
| **文档去重** | 去掉内容高度重复的 chunk | 浪费上下文窗口，干扰判断 |
| **长度控制** | 总长度不超过模型上下文限制 | 超过会截断，丢失信息 |
| **分隔清晰** | 不同文档用明显分隔符区分 | 防止模型混淆不同来源 |

#### 3.5.2 上下文窗口管理

大模型有**上下文长度限制**，需要合理分配：

```
总上下文 = System Prompt + 检索文档 + 用户问题 + 输出空间

以 8K 上下文模型为例：
├─ System Prompt:        500 tokens
├─ 检索文档 (Top 5):     3000 tokens  ← 可调整
├─ 用户问题 + 历史:      500 tokens
├─ 输出预留:             2000 tokens
└─ 安全余量:             1000 tokens
```

**如果检索文档太长怎么办？**

| 策略 | 方法 | 适用场景 |
|------|------|----------|
| **减少 Top-K** | 从 Top 10 减到 Top 5 | 文档都很长时 |
| **Map-Reduce** | 每个文档单独问模型，再汇总 | 复杂总结任务 |
| **Refine** | 逐个文档迭代精炼答案 | 需要深度分析 |
| **父子文档** | 检索用小 chunk，生成用父 chunk | 长文档场景 |

LangGraph 中可以方便地实现这些高级策略（见第六章）。


---

## 四、RAG 架构选型：传统 RAG vs GraphRAG vs AgenticRAG

企业落地时，首先要回答一个问题：**用哪种 RAG 架构？** 三种主流方案各有适用场景。

### 4.1 传统 RAG（Naive RAG）

**定义**：前面讲的完整 Pipeline（切分 → 向量 → 检索 → 生成），最经典的 RAG。

```
┌─────────────────────────────────────────┐
│            传统 RAG 架构                 │
│                                         │
│  文档 ──► 切分 ──► 向量库               │
│                      ▲                  │
│                      │ similarity      │
│  用户提问 ──► 检索 ──┘                  │
│                 │                       │
│                 ▼                       │
│              Prompt 拼接                │
│                 │                       │
│                 ▼                       │
│              大模型生成                  │
└─────────────────────────────────────────┘
```

**优点**：
- 实现简单，快速上线
- 技术栈成熟（LangChain/LlamaIndex 生态完善）
- 调试直观，问题定位容易

**缺点**：
- 无法处理跨文档的复杂关联（"A文件的第3条和B文件的第5条有什么关系？"）
- 对"全局总结类"问题效果差（"总结公司所有政策的变化趋势"）
- 没有推理能力，一次检索定终身

**适用场景**：
- FAQ 问答（单文档内找答案）
- 产品手册查询
- 规章制度检索
- 快速原型验证

---

### 4.2 GraphRAG

**定义**：在 RAG 的基础上，引入**知识图谱（Knowledge Graph）**，把文档中的实体和关系结构化存储。

```
传统 RAG:              GraphRAG:

文档A: "张三在北京工作"    实体: 张三 ──[工作于]──► 北京
文档B: "北京是总部"         实体: 北京 ──[是]──────► 总部
文档C: "总部有100人"        实体: 总部 ──[人数]────► 100
                          
                         用户问: "张三在哪个总部工作？"
                         图谱推理: 张三 → 北京 → 总部 ✅
```

**GraphRAG 的数据流**：

```
原始文档
   │
   ▼
实体抽取（Entity Extraction）
   │  "张三"(人) ──工作于──► "北京"(地点)
   │  "北京"(地点) ──是──► "总部"(组织)
   ▼
关系抽取（Relation Extraction）
   │
   ▼
构建知识图谱（Neo4j / NebulaGraph）
   │
   ├──► 社区检测（Community Detection）
   │      把相关实体聚成"社区"
   │      生成社区摘要
   │
   └──► 向量索引（实体和关系也向量化）

查询时：
   ├──► 全局查询（Global Search）: 读社区摘要回答宏观问题
   └──► 局部查询（Local Search）: 从实体出发遍历图谱回答具体问题
```

**微软 GraphRAG 的实现**：

```python
# 概念性代码，展示 GraphRAG 的核心思想
# 实际使用请参考微软 graphrag 库

# 1. 实体和关系抽取
extraction_prompt = """
从以下文档中提取所有实体和关系。
实体类型：人、组织、地点、政策、产品
关系类型：工作于、属于、位于、规定、包含

文档：{document}

输出 JSON 格式：
{
  "entities": [{"name": "...", "type": "..."}],
  "relationships": [{"source": "...", "target": "...", "type": "..."}]
}
"""

# 2. 社区摘要（Global Search 的关键）
community_prompt = """
以下是一组相关实体和关系，请生成一段总结性描述：
{community_data}
"""

# 3. 查询时选择搜索模式
def search(query, mode="local"):
    if mode == "local":
        # 局部搜索：找相关实体 → 遍历邻居 → 读原始文本
        entities = extract_entities_from_query(query)
        return graph_traversal(entities, depth=2)
    else:
        # 全局搜索：读社区摘要 → 综合回答
        relevant_communities = vector_search_communities(query)
        return synthesize_from_summaries(relevant_communities)
```

**GraphRAG 的优缺点**：

| 维度 | 评价 |
|------|------|
| **跨文档关联** | 🟢 极强，能回答"A和B有什么关系" |
| **全局总结** | 🟢 社区摘要机制擅长宏观问题 |
| **实现复杂度** | 🔴 高，需要图谱数据库 + 抽取pipeline |
| **构建成本** | 🔴 高，需要多次 LLM 调用抽取实体关系 |
| **维护成本** | 🔴 高，文档更新后图谱需要同步更新 |
| **响应延迟** | 🟡 中，图谱遍历比向量检索慢 |

**适用场景**：
- 需要理解实体关系的场景（组织架构、供应链、法律关系）
- 跨文档的复杂推理（"对比A政策和B政策的差异"）
- 全局性总结问题（"公司过去三年的战略变化"）
- 数据量 < 10 万文档（图谱构建成本高）

**不适用场景**：
- 纯 FAQ 问答（杀鸡用牛刀）
- 文档更新频繁（图谱维护成本高）
- 追求低延迟（图谱查询比向量检索慢）

---

### 4.3 AgenticRAG

**定义**：RAG + Agent（智能体）。不是"一次检索定终身"，而是让 AI **自主决定怎么检索、检索几次、什么时候停止**。

```
传统 RAG（一次检索）:           AgenticRAG（多轮检索+推理）:

用户: "年假和病假一共多少天？"    用户: "年假和病假一共多少天？"
  │                                │
  ▼                                ▼
检索 "年假和病假一共多少天"       Step 1: 思考 ──► 需要分别查"年假"和"病假"
  │                                │
  ▼                                ▼
召回一个 chunk                    Step 2: 检索 "年假天数"
（可能只讲了年假，没讲病假）        │
  │                                ▼
  ▼                              Step 3: 检索 "病假天数"
生成（可能漏掉病假信息）            │
                                   ▼
                                 Step 4: 计算总和
                                   │
                                   ▼
                                 生成 "年假5天 + 病假3天 = 8天"
```

**AgenticRAG 的核心能力**：

| 能力 | 说明 | 例子 |
|------|------|------|
| **查询分解** | 把复杂问题拆成子问题 | "A和B的差异" → 先查A，再查B |
| **多轮检索** | 根据中间结果决定下一步检索什么 | 查到"年假5天"后，发现还需要"病假" |
| **工具调用** | 不仅检索文档，还能调用计算器、API | 查到数字后自动相加 |
| **自我纠错** | 发现检索结果不够，主动补充检索 | "第一次检索没找到病假信息，再查一次" |
| **路由决策** | 判断问题是否需要检索 | "你好"→直接回答；"年假政策"→需要检索 |

**LangGraph 中实现 AgenticRAG**：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    """整个 Pipeline 的状态"""
    question: str           # 用户原始问题
    sub_questions: List[str] # 分解后的子问题
    documents: List[Document] # 检索到的文档
    generation: str         # 生成的答案
    steps: int              # 已执行的检索步数
    should_continue: bool   # 是否继续检索

# 节点1：路由判断（需不需要检索？）
def route_question(state: GraphState):
    # 简单问题直接回答，复杂问题走 RAG
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""判断以下问题是否需要检索文档才能回答。
只需回答 YES 或 NO。

问题：{state['question']}"""
    
    response = llm.invoke(prompt).content.strip()
    if response == "NO":
        return {"should_continue": False}
    return {"should_continue": True}

# 节点2：查询分解
def decompose_query(state: GraphState):
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""将以下问题分解为2-3个子问题，每个子问题单独检索。
用换行分隔。

问题：{state['question']}"""
    
    sub_questions = llm.invoke(prompt).content.strip().split("\n")
    return {"sub_questions": sub_questions}

# 节点3：多轮检索
def retrieve(state: GraphState):
    all_docs = []
    for q in state["sub_questions"]:
        docs = vectorstore.similarity_search(q, k=3)
        all_docs.extend(docs)
    
    # 去重
    seen = set()
    unique_docs = []
    for d in all_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)
    
    return {"documents": unique_docs, "steps": state["steps"] + 1}

# 节点4：评估检索结果是否足够
def grade_documents(state: GraphState):
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""根据以下文档，判断是否已经足够回答问题。
只需回答 YES 或 NO。

问题：{state['question']}
文档：{[d.page_content[:200] for d in state['documents']]}
"""
    
    response = llm.invoke(prompt).content.strip()
    if response == "YES" or state["steps"] >= 3:
        return {"should_continue": False}
    return {"should_continue": True}

# 节点5：生成答案
def generate(state: GraphState):
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"""根据以下参考文档回答问题：

参考文档：
{context}

问题：{state['question']}
"""
    
    response = ChatOpenAI(model="gpt-4o").invoke(prompt)
    return {"generation": response.content}

# 构建图
workflow = StateGraph(GraphState)

workflow.add_node("route", route_question)
workflow.add_node("decompose", decompose_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate)

workflow.set_entry_point("route")
workflow.add_conditional_edges(
    "route",
    lambda s: "decompose" if s["should_continue"] else "generate",
    {"decompose": "decompose", "generate": "generate"}
)
workflow.add_edge("decompose", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    lambda s: "retrieve" if s["should_continue"] else "generate",
    {"retrieve": "retrieve", "generate": "generate"}
)
workflow.add_edge("generate", END)

app = workflow.compile()

# 运行
result = app.invoke({"question": "年假和病假一共多少天？", "steps": 0})
print(result["generation"])
```

**AgenticRAG 的优缺点**：

| 维度 | 评价 |
|------|------|
| **复杂问题处理** | 🟢 极强，能分解、多轮、推理 |
| **灵活性** | 🟢 极高，Agent 自主决策 |
| **实现复杂度** | 🔴 高，需要设计 StateGraph、节点逻辑 |
| **成本** | 🔴 高，多轮 LLM 调用 |
| **延迟** | 🔴 高，多轮检索+推理 |
| **稳定性** | 🟡 中，Agent 可能"想太多"或陷入循环 |

**适用场景**：
- 复杂多条件查询（"2023年后入职的、技术部的、年假政策是什么？"）
- 需要计算/推理的问题（"根据A政策和B政策，我能休几天？"）
- 跨数据源的问答（文档 + API + 数据库）
- 对话式交互（多轮上下文）

**不适用场景**：
- 简单 FAQ（过度设计）
- 低延迟要求（Agent 推理需要时间）
- 成本敏感（Token 消耗大）

---

### 4.4 如何选择？

```
                    ┌─────────────────────┐
                    │  你的问题是什么类型？  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         简单事实查询      实体关系查询       复杂推理/多步骤
        "年假多少天"    "张三向谁汇报"     "年假+病假怎么算"
              │                │                │
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ 传统 RAG  │    │ GraphRAG │    │AgenticRAG│
        │  ★首选   │    │ ★首选   │    │ ★首选   │
        └──────────┘    └──────────┘    └──────────┘
```

**决策矩阵**：

| 评估维度 | 传统 RAG | GraphRAG | AgenticRAG |
|----------|----------|----------|------------|
| **实现难度** | ⭐ 低 | ⭐⭐⭐ 高 | ⭐⭐⭐ 高 |
| **维护成本** | ⭐ 低 | ⭐⭐⭐ 高 | ⭐⭐ 中 |
| **响应速度** | ⭐⭐⭐ 快 | ⭐⭐ 中 | ⭐ 慢 |
| **单次成本** | ⭐⭐⭐ 低 | ⭐⭐ 中 | ⭐ 高 |
| **简单问答** | ⭐⭐⭐ 好 | ⭐⭐⭐ 好 | ⭐⭐ 一般 |
| **跨文档关联** | ⭐ 差 | ⭐⭐⭐ 好 | ⭐⭐ 中 |
| **复杂推理** | ⭐ 差 | ⭐⭐ 中 | ⭐⭐⭐ 好 |
| **全局总结** | ⭐ 差 | ⭐⭐⭐ 好 | ⭐⭐ 中 |

**企业落地建议**：

1. **第一阶段（MVP）**：用 **传统 RAG** 快速上线，验证业务价值
2. **第二阶段（增强）**：如果用户反馈"找不到关联信息"，引入 **GraphRAG**
3. **第三阶段（智能化）**：如果用户问复杂问题，引入 **AgenticRAG**
4. **也可以组合**：GraphRAG 做知识构建，AgenticRAG 做查询推理


---

## 五、企业落地：成本与稳定性控制

RAG 系统上线后，真正的挑战是**花多少钱**和**稳不稳定**。

### 5.1 成本控制策略

**RAG 系统的成本构成**：

```
总成本 = Embedding 成本 + 向量库存储成本 + 检索成本 + LLM 生成成本 + 重排序成本
```

| 成本项 | 占比 | 控制策略 |
|--------|------|----------|
| **Embedding** | 20-30% | 本地部署开源模型（bge/m3e）；批量处理降 API 调用频率 |
| **向量库存储** | 5-10% | 选择合适的数据库；定期清理过期数据；用 PQ 压缩 |
| **检索** | <5% | ANN 索引已经很便宜；主要是内存成本 |
| **LLM 生成** | 40-60% | 用轻量模型（gpt-4o-mini / qwen-turbo）；限制 max_tokens；缓存常见问题 |
| **重排序** | 10-15% | 本地部署开源 reranker；减少 rerank 候选数量 |
| **Query 重写** | 5-10% | 只在多轮对话时启用；用轻量模型 |

**具体降本措施**：

```python
# 1. Embedding 批量处理（降低 API 调用次数）
# 坏的：每个 chunk 单独调用
for chunk in chunks:
    embedding.embed_query(chunk)  # N 次 API 调用

# 好的：批量调用
embeddings = embedding.embed_documents([c.page_content for c in chunks])  # 1 次

# 2. LLM 响应缓存
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# 3. 问题分类：简单问题不走完整 RAG Pipeline
def classify_question(question: str) -> str:
    """返回 'direct' | 'rag' | 'complex'"""
    # 用轻量模型快速分类
    # "你好" → direct（直接回答）
    # "年假多少天" → rag（标准 RAG）
    # "年假和病假加起来怎么算" → complex（AgenticRAG）
    ...

# 4. 限制 Token 消耗
llm = ChatOpenAI(
    model="gpt-4o-mini",      # 用便宜模型
    max_tokens=500,           # 限制输出长度
    temperature=0.1,          # 低温度减少"废话"
)

# 5. 动态调整检索深度
def adaptive_retrieval(question: str) -> int:
    """根据问题复杂度决定召回数量"""
    if is_simple_question(question):
        return 3   # 简单问题召回3个
    elif is_medium_question(question):
        return 5   # 中等问题召回5个
    else:
        return 10  # 复杂问题召回10个
```

### 5.2 稳定性保障

**常见故障与应对**：

| 故障类型 | 表现 | 解决方案 |
|----------|------|----------|
| **检索为空** | 没召回任何文档 | 降低相似度阈值；fallback 到 BM25；返回"未找到相关信息" |
| **检索偏差** | 召回的文档和问题无关 | 增加 Rerank 环节；调整 chunk_size；混合检索 |
| **LLM 幻觉** | 生成内容与文档矛盾 | 严格 Prompt 约束（"只根据文档回答"）；降低 temperature |
| **响应超时** | 检索或生成太慢 | 设置超时；降级到轻量模型；缓存热点问题 |
| **向量库挂了** | 检索失败 | fallback 到 BM25（基于内存或 Elasticsearch）；返回友好提示 |
| **上下文溢出** | Prompt 太长被截断 | 控制 Top-K 数量；使用 Map-Reduce；限制 chunk_size |

**稳定性代码示例**：

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def retrieve_with_fallback(query: str, vectorstore, bm25_retriever):
    """带降级策略的检索"""
    try:
        # 优先向量检索
        results = await vectorstore.asimilarity_search(query, k=5)
        if len(results) == 0:
            raise ValueError("Empty vector results")
        return results
    except Exception as e:
        # 降级到 BM25
        print(f"Vector search failed: {e}, falling back to BM25")
        return bm25_retriever.get_relevant_documents(query)

def generate_with_guardrails(documents, question, llm):
    """带安全护栏的生成"""
    if not documents:
        return "抱歉，我在知识库中没有找到相关信息。"
    
    context = "\n\n".join([d.page_content for d in documents[:5]])  # 限制上下文
    
    prompt = f"""你是一个严谨的企业助手。请严格根据参考文档回答问题。
如果文档中没有相关信息，必须回答"根据现有资料无法回答"。
绝对不要编造信息。

参考文档：
{context}

问题：{question}
"""
    
    return llm.invoke(prompt).content
```

### 5.3 参数调优指南

#### 5.3.1 Chunk Size / Overlap

```python
# 调参实验框架
def evaluate_chunk_params(docs, questions, chunk_sizes, overlaps):
    """
    用一组测试问题，评估不同切分参数的效果。
    返回每个参数组合的命中率和平均上下文完整度。
    """
    results = []
    for size in chunk_sizes:
        for overlap in overlaps:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=overlap
            )
            chunks = splitter.split_documents(docs)
            vs = PGVector.from_documents(chunks, embedding)
            
            # 评估
            score = 0
            for q in questions:
                retrieved = vs.similarity_search(q, k=3)
                if is_answer_contained(retrieved, q):
                    score += 1
            
            results.append({
                "chunk_size": size,
                "overlap": overlap,
                "hit_rate": score / len(questions),
                "num_chunks": len(chunks)
            })
    return results

# 推荐参数网格
chunk_sizes = [256, 512, 1024, 1536, 2048]
overlaps = [0, 50, 100, 200]
```

**企业调参经验**：

| 参数 | 默认值 | 调大效果 | 调小效果 |
|------|--------|----------|----------|
| `chunk_size` | 512 | 上下文更完整，但可能混入噪声 | 更精准，但可能丢失上下文 |
| `chunk_overlap` | 50 | 边界信息更安全 | 存储更省，但边界可能断裂 |

#### 5.3.2 Top-K 动态调整

**固定 Top-K 的问题**：

- Top-K=3：简单问题够了，复杂问题信息不够
- Top-K=10：复杂问题够了，简单问题浪费 Token

**动态调整策略**：

```python
def dynamic_top_k(question: str, base_k: int = 5) -> int:
    """根据问题特征动态调整召回数量"""
    
    # 策略1：问题长度
    if len(question) > 50:
        base_k += 3  # 长问题可能需要更多信息
    
    # 策略2：关键词数量（多条件查询需要更多文档）
    keywords = extract_keywords(question)
    if len(keywords) > 3:
        base_k += 2
    
    # 策略3：问题类型
    if any(w in question for w in ["对比", "区别", "差异", "和", "与"]):
        base_k += 3  # 对比类问题需要更多上下文
    
    # 策略4：根据首次检索结果的分数分布
    initial_results = vectorstore.similarity_search_with_score(question, k=base_k)
    scores = [score for _, score in initial_results]
    
    # 如果 Top-1 和 Top-5 分数差距很大，说明信息集中，可以减少 K
    if scores[0] - scores[-1] > 0.2:
        base_k = max(3, base_k - 2)
    
    # 如果分数都很接近，说明信息分散，需要增加 K
    if scores[0] - scores[-1] < 0.05:
        base_k += 3
    
    return min(base_k, 20)  # 上限保护

# 使用
k = dynamic_top_k(question)
results = vectorstore.similarity_search(question, k=k)
```

#### 5.3.3 Temperature / Max Tokens

| 参数 | 建议值 | 说明 |
|------|--------|------|
| **Temperature** | 0.0 ~ 0.3 | RAG 要求事实准确，temperature 越低越好（减少创造力，增加确定性） |
| **Max Tokens** | 500 ~ 1500 | 根据预期答案长度设置。太长浪费钱，太短可能截断 |
| **Top P** | 0.9 ~ 1.0 | 保持默认 1.0 即可，RAG 场景不需要特别调整 |

```python
# 不同场景的模型配置
configs = {
    "faq": {
        "model": "gpt-4o-mini",
        "temperature": 0.0,      # 最确定
        "max_tokens": 300,       # FAQ 答案短
    },
    "explanation": {
        "model": "gpt-4o",
        "temperature": 0.1,      # 稍灵活，用于解释性回答
        "max_tokens": 1000,
    },
    "complex_analysis": {
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 2000,
    }
}
```

#### 5.3.4 相似度阈值（Score Threshold）

过滤掉"明显不相关"的检索结果，减少噪声和 Token 浪费。

```python
# 方式1：固定阈值
results = vectorstore.similarity_search_with_score(query, k=10)
filtered = [doc for doc, score in results if score > 0.7]  # 只保留相似度 > 0.7 的

# 方式2：相对阈值（保留 Top 结果中与最佳结果的差距在一定范围内）
results = vectorstore.similarity_search_with_score(query, k=10)
if results:
    best_score = results[0][1]
    # 只保留与最佳结果差距 < 0.15 的
    filtered = [doc for doc, score in results if best_score - score < 0.15]

# 方式3：自适应阈值（根据分数分布动态调整）
import numpy as np

def adaptive_threshold(results):
    scores = [s for _, s in results]
    if len(scores) < 3:
        return results
    
    # 计算分数的"断崖"（最大差距点）
    gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
    max_gap_idx = np.argmax(gaps)
    
    # 在最大差距处截断
    return results[:max_gap_idx + 1]
```

### 5.4 评估与监控 ⭐⭐⭐ 企业级必备

评估与监控是 RAG 系统从"能用"走向"好用"的核心环节。**没有评估，你就像蒙着眼开车；没有监控，系统什么时候崩了你都不知道。**

#### 5.4.1 为什么要评估和监控？

**① RAG 是多组件系统，任何一个环节出错都会影响最终效果**

```
用户问"年假政策"
    │
    ├── 切分错了 → 关键信息被切断 → 检索不到 → 答案错误
    ├── 检索差了 → 召回无关文档 → 上下文噪声大 → 答案错误
    ├── 重排序崩了 → 相关文档排后面 → LLM看不到 → 答案错误
    ├── Prompt设计差了 → LLM不按要求答 → 答案错误
    └── LLM本身幻觉 → 编造内容 → 答案错误

问题：最终答案错了，到底哪个环节出的问题？
答案：没有评估和监控，你只能猜。
```

**② 上线后数据会漂移（Data Drift）**

| 漂移类型 | 表现 | 后果 |
|----------|------|------|
| **Query 漂移** | 用户开始问之前没预料到的问题 | 检索策略失效 |
| **文档漂移** | 知识库文档更新、过期 | 召回内容过时 |
| **分布漂移** | 用户从问"政策"变成问"实操案例" | 切分粒度可能不再合适 |
| **模型漂移** | Embedding 模型版本更新 | 向量空间分布变化 |

**③ 评估是优化的前提**

> 你不能优化你没法测量的东西。
>
> 调整 chunk_size 从 500 改到 1000，效果变好了还是变差了？没有评估指标，只能靠"感觉"。

**④ 监控是防止线上故障的唯一手段**

- 向量库挂了 → 检索全部失败 →  fallback 有没有生效？
- LLM API 限流了 → 响应超时 → 用户是不是一直在等？
- 检索分数持续下降 → 数据质量出了问题 → 什么时候开始的？

#### 5.4.2 离线评估：如何系统性地评估 RAG 效果？

离线评估 = 用**预先准备好的测试集**，在系统上线前或迭代后，批量评估各组件的性能。

##### 评估指标体系

RAG 系统可以拆成 **检索** 和 **生成** 两个阶段分别评估：

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG 评估指标体系                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  检索阶段 (Retrieval)                                        │
│  ├── Context Precision  ── 检索结果中 relevant 的比例        │
│  ├── Context Recall     ── 所有 relevant 文档被召回的比例     │
│  ├── Context Relevance ── 检索结果与问题的相关程度            │
│  └── MRR / NDCG        ── 排序质量                           │
│                                                             │
│  生成阶段 (Generation)                                       │
│  ├── Answer Relevance  ── 答案是否回答了问题                 │
│  ├── Faithfulness      ── 答案是否忠实于检索文档             │
│  └── Answer Correctness ── 答案的事实正确性                 │
│                                                             │
│  端到端 (End-to-End)                                         │
│  └── 用户满意度 / 任务完成率 / 人工评分                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

##### 检索阶段评估指标详解

**① Context Precision（上下文精确率）**

> 检索回来的 Top-K 文档中，有多少比例是真正相关的？

```
检索结果 Top-5: [相关, 不相关, 相关, 相关, 不相关]
Context Precision@5 = 3/5 = 0.6
```

```python
def context_precision(retrieved_docs: List[Document], 
                      relevant_docs: List[str]) -> float:
    """
    retrieved_docs: 检索回来的文档列表
    relevant_docs:  预先标注的正确答案文档 ID 或内容哈希
    """
    if not retrieved_docs:
        return 0.0
    
    relevant_count = sum(1 for d in retrieved_docs 
                         if d.page_content in relevant_docs)
    return relevant_count / len(retrieved_docs)
```

**② Context Recall（上下文召回率）**

> 所有应该被召回的相关文档中，实际被召回了多少？

```
应该召回的相关文档: [A, B, C, D]  （共4篇）
实际检索召回:       [A, C, E]      （A和C是对的，E不相关）
Context Recall = 2/4 = 0.5
```

```python
def context_recall(retrieved_docs: List[Document],
                   all_relevant: List[str]) -> float:
    if not all_relevant:
        return 1.0  # 没有相关文档，算满分
    
    retrieved_set = {d.page_content for d in retrieved_docs}
    relevant_retrieved = sum(1 for r in all_relevant if r in retrieved_set)
    return relevant_retrieved / len(all_relevant)
```

**③ MRR（Mean Reciprocal Rank，平均倒数排名）**

> 正确答案的平均排名倒数。排第1 → 1.0；排第2 → 0.5；排第3 → 0.33

```
问题1: 正确答案排第1 → 1/1 = 1.0
问题2: 正确答案排第3 → 1/3 = 0.33
问题3: 正确答案排第2 → 1/2 = 0.5
MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

```python
def mean_reciprocal_rank(results: List[List[Document]], 
                         ground_truths: List[str]) -> float:
    """
    results: 每个问题的检索结果列表
    ground_truths: 每个问题的正确答案内容
    """
    rr_sum = 0
    for retrieved, truth in zip(results, ground_truths):
        for rank, doc in enumerate(retrieved, 1):
            if truth in doc.page_content:
                rr_sum += 1 / rank
                break
    return rr_sum / len(results)
```

**④ NDCG（Normalized Discounted Cumulative Gain）**

> 考虑排序位置的加权指标。排得越靠前越相关，分数越高。

```python
import numpy as np

def dcg(relevances: List[float]) -> float:
    """DCG = sum(rel_i / log2(i+1))"""
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg(retrieved_docs: List[Document],
         relevance_scores: List[float]) -> float:
    """
    relevance_scores: 每个位置的相关度分数（如 0=不相关, 1=部分相关, 2=完全相关）
    """
    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

##### 生成阶段评估指标详解

**⑤ Faithfulness（忠实度/忠诚度）**

> 生成的答案中的每个陈述，是否都能被检索到的文档支持？

这是 RAG 最核心的指标——**防止幻觉**。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def evaluate_faithfulness(answer: str, contexts: List[str]) -> dict:
    """
    用 LLM 判断答案的忠实度
    返回: {"score": 0-1, "reasoning": "...", "violations": [...]}
    """
    context_text = "\n\n".join(contexts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严格的事实核查专家。
请根据"参考文档"，逐一检查"生成答案"中的每个陈述是否被文档支持。

输出 JSON 格式：
{
  "score": 0-1 之间的浮点数（1表示完全忠实，0表示完全不忠实）,
  "reasoning": "总体判断理由",
  "violations": [
    {"statement": "答案中的具体陈述", "reason": "为什么这个陈述不被文档支持"}
  ]
}"""),
        ("human", """参考文档：
{context}

生成答案：
{answer}

请输出 JSON：""")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    chain = prompt | llm
    
    import json
    response = chain.invoke({"context": context_text, "answer": answer}).content
    # 解析 JSON（实际代码需要处理格式问题）
    try:
        return json.loads(response)
    except:
        return {"score": 0.5, "reasoning": "解析失败", "violations": []}
```

**⑥ Answer Relevance（答案相关性）**

> 生成的答案是否与用户的问题相关？（避免答非所问）

```python
def evaluate_answer_relevance(question: str, answer: str) -> float:
    """
    用 LLM 评估答案与问题的相关性
    """
    prompt = f"""请判断以下答案是否直接回答了用户的问题。
只输出一个 0-1 之间的分数，1 表示完全相关，0 表示完全不相关。
不要解释，只输出数字。

问题：{question}
答案：{answer}

分数："""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    score_text = llm.invoke(prompt).content.strip()
    
    try:
        return float(score_text)
    except:
        return 0.5
```

##### RAGAS 框架：一站式 RAG 评估

**RAGAS** 是一个专门评估 RAG 系统的开源框架，封装了上述所有指标。

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
)
from datasets import Dataset

# 准备测试数据
# 格式：question, answer, contexts (检索结果), ground_truth (标准答案)
test_data = Dataset.from_dict({
    "question": ["年假有多少天？", "怎么申请病假？"],
    "answer": ["5天", "需要提前申请"],
    "contexts": [
        ["入职满1年享受5天年假", "年假需提前3天申请"],
        ["病假需提交医院证明", "病假申请流程..."]
    ],
    "ground_truth": ["入职满1年享受5天年假", "病假需提交医院证明并提前申请"]
})

# 运行评估
result = evaluate(
    dataset=test_data,
    metrics=[
        faithfulness,           # 忠实度
        answer_relevancy,       # 答案相关性
        context_precision,      # 上下文精确率
        context_recall,         # 上下文召回率
    ],
    llm=ChatOpenAI(model="gpt-4o"),
    embeddings=OpenAIEmbeddings(),
)

print(result)
# 输出示例:
# {'faithfulness': 0.85, 'answer_relevancy': 0.92, 
#  'context_precision': 0.78, 'context_recall': 0.81}
```

##### 构建测试集（Ground Truth）的方法

**方法一：人工标注（最准，最贵）**

```python
"""
标注流程：
1. 从真实用户 query 日志中抽样（如 100-500 条）
2. 对每条 query，人工在知识库中找到正确答案
3. 标注：
   - relevant_docs: 哪些文档片段包含答案
   - answer: 标准答案文本
   - relevance_level: 1(不相关) / 2(部分相关) / 3(完全相关)

格式示例：
{
  "question": "年假怎么申请？",
  "relevant_doc_ids": ["doc_003_chunk_12", "doc_003_chunk_13"],
  "answer": "需提前3天在OA系统提交申请",
  "difficulty": "easy"  // easy/medium/hard
}
"""
```

**方法二：LLM 辅助生成（快，需人工抽检）**

```python
def generate_test_cases(docs: List[Document], num_questions: int = 100):
    """
    用 LLM 基于文档自动生成问答对
    """
    prompt = """基于以下文档片段，生成 {n} 个用户可能问的问题和对应的答案。
要求：
1. 问题要覆盖文档中的关键信息点
2. 包含简单、中等、困难三种难度
3. 答案必须完全基于提供的文档

输出 JSON 数组格式：
[
  {"question": "...", "answer": "...", "source": "...", "difficulty": "easy"}
]

文档：
{docs}
"""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    sample_docs = random.sample(docs, min(20, len(docs)))
    doc_text = "\n\n".join([d.page_content[:500] for d in sample_docs])
    
    response = llm.invoke(prompt.format(n=num_questions, docs=doc_text))
    # 解析 JSON...
    return parse_qa_pairs(response.content)
```

**方法三：用户反馈收集（最真实，需时间积累）**

```python
# 在用户界面中嵌入反馈按钮
user_feedback = {
    "query_id": "uuid",
    "question": "年假怎么申请？",
    "answer": "需提前3天申请",
    "user_rating": 1,  # 1=满意, 0=不满意
    "user_comment": "答案是错的，实际是提前1天",  # 用户纠错
    "timestamp": "2024-01-15T10:30:00Z"
}
```

##### 离线评估完整代码

```python
# evaluation.py - 完整的离线评估脚本
import json
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

@dataclass
class RAGEvaluationResult:
    """单次评估结果"""
    question: str
    retrieved_docs: List[Document]
    generated_answer: str
    ground_truth_answer: str
    ground_truth_docs: List[str]
    
    # 检索指标
    context_precision: float
    context_recall: float
    mrr: float
    
    # 生成指标
    faithfulness: float
    answer_relevance: float
    
    # 综合
    overall_score: float


class RAGEvaluator:
    """RAG 系统离线评估器"""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)
    
    def evaluate_retrieval(self,
                           retrieved: List[Document],
                           ground_truth_docs: List[str]) -> Dict[str, float]:
        """评估检索质量"""
        retrieved_texts = [d.page_content for d in retrieved]
        
        # Context Precision
        relevant_count = sum(1 for t in retrieved_texts if t in ground_truth_docs)
        precision = relevant_count / len(retrieved) if retrieved else 0.0
        
        # Context Recall
        gt_set = set(ground_truth_docs)
        retrieved_set = set(retrieved_texts)
        recall = len(gt_set & retrieved_set) / len(gt_set) if gt_set else 1.0
        
        # MRR
        mrr = 0.0
        for rank, doc in enumerate(retrieved, 1):
            if doc.page_content in ground_truth_docs:
                mrr = 1.0 / rank
                break
        
        return {
            "context_precision": precision,
            "context_recall": recall,
            "mrr": mrr,
        }
    
    def evaluate_generation(self,
                            question: str,
                            answer: str,
                            contexts: List[str],
                            ground_truth: str) -> Dict[str, float]:
        """评估生成质量（使用 LLM-as-Judge）"""
        context_text = "\n\n".join(contexts)
        
        # Faithfulness
        faithfulness_prompt = f"""根据以下参考文档，判断答案是否忠实于文档内容。
只输出一个 0-1 的分数，1=完全忠实，0=完全不忠实。

参考文档：{context_text}
答案：{answer}

分数："""
        faithfulness = self._parse_score(
            self.llm.invoke(faithfulness_prompt).content
        )
        
        # Answer Relevance
        relevance_prompt = f"""判断以下答案是否直接、准确回答了问题。
只输出一个 0-1 的分数。

问题：{question}
答案：{answer}

分数："""
        relevance = self._parse_score(
            self.llm.invoke(relevance_prompt).content
        )
        
        # Answer Correctness (对比 ground truth)
        correctness_prompt = f"""比较"生成答案"和"标准答案"的事实一致性。
只输出一个 0-1 的分数。

问题：{question}
生成答案：{answer}
标准答案：{ground_truth}

分数："""
        correctness = self._parse_score(
            self.llm.invoke(correctness_prompt).content
        )
        
        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "answer_correctness": correctness,
        }
    
    def _parse_score(self, text: str) -> float:
        """从 LLM 输出中解析分数"""
        try:
            # 提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', text.strip())
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
        except:
            pass
        return 0.5
    
    def run_full_evaluation(self,
                            test_cases: List[Dict],
                            rag_pipeline_func) -> Dict:
        """
        运行完整评估
        
        test_cases: [{"question": "...", "ground_truth": "...", "ground_truth_docs": [...]}]
        rag_pipeline_func: 你的 RAG pipeline 函数，输入 question，返回 {"answer": "...", "documents": [...]}
        """
        results = []
        
        for case in test_cases:
            # 运行 RAG
            rag_result = rag_pipeline_func(case["question"])
            
            # 评估检索
            retrieval_metrics = self.evaluate_retrieval(
                rag_result["documents"],
                case.get("ground_truth_docs", [])
            )
            
            # 评估生成
            generation_metrics = self.evaluate_generation(
                case["question"],
                rag_result["answer"],
                [d.page_content for d in rag_result["documents"]],
                case["ground_truth"]
            )
            
            result = RAGEvaluationResult(
                question=case["question"],
                retrieved_docs=rag_result["documents"],
                generated_answer=rag_result["answer"],
                ground_truth_answer=case["ground_truth"],
                ground_truth_docs=case.get("ground_truth_docs", []),
                **retrieval_metrics,
                **generation_metrics,
                overall_score=(retrieval_metrics["context_recall"] + 
                              generation_metrics["faithfulness"]) / 2
            )
            results.append(result)
        
        # 汇总统计
        summary = {
            "total_cases": len(results),
            "avg_context_precision": np.mean([r.context_precision for r in results]),
            "avg_context_recall": np.mean([r.context_recall for r in results]),
            "avg_mrr": np.mean([r.mrr for r in results]),
            "avg_faithfulness": np.mean([r.faithfulness for r in results]),
            "avg_answer_relevance": np.mean([r.answer_relevance for r in results]),
            "avg_overall_score": np.mean([r.overall_score for r in results]),
            "failed_cases": [
                {"question": r.question, 
                 "answer": r.generated_answer,
                 "ground_truth": r.ground_truth_answer}
                for r in results if r.overall_score < 0.5
            ]
        }
        
        return summary


# ========== 使用示例 ==========
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    test_cases = [
        {
            "question": "年假有多少天？",
            "ground_truth": "入职满1年享受5天年假，满10年享受10天",
            "ground_truth_docs": ["年假政策：入职满1年享受5天年假。满10年享受10天。"]
        },
        # ... 更多测试用例
    ]
    
    # 假设 ask() 是你的 RAG 接口
    def my_rag_pipeline(question: str):
        # 这里调用你的实际 RAG 系统
        return {"answer": "...", "documents": [Document(...)]}
    
    summary = evaluator.run_full_evaluation(test_cases, my_rag_pipeline)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
```

#### 5.4.3 在线监控：如何实时掌握系统健康度？

在线监控 = **系统运行时**持续采集指标，及时发现问题并告警。

##### 监控分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                     在线监控体系                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户层                                                      │
│  ├── 用户满意度（👍/👎）                                     │
│  ├── 会话轮数                                                │
│  ├── 重复提问率（用户问了同样的问题）                         │
│  └── 人工介入率（用户从 AI 转人工客服）                       │
│                                                             │
│  业务层                                                      │
│  ├── 检索质量：Top-1 命中率、平均检索分数                     │
│  ├── 生成质量：Faithfulness 分布、答案长度分布                │
│  ├── 覆盖率：能回答的问题比例（vs 回答"不知道"）              │
│  └── 拒答率：fallback 到"未找到信息"的比例                    │
│                                                             │
│  系统层                                                      │
│  ├── 延迟：端到端响应时间、各节点耗时                         │
│  ├── 吞吐量：QPS、并发数                                     │
│  ├── 错误率：各环节失败比例                                   │
│  ├── Token 消耗：Embedding + LLM + Rerank                     │
│  └── 资源：CPU、内存、GPU、磁盘                               │
│                                                             │
│  数据层                                                      │
│  ├── 向量库规模：文档数、索引大小                             │
│  ├── Query 分布：高频词、新词、异常 query                     │
│  └── 数据漂移：embedding 分布变化                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

##### 系统层监控代码

```python
import time
import functools
from typing import Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class RAGMetrics:
    """RAG 系统运行时指标收集器"""
    # 延迟（毫秒）
    total_latency_ms: float = 0.0
    retrieve_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    generate_latency_ms: float = 0.0
    
    # Token 消耗
    embedding_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    
    # 检索质量
    retrieved_count: int = 0
    retrieval_scores: List[float] = field(default_factory=list)
    top1_score: float = 0.0
    
    # 生成质量
    answer_length: int = 0
    has_citation: bool = False
    
    # 结果
    was_fallback: bool = False  # 是否降级了
    error: str = ""  # 错误信息


class RAGMonitor:
    """RAG 系统监控器"""
    
    def __init__(self):
        self.metrics_history: List[RAGMetrics] = []
        self.daily_stats = defaultdict(lambda: {
            "total_requests": 0,
            "avg_latency": 0.0,
            "error_count": 0,
            "fallback_count": 0,
            "avg_retrieval_score": 0.0,
        })
    
    def timed(self, metric_name: str):
        """装饰器：自动测量函数执行时间"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    latency = (time.time() - start) * 1000
                    
                    # 存储到当前上下文
                    if hasattr(wrapper, '_current_metrics'):
                        setattr(wrapper._current_metrics, f"{metric_name}_latency_ms", latency)
                    
                    return result
                except Exception as e:
                    if hasattr(wrapper, '_current_metrics'):
                        wrapper._current_metrics.error = str(e)
                    raise
            return wrapper
        return decorator
    
    def log_query(self, metrics: RAGMetrics):
        """记录一次查询的指标"""
        self.metrics_history.append(metrics)
        
        # 更新日统计
        today = time.strftime("%Y-%m-%d")
        stats = self.daily_stats[today]
        stats["total_requests"] += 1
        
        # 滑动平均更新
        n = stats["total_requests"]
        stats["avg_latency"] = (stats["avg_latency"] * (n-1) + metrics.total_latency_ms) / n
        stats["avg_retrieval_score"] = (stats["avg_retrieval_score"] * (n-1) + 
                                         (metrics.top1_score if metrics.retrieval_scores else 0)) / n
        
        if metrics.error:
            stats["error_count"] += 1
        if metrics.was_fallback:
            stats["fallback_count"] += 1
        
        # 实时告警检查
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: RAGMetrics):
        """检查是否需要告警"""
        alerts = []
        
        # 延迟告警
        if metrics.total_latency_ms > 10000:  # 10秒
            alerts.append(f"[ALERT] 响应延迟过高: {metrics.total_latency_ms:.0f}ms")
        
        # 检索质量告警
        if metrics.retrieval_scores and metrics.top1_score < 0.3:
            alerts.append(f"[ALERT] Top-1 检索分数过低: {metrics.top1_score:.3f}")
        
        # 错误告警
        if metrics.error:
            alerts.append(f"[ALERT] 处理出错: {metrics.error}")
        
        # 降级告警
        if metrics.was_fallback:
            alerts.append("[WARNING] 触发了降级策略")
        
        for alert in alerts:
            print(alert)
            # 实际生产环境：发送到钉钉/企业微信/飞书/Sentry
    
    def get_daily_report(self, date: str = None) -> dict:
        """生成日报"""
        date = date or time.strftime("%Y-%m-%d")
        stats = self.daily_stats[date]
        
        total = stats["total_requests"]
        if total == 0:
            return {"message": "今日无数据"}
        
        return {
            "date": date,
            "total_requests": total,
            "avg_latency_ms": round(stats["avg_latency"], 2),
            "error_rate": round(stats["error_count"] / total * 100, 2),
            "fallback_rate": round(stats["fallback_count"] / total * 100, 2),
            "avg_retrieval_score": round(stats["avg_retrieval_score"], 3),
        }


# ========== 在 LangGraph 中集成监控 ==========
from langgraph.graph import StateGraph

class MonitoredRAGState(dict):
    """带监控的 State"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = RAGMetrics()
        self._start_time = time.time()

def monitored_retrieve(state: MonitoredRAGState):
    """带监控的检索节点"""
    start = time.time()
    
    # 原有检索逻辑...
    question = state.get("question", "")
    docs = vectorstore.similarity_search(question, k=5)
    
    # 记录指标
    state.metrics.retrieve_latency_ms = (time.time() - start) * 1000
    state.metrics.retrieved_count = len(docs)
    if docs:
        # 假设有分数（不同数据库获取方式不同）
        state.metrics.retrieval_scores = [0.8] * len(docs)  # 示例
        state.metrics.top1_score = 0.8
    
    return {"documents": docs, "metrics": state.metrics}

def monitored_generate(state: MonitoredRAGState):
    """带监控的生成节点"""
    start = time.time()
    
    # 原有生成逻辑...
    answer = "..."
    
    state.metrics.generate_latency_ms = (time.time() - start) * 1000
    state.metrics.total_latency_ms = (time.time() - state._start_time) * 1000
    state.metrics.answer_length = len(answer)
    state.metrics.has_citation = "[来源" in answer or "[文档" in answer
    
    return {"generation": answer, "metrics": state.metrics}
```

##### 与 LangSmith / Langfuse 集成

```python
# ========== LangSmith 集成（OpenAI 出品，功能全）==========
# pip install langsmith
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "company-rag-prod"

# 自动追踪所有 LangChain 调用
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
# 每次调用会自动在 LangSmith 上记录 trace


# ========== Langfuse 集成（开源，可自托管）==========
# pip install langfuse
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse(
    public_key="pk-xxx",
    secret_key="sk-xxx",
    host="https://cloud.langfuse.com"  # 或自托管地址
)

@observe()
def rag_pipeline(question: str):
    """整个 RAG pipeline 会被自动追踪"""
    # 检索
    docs = retriever.get_relevant_documents(question)
    
    # 生成
    answer = generate_answer(docs, question)
    
    # 手动记录分数
    langfuse.score(
        trace_id=langfuse.get_current_trace_id(),
        name="retrieval_quality",
        value=0.85,
        comment="Top-3 都相关"
    )
    
    return answer


# ========== 自定义 Dashboard（用 Prometheus + Grafana）==========
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义指标
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests', ['status'])
REQUEST_LATENCY = Histogram('rag_request_duration_seconds', 'Request latency')
RETRIEVAL_SCORE = Gauge('rag_retrieval_top1_score', 'Top-1 retrieval score')
VECTOR_DB_SIZE = Gauge('rag_vector_db_documents', 'Number of documents in vector DB')

def record_metrics(metrics: RAGMetrics):
    """将指标推送到 Prometheus"""
    REQUEST_LATENCY.observe(metrics.total_latency_ms / 1000)
    
    if metrics.error:
        REQUEST_COUNT.labels(status='error').inc()
    else:
        REQUEST_COUNT.labels(status='success').inc()
    
    if metrics.retrieval_scores:
        RETRIEVAL_SCORE.set(metrics.top1_score)

# 启动 metrics 服务（端口 8000）
# start_http_server(8000)
```

##### 监控 Dashboard 设计

```
┌────────────────────────────────────────────────────────────────┐
│                    RAG 系统监控大屏                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ QPS: 120    │  │ 平均延迟    │  │ 错误率      │           │
│  │ ↑ 15%       │  │ 1.2s        │  │ 0.5%        │           │
│  │             │  │ ↑ 0.3s      │  │ ↓ 0.2%      │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                │
│  ┌─────────────────────────┐  ┌─────────────────────────┐     │
│  │  检索质量趋势（24h）      │  │  生成质量趋势（24h）      │     │
│  │  Top-1 命中率: 78%       │  │  Faithfulness: 0.85      │     │
│  │  Avg Score: 0.72         │  │  Relevance: 0.91         │     │
│  │  [████████░░░░░░░░]      │  │  [██████████░░░░]        │     │
│  └─────────────────────────┘  └─────────────────────────┘     │
│                                                                │
│  ┌─────────────────────────┐  ┌─────────────────────────┐     │
│  │  Token 消耗（按组件）     │  │  用户反馈分布            │     │
│  │  Embedding:  35% ████    │  │  👍 满意: 82%            │     │
│  │  LLM:        55% ██████  │  │  👎 不满意: 12%          │     │
│  │  Rerank:     10% █       │  │  ⚠️ 需改进: 6%           │     │
│  └─────────────────────────┘  └─────────────────────────┘     │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  近期 Bad Cases（Top 5）                                   │   │
│  │  1. [检索失败] "加班费" → 召回"加班餐补"文档              │   │
│  │  2. [幻觉] "病假"回答中包含未提及的"需三甲医院证明"        │   │
│  │  3. [拒答] "绩效奖金" → 回答"未找到信息"（实际有文档）     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### 5.4.4 持续优化闭环：从反馈到改进

**Bad Case 分析模板**：

```
Bad Case 分析单
───────────────
编号: BC-2024-0115-001
时间: 2024-01-15 14:32
用户问题: "加班费怎么算？"
系统答案: "工作日加班按1.5倍工资计算"
正确/期望答案: "工作日加班1.5倍，周末2倍，节假日3倍"

问题分类（打勾）:
  [✓] 检索失败（没召回正确文档）
  [ ] 检索偏差（召回的不相关）
  [ ] 重排序错误（相关文档排后面）
  [ ] LLM幻觉（编造了不存在的信息）
  [ ] 答案不完整（漏了关键信息）
  [ ] 切分问题（信息被切断了）
  [ ] 文档缺失（知识库里本来就没有）

根因分析:
  用户问"加班费"，系统召回了"加班政策.pdf"的第1页，
  但正确的"倍数说明"在第3页。原因是 chunk_size=500，
  第1页只包含"总则"部分，倍数说明被切到了另一个 chunk。

改进措施:
  1. 调大 chunk_overlap 从 50 到 150
  2. 对该文档做语义切分
  3. 补充 FAQ 问答对

责任人: 张三
预计完成: 2024-01-17
验证方式: 重新跑评估集，该 case 的 recall 应 > 0.8
```

**A/B 测试框架**：

```python
class RAGABTest:
    """RAG A/B 测试框架"""
    
    def __init__(self, 
                 variant_a_pipeline,  # 当前版本
                 variant_b_pipeline,  # 新版本
                 traffic_split: float = 0.1):  # 10% 流量走 B
        self.variant_a = variant_a_pipeline
        self.variant_b = variant_b_pipeline
        self.split = traffic_split
        self.results = {"A": [], "B": []}
    
    def route(self, question: str, user_id: str) -> str:
        """根据用户 ID 哈希分流"""
        import hashlib
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        
        if hash_val % 100 < int(self.split * 100):
            variant = "B"
            answer = self.variant_b(question)
        else:
            variant = "A"
            answer = self.variant_a(question)
        
        return variant, answer
    
    def collect_feedback(self, variant: str, 
                        question: str, 
                        answer: str, 
                        user_rating: int):
        """收集用户反馈"""
        self.results[variant].append({
            "question": question,
            "answer": answer,
            "rating": user_rating,
        })
    
    def get_comparison_report(self) -> dict:
        """生成 A/B 对比报告"""
        import numpy as np
        
        ratings_a = [r["rating"] for r in self.results["A"]]
        ratings_b = [r["rating"] for r in self.results["B"]]
        
        return {
            "sample_size": {"A": len(ratings_a), "B": len(ratings_b)},
            "avg_rating": {
                "A": np.mean(ratings_a) if ratings_a else 0,
                "B": np.mean(ratings_b) if ratings_b else 0,
            },
            "satisfaction_rate": {
                "A": sum(1 for r in ratings_a if r >= 4) / len(ratings_a) if ratings_a else 0,
                "B": sum(1 for r in ratings_b if r >= 4) / len(ratings_b) if ratings_b else 0,
            },
            "recommendation": "B" if (np.mean(ratings_b) if ratings_b else 0) > 
                                      (np.mean(ratings_a) if ratings_a else 0) + 0.1 else "A"
        }
```

**持续优化闭环流程**：

```
            ┌─────────────────────────────────────────────┐
            │               线上运行                        │
            └──────────────┬──────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │  收集数据                     │
            │  ├── 用户反馈（👍/👎）         │
            │  ├── 监控指标（延迟/分数）      │
            │  └── 日志记录（query/answer）   │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │  发现 Bad Case                │
            │  ├── 用户投诉                  │
            │  ├── 监控告警（分数骤降）       │
            │  └── 定期抽查评估集            │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │  根因分析                     │
            │  ├── 是切分问题？              │
            │  ├── 是检索问题？              │
            │  ├── 是重排序问题？            │
            │  ├── 是 Prompt 问题？          │
            │  └── 是 LLM 问题？             │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │  制定改进方案                  │
            │  ├── 调 chunk_size/overlap     │
            │  ├── 换 embedding 模型         │
            │  ├── 加混合检索/重排序          │
            │  ├── 优化 Prompt 模板          │
            │  └── 补充/修正文档             │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │  离线评估验证                  │
            │  ├── 跑测试集看指标变化         │
            │  ├── 检查 bad case 是否修复     │
            │  └── 对比新旧版本差异           │
            └──────────────┬───────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        [指标未提升]               [指标提升]
              │                         │
              ▼                         ▼
        重新分析根因               A/B 测试（小流量）
              │                         │
              │    ┌────────────────────┘
              │    ▼
              │  全量上线
              │    │
              └────┘
                   ▼
            回到"线上运行"，循环往复
```

> **企业落地建议**：
> 1. **评估先行**：上线前必须有 100+ 条测试用例，覆盖常见和边缘场景
> 2. **监控必建**：至少监控延迟、错误率、检索分数三个核心指标
> 3. **Bad Case 驱动**：每周评审 Top-10 用户不满意的问题，持续修复
> 4. **版本管理**：每次参数调整都要记录，方便回滚和对比
> 5. **工具选型**：小团队用 LangSmith/Langfuse，大团队自建 Prometheus + Grafana + ELK

## 六、完整 LangGraph 实战代码

LangGraph 是 LangChain 的团队推出的**状态图框架**，特别适合构建复杂的 RAG Pipeline。相比传统链式调用，LangGraph 的优势：

- **可视化流程**：每个节点（加载、切分、检索、生成）清晰可追踪
- **状态管理**：中间结果（检索到的文档、重排序后的结果）在 State 中流转
- **条件分支**：根据中间结果决定走哪条路径（是否需要重试、是否需要重排序）
- **循环支持**：检索不够？自动再来一轮

### 6.1 离线 Pipeline：文档 → 向量库

```python
# ingest.py - 离线数据摄入脚本
import os
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

# ========== 配置 ==========
DATA_DIR = "./knowledge_base"      # 原始文档目录
CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/rag_db"             # 向量库持久化目录
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"

# ========== 1. 加载文档 ==========
def load_documents(data_dir: str) -> List[Document]:
    """递归加载目录下的所有支持的文档"""
    all_docs = []
    
    # PDF
    if any(f.endswith('.pdf') for f in os.listdir(data_dir)):
        pdf_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        all_docs.extend(pdf_loader.load())
    
    # TXT / Markdown
    txt_loader = DirectoryLoader(
        data_dir, glob="**/*.{txt,md,markdown}", loader_cls=TextLoader
    )
    all_docs.extend(txt_loader.load())
    
    print(f"Loaded {len(all_docs)} raw documents")
    return all_docs

# ========== 2. 数据清洗 ==========
class DocumentCleaner:
    """可配置的企业级文档清洗器"""
    
    def __init__(self, min_doc_length: int = 50, dedup_exact: bool = True):
        self.min_doc_length = min_doc_length
        self.dedup_exact = dedup_exact
    
    def clean_text(self, text: str) -> str:
        import re
        # 去除零宽字符
        for c in '\u200b\u200c\u200d\ufeff\u2060':
            text = text.replace(c, '')
        # 去除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除 URL
        text = re.sub(r'https?://\S+', '[URL]', text)
        # 去除邮箱
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        # 去除页码
        text = re.sub(r'\n?\s*[-–—]\s*\d+\s*[-–—]\s*\n?', '\n', text)
        text = re.sub(r'\n?\s*Page\s+\d+.*?\n?', '\n', text, flags=re.I)
        text = re.sub(r'\n?\s*第\s*\d+\s*页\s*\n?', '\n', text)
        # 去除版权
        text = re.sub(r'Copyright\s+©.*?\.?\s*\n?', '', text, flags=re.I)
        text = re.sub(r'All\s+rights\s+reserved.*?\n?', '', text, flags=re.I)
        # 规范化空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()
    
    def clean(self, docs: List[Document]) -> List[Document]:
        cleaned = []
        for d in docs:
            t = self.clean_text(d.page_content)
            if t and len(t) >= self.min_doc_length:
                cleaned.append(Document(page_content=t, metadata=d.metadata))
        
        # 精确去重
        if self.dedup_exact:
            import hashlib
            seen = set()
            unique = []
            for d in cleaned:
                h = hashlib.md5(d.page_content.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    unique.append(d)
            cleaned = unique
        
        print(f"Cleaned: {len(docs)} -> {len(cleaned)}")
        return cleaned

# ========== 3. 文档切分 ==========
def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# ========== 4. 向量化 + 入库 ==========
def build_vectorstore(chunks: List[Document]) -> PGVector:
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(CONNECTION_STRING):
        import shutil
        shutil.rmtree(CONNECTION_STRING)
    
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embedding,
        connection_string=CONNECTION_STRING,
        collection_name="company_knowledge",
        pre_delete_collection=True,  # 重建时先清空旧数据
    )
    
    print(f"Vectorstore built at {CONNECTION_STRING}")
    return vectorstore

# ========== 主流程 ==========
def main():
    print("=" * 50)
    print("RAG Knowledge Base Ingestion Pipeline")
    print("=" * 50)
    
    docs = load_documents(DATA_DIR)
    
    cleaner = DocumentCleaner(min_doc_length=30, dedup_exact=True)
    clean_docs = cleaner.clean(docs)
    
    chunks = split_documents(clean_docs)
    
    vectorstore = build_vectorstore(chunks)
    
    print("\n✅ Ingestion complete!")
    
    results = vectorstore.similarity_search("年假政策", k=2)
    print(f"\nTest query results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.metadata.get('source')}] {r.page_content[:100]}...")

if __name__ == "__main__":
    main()
```

### 6.2 在线 Pipeline：查询 → 生成

```python
# query.py - 基础 RAG 查询（传统 RAG）
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langgraph.graph import StateGraph, END

# ========== 配置 ==========
CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/rag_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5

# ========== State 定义 ==========
class RAGState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    generation: str
    sources: List[str]

# ========== 组件初始化 ==========
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embedding,
    collection_name="company_knowledge",
)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)

# ========== 节点函数 ==========
def retrieve(state: RAGState) -> RAGState:
    question = state.get("rewritten_question") or state["question"]
    docs = vectorstore.similarity_search(question, k=TOP_K)
    print(f"[Retrieve] Found {len(docs)} docs")
    return {"documents": docs}

def generate(state: RAGState) -> RAGState:
    question = state["question"]
    documents = state["documents"]
    
    context_parts = []
    sources = []
    for i, doc in enumerate(documents, 1):
        source = f"{doc.metadata.get('source', '未知')}"
        page = doc.metadata.get('page', '?')
        context_parts.append(f"[文档{i}] 来源: {source} 第{page}页\n{doc.page_content}")
        sources.append(f"{source} 第{page}页")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严谨的企业知识助手。请严格根据参考文档回答问题。

规则：
1. 只使用参考文档中的信息，绝对不要编造
2. 如果文档中没有答案，明确回答"根据现有资料，我无法回答这个问题"
3. 引用来源格式：[文档1]、[文档2]
4. 回答要简洁准确

参考文档：
{context}"""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {"generation": response.content, "sources": list(set(sources))}

# ========== 构建 LangGraph ==========
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

# ========== 运行 ==========
def ask(question: str) -> dict:
    result = app.invoke({"question": question})
    return {
        "answer": result["generation"],
        "sources": result["sources"],
        "retrieved_count": len(result["documents"])
    }

if __name__ == "__main__":
    for q in ["公司年假有多少天？", "怎么申请病假？", "加班费怎么算？"]:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        result = ask(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
```

### 6.3 混合检索 + 重排序完整版

```python
# advanced_rag.py - 生产级 RAG（混合检索 + 重排序 + Query 重写）
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langgraph.graph import StateGraph, END
import numpy as np

# ========== 配置 ==========
CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/rag_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RETRIEVE_K = 15
RERANK_TOP_N = 5
FINAL_TOP_K = 5

# ========== State 定义 ==========
class AdvancedRAGState(TypedDict):
    question: str
    rewritten_question: str
    needs_rewrite: bool
    documents: List[Document]
    reranked_documents: List[Document]
    generation: str
    sources: List[str]

# ========== 组件初始化 ==========
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)

vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embedding,
    collection_name="company_knowledge",
)

# 混合检索器
# pgvector 没有 get() 方法，通过检索获取所有文档用于 BM25 索引
_all_docs = vectorstore.similarity_search("", k=10000)
bm25_retriever = BM25Retriever.from_documents(_all_docs, k=RETRIEVE_K)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

# 重排序器
reranker_model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
compressor = CrossEncoderReranker(model=reranker_model, top_n=RERANK_TOP_N)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever,
)

# ========== 节点函数 ==========
def analyze_query(state: AdvancedRAGState) -> AdvancedRAGState:
    question = state["question"]
    needs_rewrite = (
        len(question) < 15 or
        any(w in question for w in ["这个", "那个", "它", "他", "她", "怎么", "咋"])
    )
    return {"needs_rewrite": needs_rewrite}

def rewrite_query(state: AdvancedRAGState) -> AdvancedRAGState:
    if not state["needs_rewrite"]:
        return {"rewritten_question": state["question"]}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个查询优化专家。请把用户问题改写成更正式、明确的查询语句。"),
        ("human", "用户问题：{question}\n\n改写后的查询：")
    ])
    
    chain = prompt | ChatOpenAI(model="gpt-4o-mini")
    rewritten = chain.invoke({"question": state["question"]}).content.strip()
    print(f"[Rewrite] '{state['question']}' -> '{rewritten}'")
    return {"rewritten_question": rewritten}

def hybrid_retrieve(state: AdvancedRAGState) -> AdvancedRAGState:
    question = state["rewritten_question"]
    docs = ensemble_retriever.get_relevant_documents(question)
    print(f"[Hybrid Retrieve] Found {len(docs)} docs")
    return {"documents": docs}

def rerank(state: AdvancedRAGState) -> AdvancedRAGState:
    question = state["rewritten_question"]
    reranked = compression_retriever.get_relevant_documents(question)
    final_docs = reranked[:FINAL_TOP_K]
    print(f"[Rerank] Kept top {len(final_docs)} docs")
    return {"reranked_documents": final_docs}

def generate(state: AdvancedRAGState) -> AdvancedRAGState:
    question = state["question"]
    documents = state["reranked_documents"]
    
    if not documents:
        return {"generation": "抱歉，我在知识库中没有找到相关信息。", "sources": []}
    
    context_parts = []
    sources = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', '未知')
        page = doc.metadata.get('page', '?')
        context_parts.append(f"[文档{i}] 来源: {source} 第{page}页\n{doc.page_content}")
        sources.append(f"{source} 第{page}页")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严谨的企业知识助手。请严格根据参考文档回答问题。

规则：
1. 只使用参考文档中的信息，绝对不要编造
2. 如果文档中没有明确答案，回答"根据现有资料，我无法完整回答这个问题"
3. 引用来源格式：[文档1]、[文档2]
4. 回答要简洁准确

参考文档：
{context}"""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {"generation": response.content, "sources": list(set(sources))}

# ========== 构建 LangGraph ==========
workflow = StateGraph(AdvancedRAGState)
workflow.add_node("analyze", analyze_query)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("retrieve", hybrid_retrieve)
workflow.add_node("rerank", rerank)
workflow.add_node("generate", generate)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ========== 运行接口 ==========
def advanced_ask(question: str) -> dict:
    result = app.invoke({
        "question": question,
        "rewritten_question": "",
        "needs_rewrite": False,
        "documents": [],
        "reranked_documents": [],
        "generation": "",
        "sources": [],
    })
    
    return {
        "question": question,
        "rewritten_question": result["rewritten_question"],
        "answer": result["generation"],
        "sources": result["sources"],
        "retrieved": len(result["documents"]),
        "final_docs": len(result["reranked_documents"]),
    }

if __name__ == "__main__":
    for q in ["年假怎么申请？", "那个请假政策，能详细说说吗？", "加班费和调休怎么换算？"]:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        result = advanced_ask(q)
        print(f"Rewritten: {result['rewritten_question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Retrieved: {result['retrieved']} -> Final: {result['final_docs']}")
```

### LangGraph 可视化

```python
# 把图结构可视化出来
print(app.get_graph().draw_mermaid())
```

生成的 Mermaid 图大致如下：

```mermaid
graph TD
    A[analyze_query] --> B[rewrite_query]
    B --> C[hybrid_retrieve]
    C --> D[rerank]
    D --> E[generate]
    E --> F[END]
```

---

## 附录：RAG 流程全景图

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        企业 RAG 系统全景图                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │                         离线模式（数据准备）                           │   ║
║  │                                                                      │   ║
║  │   文档源                                                              │   ║
║  │   ├── PDF / Word / TXT / Markdown / HTML                            │   ║
║  │   ├── 数据库（MySQL / PG / MongoDB）                                 │   ║
║  │   ├── API（Confluence / Notion / 钉钉文档）                          │   ║
║  │   └── ...                                                            │   ║
║  │       │                                                              │   ║
║  │       ▼                                                              │   ║
║  │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │   ║
║  │   │  加载文档 │───►│  数据清洗 │───►│  文档切分 │───►│  向量化  │     │   ║
║  │   │ Loaders  │    │ Cleaner  │    │ Splitters│    │Embedding │     │   ║
║  │   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │   ║
║  │                        │                    │                       │   ║
║  │        ┌───────────────┘                   │                       │   ║
║  │        ▼                                  ▼                       │   ║
║  │   去HTML/URL/邮箱                    固定大小/递归/语义/父子         │   ║
║  │   去页码/版权/零宽字符               chunk_size + overlap          │   ║
║  │   去重/过滤/质量检查                                                │   ║
║  │   保留表格/代码结构                                                 │   ║
║  │                                                                      │   ║
║  │   向量化: OpenAI / bge / m3e / Jina                                │   ║
║  │   向量库: pgvector / Milvus / Qdrant / Pinecone                      │   ║
║  │   ANN索引: HNSW / IVF / Flat                                        │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                    │                                         ║
║                                    ▼                                         ║
║                          ┌──────────────────┐                               ║
║                          │    向量知识库     │                               ║
║                          │  (Embeddings +   │                               ║
║                          │   Metadata)      │                               ║
║                          └────────┬─────────┘                               ║
║                                   │                                          ║
║  ┌────────────────────────────────┼──────────────────────────────────────┐  ║
║  │                         在线模式（查询回答）                            │  ║
║  │                                │                                       │  ║
║  │   用户提问                     │                                       │  ║
║  │      │                         │                                       │  ║
║  │      ▼                         ▼                                       │  ║
║  │   ┌──────────┐    ┌──────────────────────┐                            │  ║
║  │   │Query重写 │    │      检索层          │                            │  ║
║  │   │          │    │  ┌──────┐ ┌──────┐  │                            │  ║
║  │   ├─指代消解 │    │  │ BM25 │ │向量  │  │                            │  ║
║  │   ├─HyDE     │───►│  │检索  │ │检索  │  │                            │  ║
║  │   ├─查询扩展 │    │  └──┬───┘ └──┬───┘  │                            │  ║
║  │   └─StepBack │    │     └────┬───┘     │                            │  ║
║  │              │    │     混合检索(Ensemble│                            │  ║
║  │              │    │     / RRF融合)       │                            │  ║
║  │              │    └──────────┬───────────┘                            │  ║
║  │              │               ▼                                        │  ║
║  │              │    ┌──────────────────────┐                           │  ║
║  │              │    │     重排序层          │                           │  ║
║  │              │    │  Cohere / bge-reranker│                           │  ║
║  │              │    │  Cross-Encoder       │                           │  ║
║  │              │    └──────────┬───────────┘                           │  ║
║  │              │               ▼                                        │  ║
║  │              │    ┌──────────────────────┐                           │  ║
║  │              └───►│     Prompt拼接        │                           │  ║
║  │                   │  System Prompt +      │                           │  ║
║  │                   │  Context (Top-K docs) │                           │  ║
║  │                   └──────────┬───────────┘                           │  ║
║  │                              ▼                                        │  ║
║  │                   ┌──────────────────────┐                           │  ║
║  │                   │    大模型生成         │                           │  ║
║  │                   │  GPT-4o / Claude /    │                           │  ║
║  │                   │  Qwen / DeepSeek      │                           │  ║
║  │                   └──────────┬───────────┘                           │  ║
║  │                              ▼                                        │  ║
║  │                        最终答案                                        │  ║
║  │                   （带引用溯源）                                        │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │                         架构选型层                                      │  ║
║  │                                                                      │  ║
║  │   传统 RAG ◄──── 简单问答、快速上线                                    │  ║
║  │   GraphRAG ◄─── 跨文档关联、全局总结                                   │  ║
║  │   AgenticRAG ◄─ 复杂推理、多轮检索、工具调用                           │  ║
║  │                                                                      │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │                         运维监控层                                      │  ║
║  │                                                                      │  ║
║  │   成本控制 ◄─── Embedding / LLM / 重排序 / 缓存策略                    │  ║
║  │   稳定性  ◄─── 降级策略 / 超时控制 / 异常处理 / 护栏机制                │  ║
║  │   参数调优 ◄─── Chunk/Overlap / Top-K / Temperature / 相似度阈值       │  ║
║  │   评估监控 ◄─── Recall@K / MRR / Answer Accuracy / Faithfulness        │  ║
║  │                                                                      │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                             ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

> **写在最后**
>
> RAG 不是"把文档丢进向量库就完事"的简单技术，而是一个需要**精细调优的系统工程**。
>
> **数据清洗是整个 Pipeline 中最容易被忽视、但回报最高的环节。** 再强的模型、再精妙的检索策略，如果喂进去的是脏数据，产出也只能是垃圾。业界常说的 "Garbage In, Garbage Out"，在 RAG 中体现得淋漓尽致。
>
> 企业落地的正确姿势：
> 1. **先跑通**：用传统 RAG + 递归切分 + 向量检索，快速验证价值
> 2. **重清洗**：投入时间建立清洗 Pipeline，去重、去噪、保留结构
> 3. **再优化**：加混合检索、加重排序、调 chunk 参数
> 4. **后智能**：根据业务复杂度，引入 Query 重写、AgenticRAG
> 5. **保稳定**：加监控、加降级、控制成本
>
> 记住：**没有最好的 RAG，只有最适合你业务场景的 RAG。数据质量永远是第一优先级。**
