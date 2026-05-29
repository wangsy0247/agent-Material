"""
RAG 路由：支持 SSE 流式查询和文档检索
"""

import json
import time
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from ..schemas import (
    RAGQueryRequest,
    RAGRetrieveRequest,
    RAGRetrieveResponse,
    RAGQueryResponse,
    DocumentSource,
    SSEEvent,
)
from ..auth.dependencies import get_rag_service, get_current_user
from ..auth.schemas import UserInfo
from agent_material.utils.logger import logger

router = APIRouter(prefix="/rag", tags=["RAG 知识库"])


def format_sse_event(event_type: str, content: str | dict | list | None = None) -> str:
    """将数据格式化为 SSE 事件字符串"""
    event = SSEEvent(type=event_type, content=content, timestamp=time.time())
    return f"data: {event.model_dump_json()}\n\n"


def _build_context_with_sources(docs: list) -> tuple[str, list[DocumentSource]]:
    """构建 RAG 上下文和来源信息
    
    Args:
        docs: 检索到的文档列表
        
    Returns:
        (context_str, sources_list): 拼接后的上下文字符串和来源信息列表
    """
    context = ""
    sources = []
    
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        source_name = os.path.basename(source) if isinstance(source, str) else "未知来源"
        score = doc.metadata.get("score", 0.0)
        
        context += f"[参考资料{i}] 来源：{source_name}\n{doc.page_content}\n\n"
        
        sources.append(DocumentSource(
            index=i,
            source=source_name,
            score=round(score, 4),
            snippet=doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
        ))
    
    return context, sources


@router.post("/retrieve", response_model=RAGRetrieveResponse, summary="检索相关文档")
async def rag_retrieve(
    request: RAGRetrieveRequest,
    current_user: UserInfo = Depends(get_current_user),
    rag_service=Depends(get_rag_service),
):
    """
    仅执行文档检索，返回相关文档列表（非流式）
    
    适用于需要在前端展示"参考了哪些资料"的场景
    """
    import time
    start = time.time()
    try:
        logger.info(f"[RAG Retrieve] 用户={current_user.user_id} | 查询: {request.question[:50]}...")
        
        docs = rag_service.retriever_docx(request.question)
        elapsed = time.time() - start
        
        if not docs:
            logger.info(f"[RAG Retrieve] 未检索到文档 | 耗时: {elapsed:.2f}s")
            return RAGRetrieveResponse(
                code=200,
                message="未检索到相关文档",
                data=[],
                total=0,
            )
        
        _, sources = _build_context_with_sources(docs)
        
        logger.info(f"[RAG Retrieve] 检索完成 | 返回 {len(sources)} 条 | 总耗时: {elapsed:.2f}s")
        return RAGRetrieveResponse(
            code=200,
            message=f"检索到 {len(sources)} 条相关文档",
            data=sources,
            total=len(sources),
        )
        
    except Exception as e:
        logger.error(f"[RAG Retrieve] 失败 | 耗时: {time.time() - start:.2f}s | 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", summary="RAG 问答（SSE 流式）")
async def rag_query(
    request: RAGQueryRequest,
    current_user: UserInfo = Depends(get_current_user),
    rag_service=Depends(get_rag_service),
):
    """
    RAG 知识库问答，支持 SSE 流式输出
    
    执行流程：
    1. 检索相关文档（混合检索 + 重排序）
    2. 推送检索到的来源
    3. 流式生成答案
    
    **流式事件类型说明：**
    - `start`: 开始处理
    - `status`: 状态更新（如"正在检索..."）
    - `sources`: 检索到的文档来源
    - `token`: 答案的实时 token
    - `done`: 完成
    - `error`: 错误
    
    **前端消费示例（JavaScript）：**
    ```javascript
    const response = await fetch('/api/v1/rag/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + token
        },
        body: JSON.stringify({question: '石墨烯的性质', stream: true})
    });
    const reader = response.body.getReader();
    // 逐块读取并解析 SSE 事件...
    ```
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    logger.info(f"[RAG Query] 用户={current_user.user_id} | 查询: {request.question[:50]}...")
    
    # 非流式模式
    if not request.stream:
        try:
            answer = rag_service.get_summary(request.question)
            
            # 重新检索获取来源信息
            docs = rag_service.retriever_docx(request.question)
            _, sources = _build_context_with_sources(docs) if docs else ([], [])
            
            return RAGQueryResponse(answer=answer, sources=sources)
            
        except Exception as e:
            logger.error(f"[RAG Query] 非流式调用失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # 流式模式：SSE
    async def event_stream():
        import time
        start = time.time()
        logger.info(f"[RAG SSE] 用户={current_user.user_id} | question={request.question[:50]}...")
        
        yield format_sse_event("start")
        
        try:
            # 1. 检索阶段
            yield format_sse_event("status", "正在检索相关文献...")
            
            t0 = time.time()
            docs = rag_service.retriever_docx(request.question)
            retrieve_time = time.time() - t0
            logger.info(f"[RAG SSE] 检索完成 | 文档数={len(docs)} | 耗时={retrieve_time:.2f}s")
            
            if not docs:
                yield format_sse_event(
                    "status",
                    "未检索到相关文档，将基于通用知识回答"
                )
                yield format_sse_event(
                    "token",
                    "根据现有资料无法回答该问题。"
                )
                yield format_sse_event("done")
                return
            
            # 2. 推送来源
            context, sources = _build_context_with_sources(docs)
            yield format_sse_event("sources", [s.model_dump() for s in sources])
            
            # 3. 生成阶段
            yield format_sse_event("status", "正在生成答案...")
            
            t0 = time.time()
            for chunk in rag_service.chain.stream(
                {"input": request.question, "context": context}
            ):
                if chunk:
                    yield format_sse_event("token", chunk)
            gen_time = time.time() - t0
            
            total_time = time.time() - start
            yield format_sse_event("done")
            logger.info(
                f"[RAG SSE] 请求完成 | user={current_user.user_id} | "
                f"sources={len(sources)} | 检索={retrieve_time:.2f}s | "
                f"生成={gen_time:.2f}s | 总耗时={total_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"[RAG SSE] 请求失败 | 耗时={time.time() - start:.2f}s | 错误: {e}")
            yield format_sse_event("error", {"message": str(e), "type": type(e).__name__})
            yield format_sse_event("done")
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
