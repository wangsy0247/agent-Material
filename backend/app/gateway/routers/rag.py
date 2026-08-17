"""
RAG 路由（REST API 直连版）

为前端 RAG 管理页面提供直接的 REST API 调用能力，
绕过 MCP 协议，直接调用 rag_engine 内部模块。
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import get_current_admin, get_current_user
from ..auth.schemas import UserInfo
from ..schemas import BaseResponse
from ..sse import HEARTBEAT_INTERVAL, format_heartbeat, format_sse_event as _sse

if TYPE_CHECKING:
    pass

logger = logging.getLogger("rag_router")

router = APIRouter(prefix="/rag", tags=["RAG 知识库"])

_MAX_UPLOAD_FILES = 10
_MAX_UPLOAD_SIZE = 50 * 1024 * 1024
_UPLOAD_CHUNK_SIZE = 1024 * 1024
_upload_commit_lock = asyncio.Lock()


def _internal_error(message: str, error: Exception) -> HTTPException:
    """记录内部异常，但不向 API 客户端暴露路径、SQL 或堆栈。"""
    error_id = uuid.uuid4().hex
    logger.exception("%s error_id=%s", message, error_id, exc_info=error)
    return HTTPException(
        status_code=500,
        detail=f"{message}，错误编号: {error_id}",
    )


def _single_retrieval_enabled() -> bool:
    """single_retrieval_sse 开关：开启时 /rag/query 与 /rag/search 走单次检索新路径。"""
    try:
        from agent_material.utils import config_handler

        return bool(
            config_handler.vector_store_config.get("feature_flags", {}).get(
                "single_retrieval_sse", False
            )
        )
    except Exception:
        return False


def _feature_flag(name: str, default: bool = False) -> bool:
    """读取 vector_store_config.feature_flags 开关（调用时取，便于测试替换）。"""
    try:
        from agent_material.utils import config_handler

        return bool(
            config_handler.vector_store_config.get("feature_flags", {}).get(
                name, default
            )
        )
    except Exception:
        return default

# ------------------------------------------------------------------
# 延迟初始化（避免模块导入时触发重量级初始化）
# ------------------------------------------------------------------
_VectorStore = None
_RagService = None


def _get_vector_store():
    global _VectorStore
    if _VectorStore is None:
        import os
        import faulthandler
        faulthandler.enable()

        # 诊断：检测 libffi 冲突（已知 WSL+Anaconda 会导致 psycopg segfault）
        ld_preload = os.environ.get("LD_PRELOAD", "")
        if "libffi" not in ld_preload:
            import logging
            logging.warning(
                "[RAG] LD_PRELOAD 未设置 libffi 系统库。"
                "如果后续出现 Segmentation fault，请使用 scripts/start_server.sh 启动，"
                "或手动执行: export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7"
            )

        from agent_material.rag_engine.vector_store import get_vector_store

        # 进程内统一单例（WRK-001）：与 Agent Tool / RagService 共享同一实例
        _VectorStore = get_vector_store()
    return _VectorStore


def _get_rag_service():
    global _RagService
    if _RagService is None:
        from agent_material.rag_engine.rag_service import RagService

        _RagService = RagService()
    return _RagService


# ------------------------------------------------------------------
# 请求模型
# ------------------------------------------------------------------


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="检索查询")


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="用户问题")


class RAGAddDocumentsRequest(BaseModel):
    documents: list[str] = Field(..., min_length=1, description="文本内容列表")
    sources: list[str] | None = Field(default=None, description="来源标识列表")


class RAGDeleteRequest(BaseModel):
    source: str = Field(..., min_length=1, description="来源路径或标识")


class RAGBatchDeleteRequest(BaseModel):
    sources: list[str] = Field(..., min_length=1, description="要删除的来源路径列表")


class RAGAddChunksRequest(BaseModel):
    chunks: list[str] = Field(..., min_length=1, description="已切分好的文本片段列表")
    sources: list[str] | None = Field(default=None, description="来源标识列表，与 chunks 一一对应；不传则自动分配")
    metadatas: list[dict] | None = Field(default=None, description="额外元数据列表，与 chunks 一一对应")


# ------------------------------------------------------------------
# 路由
# ------------------------------------------------------------------


@router.get("/stats", summary="获取 RAG 统计信息", response_model=BaseResponse)
async def rag_stats(current_user: UserInfo = Depends(get_current_user)):
    """获取向量库统计信息：文档总数、配置参数等。"""
    try:
        store = _get_vector_store()
        from agent_material.utils.config_handler import vector_store_config

        def _mask_conn_str(conn_str: str) -> str:
            try:
                if "@" in conn_str:
                    prefix, rest = conn_str.split("@", 1)
                    if ":" in prefix:
                        scheme_user, _ = prefix.rsplit(":", 1)
                        return f"{scheme_user}:****@{rest}"
                return conn_str
            except Exception:
                return "***masked***"

        stats = {
            "document_count": store.get_document_count(),
            "collection_name": vector_store_config.get("character_name", "agent"),
            "connection_string_masked": _mask_conn_str(
                vector_store_config.get("connection_string", "")
            ),
            "splitter_type": vector_store_config.get("splitter_type", "recursive"),
            "chunk_size": vector_store_config.get("chunk_size"),
            "chunk_overlap": vector_store_config.get("chunk_overlap"),
            "vector_weight": vector_store_config.get("vector_weight", 0.7),
            "bm25_weight": vector_store_config.get("bm25_weight", 0.3),
            "enable_reranker": vector_store_config.get("enable_reranker", True),
            "top_n": vector_store_config.get("top_n", 3),
            "score_threshold": vector_store_config.get("score_threshold", 0.3),
            "data_directory": vector_store_config.get("data_directory", "data/"),
        }
        return BaseResponse(code=200, message="success", data=stats)
    except Exception as e:
        raise _internal_error("获取统计信息失败", e)


@router.post("/search", summary="检索相关文档", response_model=BaseResponse)
async def rag_search(
    request: RAGSearchRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """根据查询内容从向量数据库检索最相关的文档片段。

    single_retrieval_sse 开关开启时走 service.prepare 的请求级检索管道
    （不生成回答），返回 resolved_parents（父块级，与 /rag/query 的生成上下文
    同源；无父块时回退 reranked_children 子块），并附带溯源字段与降级事件。
    """
    try:
        service = _get_rag_service()

        # ── 新路径：请求级检索管道（单次检索，含全局 Rerank 与父块回溯）──
        if _single_retrieval_enabled():
            ctx = await asyncio.to_thread(service.prepare, request.query)
            docs = ctx.resolved_parents or ctx.reranked_children
            if not docs:
                return BaseResponse(code=200, message="未检索到相关文档", data=[])

            results = []
            for i, doc in enumerate(docs, start=1):
                m = doc.metadata or {}
                source = m.get("source", "未知来源")
                results.append(
                    {
                        "index": i,
                        "rank": i,
                        "chunk_id": m.get("chunk_id"),
                        "parent_id": m.get("parent_id"),
                        "document_id": m.get("document_id"),
                        "version_id": m.get("version_id"),
                        "source": source,
                        "source_filename": m.get("source_filename")
                        or (os.path.basename(source) if isinstance(source, str) else None),
                        "page_start": m.get("page_start"),
                        "page_end": m.get("page_end"),
                        "page_unknown": bool(m.get("page_unknown", False)),
                        "heading_path": m.get("heading_path"),
                        "rerank_score": m.get("rerank_score"),
                        "score": m.get(
                            "rerank_score", m.get("rrf_score", m.get("score", 0.0))
                        ),
                        "snippet": doc.page_content[:300] + "..."
                        if len(doc.page_content) > 300
                        else doc.page_content,
                        "metadata": m,
                    }
                )
            resp = BaseResponse(
                code=200, message="success", data=results, total=len(results)
            )
            if ctx.degradation_events:
                resp.degradation_events = ctx.degradation_events
            logger.info("rag_search_summary %s", ctx.summary())
            return resp

        # ── 旧路径（回滚兼容）──
        docs = service.retriever_docx(request.query)
        if not docs:
            return BaseResponse(code=200, message="未检索到相关文档", data=[])

        results = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "未知来源")
            score = doc.metadata.get("score", 0.0)
            results.append(
                {
                    "index": i,
                    "source": source,
                    "score": score,
                    "snippet": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return BaseResponse(code=200, message="success", data=results, total=len(results))
    except Exception as e:
        raise _internal_error("检索失败", e)


@router.post("/query", summary="RAG 问答（SSE 流式）")
async def rag_query(
    request: RAGQueryRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    RAG 问答：检索相关文档并流式生成回答。

    SSE 事件类型：
    - `start`: 开始处理
    - `status`: 阶段切换（{"phase": "retrieving" | "generating"}）
    - `sources`: 检索到的引用来源
    - `token`: 大模型实时生成的 token（data.index 从 1 递增）
    - `done`: 完成
    - `error`: 错误
    - `cancelled`: 客户端断连/中止导致的取消（终态）
    """

    async def event_generator():
        # 帧 id 计数器：run 内单调递增，供前端记录 lastEventId
        event_id = 0
        # token 序号：run 内从 1 递增，供前端乱序/丢帧检测
        token_index = 0

        def next_sse(event_type: str, content=None, index=None) -> str:
            nonlocal event_id
            event_id += 1
            return _sse(event_type, content, event_id, index=index)

        def token_sse(text: str) -> str:
            nonlocal token_index
            token_index += 1
            return next_sse("token", text, index=token_index)

        ctx = None
        # 客户端断连/取消标志（Spec V2 12.3）：async 侧捕获取消后 set，
        # 消费线程每轮检查并尽快 break + 关闭底层 stream，终止 LLM 生成
        cancel_event = threading.Event()
        try:
            # RagService 首次初始化是重量级同步操作，放线程避免阻塞 event loop
            service = await asyncio.to_thread(_get_rag_service)
            use_new = _single_retrieval_enabled()

            # 1. 阶段事件：开始检索
            yield next_sse("status", {"phase": "retrieving"})

            if use_new:
                # 新路径（ONL-002）：全程只调一次 service.prepare，
                # sources 与 LLM 上下文来自同一份冻结上下文
                ctx = await asyncio.to_thread(service.prepare, request.query)
                # chat/irrelevant 无检索结果；rag 空结果发空 sources 数组
                no_context = ctx.route_decision == "rag" and not ctx.final_context
                sources = ctx.sources
            else:
                # 旧路径（回滚兼容）：先检索一次构造 sources
                docs = await asyncio.to_thread(service.retriever_docx, request.query)
                no_context = not docs
                sources = []
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "未知来源")
                    score = doc.metadata.get("score", 0.0)
                    sources.append(
                        {
                            "index": i,
                            "source": source,
                            "score": score,
                            "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        }
                    )

            # 2. 发送 start 事件
            yield next_sse("start", {"query": request.query})

            # 3. 发送 sources 事件
            yield next_sse("sources", sources)

            if no_context:
                yield token_sse("根据现有资料无法回答该问题。")
                yield next_sse("done")
                if ctx is not None:
                    logger.info("rag_query_summary %s", ctx.summary())
                return

            # 4. 阶段事件：开始生成
            yield next_sse("status", {"phase": "generating"})

            # 5. 流式生成：stream_answer/get_summary_stream 是同步生成器函数（惰性），
            # 整个迭代都在线程中执行，经 asyncio.Queue 桥接回 async 侧逐 chunk 消费，
            # 避免普通 for 迭代独占 event loop。
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = None  # 结束哨兵
            # 取消标志见生成器顶部 cancel_event（线程每轮检查）

            def _consume_stream():
                stream = None
                try:
                    if use_new:
                        # 不再检索，仅消费冻结上下文
                        stream = service.stream_answer(ctx)
                    else:
                        stream = service.get_summary_stream(request.query)
                    for chunk in stream:
                        if cancel_event.is_set():
                            logger.info("[RAG] 检测到取消标志，终止后台 LLM 流式生成")
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as exc:  # 异常经队列传回 async 侧 re-raise
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    # 尽快关闭底层 stream/response（能关则关），释放 LLM 连接
                    if stream is not None and hasattr(stream, "close"):
                        try:
                            stream.close()
                        except Exception:
                            pass
                    # 无论正常/异常/取消都放哨兵，保证 async 侧不泄漏等待
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

            consume_task = asyncio.create_task(asyncio.to_thread(_consume_stream))
            try:
                while True:
                    try:
                        # 超时发注释心跳帧：长时间无 token 时防代理/浏览器判死断连
                        item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL)
                    except TimeoutError:
                        yield format_heartbeat()
                        continue
                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield token_sse(item)
            finally:
                # 先置取消标志（取消路径下让消费线程尽快退出），再收尾线程任务
                # （正常路径线程已结束，set 幂等无害）
                cancel_event.set()
                await asyncio.shield(asyncio.gather(consume_task, return_exceptions=True))

            yield next_sse("done")
            if ctx is not None:
                logger.info("rag_query_summary %s", ctx.summary())
        except asyncio.CancelledError:
            # 客户端断连/中止：先置取消标志终止后台 LLM 生成线程（若已进入
            # 生成阶段），再尽力向当前连接发 cancelled 帧（连接已断开时
            # 发送会失败，静默忽略），随后原样抛出结束流
            cancel_event.set()
            try:
                yield next_sse("cancelled", {"message": "run cancelled"})
            except BaseException:
                pass
            raise
        except Exception as e:
            error_id = uuid.uuid4().hex
            logger.exception("RAG 流式问答失败 error_id=%s", error_id, exc_info=e)
            # error 为终态：发完 error 帧直接结束，不再依赖连接关闭收尾
            yield next_sse(
                "error",
                {"message": "检索或生成失败", "error_id": error_id},
            )
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 nginx 代理缓冲，保证实时推送
        },
    )


@router.post("/load-documents", summary="扫描并加载文档", response_model=BaseResponse)
async def rag_load_documents(current_user: UserInfo = Depends(get_current_admin)):
    """扫描配置的 data/ 目录，为新增/修改/消失的文档创建索引任务（异步执行）。

    Spec V2 11：返回"任务已接收"语义 —— data 含 task_ids，
    最终索引结果通过 GET /rag/tasks/{task_id} 查询。
    """
    try:
        store = _get_vector_store()
        task_ids = store.load_document() or []
        return BaseResponse(
            code=200,
            message=f"已接收 {len(task_ids)} 个索引任务，后台异步执行，"
                    "结果请通过 /rag/tasks/{task_id} 查询",
            data={
                "task_ids": task_ids,
                "accepted": len(task_ids),
            },
        )
    except Exception as e:
        raise _internal_error("加载文档失败", e)


@router.get("/tasks/{task_id}", summary="查询索引任务状态", response_model=BaseResponse)
async def rag_task_status(
    task_id: int,
    current_user: UserInfo = Depends(get_current_admin),
):
    """查询 PostgreSQL 控制面的后台索引任务状态。

    返回任务类型、状态、当前阶段、重试次数、失败原因等（可运维性要求）。
    """
    try:
        store = _get_vector_store()
        status = store.get_task_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        return BaseResponse(code=200, message="success", data=status)
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error("查询任务状态失败", e)


@router.post("/tasks/{task_id}/retry", summary="重试失败的索引任务", response_model=BaseResponse)
async def rag_retry_task(
    task_id: int,
    current_user: UserInfo = Depends(get_current_admin),
):
    """显式重置 failed 任务的重试预算并重新入队；非 failed 状态返回 409。"""
    try:
        task = _get_vector_store().retry_failed_task(task_id)
        return BaseResponse(code=200, message="任务已重新入队", data=task)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error("重试任务失败", e)


@router.get("/kb-state", summary="查询知识库 revision 状态", response_model=BaseResponse)
async def rag_kb_state(current_user: UserInfo = Depends(get_current_admin)):
    """返回 rag_kb_state 的 active_revision / bm25_revision。"""
    try:
        store = _get_vector_store()
        state = store.get_kb_state()
        return BaseResponse(code=200, message="success", data=state)
    except Exception as e:
        raise _internal_error("查询知识库状态失败", e)


@router.get("/consistency-check", summary="在线数据一致性检测", response_model=BaseResponse)
async def rag_consistency_check(current_user: UserInfo = Depends(get_current_admin)):
    """执行一致性检测 SQL（Spec 15 可运维性），结果均应为 0：

    - multiple_active_version_documents：同一 document_id 多个 active version 的文档数
    - orphan_active_children：active child 找不到 active parent 的子块数
    """
    try:
        store = _get_vector_store()
        result = await asyncio.to_thread(store.run_consistency_check)
        return BaseResponse(code=200, message="success", data=result)
    except Exception as e:
        raise _internal_error("一致性检测失败", e)


@router.post("/add-documents", summary="手动添加文档", response_model=BaseResponse)
async def rag_add_documents(
    request: RAGAddDocumentsRequest,
    current_user: UserInfo = Depends(get_current_admin),
):
    """手动向向量库添加文本内容。

    文本封装为 ingest_text 任务进入统一版本化索引队列
    （切分 → staging → validate → activate），返回 task_ids，
    最终结果通过 /rag/tasks/{task_id} 查询。
    """
    try:
        sources = request.sources or [f"manual_{i}" for i in range(len(request.documents))]
        if len(sources) != len(request.documents):
            raise HTTPException(status_code=400, detail="sources 长度必须与 documents 一致")

        store = _get_vector_store()

        task_ids = []
        for text, src in zip(request.documents, sources):
            task_id = store.enqueue_ingest_text(text, src)
            if task_id is not None:
                task_ids.append(task_id)
        return BaseResponse(
            code=200,
            message=f"已接收 {len(task_ids)} 个文本入库任务，后台异步执行，"
                    "结果请通过 /rag/tasks/{task_id} 查询",
            data={
                "task_ids": task_ids,
                "accepted": len(task_ids),
                "deduplicated": len(request.documents) - len(task_ids),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error("添加文档失败", e)


@router.post("/upload", summary="上传文件加入知识库", response_model=BaseResponse)
async def rag_upload_files(
    files: list[UploadFile] = File(...),
    current_user: UserInfo = Depends(get_current_admin),
):
    """上传文件（PDF 等 allow_type 白名单类型）加入知识库。

    文件保存到配置的 document_directory（data/）后封装 ingest_file 任务
    进入统一版本化索引队列（解析 → 切分 → staging → validate → activate），
    与目录扫描同一 canonical/dedup 契约：同名同内容幂等跳过，同名不同
    内容产生新版本并原子替换旧版本。返回逐文件明细与 task_ids。
    """
    from agent_material.utils.config_handler import vector_store_config
    from agent_material.utils.path_tools import get_abs_path

    if len(files) > _MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"单次最多上传 {_MAX_UPLOAD_FILES} 个文件",
        )

    allow_types = {
        ext.lower() for ext in vector_store_config.get("allow_type", [".pdf"])
    }
    doc_dir = Path(
        get_abs_path(vector_store_config.get("document_directory", "data"))
    ).resolve()
    doc_dir.mkdir(parents=True, exist_ok=True)

    store = _get_vector_store()
    results = []
    for uf in files:
        # 同时处理 POSIX 与 Windows 客户端传来的路径分隔符。
        filename = (uf.filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not filename or filename.startswith(".") or ".." in filename:
            results.append(
                {"filename": uf.filename, "status": "rejected",
                 "reason": "文件名非法"}
            )
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allow_types:
            results.append(
                {"filename": filename, "status": "rejected",
                 "reason": f"不支持的文件类型 {ext}，允许: {sorted(allow_types)}"}
            )
            continue

        temp_path: Path | None = None
        backup_path: Path | None = None
        try:
            size = 0
            header = bytearray()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".upload-",
                suffix=".tmp",
                dir=doc_dir,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                while chunk := await uf.read(_UPLOAD_CHUNK_SIZE):
                    size += len(chunk)
                    if size > _MAX_UPLOAD_SIZE:
                        break
                    if len(header) < 1024:
                        header.extend(chunk[: 1024 - len(header)])
                    temp_file.write(chunk)

            if size == 0:
                results.append(
                    {"filename": filename, "status": "rejected", "reason": "空文件"}
                )
                continue
            if size > _MAX_UPLOAD_SIZE:
                results.append(
                    {
                        "filename": filename,
                        "status": "rejected",
                        "reason": f"文件超过 {_MAX_UPLOAD_SIZE // (1024 * 1024)}MB 上限",
                    }
                )
                continue
            if ext == ".pdf" and b"%PDF-" not in bytes(header):
                results.append(
                    {
                        "filename": filename,
                        "status": "rejected",
                        "reason": "文件内容不是有效的 PDF",
                    }
                )
                continue

            save_path = (doc_dir / filename).resolve()
            if save_path.parent != doc_dir:
                results.append(
                    {"filename": filename, "status": "rejected", "reason": "文件名非法"}
                )
                continue

            # 替换与入队视为一个临界区；入队失败时恢复原文件。
            async with _upload_commit_lock:
                if save_path.exists():
                    backup_path = doc_dir / f".{filename}.{uuid.uuid4().hex}.bak"
                    os.replace(save_path, backup_path)
                try:
                    os.replace(temp_path, save_path)
                    temp_path = None
                    task_id = await asyncio.to_thread(
                        store.enqueue_ingest_file, str(save_path)
                    )
                except Exception:
                    save_path.unlink(missing_ok=True)
                    if backup_path is not None and backup_path.exists():
                        os.replace(backup_path, save_path)
                        backup_path = None
                    raise
                else:
                    if backup_path is not None:
                        backup_path.unlink(missing_ok=True)
                        backup_path = None

            results.append(
                {
                    "filename": filename,
                    "status": "accepted" if task_id is not None else "deduplicated",
                    "task_id": task_id,
                }
            )
        except Exception:
            logger.exception("文件上传入队失败: %s", filename)
            results.append(
                {
                    "filename": filename,
                    "status": "failed",
                    "reason": "文件保存或索引任务创建失败",
                }
            )
        finally:
            await uf.close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if backup_path is not None:
                logger.error("上传回滚未完成，保留备份文件: %s", backup_path)

    accepted = [r["task_id"] for r in results if r.get("task_id") is not None]
    return BaseResponse(
        code=200,
        message=f"已接收 {len(accepted)} 个索引任务，后台异步执行，"
                "结果请通过 /rag/tasks/{task_id} 查询",
        data={"files": results, "task_ids": accepted, "accepted": len(accepted)},
    )


@router.post("/add-chunks", summary="已移除：禁止直接写入 chunks", response_model=BaseResponse)
async def rag_add_chunks(
    request: RAGAddChunksRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """该入口无法满足父子/版本契约，永久返回 HTTP 410。"""
    raise HTTPException(
        status_code=410,
        detail="直接写入 chunks 的接口已移除，请使用 /rag/add-documents",
    )


@router.get("/sources", summary="获取所有文献来源", response_model=BaseResponse)
async def rag_sources(current_user: UserInfo = Depends(get_current_user)):
    """获取向量库中所有唯一的文献来源列表（含片段数）。"""
    try:
        store = _get_vector_store()
        sources = store.get_all_sources()
        return BaseResponse(
            code=200,
            message="success",
            data=sources,
            total=len(sources),
        )
    except Exception as e:
        raise _internal_error("获取文献列表失败", e)


@router.post("/delete-by-source", summary="按来源删除文档", response_model=BaseResponse)
async def rag_delete_by_source(
    request: RAGDeleteRequest,
    current_user: UserInfo = Depends(get_current_admin),
):
    """根据来源路径删除向量库中的文档片段。

    创建 delete_document 任务并返回 task_id；实际删除由后台 Worker
    异步执行，source 不存在时任务幂等完成。
    """
    try:
        store = _get_vector_store()
        result = store.enqueue_delete_by_source(request.source)
        return BaseResponse(
            code=200,
            message=f"已接收来源为 '{request.source}' 的删除任务，后台异步执行，"
                    "结果请通过 /rag/tasks/{task_id} 查询",
            data=result,
        )
    except Exception as e:
        raise _internal_error("删除文档失败", e)


@router.post("/delete-sources", summary="批量删除文献", response_model=BaseResponse)
async def rag_delete_sources(
    request: RAGBatchDeleteRequest,
    current_user: UserInfo = Depends(get_current_admin),
):
    """批量删除向量库中的文献。

    每个 source 独立创建 delete_document 任务，返回 accepted、
    deduplicated 和 failed 明细，不同步执行物理删除。
    """
    try:
        store = _get_vector_store()
        results = store.enqueue_batch_delete(request.sources)
        accepted = sum(1 for r in results if r["status"] == "accepted")
        deduplicated = sum(1 for r in results if r["status"] == "deduplicated")
        failed = [r for r in results if r["status"] == "failed"]
        msg = f"已接收 {accepted} 个删除任务"
        if deduplicated:
            msg += f"，{deduplicated} 个与进行中任务重复（去重）"
        if failed:
            msg += f"，{len(failed)} 个创建失败"
        msg += "，后台异步执行，结果请通过 /rag/tasks/{task_id} 查询"
        return BaseResponse(
            code=200,
            message=msg,
            data={
                "results": results,
                "accepted": accepted,
                "deduplicated": deduplicated,
                "failed": failed,
            },
        )
    except Exception as e:
        raise _internal_error("批量删除失败", e)
