/**
 * RAG 知识库管理页面前端逻辑
 * 功能：统计概览、检索测试、RAG 问答、文档管理
 */

const API_BASE = '/api/v1';

// 状态
const state = {
    token: localStorage.getItem('agent_token') || null,
    refreshToken: localStorage.getItem('agent_refresh_token') || null,
    currentTab: 'overview',
    isQuerying: false,
    abortController: null,
    selectedDocs: new Set(),
    allDocs: [],
};

// DOM 缓存
const $ = (id) => document.getElementById(id);

// ============================================
// 认证 & API
// ============================================

// 登录失效：提示并跳回登录页（稍作延迟让用户看到提示）
function relogin() {
    Toast.error('登录已过期，请重新登录');
    setTimeout(() => { window.location.href = '/'; }, 1200);
}

// 认证上下文（公共逻辑见 common.js AppCommon.createAuthContext）：
// token 存取注入本页 state，刷新失败/最终 401 走 relogin()
const authCtx = AppCommon.createAuthContext({
    getToken: () => state.token,
    getRefreshToken: () => state.refreshToken,
    setTokens(access, refresh) {
        state.token = access;
        state.refreshToken = refresh;
        localStorage.setItem('agent_token', access);
        localStorage.setItem('agent_refresh_token', refresh);
    },
    onAuthFail: () => relogin(),
});

async function apiFetch(path, options = {}) {
    await authCtx.ensureTokenValid();
    const headers = { 'Content-Type': 'application/json' };
    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
    const res = await fetch(`${API_BASE}${path}`, { headers, ...options });
    if (res.status === 401) {
        if (state.refreshToken) {
            try {
                await authCtx.refresh();
                return apiFetch(path, options);
            } catch {
                relogin();
                throw AppCommon.friendlyError('登录已过期');
            }
        }
        relogin();
        throw AppCommon.friendlyError('登录已过期');
    }
    if (!res.ok) {
        const err = await res.text();
        const error = new Error(err);
        error.status = res.status; // 带状态码，供 formatError 友好化
        throw error;
    }
    return res.json();
}

// ============================================
// 工具函数
// ============================================

function setLoading(el, text = '加载中...') {
    el.innerHTML = `<div class="rag-loading"><div class="rag-spinner"></div><span>${text}</span></div>`;
}

function setError(el, text) {
    el.innerHTML = `<div class="rag-error">❌ ${AppCommon.escapeHtml(text)}</div>`;
}

function setSuccess(el, text) {
    el.innerHTML = `<div class="rag-success">✅ ${AppCommon.escapeHtml(text)}</div>`;
}

// ============================================
// Tab 切换
// ============================================

function switchTab(tab) {
    state.currentTab = tab;
    document.querySelectorAll('.rag-nav-btn').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    document.querySelectorAll('.rag-section').forEach((sec) => {
        sec.classList.toggle('active', sec.id === `tab-${tab}`);
    });
    if (tab === 'overview') loadStats();
    if (tab === 'docs') loadDocsList();
}

// ============================================
// 概览：统计信息
// ============================================

async function loadStats() {
    const grid = $('statsGrid');
    setLoading(grid, '加载统计中...');
    try {
        const res = await apiFetch('/rag/stats');
        const d = res.data || {};
        grid.innerHTML = `
            <div class="rag-stat-item">
                <div class="rag-stat-label">向量库文档数</div>
                <div class="rag-stat-value">${AppCommon.escapeHtml(String(d.document_count ?? '-'))}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">Collection</div>
                <div class="rag-stat-value" style="font-size:14px;">${AppCommon.escapeHtml(d.collection_name || '-')}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">Chunk Size</div>
                <div class="rag-stat-value">${AppCommon.escapeHtml(String(d.chunk_size ?? '-'))}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">重排序</div>
                <div class="rag-stat-value">${d.enable_reranker ? '启用' : '关闭'}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">Top N</div>
                <div class="rag-stat-value">${AppCommon.escapeHtml(String(d.top_n ?? '-'))}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">相似度阈值</div>
                <div class="rag-stat-value">${AppCommon.escapeHtml(String(d.score_threshold ?? '-'))}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">向量权重</div>
                <div class="rag-stat-value">${AppCommon.escapeHtml(String(d.vector_weight ?? '-'))}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">BM25 权重</div>
                <div class="rag-stat-value">${AppCommon.escapeHtml(String(d.bm25_weight ?? '-'))}</div>
            </div>
        `;
        // 配置表
        const tbody = $('configTable').querySelector('tbody');
        const configItems = [
            ['Collection Name', d.collection_name],
            ['Data Directory', d.data_directory],
            ['Connection', d.connection_string_masked],
            ['Splitter Type', d.splitter_type],
            ['Chunk Size', d.chunk_size],
            ['Chunk Overlap', d.chunk_overlap],
            ['Vector Weight', d.vector_weight],
            ['BM25 Weight', d.bm25_weight],
            ['Enable Reranker', d.enable_reranker],
            ['Top N', d.top_n],
            ['Score Threshold', d.score_threshold],
        ];
        tbody.innerHTML = configItems.map(([k, v]) => `
            <tr><td>${AppCommon.escapeHtml(k)}</td><td>${AppCommon.escapeHtml(String(v ?? '-'))}</td></tr>
        `).join('');
    } catch (e) {
        setError(grid, AppCommon.formatError(e));
    }
}

// ============================================
// 检索
// ============================================

async function doSearch() {
    const input = $('searchInput');
    const query = input.value.trim();
    if (!query) return;
    const status = $('searchStatus');
    const results = $('searchResults');
    setLoading(status, '检索中...');
    results.innerHTML = '';
    try {
        const res = await apiFetch('/rag/search', {
            method: 'POST',
            body: JSON.stringify({ query }),
        });
        status.innerHTML = '';
        const data = res.data || [];
        if (!data.length) {
            results.innerHTML = '<div class="rag-empty">未检索到相关文档</div>';
            return;
        }
        results.innerHTML = data.map((item) => `
            <div class="rag-result-item">
                <div class="rag-result-header">
                    <span class="rag-result-source">#${AppCommon.escapeHtml(String(item.index))} ${AppCommon.escapeHtml(item.source)}</span>
                    <span class="rag-result-score">相似度 ${((item.score || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="rag-result-snippet">${AppCommon.escapeHtml(item.snippet)}</div>
            </div>
        `).join('');
    } catch (e) {
        status.innerHTML = '';
        setError(results, AppCommon.formatError(e));
    }
}

// ============================================
// 问答（SSE 流式）
// ============================================

async function doQuery() {
    const input = $('queryInput');
    const query = input.value.trim();
    if (!query || state.isQuerying) return;

    const status = $('queryStatus');
    const resultEl = $('queryResult');
    state.isQuerying = true;
    state.abortController = new AbortController();
    setLoading(status, '检索并生成中...');

    resultEl.innerHTML = `
        <div class="rag-answer" id="queryAnswer"></div>
        <div class="rag-sources-inline" id="querySources" style="display:none;">
            <div class="rag-sources-inline-title">📚 引用来源</div>
            <div id="querySourcesList"></div>
        </div>
    `;
    const answerEl = $('queryAnswer');
    const sourcesWrap = $('querySources');
    const sourcesList = $('querySourcesList');

    const { handler, typewriter } = createQueryEventHandler(answerEl, sourcesWrap, sourcesList, status);

    try {
        await authCtx.sseConnect({
            url: `${API_BASE}/rag/query`,
            body: { query },
            onEvent: handler,
            signal: state.abortController.signal,
        });
    } catch (e) {
        status.innerHTML = '';
        if (e.name === 'AbortError') {
            // 用户中止：保留已生成内容 + "已停止"标记
            typewriter.finish();
            appendStoppedMark(answerEl);
        } else {
            setError(resultEl, AppCommon.formatError(e));
        }
    } finally {
        state.isQuerying = false;
        state.abortController = null;
    }
}

// 在回答区尾部追加"已停止"标记（cancelled / 用户中止共用）
function appendStoppedMark(answerEl) {
    if (answerEl.querySelector('.rag-stopped-mark')) return;
    const mark = document.createElement('p');
    mark.className = 'rag-stopped-mark';
    mark.textContent = '⏹️ 已停止';
    answerEl.appendChild(mark);
}

// 构建问答事件处理器：fullText 累积收在闭包内
// 流式期间 Typewriter 增量 append 文本节点（不再逐 token 全量重写 innerHTML）
function createQueryEventHandler(answerEl, sourcesWrap, sourcesList, statusEl) {
    let fullText = '';
    let streamSpan = null;
    let streamNode = null;

    const PHASE_LABELS = { retrieving: '🔍 检索中…', generating: '✍️ 生成中…' };
    let phaseShown = false;
    function setPhase(phase) {
        if (phase) phaseShown = true;
        if (statusEl) statusEl.textContent = PHASE_LABELS[phase] || phase || '';
    }

    const typewriter = StreamClient.Typewriter((delta) => {
        if (!streamNode) {
            answerEl.innerHTML = '';
            streamSpan = document.createElement('span');
            streamSpan.className = 'rag-stream-text';
            streamNode = document.createTextNode('');
            streamSpan.appendChild(streamNode);
            answerEl.appendChild(streamSpan);
        }
        streamNode.appendData(delta);
    });

    function handler(event) {
        if (!event) return;
        switch (event.type) {
            case 'start':
                break;
            case 'status': {
                // 阶段状态（检索中/生成中）
                if (event.content?.phase) setPhase(event.content.phase);
                break;
            }
            case 'sources': {
                const sources = Array.isArray(event.content) ? event.content : [];
                if (sources.length) {
                    sourcesWrap.style.display = 'block';
                    sourcesList.innerHTML = sources.map((s) => `
                        <div class="rag-result-item" style="margin-bottom:6px;padding:10px;">
                            <div class="rag-result-header">
                                <span class="rag-result-source">#${AppCommon.escapeHtml(String(s.index))} ${AppCommon.escapeHtml(s.source)}</span>
                                <span class="rag-result-score">相似度 ${((s.score || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div class="rag-result-snippet">${AppCommon.escapeHtml(s.snippet)}</div>
                        </div>
                    `).join('');
                }
                break;
            }
            case 'token': {
                const token = event.content || '';
                // 旧后端不发 phase：首个 token 到达时清掉初始 loading
                if (!phaseShown && statusEl) statusEl.innerHTML = '';
                fullText += token;
                typewriter.push(token);
                break;
            }
            case 'answer': {
                // 完整答案兜底：仅在没有任何 token 累积时整体渲染
                if (!fullText && event.content) {
                    const text = typeof event.content === 'string' ? event.content : (event.content.text || '');
                    if (text) {
                        fullText = text;
                        typewriter.push(text);
                    }
                }
                break;
            }
            case 'done': {
                typewriter.finish();
                setPhase('');
                // 最终渲染 Markdown（renderMarkdown 内已过 DOMPurify）
                answerEl.innerHTML = AppCommon.renderMarkdown(fullText);
                AppCommon.highlightCodeBlocks(answerEl); // 异步补高亮（hljs 按需加载）
                break;
            }
            case 'cancelled': {
                // 用户中止/断连：保留已生成内容 + "已停止"标记
                typewriter.finish();
                setPhase('');
                appendStoppedMark(answerEl);
                break;
            }
            case 'error': {
                const msg = event.content?.message || '未知错误';
                typewriter.finish();
                setPhase('');
                // error 为终态：先把已累积文本定格为纯文本，再追加错误提示，避免半渲染状态
                answerEl.innerHTML = AppCommon.escapeHtml(fullText).replace(/\n/g, '<br>')
                    + `<p style="color:var(--error)">❌ 错误: ${AppCommon.escapeHtml(msg)}</p>`;
                break;
            }
        }
    }
    return { handler, typewriter };
}

// ============================================
// 文档管理
// ============================================

async function doLoadDocuments() {
    const btn = $('btnLoadDocs');
    const result = $('loadDocsResult');
    btn.disabled = true;
    setLoading(result, '扫描并加载中，请稍候...');
    try {
        const res = await apiFetch('/rag/load-documents', { method: 'POST' });
        const d = res.data || {};
        setSuccess(result, `已接收 ${d.accepted ?? (d.task_ids || []).length} 个索引任务，后台异步执行中`);
        if (state.currentTab === 'overview') loadStats();
    } catch (e) {
        setError(result, AppCommon.formatError(e));
    } finally {
        btn.disabled = false;
    }
}

async function doAddDocuments() {
    const text = $('addDocsText').value.trim();
    const source = $('addDocsSource').value.trim();
    const result = $('addDocsResult');
    if (!text) {
        setError(result, '请输入要添加的文本内容');
        return;
    }
    const btn = $('btnAddDocs');
    btn.disabled = true;
    setLoading(result, '添加中...');
    try {
        const res = await apiFetch('/rag/add-documents', {
            method: 'POST',
            body: JSON.stringify({
                documents: [text],
                sources: source ? [source] : undefined,
            }),
        });
        setSuccess(result, res.message);
        $('addDocsText').value = '';
        $('addDocsSource').value = '';
        if (state.currentTab === 'overview') loadStats();
    } catch (e) {
        setError(result, AppCommon.formatError(e));
    } finally {
        btn.disabled = false;
    }
}

async function doDeleteBySource() {
    const source = $('deleteSource').value.trim();
    const result = $('deleteResult');
    if (!source) {
        setError(result, '请输入来源路径或标识');
        return;
    }
    if (!(await Toast.confirm(`确定要删除来源为 "${source}" 的所有文档片段吗？`))) return;

    const btn = $('btnDeleteSource');
    btn.disabled = true;
    setLoading(result, '删除中...');
    try {
        const res = await apiFetch('/rag/delete-by-source', {
            method: 'POST',
            body: JSON.stringify({ source }),
        });
        setSuccess(result, res.message);
        $('deleteSource').value = '';
        if (state.currentTab === 'overview') loadStats();
    } catch (e) {
        setError(result, AppCommon.formatError(e));
    } finally {
        btn.disabled = false;
    }
}

// ============================================
// 事件绑定
// ============================================

function bindEvents() {
    // Tab 切换
    document.querySelectorAll('.rag-nav-btn').forEach((btn) => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // 检索
    $('btnSearch').addEventListener('click', doSearch);
    $('searchInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doSearch();
    });

    // 问答
    $('btnQuery').addEventListener('click', doQuery);
    $('queryInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doQuery();
    });

    // 文档管理
    $('btnLoadDocs').addEventListener('click', doLoadDocuments);
    $('btnAddDocs').addEventListener('click', doAddDocuments);
    $('btnDeleteSource').addEventListener('click', doDeleteBySource);

    // 文献列表
    $('btnRefreshDocs').addEventListener('click', loadDocsList);
    $('checkAllDocs').addEventListener('change', toggleSelectAll);
    $('checkAllDocsHeader').addEventListener('change', toggleSelectAll);
    $('btnBatchDeleteDocs').addEventListener('click', doBatchDelete);

    // 批量入库 chunks
    $('btnAddChunks').addEventListener('click', doAddChunks);
}

// ============================================
// 初始化
// ============================================
function init() {
    if (!state.token) {
        Toast.error('请先登录');
        setTimeout(() => { window.location.href = '/'; }, 1200);
        return;
    }
    bindEvents();
    loadStats();
    console.log('📚 RAG 知识库页面已加载');
}

// ============================================
// 文献列表
// ============================================

async function loadDocsList() {
    const status = $('docsListStatus');
    const tbody = $('docsTableBody');
    setLoading(status, '加载文献列表...');
    try {
        const res = await apiFetch('/rag/sources');
        const data = res.data || [];
        state.allDocs = data;
        state.selectedDocs.clear();
        updateBatchDeleteUI();
        status.innerHTML = '';
        if (!data.length) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:24px;">暂无文献</td></tr>`;
            return;
        }
        tbody.innerHTML = data.map((item, i) => `
            <tr data-source="${AppCommon.escapeHtml(item.source)}">
                <td class="col-check"><input type="checkbox" class="rag-doc-checkbox doc-checkbox" data-source="${AppCommon.escapeHtml(item.source)}"></td>
                <td class="col-num">${i + 1}</td>
                <td class="rag-doc-source">${AppCommon.escapeHtml(item.source)}</td>
                <td class="col-count">${AppCommon.escapeHtml(String(item.chunk_count ?? '-'))}</td>
                <td class="col-action">
                    <button class="rag-btn-small rag-btn-danger-small btn-delete-doc" data-source="${AppCommon.escapeHtml(item.source)}">删除</button>
                </td>
            </tr>
        `).join('');

        // 绑定单个 checkbox
        tbody.querySelectorAll('.doc-checkbox').forEach((cb) => {
            cb.addEventListener('change', () => {
                const src = cb.dataset.source;
                if (cb.checked) state.selectedDocs.add(src);
                else state.selectedDocs.delete(src);
                updateBatchDeleteUI();
            });
        });

        // 绑定单个删除按钮
        tbody.querySelectorAll('.btn-delete-doc').forEach((btn) => {
            btn.addEventListener('click', () => doDeleteSingleDoc(btn.dataset.source));
        });
    } catch (e) {
        status.innerHTML = '';
        setError(status, AppCommon.formatError(e));
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--error);padding:24px;">加载失败: ${AppCommon.escapeHtml(AppCommon.formatError(e))}</td></tr>`;
    }
}

function updateBatchDeleteUI() {
    const count = state.selectedDocs.size;
    $('batchDeleteCount').textContent = count;
    $('btnBatchDeleteDocs').disabled = count === 0;
    // 同步全选 checkbox 状态
    const allCount = state.allDocs.length;
    const checked = allCount > 0 && count === allCount;
    $('checkAllDocs').checked = checked;
    $('checkAllDocsHeader').checked = checked;
}

function toggleSelectAll(e) {
    const checked = e.target.checked;
    $('checkAllDocs').checked = checked;
    $('checkAllDocsHeader').checked = checked;
    if (checked) {
        state.allDocs.forEach((d) => state.selectedDocs.add(d.source));
    } else {
        state.selectedDocs.clear();
    }
    // 同步表格中所有 checkbox
    $('docsTableBody').querySelectorAll('.doc-checkbox').forEach((cb) => {
        cb.checked = checked;
    });
    updateBatchDeleteUI();
}

async function doDeleteSingleDoc(source) {
    if (!(await Toast.confirm(`确定要删除文献 "${source}" 吗？`))) return;
    const status = $('docsListStatus');
    setLoading(status, '删除中...');
    try {
        const res = await apiFetch('/rag/delete-by-source', {
            method: 'POST',
            body: JSON.stringify({ source }),
        });
        setSuccess(status, res.message);
        state.selectedDocs.delete(source);
        loadDocsList();
        if (state.currentTab === 'overview') loadStats();
    } catch (e) {
        setError(status, AppCommon.formatError(e));
    }
}

async function doBatchDelete() {
    const sources = Array.from(state.selectedDocs);
    if (!sources.length) return;
    if (!(await Toast.confirm(`确定要删除选中的 ${sources.length} 篇文献吗？此操作不可恢复。`))) return;
    const status = $('docsListStatus');
    setLoading(status, '批量删除中...');
    try {
        const res = await apiFetch('/rag/delete-sources', {
            method: 'POST',
            body: JSON.stringify({ sources }),
        });
        setSuccess(status, res.message);
        state.selectedDocs.clear();
        loadDocsList();
        if (state.currentTab === 'overview') loadStats();
    } catch (e) {
        setError(status, AppCommon.formatError(e));
    }
}

async function doAddChunks() {
    const text = $('addChunksText').value.trim();
    const prefix = $('addChunksSource').value.trim();
    const result = $('addChunksResult');
    if (!text) {
        setError(result, '请输入要入库的 chunks');
        return;
    }
    const chunks = text.split('\n').map((s) => s.trim()).filter((s) => s.length > 0);
    if (!chunks.length) {
        setError(result, '没有有效的 chunks');
        return;
    }
    const btn = $('btnAddChunks');
    btn.disabled = true;
    setLoading(result, `正在入库 ${chunks.length} 条 chunks...`);
    try {
        const sources = prefix ? chunks.map((_, i) => `${prefix}_${i + 1}`) : undefined;
        const res = await apiFetch('/rag/add-chunks', {
            method: 'POST',
            body: JSON.stringify({ chunks, sources }),
        });
        setSuccess(result, res.message);
        $('addChunksText').value = '';
        $('addChunksSource').value = '';
        if (state.currentTab === 'overview') loadStats();
        if (state.currentTab === 'docs') loadDocsList();
    } catch (e) {
        setError(result, AppCommon.formatError(e));
    } finally {
        btn.disabled = false;
    }
}

document.addEventListener('DOMContentLoaded', init);
