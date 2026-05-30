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
};

// DOM 缓存
const $ = (id) => document.getElementById(id);

// ============================================
// 认证 & API
// ============================================

async function _doRefresh() {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: state.refreshToken }),
    });
    if (!res.ok) throw new Error('刷新令牌失效');
    const data = await res.json();
    state.token = data.access_token;
    state.refreshToken = data.refresh_token;
    localStorage.setItem('agent_token', data.access_token);
    localStorage.setItem('agent_refresh_token', data.refresh_token);
    return data.access_token;
}

async function ensureTokenValid() {
    if (!state.token || !state.refreshToken) return;
    try {
        const payload = JSON.parse(atob(state.token.split('.')[1]));
        const exp = payload.exp * 1000;
        if (Date.now() + 5 * 60 * 1000 >= exp) {
            await _doRefresh();
        }
    } catch (e) {
        console.warn('Token 解析失败', e);
    }
}

async function apiFetch(path, options = {}) {
    await ensureTokenValid();
    const headers = { 'Content-Type': 'application/json' };
    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
    const res = await fetch(`${API_BASE}${path}`, { headers, ...options });
    if (res.status === 401) {
        if (state.refreshToken) {
            try {
                await _doRefresh();
                return apiFetch(path, options);
            } catch {
                alert('登录已过期，请重新登录');
                window.location.href = '/';
                throw new Error('登录已过期');
            }
        }
        alert('登录已过期，请重新登录');
        window.location.href = '/';
        throw new Error('登录已过期');
    }
    if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
    }
    return res.json();
}

// ============================================
// 工具函数
// ============================================

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function renderMarkdown(text) {
    if (!text) return '';
    marked.setOptions({ breaks: true, gfm: true });
    let html = marked.parse(text);
    html = html.replace(/<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g, (match, lang, code) => {
        const highlighted = hljs.highlightAuto(code, lang ? [lang] : undefined).value;
        return `<pre><code class="hljs language-${lang || ''}">${highlighted}</code></pre>`;
    });
    return html;
}

function setLoading(el, text = '加载中...') {
    el.innerHTML = `<div class="rag-loading"><div class="rag-spinner"></div><span>${text}</span></div>`;
}

function setError(el, text) {
    el.innerHTML = `<div class="rag-error">❌ ${escapeHtml(text)}</div>`;
}

function setSuccess(el, text) {
    el.innerHTML = `<div class="rag-success">✅ ${escapeHtml(text)}</div>`;
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
                <div class="rag-stat-value">${d.document_count ?? '-'}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">Collection</div>
                <div class="rag-stat-value" style="font-size:14px;">${escapeHtml(d.collection_name || '-')}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">Chunk Size</div>
                <div class="rag-stat-value">${d.chunk_size ?? '-'}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">重排序</div>
                <div class="rag-stat-value">${d.enable_reranker ? '启用' : '关闭'}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">Top N</div>
                <div class="rag-stat-value">${d.top_n ?? '-'}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">相似度阈值</div>
                <div class="rag-stat-value">${d.score_threshold ?? '-'}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">向量权重</div>
                <div class="rag-stat-value">${d.vector_weight ?? '-'}</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-label">BM25 权重</div>
                <div class="rag-stat-value">${d.bm25_weight ?? '-'}</div>
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
            <tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v ?? '-'))}</td></tr>
        `).join('');
    } catch (e) {
        setError(grid, e.message);
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
                    <span class="rag-result-source">#${item.index} ${escapeHtml(item.source)}</span>
                    <span class="rag-result-score">相似度 ${((item.score || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="rag-result-snippet">${escapeHtml(item.snippet)}</div>
            </div>
        `).join('');
    } catch (e) {
        status.innerHTML = '';
        setError(results, e.message);
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
    let fullText = '';

    const headers = { 'Content-Type': 'application/json' };
    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

    try {
        await ensureTokenValid();
        if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

        const res = await fetch(`${API_BASE}/rag/query`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ query }),
            signal: state.abortController.signal,
        });
        if (!res.ok) {
            const err = await res.text();
            throw new Error(err);
        }
        status.innerHTML = '';

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        await new Promise((resolve, reject) => {
            const read = () => {
                reader.read().then(({ done, value }) => {
                    if (done) { resolve(); return; }
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop() || '';

                    for (const chunk of lines) {
                        const line = chunk.trim();
                        if (!line.startsWith('data:')) continue;
                        try {
                            const data = JSON.parse(line.slice(5).trim());
                            handleQueryEvent(data, answerEl, sourcesWrap, sourcesList, fullText);
                            if (data.type === 'token' && typeof data.content === 'string') {
                                fullText += data.content;
                            }
                        } catch (e) {
                            console.warn('SSE parse error:', line, e);
                        }
                    }
                    read();
                }).catch(reject);
            };
            read();
        });
    } catch (e) {
        if (e.name === 'AbortError') {
            status.innerHTML = '';
            answerEl.innerHTML += '<p style="color:var(--text-secondary)">⏹️ 已中断</p>';
        } else {
            status.innerHTML = '';
            setError(resultEl, e.message);
        }
    } finally {
        state.isQuerying = false;
        state.abortController = null;
    }
}

function handleQueryEvent(event, answerEl, sourcesWrap, sourcesList, currentText) {
    switch (event.type) {
        case 'start':
            break;
        case 'sources': {
            const sources = Array.isArray(event.content) ? event.content : [];
            if (sources.length) {
                sourcesWrap.style.display = 'block';
                sourcesList.innerHTML = sources.map((s) => `
                    <div class="rag-result-item" style="margin-bottom:6px;padding:10px;">
                        <div class="rag-result-header">
                            <span class="rag-result-source">#${s.index} ${escapeHtml(s.source)}</span>
                            <span class="rag-result-score">相似度 ${((s.score || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div class="rag-result-snippet">${escapeHtml(s.snippet)}</div>
                    </div>
                `).join('');
            }
            break;
        }
        case 'token': {
            const token = event.content || '';
            // 流式时显示纯文本，最后渲染 Markdown
            answerEl.innerHTML = escapeHtml(currentText + token).replace(/\n/g, '<br>');
            break;
        }
        case 'done': {
            // 最终渲染 Markdown
            answerEl.innerHTML = renderMarkdown(currentText);
            break;
        }
        case 'error': {
            const msg = event.content?.message || '未知错误';
            answerEl.innerHTML += `<p style="color:var(--error)">❌ 错误: ${escapeHtml(msg)}</p>`;
            break;
        }
    }
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
        setSuccess(result, `加载完成 | 加载前: ${d.before} 条 | 加载后: ${d.after} 条 | 新增: ${d.loaded} 条`);
        if (state.currentTab === 'overview') loadStats();
    } catch (e) {
        setError(result, e.message);
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
        setError(result, e.message);
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
    if (!confirm(`确定要删除来源为 "${source}" 的所有文档片段吗？`)) return;

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
        setError(result, e.message);
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
}

// ============================================
// 初始化
// ============================================
function init() {
    if (!state.token) {
        alert('请先登录');
        window.location.href = '/';
        return;
    }
    bindEvents();
    loadStats();
    console.log('📚 RAG 知识库页面已加载');
}

document.addEventListener('DOMContentLoaded', init);
