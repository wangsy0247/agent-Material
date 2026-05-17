/**
 * 材料科学 Agent 前端主逻辑
 * 功能：会话管理、SSE 流式对话、Markdown 渲染、参考资料展示
 */

// ============================================
// 配置
// ============================================
const API_BASE = '/api/v1';

// ============================================
// 状态管理
// ============================================
const state = {
    currentSessionId: null,
    sessions: [],
    isStreaming: false,
    isSwitchingSession: false,
    sources: [],
    abortController: null,
};

// ============================================
// DOM 元素缓存
// ============================================
const $ = (id) => document.getElementById(id);
const dom = {
    sidebarLeft: $('sidebarLeft'),
    sidebarRight: $('sidebarRight'),
    overlay: $('overlay'),
    sessionList: $('sessionList'),
    chatMessages: $('chatMessages'),
    chatTitle: $('chatTitle'),
    chatSubtitle: $('chatSubtitle'),
    chatInput: $('chatInput'),
    btnSend: $('btnSend'),
    btnNewSession: $('btnNewSession'),
    btnToggleLeft: $('btnToggleLeft'),
    btnToggleRight: $('btnToggleRight'),
    sourcesList: $('sourcesList'),
    sourceCount: $('sourceCount'),
    welcomeScreen: $('welcomeScreen'),
    memoryStatus: $('memoryStatus'),
};

// ============================================
// API 封装
// ============================================
const api = {
    async fetchJson(path, options = {}) {
        const res = await fetch(`${API_BASE}${path}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        if (!res.ok) {
            const err = await res.text();
            throw new Error(err);
        }
        return res.json();
    },

    // 会话管理
    listSessions() {
        return this.fetchJson('/memory/sessions');
    },
    createSession(title = '') {
        return this.fetchJson('/memory/sessions', {
            method: 'POST',
            body: JSON.stringify({ title, source: 'web' }),
        });
    },
    loadSession(sessionId) {
        return this.fetchJson(`/memory/sessions/${sessionId}/load`, { method: 'POST' });
    },
    endSession(sessionId) {
        return this.fetchJson(`/memory/sessions/${sessionId}/end`, { method: 'POST' });
    },

    // 获取会话消息历史
    getSessionMessages(sessionId) {
        return this.fetchJson(`/memory/sessions/${sessionId}/messages`);
    },

    // RAG 检索（非流式，返回参考文献列表）
    ragRetrieve(question, topK = 5) {
        return this.fetchJson('/rag/retrieve', {
            method: 'POST',
            body: JSON.stringify({ question, top_k: topK }),
        });
    },

    // Agent 对话（SSE）
    chatStream(question, sessionId, onEvent) {
        const body = JSON.stringify({ question, session_id: sessionId || 'default', stream: true });
        return new Promise((resolve, reject) => {
            fetch(`${API_BASE}/agent/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body,
            })
                .then((res) => {
                    if (!res.ok) {
                        res.text().then((t) => reject(new Error(t)));
                        return;
                    }
                    const reader = res.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';

                    const read = () => {
                        reader.read().then(({ done, value }) => {
                            if (done) {
                                resolve();
                                return;
                            }
                            buffer += decoder.decode(value, { stream: true });
                            const lines = buffer.split('\n\n');
                            buffer = lines.pop() || '';

                            for (const chunk of lines) {
                                const line = chunk.trim();
                                if (!line.startsWith('data:')) continue;
                                try {
                                    const data = JSON.parse(line.slice(5).trim());
                                    onEvent(data);
                                } catch (e) {
                                    console.warn('SSE parse error:', line, e);
                                }
                            }
                            read();
                        }).catch(reject);
                    };
                    read();
                })
                .catch(reject);
        });
    },
};

// ============================================
// 工具函数
// ============================================
function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    return d.toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================
// Markdown 渲染
// ============================================
function renderMarkdown(text) {
    if (!text) return '';
    // 配置 marked
    marked.setOptions({
        breaks: true,
        gfm: true,
    });
    let html = marked.parse(text);
    // 高亮代码块
    html = html.replace(/<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g, (match, lang, code) => {
        const highlighted = hljs.highlightAuto(code, lang ? [lang] : undefined).value;
        return `<pre><code class="hljs language-${lang || ''}">${highlighted}</code></pre>`;
    });
    return html;
}

// ============================================
// 会话管理
// ============================================
async function loadSessions() {
    try {
        const res = await api.listSessions();
        state.sessions = res.data || [];
        renderSessionList();
    } catch (e) {
        console.error('加载会话失败:', e);
        dom.sessionList.innerHTML = '<div class="empty-state">加载失败，请刷新重试</div>';
    }
}

function renderSessionList() {
    if (!state.sessions.length) {
        dom.sessionList.innerHTML = '<div class="empty-state">暂无会话<br><small>点击上方按钮创建</small></div>';
        return;
    }
    dom.sessionList.innerHTML = state.sessions.map((s) => {
        const isActive = s.session_id === state.currentSessionId;
        return `
            <div class="session-item ${isActive ? 'active' : ''}" data-id="${s.session_id}">
                <div class="session-title">${escapeHtml(s.title || '未命名会话')}</div>
                <div class="session-meta">
                    <span>${formatTime(s.updated_at)} · ${s.message_count} 条消息</span>
                    <span class="session-badge ${s.is_active ? '' : 'inactive'}"></span>
                </div>
            </div>
        `;
    }).join('');

    // 绑定点击事件
    dom.sessionList.querySelectorAll('.session-item').forEach((el) => {
        el.addEventListener('click', () => switchSession(el.dataset.id));
    });
}

async function createNewSession() {
    try {
        const res = await api.createSession('新会话');
        const sessionId = res.data?.session_id;
        if (sessionId) {
            state.currentSessionId = sessionId;
            await loadSessions();
            clearChat();
            updateChatHeader('新会话', '开始对话');
            closeSidebars();
        }
    } catch (e) {
        alert('创建会话失败: ' + e.message);
    }
}

async function switchSession(sessionId) {
    // 防止重复点击：正在切换中、或点击的就是当前会话
    if (state.isSwitchingSession || sessionId === state.currentSessionId) return;
    state.isSwitchingSession = true;

    try {
        // 1. 后端加载会话（恢复短期记忆）
        await api.loadSession(sessionId);
        state.currentSessionId = sessionId;
        renderSessionList();
        clearChat();
        closeSidebars();

        // 2. 获取并渲染历史消息
        const msgRes = await api.getSessionMessages(sessionId);
        const history = msgRes.data || [];
        renderHistoryMessages(history);

        // 3. 更新标题
        const session = state.sessions.find((s) => s.session_id === sessionId);
        updateChatHeader(
            session?.title || '会话',
            session ? `${session.message_count} 条消息 · ${session.total_tokens} tokens` : ''
        );
    } catch (e) {
        console.error('切换会话失败:', e);
        appendErrorMessage('切换会话失败: ' + e.message);
    } finally {
        state.isSwitchingSession = false;
    }
}

function renderHistoryMessages(messages) {
    if (!messages.length) {
        dom.chatMessages.innerHTML = '<div class="empty-state">该会话暂无消息</div>';
        return;
    }
    for (const m of messages) {
        if (m.role === 'user') {
            appendUserMessage(m.content);
        } else if (m.role === 'assistant') {
            appendAssistantMessage(m.content);
        } else if (m.role === 'system') {
            // 系统消息（如压缩摘要）不显示在聊天界面
        }
    }
    scrollToBottom();
}

function appendAssistantMessage(text) {
    const id = 'msg-assistant-' + Date.now() + Math.random().toString(36).slice(2, 6);
    const html = `
        <div class="message assistant" id="${id}">
            <div class="message-avatar">🔬</div>
            <div class="message-content-wrap">
                <div class="message-content">${renderMarkdown(text)}</div>
            </div>
        </div>
    `;
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return id;
}

// ============================================
// 聊天界面
// ============================================
function clearChat() {
    dom.chatMessages.innerHTML = '';
    dom.sourcesList.innerHTML = '<div class="empty-state"><div class="empty-icon">📖</div><div>发送问题后将显示相关文献</div></div>';
    dom.sourceCount.textContent = '';
    state.sources = [];
}

function updateChatHeader(title, subtitle) {
    dom.chatTitle.textContent = title || '材料科学 Agent';
    dom.chatSubtitle.textContent = subtitle || '';
}

function appendUserMessage(text) {
    const id = 'msg-' + Date.now();
    const html = `
        <div class="message user" id="${id}">
            <div class="message-avatar">👤</div>
            <div class="message-content-wrap">
                <div class="message-content">${escapeHtml(text)}</div>
            </div>
        </div>
    `;
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return id;
}

function appendAssistantPlaceholder() {
    const id = 'msg-assistant-' + Date.now();
    const html = `
        <div class="message assistant" id="${id}">
            <div class="message-avatar">🔬</div>
            <div class="message-content-wrap">
                <div class="message-content">
                    <div class="typing-indicator">
                        <span>思考中</span>
                        <div class="dots"><span></span><span></span><span></span></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return id;
}

function updateAssistantMessage(msgId, text, isFinal = false) {
    const el = document.querySelector(`#${msgId} .message-content`);
    if (!el) return;
    if (isFinal) {
        el.innerHTML = renderMarkdown(text);
    } else {
        // 流式更新时显示纯文本（避免频繁渲染 Markdown）
        el.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
    }
    scrollToBottom();
}

function appendStatusMessage(text) {
    const id = 'status-' + Date.now();
    const html = `
        <div class="message assistant" id="${id}">
            <div class="message-avatar">ℹ️</div>
            <div class="message-content-wrap">
                <div class="message-content" style="font-size:13px;color:var(--text-secondary)">${escapeHtml(text)}</div>
            </div>
        </div>
    `;
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return id;
}

function appendErrorMessage(text) {
    const html = `
        <div class="message assistant">
            <div class="message-avatar">⚠️</div>
            <div class="message-content-wrap">
                <div class="message-content error-message">${escapeHtml(text)}</div>
            </div>
        </div>
    `;
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function scrollToBottom() {
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

// ============================================
// 参考资料
// ============================================
function renderSources(sources) {
    state.sources = sources || [];
    dom.sourceCount.textContent = state.sources.length ? `${state.sources.length} 篇` : '';

    if (!state.sources.length) {
        dom.sourcesList.innerHTML = '<div class="empty-state"><div class="empty-icon">📖</div><div>发送问题后将显示相关文献</div></div>';
        return;
    }

    dom.sourcesList.innerHTML = state.sources.map((s, i) => `
        <div class="source-item" title="点击展开">
            <div class="source-header">
                <span class="source-index">#${s.index}</span>
                <span class="source-score">相似度 ${(s.score * 100).toFixed(1)}%</span>
            </div>
            <div class="source-name">${escapeHtml(s.source)}</div>
            <div class="source-snippet">${escapeHtml(s.snippet)}</div>
        </div>
    `).join('');
}

// ============================================
// 发送消息（核心逻辑）
// ============================================
async function sendMessage() {
    const text = dom.chatInput.value.trim();
    if (!text || state.isStreaming) return;

    // 隐藏欢迎界面
    if (dom.welcomeScreen) dom.welcomeScreen.style.display = 'none';

    // 清空输入
    dom.chatInput.value = '';
    dom.chatInput.style.height = 'auto';
    dom.btnSend.disabled = true;
    state.isStreaming = true;

    // 显示用户消息
    appendUserMessage(text);

    // ---- 并行调用 RAG 检索，填充右栏参考文献 ----
    api.ragRetrieve(text, 5)
        .then((res) => {
            const sources = res.data || [];
            if (sources.length) {
                renderSources(sources);
            } else {
                dom.sourcesList.innerHTML = '<div class="empty-state"><div class="empty-icon">📖</div><div>未检索到相关文献</div></div>';
                dom.sourceCount.textContent = '0 篇';
            }
        })
        .catch((e) => {
            console.error('RAG 检索失败:', e);
            dom.sourcesList.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><div>文献检索失败</div></div>';
        });

    // 显示助手占位
    const assistantId = appendAssistantPlaceholder();
    let fullAnswer = '';
    let thinkingText = '';

    try {
        await api.chatStream(text, state.currentSessionId, (event) => {
            switch (event.type) {
                case 'start': {
                    const sid = event.content?.session_id;
                    if (sid && !state.currentSessionId) {
                        state.currentSessionId = sid;
                        loadSessions();
                    }
                    break;
                }
                case 'status': {
                    const status = event.content;
                    if (status?.memory) {
                        const info = status.memory.join(' · ');
                        dom.memoryStatus.innerHTML = `<span class="dot"></span><span>${info}</span>`;
                    }
                    break;
                }
                case 'thinking': {
                    thinkingText += event.content || '';
                    break;
                }
                case 'token': {
                    const token = event.content || '';
                    fullAnswer += token;
                    updateAssistantMessage(assistantId, fullAnswer, false);
                    break;
                }
                case 'answer': {
                    // 完整答案片段（流式模式下通常由 token 组成）
                    break;
                }
                case 'sources': {
                    const sources = Array.isArray(event.content) ? event.content : [];
                    renderSources(sources);
                    break;
                }
                case 'error': {
                    const err = event.content?.message || '未知错误';
                    updateAssistantMessage(assistantId, `❌ 错误: ${err}`, true);
                    break;
                }
                case 'done': {
                    // 最终渲染 Markdown
                    updateAssistantMessage(assistantId, fullAnswer, true);
                    // 如果有思考过程，追加折叠框
                    if (thinkingText) {
                        const contentEl = document.querySelector(`#${assistantId} .message-content`);
                        if (contentEl) {
                            contentEl.insertAdjacentHTML('afterbegin', `
                                <details class="thinking-box">
                                    <summary>💡 思考过程</summary>
                                    <pre>${escapeHtml(thinkingText)}</pre>
                                </details>
                            `);
                        }
                    }
                    break;
                }
            }
        });
    } catch (e) {
        console.error('对话失败:', e);
        updateAssistantMessage(assistantId, `❌ 请求失败: ${e.message}`, true);
    } finally {
        state.isStreaming = false;
        dom.btnSend.disabled = false;
        scrollToBottom();
    }
}

// ============================================
// 侧边栏控制
// ============================================
function toggleSidebar(side) {
    const el = side === 'left' ? dom.sidebarLeft : dom.sidebarRight;
    const isOpen = el.classList.contains('open');
    if (isOpen) {
        el.classList.remove('open');
        dom.overlay.classList.remove('active');
    } else {
        closeSidebars();
        el.classList.add('open');
        dom.overlay.classList.add('active');
    }
}

function closeSidebars() {
    dom.sidebarLeft.classList.remove('open');
    dom.sidebarRight.classList.remove('open');
    dom.overlay.classList.remove('active');
}

// ============================================
// 事件绑定
// ============================================
function bindEvents() {
    // 发送按钮
    dom.btnSend.addEventListener('click', sendMessage);

    // 输入框 Enter 发送 / Shift+Enter 换行
    dom.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 输入框自适应高度
    dom.chatInput.addEventListener('input', () => {
        dom.chatInput.style.height = 'auto';
        dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 200) + 'px';
    });

    // 新建会话
    dom.btnNewSession.addEventListener('click', createNewSession);

    // 侧边栏切换
    dom.btnToggleLeft.addEventListener('click', () => toggleSidebar('left'));
    dom.btnToggleRight.addEventListener('click', () => toggleSidebar('right'));
    dom.overlay.addEventListener('click', closeSidebars);

    // 欢迎界面示例卡片
    document.querySelectorAll('.example-card').forEach((card) => {
        card.addEventListener('click', () => {
            dom.chatInput.value = card.dataset.text;
            sendMessage();
        });
    });
}

// ============================================
// 初始化
// ============================================
function init() {
    bindEvents();
    loadSessions();
    console.log('🔬 材料科学 Agent 前端已加载');
}

document.addEventListener('DOMContentLoaded', init);
