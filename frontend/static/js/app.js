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
    pendingApproval: null,  // { session_id, tool_calls, message }
    // 认证状态
    token: localStorage.getItem('agent_token') || null,
    refreshToken: localStorage.getItem('agent_refresh_token') || null,
    user: null,
    isLoggedIn: false,
    lastSessionId: localStorage.getItem('agent_last_session_id') || null,
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
    approvalPanel: $('approvalPanel'),
    approvalDesc: $('approvalDesc'),
    approvalDetail: $('approvalDetail'),
    btnApprove: $('btnApprove'),
    btnReject: $('btnReject'),
    chatInputArea: document.querySelector('.chat-input-area'),
};

// ============================================
// API 封装
// ============================================
let _refreshingPromise = null;

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
        // Token 将在 5 分钟内过期时自动刷新
        if (Date.now() + 5 * 60 * 1000 >= exp) {
            if (!_refreshingPromise) {
                _refreshingPromise = _doRefresh().finally(() => { _refreshingPromise = null; });
            }
            await _refreshingPromise;
        }
    } catch (e) {
        console.warn('Token 解析失败', e);
    }
}

const api = {
    async fetchJson(path, options = {}) {
        const doFetch = async () => {
            await ensureTokenValid();
            const headers = { 'Content-Type': 'application/json' };
            if (state.token) {
                headers['Authorization'] = `Bearer ${state.token}`;
            }
            const res = await fetch(`${API_BASE}${path}`, {
                headers,
                ...options,
            });
            if (res.status === 401) {
                if (state.refreshToken) {
                    if (!_refreshingPromise) {
                        _refreshingPromise = _doRefresh()
                            .catch((err) => { _refreshingPromise = null; throw err; })
                            .finally(() => { _refreshingPromise = null; });
                    }
                    try {
                        await _refreshingPromise;
                        // 刷新成功后重试原请求
                        return doFetch();
                    } catch (refreshErr) {
                        logout();
                        throw new Error('登录已过期，请重新登录');
                    }
                }
                logout();
                throw new Error('登录已过期，请重新登录');
            }
            if (!res.ok) {
                const err = await res.text();
                throw new Error(err);
            }
            return res.json();
        };
        return doFetch();
    },

    // 认证
    login(username, password) {
        return this.fetchJson('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ username, password }),
        });
    },
    register(username, password, email) {
        return this.fetchJson('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ username, password, email }),
        });
    },
    me() {
        return this.fetchJson('/auth/me');
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

    // Agent 对话（SSE）
    chatStream(question, sessionId, onEvent, signal) {
        const body = JSON.stringify({ question, session_id: sessionId || 'default', stream: true });
        const headers = { 'Content-Type': 'application/json' };
        if (state.token) {
            headers['Authorization'] = `Bearer ${state.token}`;
        }
        return new Promise((resolve, reject) => {
            ensureTokenValid().then(() => {
                if (state.token) {
                    headers['Authorization'] = `Bearer ${state.token}`;
                }
                return fetch(`${API_BASE}/agent/chat`, {
                    method: 'POST',
                    headers,
                    body,
                    signal,
                });
            }).then((res) => {
                if (!res.ok) {
                    if (res.status === 401) {
                        logout();
                        reject(new Error('登录已过期，请重新登录'));
                        return;
                    }
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
            }).catch(reject);
        });
    },

    // HITL 审批恢复（SSE）
    approveStream(sessionId, action, comment, onEvent, signal) {
        const body = JSON.stringify({ session_id: sessionId, action, comment: comment || '' });
        const headers = { 'Content-Type': 'application/json' };
        if (state.token) {
            headers['Authorization'] = `Bearer ${state.token}`;
        }
        return new Promise((resolve, reject) => {
            ensureTokenValid().then(() => {
                if (state.token) {
                    headers['Authorization'] = `Bearer ${state.token}`;
                }
                return fetch(`${API_BASE}/agent/approve`, {
                    method: 'POST',
                    headers,
                    body,
                    signal,
                });
            }).then((res) => {
                if (!res.ok) {
                    if (res.status === 401) {
                        logout();
                        reject(new Error('登录已过期，请重新登录'));
                        return;
                    }
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
            }).catch(reject);
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
async function loadSessions(autoRestore = false) {
    try {
        const res = await api.listSessions();
        state.sessions = res.data || [];
        renderSessionList();

        // 自动恢复上次会话（仅在初始化时触发）
        if (autoRestore && state.lastSessionId) {
            const exists = state.sessions.find((s) => s.session_id === state.lastSessionId);
            if (exists && state.lastSessionId !== state.currentSessionId) {
                await switchSession(state.lastSessionId);
            } else if (!exists) {
                localStorage.removeItem('agent_last_session_id');
                state.lastSessionId = null;
            }
        }
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
            state.lastSessionId = null;
            localStorage.removeItem('agent_last_session_id');
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
        // 如果正在流式输出或等待审批，先清理状态
        if ((state.isStreaming || state.pendingApproval) && state.abortController) {
            state.abortController.abort();
        }
        state.isStreaming = false;
        state.pendingApproval = null;
        dom.btnSend.disabled = false;
        hideApprovalPanel();

        // 1. 后端加载会话（恢复短期记忆）
        await api.loadSession(sessionId);
        state.currentSessionId = sessionId;
        state.lastSessionId = sessionId;
        localStorage.setItem('agent_last_session_id', sessionId);
        await loadSessions(); // 刷新列表以获取最新的消息计数和时间
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
// HITL 审批面板
// ============================================
let _approvalMsgId = null;

function showApprovalPanel(data) {
    const toolCalls = data.tool_calls || [];
    const message = data.message || 'Agent 请求执行敏感操作，请确认是否允许。';

    // ---- 1. 更新顶部审批面板（备选） ----
    dom.approvalDesc.textContent = message;
    let detailHtml = '';
    if (toolCalls.length > 0) {
        detailHtml = toolCalls.map((tc, i) => {
            const name = tc.name || tc.function?.name || '未知工具';
            const args = tc.arguments || tc.function?.arguments || tc.args || '{}';
            let argsStr = '';
            try {
                argsStr = typeof args === 'string' ? args : JSON.stringify(args, null, 2);
            } catch {
                argsStr = String(args);
            }
            return `工具 #${i + 1}: ${name}\n参数:\n${argsStr}`;
        }).join('\n\n');
    } else {
        detailHtml = '（未获取到工具调用详情）';
    }
    dom.approvalDetail.textContent = detailHtml;
    dom.approvalPanel.style.display = 'block';
    dom.chatInputArea = dom.chatInputArea || document.querySelector('.chat-input-area');
    if (dom.chatInputArea) dom.chatInputArea.classList.add('pending-approval');
    dom.btnApprove.disabled = false;
    dom.btnReject.disabled = false;

    // ---- 2. 在消息流中插入可交互的审批气泡（确保用户一定能看到） ----
    _approvalMsgId = 'msg-approval-' + Date.now();
    const toolsHtml = toolCalls.map((tc) => {
        const name = tc.name || '未知工具';
        const argsStr = JSON.stringify(tc.args || tc.arguments || {}, null, 2);
        return `<div class="approval-tool-item"><code>${name}</code><pre>${escapeHtml(argsStr)}</pre></div>`;
    }).join('');

    const html = `
        <div class="message assistant approval-message" id="${_approvalMsgId}">
            <div class="message-avatar">⚠️</div>
            <div class="message-content-wrap">
                <div class="message-content">
                    <div class="approval-bubble">
                        <div class="approval-bubble-title">🔒 敏感操作需要确认</div>
                        <div class="approval-bubble-desc">${escapeHtml(message)}</div>
                        <div class="approval-bubble-tools">${toolsHtml}</div>
                        <div class="approval-bubble-actions">
                            <button class="btn-bubble-approve" data-action="approve">✓ 确认执行</button>
                            <button class="btn-bubble-reject" data-action="reject">✕ 拒绝</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    scrollToBottom();

    // 绑定气泡内按钮事件
    const msgEl = document.getElementById(_approvalMsgId);
    if (msgEl) {
        msgEl.querySelectorAll('button[data-action]').forEach((btn) => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                // 禁用按钮防止重复点击
                msgEl.querySelectorAll('button[data-action]').forEach((b) => b.disabled = true);
                handleApproval(action);
            });
        });
    }
}

function hideApprovalPanel() {
    dom.approvalPanel.style.display = 'none';
    if (dom.chatInputArea) dom.chatInputArea.classList.remove('pending-approval');
    dom.btnApprove.disabled = false;
    dom.btnReject.disabled = false;

    // 隐藏消息流中的审批气泡（可选：保留历史记录，仅禁用按钮）
    if (_approvalMsgId) {
        const msgEl = document.getElementById(_approvalMsgId);
        if (msgEl) {
            msgEl.querySelectorAll('button[data-action]').forEach((b) => {
                b.disabled = true;
                b.style.opacity = '0.5';
            });
        }
        _approvalMsgId = null;
    }
}

async function handleApproval(action) {
    if (!state.pendingApproval) return;

    const { session_id } = state.pendingApproval;
    const comment = action === 'approve' ? '用户确认执行' : '用户拒绝执行';

    // 禁用按钮防止重复点击
    dom.btnApprove.disabled = true;
    dom.btnReject.disabled = true;

    // 隐藏审批面板
    hideApprovalPanel();

    // 在聊天区显示用户的选择
    const actionText = action === 'approve' ? '✓ 已确认执行该操作' : '✕ 已拒绝执行该操作';
    appendStatusMessage(actionText);

    // 显示新的助手占位（继续等待后续输出）
    const assistantId = appendAssistantPlaceholder();
    let fullAnswer = '';
    let thinkingText = '';

    state.abortController = new AbortController();

    try {
        await api.approveStream(session_id, action, comment, (event) => {
            switch (event.type) {
                case 'start': {
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
                case 'sources': {
                    const sources = Array.isArray(event.content) ? event.content : [];
                    renderSources(sources);
                    break;
                }
                case 'approval_required': {
                    // 再次遇到审批（理论上不太可能）
                    state.pendingApproval = event.content;
                    showApprovalPanel(event.content);
                    break;
                }
                case 'error': {
                    const err = event.content?.message || '未知错误';
                    updateAssistantMessage(assistantId, `❌ 错误: ${err}`, true);
                    break;
                }
                case 'done': {
                    updateAssistantMessage(assistantId, fullAnswer, true);
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
                    loadSessions();
                    break;
                }
            }
        }, state.abortController.signal);
    } catch (e) {
        if (e.name === 'AbortError') {
            console.log('审批恢复已中止');
            updateAssistantMessage(assistantId, '⏹️ 操作已中断', true);
        } else {
            console.error('审批恢复失败:', e);
            updateAssistantMessage(assistantId, `❌ 恢复失败: ${e.message}`, true);
        }
        loadSessions();
    } finally {
        state.isStreaming = false;
        state.pendingApproval = null;
        dom.btnSend.disabled = false;
        state.abortController = null;
        scrollToBottom();
    }
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

    // 如果存在待审批的操作，提示用户先处理
    if (state.pendingApproval) {
        appendStatusMessage('⏳ 请先处理上方的敏感操作确认请求');
        return;
    }

    // 隐藏欢迎界面
    if (dom.welcomeScreen) dom.welcomeScreen.style.display = 'none';

    // 清空输入
    dom.chatInput.value = '';
    dom.chatInput.style.height = 'auto';
    dom.btnSend.disabled = true;
    state.isStreaming = true;

    // 显示用户消息
    appendUserMessage(text);

    // 显示助手占位
    const assistantId = appendAssistantPlaceholder();
    let fullAnswer = '';
    let thinkingText = '';

    state.abortController = new AbortController();

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
                case 'approval_required': {
                    // HITL 中断：保存审批状态并显示审批面板
                    state.pendingApproval = event.content;
                    showApprovalPanel(event.content);
                    // 将思考中占位替换为提示信息
                    updateAssistantMessage(assistantId, '⏳ 等待人工确认...', true);
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
                    // 刷新会话列表，更新消息计数和时间
                    loadSessions();
                    break;
                }
            }
        });
    } catch (e) {
        if (e.name === 'AbortError') {
            console.log('对话已中止');
            updateAssistantMessage(assistantId, '⏹️ 对话已中断', true);
        } else {
            console.error('对话失败:', e);
            updateAssistantMessage(assistantId, `❌ 请求失败: ${e.message}`, true);
        }
        // 刷新会话列表（即使失败也要更新时间显示）
        loadSessions();
    } finally {
        // 如果触发了审批，保持 streaming 状态直到用户处理
        if (!state.pendingApproval) {
            state.isStreaming = false;
            dom.btnSend.disabled = false;
        }
        state.abortController = null;
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

    // HITL 审批按钮
    if (dom.btnApprove) {
        dom.btnApprove.addEventListener('click', () => handleApproval('approve'));
    }
    if (dom.btnReject) {
        dom.btnReject.addEventListener('click', () => handleApproval('reject'));
    }
}

// ============================================
// 认证模块
// ============================================

function showAuthModal() {
    $('authOverlay').style.display = 'flex';
}

function hideAuthModal() {
    $('authOverlay').style.display = 'none';
}

function logout() {
    state.token = null;
    state.refreshToken = null;
    state.user = null;
    state.isLoggedIn = false;
    state.lastSessionId = null;
    localStorage.removeItem('agent_token');
    localStorage.removeItem('agent_refresh_token');
    localStorage.removeItem('agent_last_session_id');
    showAuthModal();
    updateUserBar();
}

function updateUserBar() {
    let bar = document.querySelector('.user-bar');
    if (!bar) {
        bar = document.createElement('div');
        bar.className = 'user-bar';
        const footer = document.querySelector('.sidebar-footer');
        if (footer) {
            footer.parentNode.insertBefore(bar, footer.nextSibling);
        }
    }
    if (state.isLoggedIn && state.user) {
        bar.innerHTML = `
            <span class="user-name">👤 ${escapeHtml(state.user.username)}</span>
            <button class="logout-btn" id="btnLogout">退出</button>
        `;
        document.getElementById('btnLogout')?.addEventListener('click', logout);
    } else {
        bar.innerHTML = '';
    }
}

async function checkAuth() {
    if (!state.token) {
        showAuthModal();
        return;
    }
    try {
        const data = await api.me();
        state.user = data;
        state.isLoggedIn = true;
        hideAuthModal();
        updateUserBar();
        loadSessions(true); // true = 自动恢复上次会话
    } catch (e) {
        console.warn('Token 校验失败:', e);
        logout();
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const btn = $('loginBtn');
    const errEl = $('loginError');
    errEl.textContent = '';
    btn.disabled = true;
    btn.textContent = '登录中...';

    try {
        const data = await api.login($('loginUsername').value.trim(), $('loginPassword').value);
        state.token = data.access_token;
        state.refreshToken = data.refresh_token;
        state.user = data.user;
        state.isLoggedIn = true;
        localStorage.setItem('agent_token', data.access_token);
        localStorage.setItem('agent_refresh_token', data.refresh_token);
        hideAuthModal();
        updateUserBar();
        loadSessions();
        $('loginPassword').value = '';
    } catch (e) {
        errEl.textContent = e.message || '登录失败';
    } finally {
        btn.disabled = false;
        btn.textContent = '登录';
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const btn = $('registerBtn');
    const errEl = $('registerError');
    errEl.textContent = '';
    btn.disabled = true;
    btn.textContent = '注册中...';

    try {
        const email = $('registerEmail').value.trim() || null;
        await api.register($('registerUsername').value.trim(), $('registerPassword').value, email);
        errEl.className = 'auth-success';
        errEl.textContent = '注册成功，请登录';
        $('registerPassword').value = '';
        // 切换到登录页
        setTimeout(() => switchAuthTab('login'), 800);
    } catch (e) {
        errEl.className = 'auth-error';
        errEl.textContent = e.message || '注册失败';
    } finally {
        btn.disabled = false;
        btn.textContent = '注册';
    }
}

function switchAuthTab(tab) {
    document.querySelectorAll('.auth-tab').forEach((t) => {
        t.classList.toggle('active', t.dataset.tab === tab);
    });
    $('loginForm').classList.toggle('hidden', tab !== 'login');
    $('registerForm').classList.toggle('hidden', tab !== 'register');
    $('loginError').textContent = '';
    $('registerError').textContent = '';
}

function bindAuthEvents() {
    $('loginForm').addEventListener('submit', handleLogin);
    $('registerForm').addEventListener('submit', handleRegister);
    $('tabLogin').addEventListener('click', () => switchAuthTab('login'));
    $('tabRegister').addEventListener('click', () => switchAuthTab('register'));
}

// ============================================
// 初始化
// ============================================
function init() {
    bindEvents();
    bindAuthEvents();
    checkAuth();
    console.log('🔬 材料科学 Agent 前端已加载');
}

document.addEventListener('DOMContentLoaded', init);
