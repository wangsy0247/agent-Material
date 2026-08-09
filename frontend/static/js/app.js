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
    staleBanner: $('staleBanner'),
    backToBottom: $('backToBottom'),
    chatInputArea: document.querySelector('.chat-input-area'),
};

// ============================================
// API 封装
// ============================================
// 认证上下文（公共逻辑见 common.js AppCommon.createAuthContext）：
// token 存取注入本页 state，刷新失败/最终 401 走 logout()
const authCtx = AppCommon.createAuthContext({
    getToken: () => state.token,
    getRefreshToken: () => state.refreshToken,
    setTokens(access, refresh) {
        state.token = access;
        state.refreshToken = refresh;
        localStorage.setItem('agent_token', access);
        localStorage.setItem('agent_refresh_token', refresh);
    },
    onAuthFail: () => logout(),
});

const api = {
    async fetchJson(path, options = {}) {
        const doFetch = async () => {
            await authCtx.ensureTokenValid();
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
                    try {
                        await authCtx.refresh();
                        // 刷新成功后重试原请求
                        return doFetch();
                    } catch {
                        logout();
                        throw AppCommon.AppCommon.friendlyError('登录已过期，请重新登录');
                    }
                }
                logout();
                throw AppCommon.AppCommon.friendlyError('登录已过期，请重新登录');
            }
            if (!res.ok) {
                const err = await res.text();
                const error = new Error(err);
                error.status = res.status; // 带状态码，供 formatError 友好化
                throw error;
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
    deleteSession(sessionId) {
        return this.fetchJson(`/memory/sessions/${sessionId}`, { method: 'DELETE' });
    },

    // 获取会话消息历史
    getSessionMessages(sessionId) {
        return this.fetchJson(`/memory/sessions/${sessionId}/messages`);
    },

    // Agent 对话（SSE）；sseOpts 可传 onStale/staleMs 连接活性监视参数
    chatStream(question, sessionId, onEvent, signal, sseOpts = {}) {
        return authCtx.sseConnect({
            url: `${API_BASE}/agent/chat`,
            body: { question, session_id: sessionId || 'default', stream: true },
            onEvent,
            signal,
            ...sseOpts,
        });
    },

    // HITL 审批恢复（SSE）
    approveStream(sessionId, action, comment, onEvent, signal) {
        return authCtx.sseConnect({
            url: `${API_BASE}/agent/approve`,
            body: { session_id: sessionId, action, comment: comment || '' },
            onEvent,
            signal,
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

// runState 持久化（sessionStorage 高频写入节流）：
// 默认 300ms 节流（trailing，合并同一 runState 的连续写入，保证最后状态落盘）；
// immediate=true 时立即写入并取消挂起的节流（终态路径使用，避免刷新丢失最终状态）
const _persistTimers = new WeakMap(); // runState -> timerId

function persistRunState(runState, immediate = false) {
    const write = () => sessionStorage.setItem('agent_run', JSON.stringify(runState));
    if (immediate) {
        cancelPersistRunState(runState);
        write();
        return;
    }
    if (_persistTimers.has(runState)) return; // 已有挂起写入，合并
    _persistTimers.set(runState, setTimeout(() => {
        _persistTimers.delete(runState);
        write();
    }, 300));
}

// 取消挂起的节流写入（removeItem 前调用，防止尾写复活已清除状态）
function cancelPersistRunState(runState) {
    const t = _persistTimers.get(runState);
    if (t) {
        clearTimeout(t);
        _persistTimers.delete(runState);
    }
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
        Toast.error('会话列表加载失败');
        // 列表区显示可点击的重试入口
        dom.sessionList.innerHTML = '<div class="empty-state session-list-retry">加载失败，点击重试</div>';
        dom.sessionList.querySelector('.session-list-retry')?.addEventListener('click', () => loadSessions());
    }
}

// 会话列表渲染：keyed diff——以 session_id 为 key 复用现有 DOM 节点，
// 只更新变化的标题/时间/计数文本；点击走容器级事件委托（见 bindEvents）
function renderSessionList() {
    if (!state.sessions.length) {
        dom.sessionList.innerHTML = '<div class="empty-state">暂无会话<br><small>点击上方按钮创建</small></div>';
        return;
    }
    // 清掉 empty-state / 加载失败占位
    if (dom.sessionList.querySelector('.empty-state')) dom.sessionList.innerHTML = '';

    const seen = new Set();
    let prev = null; // 上一个就位节点，用于保持与后端返回顺序一致
    for (const s of state.sessions) {
        const sid = String(s.session_id);
        seen.add(sid);
        let el = dom.sessionList.querySelector(`[data-id="${CSS.escape(sid)}"]`);
        if (!el) {
            // 新会话：创建节点（结构与原 innerHTML 模板一致）
            el = document.createElement('div');
            el.className = 'session-item';
            el.dataset.id = sid;
            el.innerHTML = `
                <div class="session-title"></div>
                <div class="session-meta">
                    <span class="session-meta-text"></span>
                    <span class="session-badge"></span>
                </div>
                <button class="session-delete" title="删除会话" type="button">🗑</button>
            `;
        }
        // 只更新变化的字段（textContent 赋值，天然防 XSS）
        const title = s.title || '未命名会话';
        const titleEl = el.querySelector('.session-title');
        if (titleEl.textContent !== title) titleEl.textContent = title;
        const meta = `${formatTime(s.updated_at)} · ${s.message_count ?? 0} 条消息`;
        const metaEl = el.querySelector('.session-meta-text');
        if (metaEl.textContent !== meta) metaEl.textContent = meta;
        el.querySelector('.session-badge').classList.toggle('inactive', !s.is_active);
        el.classList.toggle('active', s.session_id === state.currentSessionId);

        // 移动到正确位置（保持列表顺序与后端返回一致）
        if (prev) {
            if (el.previousElementSibling !== prev) prev.after(el);
        } else if (el !== dom.sessionList.firstElementChild) {
            dom.sessionList.prepend(el);
        }
        prev = el;
    }
    // 移除已消失的会话
    dom.sessionList.querySelectorAll('.session-item').forEach((el) => {
        if (!seen.has(el.dataset.id)) el.remove();
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
        Toast.error('创建会话失败: ' + AppCommon.formatError(e));
    }
}

// 删除会话：确认后调 DELETE API（后端连带删除 SQLite 中会话与全部消息），
// 删除的是当前会话时回到新会话视图
async function deleteSession(sessionId) {
    const sid = String(sessionId);
    const s = state.sessions.find((x) => String(x.session_id) === sid);
    const title = s?.title || '该会话';
    if (!(await Toast.confirm(`确定要删除会话"${title}"吗？会话及其所有消息将被删除，此操作不可恢复。`))) return;

    try {
        await api.deleteSession(sid);
        state.sessions = state.sessions.filter((x) => String(x.session_id) !== sid);

        // 清理该会话可能残留的续流状态
        try {
            const run = JSON.parse(sessionStorage.getItem('agent_run') || 'null');
            if (run && String(run.sessionId) === sid) sessionStorage.removeItem('agent_run');
        } catch { /* 忽略解析失败 */ }

        // 删除的是当前会话：回到新会话视图
        if (String(state.currentSessionId) === sid) {
            state.currentSessionId = null;
            state.lastSessionId = null;
            localStorage.removeItem('agent_last_session_id');
            clearChat();
            updateChatHeader('新会话', '开始对话');
            if (dom.welcomeScreen) dom.welcomeScreen.style.display = '';
        }

        renderSessionList();
        Toast.info('会话已删除');
    } catch (e) {
        Toast.error('删除会话失败: ' + AppCommon.formatError(e));
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
        setSendBtnStop(false);
        dom.btnSend.disabled = false;
        clearApprovalBubble();
        setStreamScroll(false);
        hideStaleBanner();

        // 1. 后端加载会话（恢复短期记忆）
        await api.loadSession(sessionId);
        state.currentSessionId = sessionId;
        state.lastSessionId = sessionId;
        localStorage.setItem('agent_last_session_id', sessionId);
        await loadSessions(); // 刷新列表以获取最新的消息计数和时间
        clearChat();
        closeSidebars();
        // 历史加载期间显示骨架屏（renderHistoryMessages 渲染时移除）
        showSkeletonMessages();

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
        // 清掉骨架屏再显示错误
        dom.chatMessages.innerHTML = '';
        appendErrorMessage('切换会话失败: ' + AppCommon.formatError(e));
    } finally {
        state.isSwitchingSession = false;
    }
}

// 会话切换加载历史期间的骨架屏（3 条灰色占位气泡，渲染历史时整体移除）
function showSkeletonMessages() {
    dom.chatMessages.innerHTML = Array.from({ length: 3 }, (_, i) => `
        <div class="message ${i % 2 ? 'user' : 'assistant'} skeleton-message">
            <div class="message-avatar skeleton-avatar"></div>
            <div class="message-content-wrap">
                <div class="message-content skeleton-block"></div>
            </div>
        </div>
    `).join('');
}

// ---- 历史消息分页懒渲染 ----
// 首屏只渲染最近 30 条（每条都要跑 marked + DOMPurify，长会话一次性渲染会卡顿）；
// 更早的消息通过顶部"加载更早消息"占位条（点击或滚动到顶触发）向前分页渲染
const HISTORY_PAGE_SIZE = 30;
let _historyAll = [];     // 当前会话全部历史（分页数据源）
let _historyRendered = 0; // 已渲染条数（从尾部向前渲染）

function renderHistoryMessages(messages) {
    // 先清掉骨架屏/残留内容（历史为空的分支也会整体覆盖，这里统一处理）
    dom.chatMessages.innerHTML = '';
    _historyAll = messages;
    _historyRendered = 0;
    if (!messages.length) {
        dom.chatMessages.innerHTML = '<div class="empty-state">该会话暂无消息</div>';
        return;
    }
    // 首屏只渲染最近一页
    renderHistoryPage(false);
    if (_historyRendered < _historyAll.length) showHistoryMore();
    scrollToBottom();
}

// 渲染一页历史；prepend=true 时插入到已有内容之前并补偿 scrollTop 保持视口位置
function renderHistoryPage(prepend) {
    const end = _historyAll.length - _historyRendered;
    const start = Math.max(0, end - HISTORY_PAGE_SIZE);
    const page = _historyAll.slice(start, end);
    if (!page.length) return;
    _historyRendered += page.length;

    // 构造本页 HTML（结构与 appendUserMessage/appendAssistantMessage 保持一致）
    const html = page.map((m) => {
        if (m.role === 'user') return buildUserMessage(m.content).html;
        if (m.role === 'assistant') return buildAssistantMessage(m.content).html;
        return ''; // 系统消息（如压缩摘要）不显示在聊天界面
    }).join('');

    if (prepend) {
        const more = dom.chatMessages.querySelector('.history-more');
        // 记录插入前高度，插入后补偿滚动差值（用户视口不跳动）
        const prevHeight = dom.chatMessages.scrollHeight;
        if (more) {
            more.insertAdjacentHTML('afterend', html);
        } else {
            dom.chatMessages.insertAdjacentHTML('afterbegin', html);
        }
        dom.chatMessages.scrollTop += dom.chatMessages.scrollHeight - prevHeight;
    } else {
        dom.chatMessages.insertAdjacentHTML('beforeend', html);
    }
    // 本页代码块异步补高亮（hljs 按需加载）
    AppCommon.highlightCodeBlocks(dom.chatMessages);
}

// 顶部"加载更早消息"占位条
function showHistoryMore() {
    if (dom.chatMessages.querySelector('.history-more')) return;
    const bar = document.createElement('div');
    bar.className = 'history-more';
    bar.textContent = '↑ 加载更早消息';
    bar.addEventListener('click', loadEarlierHistory);
    dom.chatMessages.prepend(bar);
}

function loadEarlierHistory() {
    if (_historyRendered >= _historyAll.length) return;
    renderHistoryPage(true);
    // 全部加载完移除占位条
    if (_historyRendered >= _historyAll.length) {
        dom.chatMessages.querySelector('.history-more')?.remove();
    }
}

function buildAssistantMessage(text) {
    const id = 'msg-assistant-' + crypto.randomUUID();
    const html = `
        <div class="message assistant" id="${id}">
            <div class="message-avatar">🔬</div>
            <div class="message-content-wrap">
                <div class="message-content">${AppCommon.renderMarkdown(text)}</div>
            </div>
        </div>
    `;
    return { id, html };
}

function appendAssistantMessage(text) {
    const { id, html } = buildAssistantMessage(text);
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    AppCommon.highlightCodeBlocks(document.getElementById(id)); // 异步补高亮（hljs 按需加载）
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

function buildUserMessage(text) {
    const id = 'msg-' + crypto.randomUUID();
    const html = `
        <div class="message user" id="${id}">
            <div class="message-avatar">👤</div>
            <div class="message-content-wrap">
                <div class="message-content">${AppCommon.escapeHtml(text)}</div>
            </div>
        </div>
    `;
    return { id, html };
}

function appendUserMessage(text) {
    const { id, html } = buildUserMessage(text);
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return id;
}

function appendAssistantPlaceholder() {
    const id = 'msg-assistant-' + crypto.randomUUID();
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
        el.innerHTML = AppCommon.renderMarkdown(text);
        AppCommon.highlightCodeBlocks(el); // 异步补高亮（hljs 按需加载）
    } else {
        // 流式更新时显示纯文本（避免频繁渲染 Markdown）
        el.innerHTML = AppCommon.escapeHtml(text).replace(/\n/g, '<br>');
    }
    autoScroll();
}

function appendStatusMessage(text) {
    const id = 'status-' + crypto.randomUUID();
    const html = `
        <div class="message assistant" id="${id}">
            <div class="message-avatar">ℹ️</div>
            <div class="message-content-wrap">
                <div class="message-content" style="font-size:13px;color:var(--text-secondary)">${AppCommon.escapeHtml(text)}</div>
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
                <div class="message-content error-message">${AppCommon.escapeHtml(text)}</div>
            </div>
        </div>
    `;
    dom.chatMessages.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

// ---- 滚动：吸底逻辑 ----
// 用户上翻后暂停自动吸底；距底 <50px 才认为"在底部"
let _stickToBottom = true;

function isNearBottom() {
    const el = dom.chatMessages;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 50;
}

// 无条件滚到底（用户主动动作：发消息、点"回到底部"、切换会话）
function scrollToBottom() {
    _stickToBottom = true;
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    updateBackToBottom();
}

// 流式渲染期间的自动滚动：仅在用户原本就接近底部时吸底
function autoScroll() {
    if (_stickToBottom) {
        dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    }
    updateBackToBottom();
}

function updateBackToBottom() {
    if (dom.backToBottom) {
        dom.backToBottom.style.display = _stickToBottom ? 'none' : 'flex';
    }
}

// 流式期间关闭 smooth 滚动动画（style.css 的 scroll-behavior: smooth 会让 scrollTop 赋值变成动画）
function setStreamScroll(on) {
    dom.chatMessages.classList.toggle('streaming', on);
}

// 连接异常横幅（onStale 触发显示，续流恢复后隐藏）
function showStaleBanner() {
    if (dom.staleBanner) dom.staleBanner.style.display = 'block';
}

function hideStaleBanner() {
    if (dom.staleBanner) dom.staleBanner.style.display = 'none';
}

// ---- 发送/停止按钮状态 ----
// 流式期间 btnSend 变为"停止"态（红色 ⏹）：点击中止当前流而非禁用，让用户可主动取消
const SEND_ICON_HTML = dom.btnSend.innerHTML;
const STOP_ICON_HTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>';

function setSendBtnStop(on) {
    dom.btnSend.classList.toggle('btn-stop', on);
    dom.btnSend.title = on ? '停止生成' : '发送';
    dom.btnSend.innerHTML = on ? STOP_ICON_HTML : SEND_ICON_HTML;
}

// 用户主动中止的收尾：保留已生成内容，尾部加"已停止"标记（与 cancelled 事件路径一致）
function markMessageStopped(assistantId, emptyText = '⏹️ 对话已中断') {
    const el = document.querySelector(`#${assistantId} .message-content`);
    if (!el || el.querySelector('.stopped-mark')) return;
    el.querySelector('.typing-indicator')?.remove();
    // 尚无任何内容时显示占位文案
    if (!el.textContent.trim()) {
        el.textContent = emptyText;
        return;
    }
    const mark = document.createElement('div');
    mark.className = 'stopped-mark';
    mark.textContent = '⏹️ 已停止';
    el.appendChild(mark);
}

// ============================================
// Agent 事件分发（sendMessage / handleApproval / 刷新续流共用）
// 按六个渲染通道组织：phase / content / tools / sources / approval / terminal
// ============================================
// opts.onEventId(id)：每个事件帧 id 回调（用于断点续流的 lastEventId 跟踪）
// opts.onSessionId(sid)：start 事件带回 session_id 时回调
// opts.onGap()：token 序号跳号时回调（触发断帧续流）
// opts.onCancelled()：收到 cancelled 终结事件时回调（清理续流状态）
// opts.onRetry()：错误卡片"重试"按钮回调（缺省则不显示按钮）

// 每个气泡当前生效的处理器代次：断帧/断流重续时作废旧处理器，
// 防止新旧两个连接的事件同时写同一气泡
const _handlerEpoch = {};

function createAgentEventHandler(assistantId, opts = {}) {
    let fullAnswer = '';
    let thinkingText = '';
    // error 为终态标志：防御后端新旧行为差异（旧行为 error 后可能仍发 done）
    let errored = false;
    // 流是否已到终态（done/error/cancelled/approval_required）：
    // 用于区分"用户点停止"与"正常结束后 abort 信号滞后"，避免给已完成消息误加"已停止"标记
    let terminated = false;
    // token 序号跟踪：null 表示尚未建立基线（旧后端/回放帧没有 index，直接跳过检测）
    let expectedIndex = null;
    let gapWarned = false;

    const epoch = (_handlerEpoch[assistantId] = (_handlerEpoch[assistantId] || 0) + 1);
    const getContentEl = () => document.querySelector(`#${assistantId} .message-content`);

    // ---- content 通道：流式文本增量 append（避免每帧全量重写 innerHTML 的 O(n²)） ----
    let streamTextEl = null;
    let streamTextNode = null;
    function ensureStreamText() {
        if (streamTextNode) return;
        const el = getContentEl();
        if (!el) return;
        // 首次有内容时移除"思考中"占位
        el.querySelector('.typing-indicator')?.remove();
        streamTextEl = document.createElement('div');
        streamTextEl.className = 'stream-text';
        streamTextNode = document.createTextNode('');
        streamTextEl.appendChild(streamTextNode);
        el.appendChild(streamTextEl);
    }

    // 打字机：token 先进缓冲，rAF 平滑吐字；onRender 语义为增量（delta 追加到文本节点）
    const typewriter = StreamClient.Typewriter((delta) => {
        ensureStreamText();
        if (streamTextNode) streamTextNode.appendData(delta);
        autoScroll();
    });

    // ---- phase 通道：助手气泡头部的小阶段标签 ----
    const PHASE_LABELS = { retrieving: '🔍 检索中', generating: '✍️ 生成中' };
    function updatePhaseLabel(phase) {
        const el = getContentEl();
        if (!el) return;
        let label = el.querySelector('.phase-label');
        if (!label) {
            label = document.createElement('div');
            label.className = 'phase-label';
            el.prepend(label);
        }
        label.textContent = PHASE_LABELS[phase] || phase;
    }
    function clearPhaseLabel() {
        getContentEl()?.querySelector('.phase-label')?.remove();
    }

    // ---- tools 通道：tool_call/tool_result 按 name+出现顺序配对成卡片 ----
    const toolCards = []; // { name, args, result, truncated, summaryEl, detailEl }
    let toolAreaEl = null;

    function ensureToolArea() {
        if (toolAreaEl) return toolAreaEl;
        const el = getContentEl();
        if (!el) return null;
        el.querySelector('.typing-indicator')?.remove();
        toolAreaEl = document.createElement('div');
        toolAreaEl.className = 'tool-cards';
        el.appendChild(toolAreaEl);
        return toolAreaEl;
    }

    // 折叠摘要的关键参数：取 args 的第一个字符串值（截断 30 字）
    function firstStringArg(args) {
        if (args == null) return '';
        if (typeof args === 'string') {
            try { args = JSON.parse(args); } catch { return args.slice(0, 30); }
        }
        if (typeof args === 'object') {
            for (const k of Object.keys(args)) {
                if (typeof args[k] === 'string') return args[k].slice(0, 30);
            }
            try { return JSON.stringify(args).slice(0, 30); } catch { return ''; }
        }
        return String(args).slice(0, 30);
    }

    function formatArgs(args) {
        if (args == null) return '';
        if (typeof args === 'string') return args;
        try { return JSON.stringify(args, null, 2); } catch { return String(args); }
    }

    // 折叠态单行摘要：[工具名] 关键参数 -> 结果摘要（结果截断 50 字）
    function updateToolSummary(card) {
        let text = `🔧 [${card.name}]`;
        const argPreview = firstStringArg(card.args);
        if (argPreview) text += ` ${argPreview}`;
        text += card.result != null ? ` -> ${String(card.result).slice(0, 50)}` : '（执行中…）';
        if (card.truncated) text += ' ✂';
        card.summaryEl.textContent = text;
    }

    function addToolCard(content) {
        const area = ensureToolArea();
        if (!area) return null;
        const card = {
            name: content.name || '未知工具',
            args: content.args,
            result: null,
            truncated: !!content.truncated,
        };
        const el = document.createElement('div');
        el.className = 'tool-card';
        const summary = document.createElement('div');
        summary.className = 'tool-card-summary';
        summary.title = '点击展开/收起';
        const detail = document.createElement('div');
        detail.className = 'tool-card-detail';
        detail.style.display = 'none';
        el.appendChild(summary);
        el.appendChild(detail);
        // 点击折叠摘要切换展开/收起
        summary.addEventListener('click', () => {
            detail.style.display = detail.style.display === 'none' ? 'block' : 'none';
        });
        card.summaryEl = summary;
        card.detailEl = detail;
        updateToolSummary(card);
        toolCards.push(card);
        area.appendChild(el);
        return card;
    }

    function fillToolResult(content) {
        // 按 name+出现顺序配对：取最早一张同名且尚无结果的卡片
        let card = toolCards.find((c) => c.name === content.name && c.result === null);
        if (!card) {
            // 没有匹配的调用卡片（如回放缺帧）：补一张只有结果的卡片
            card = addToolCard({ name: content.name });
            if (!card) return;
        }
        card.result = content.content ?? '';
        card.truncated = !!content.truncated;
        updateToolSummary(card);
        // 展开区：完整参数 + 完整结果 + 截断标记（均用 textContent，防 XSS）
        const detail = card.detailEl;
        detail.innerHTML = '';
        if (card.args != null) {
            const argsPre = document.createElement('pre');
            argsPre.textContent = formatArgs(card.args);
            detail.appendChild(argsPre);
        }
        const resultPre = document.createElement('pre');
        resultPre.textContent = String(card.result);
        detail.appendChild(resultPre);
        if (card.truncated) {
            const mark = document.createElement('div');
            mark.className = 'tool-card-truncated';
            mark.textContent = '✂ 结果已截断';
            detail.appendChild(mark);
        }
    }

    // 工具事件合并：300ms 窗口内多条 tool_call/tool_result 批量处理
    const mergeToolEvents = StreamClient.throttleMerge((batch) => {
        for (const item of batch) {
            if (item.type === 'tool_call') addToolCard(item.content || {});
            else if (item.type === 'tool_result') fillToolResult(item.content || {});
        }
        autoScroll();
    });

    // 序号跳号：警告 + 触发一次断帧续流（每次流只处理一次）
    function onIndexGap() {
        if (gapWarned) return;
        gapWarned = true;
        const el = getContentEl();
        if (el && !el.querySelector('.stream-warn')) {
            const warn = document.createElement('div');
            warn.className = 'stream-warn';
            warn.textContent = '⚠️ 内容可能不完整，正在尝试恢复…';
            el.appendChild(warn);
        }
        if (opts.onGap) opts.onGap();
    }

    // terminal 通道：消息内嵌错误卡片（红框 + 错误信息 + 重试按钮）
    function showInlineError(errMsg) {
        // 后端异常原文可能很长（traceback 等），超过 200 字不上屏
        if (typeof errMsg === 'string' && errMsg.length > 200) {
            console.warn('后端原始错误:', errMsg);
            errMsg = '服务异常，请重试';
        }
        const el = getContentEl();
        if (!el) return;
        el.querySelector('.typing-indicator')?.remove();
        const card = document.createElement('div');
        card.className = 'stream-error-card';
        const text = document.createElement('div');
        text.className = 'stream-error-text';
        text.textContent = `❌ ${errMsg}`;
        card.appendChild(text);
        if (opts.onRetry) {
            const btn = document.createElement('button');
            btn.className = 'stream-retry-btn';
            btn.textContent = '重试';
            btn.addEventListener('click', () => opts.onRetry());
            card.appendChild(btn);
        }
        el.appendChild(card);
        autoScroll();
    }

    const handler = (event) => {
        if (_handlerEpoch[assistantId] !== epoch) return; // 已作废（断帧续流后旧连接的事件）
        if (typeof event.id === 'number' && opts.onEventId) opts.onEventId(event.id);
        switch (event.type) {
            case 'start': {
                const sid = event.content?.session_id;
                if (sid) {
                    if (!state.currentSessionId) {
                        state.currentSessionId = sid;
                        loadSessions();
                    }
                    if (opts.onSessionId) opts.onSessionId(sid);
                }
                break;
            }
            // ---- phase 通道 ----
            case 'status': {
                const status = event.content;
                if (status?.memory) {
                    // 记忆状态：后端字符串必须转义后再拼接（防 XSS）
                    const info = status.memory.map((m) => AppCommon.escapeHtml(String(m))).join(' · ');
                    dom.memoryStatus.innerHTML = `<span class="dot"></span><span>${info}</span>`;
                }
                if (status?.phase) updatePhaseLabel(status.phase);
                break;
            }
            case 'thinking': {
                thinkingText += event.content || '';
                break;
            }
            // ---- tools 通道 ----
            case 'tool_call':
            case 'tool_result': {
                mergeToolEvents(event);
                break;
            }
            // ---- content 通道 ----
            case 'token': {
                const token = event.content || '';
                // 序号跳号检测：以首个收到的序号建立基线（续流回放可能从中间开始）
                if (typeof event.index === 'number') {
                    if (expectedIndex === null) {
                        expectedIndex = event.index + 1;
                    } else {
                        if (event.index > expectedIndex) onIndexGap();
                        if (event.index >= expectedIndex) expectedIndex = event.index + 1;
                    }
                }
                fullAnswer += token;
                typewriter.push(token);
                break;
            }
            case 'answer': {
                // 完整答案兜底（deprecated 事件）：仅在没有任何 token 累积时整体渲染
                if (!fullAnswer && event.content) {
                    const text = typeof event.content === 'string' ? event.content : (event.content.text || '');
                    if (text) {
                        fullAnswer = text;
                        typewriter.push(text);
                    }
                }
                break;
            }
            // ---- sources 通道 ----
            case 'sources': {
                const sources = Array.isArray(event.content) ? event.content : [];
                renderSources(sources);
                break;
            }
            // ---- approval 通道 ----
            case 'approval_required': {
                // HITL 中断：停打字机、输入框置等待审批态、消息流内嵌审批气泡
                terminated = true;
                state.pendingApproval = event.content;
                typewriter.finish();
                clearPhaseLabel();
                // 等待审批期间：按钮恢复"发送"态但禁用（审批期点停止语义复杂，保持禁用现状）
                setSendBtnStop(false);
                dom.btnSend.disabled = true;
                // 气泡内追加等待提示（保留工具卡片等已有内容，不做整覆渲染）
                const el = getContentEl();
                if (el) {
                    el.querySelector('.typing-indicator')?.remove();
                    const note = document.createElement('div');
                    note.className = 'approval-waiting-note';
                    note.textContent = '⏳ 等待人工确认...';
                    el.appendChild(note);
                }
                showApprovalBubble(event.content);
                break;
            }
            // ---- terminal 通道 ----
            case 'error': {
                terminated = true;
                errored = true;
                typewriter.finish();
                clearPhaseLabel();
                const err = event.content?.message || '未知错误';
                showInlineError(err);
                break;
            }
            case 'done': {
                terminated = true;
                const finalText = typewriter.finish() || fullAnswer;
                // done 与 error 互斥；若已收到 error 则跳过覆盖渲染
                if (errored) break;
                clearPhaseLabel();
                // 最终渲染 Markdown（renderMarkdown 内已过 DOMPurify）；
                // 只替换正文区域，保留工具卡片等附加元素
                const el = getContentEl();
                if (el) {
                    if (streamTextEl) {
                        streamTextEl.classList.remove('stream-text');
                        streamTextEl.innerHTML = AppCommon.renderMarkdown(finalText);
                    } else {
                        el.innerHTML = AppCommon.renderMarkdown(finalText);
                    }
                    AppCommon.highlightCodeBlocks(el); // 异步补高亮（hljs 按需加载）
                }
                // 如果有思考过程，追加折叠框
                if (thinkingText) {
                    const contentEl = getContentEl();
                    if (contentEl) {
                        contentEl.insertAdjacentHTML('afterbegin', `
                            <details class="thinking-box">
                                <summary>💡 思考过程</summary>
                                <pre>${AppCommon.escapeHtml(thinkingText)}</pre>
                            </details>
                        `);
                    }
                }
                autoScroll();
                // 刷新会话列表，更新消息计数和时间
                loadSessions();
                break;
            }
            case 'cancelled': {
                // 用户中止/断连：保留已生成内容，尾部加"已停止"标记；不触发续流
                terminated = true;
                typewriter.finish();
                clearPhaseLabel();
                const el = getContentEl();
                if (el) {
                    el.querySelector('.typing-indicator')?.remove();
                    if (!el.querySelector('.stopped-mark')) {
                        const mark = document.createElement('div');
                        mark.className = 'stopped-mark';
                        mark.textContent = '⏹️ 已停止';
                        el.appendChild(mark);
                    }
                }
                if (opts.onCancelled) opts.onCancelled();
                autoScroll();
                break;
            }
        }
    };
    return { handler, typewriter, isTerminated: () => terminated };
}

// ============================================
// HITL 审批气泡（仅消息流内嵌，顶部 sticky 面板已移除）
// ============================================
let _approvalMsgId = null;

function showApprovalBubble(data) {
    const toolCalls = data.tool_calls || [];
    const message = data.message || 'Agent 请求执行敏感操作，请确认是否允许。';

    // 输入框置"等待审批"禁用态
    if (dom.chatInputArea) dom.chatInputArea.classList.add('pending-approval');
    dom.chatInput.disabled = true;
    dom.chatInput.placeholder = '等待人工审批...';

    // 在消息流中插入可交互的审批气泡
    _approvalMsgId = 'msg-approval-' + crypto.randomUUID();
    const toolsHtml = toolCalls.map((tc) => {
        const name = tc.name || '未知工具';
        const argsStr = JSON.stringify(tc.args || tc.arguments || {}, null, 2);
        return `<div class="approval-tool-item"><code>${AppCommon.escapeHtml(name)}</code><pre>${AppCommon.escapeHtml(argsStr)}</pre></div>`;
    }).join('');

    const html = `
        <div class="message assistant approval-message" id="${_approvalMsgId}">
            <div class="message-avatar">⚠️</div>
            <div class="message-content-wrap">
                <div class="message-content">
                    <div class="approval-bubble">
                        <div class="approval-bubble-title">🔒 敏感操作需要确认</div>
                        <div class="approval-bubble-desc">${AppCommon.escapeHtml(message)}</div>
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

function clearApprovalBubble() {
    // 恢复输入框
    if (dom.chatInputArea) dom.chatInputArea.classList.remove('pending-approval');
    dom.chatInput.disabled = false;
    dom.chatInput.placeholder = '输入问题，按 Enter 发送，Shift+Enter 换行...';

    // 保留审批气泡历史记录，仅禁用按钮
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

    // 关闭审批气泡并恢复输入框
    clearApprovalBubble();

    // 在聊天区显示用户的选择
    const actionText = action === 'approve' ? '✓ 已确认执行该操作' : '✕ 已拒绝执行该操作';
    appendStatusMessage(actionText);

    // 显示新的助手占位（继续等待后续输出）
    const assistantId = appendAssistantPlaceholder();
    const { handler, typewriter, isTerminated } = createAgentEventHandler(assistantId);

    // 审批恢复流同样可中止：按钮进入"停止"态
    state.isStreaming = true;
    setSendBtnStop(true);
    dom.btnSend.disabled = false;
    const ctrl = new AbortController();
    state.abortController = ctrl;
    setStreamScroll(true);

    try {
        await api.approveStream(session_id, action, comment, handler, ctrl.signal);
        // 用户点"停止"且连接干净关闭（不抛 AbortError）时在这里收尾
        if (ctrl.signal.aborted && !isTerminated()) {
            typewriter.finish();
            markMessageStopped(assistantId, '⏹️ 操作已中断');
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            console.log('审批恢复已中止');
            if (!isTerminated()) {
                typewriter.finish();
                markMessageStopped(assistantId, '⏹️ 操作已中断');
            }
        } else {
            console.error('审批恢复失败:', e);
            updateAssistantMessage(assistantId, `❌ 恢复失败: ${AppCommon.formatError(e)}`, true);
        }
        loadSessions();
    } finally {
        state.isStreaming = false;
        state.pendingApproval = null;
        setSendBtnStop(false);
        dom.btnSend.disabled = false;
        state.abortController = null;
        setStreamScroll(false);
        autoScroll();
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
        <div class="source-item" title="点击展开/收起">
            <div class="source-header">
                <span class="source-index">#${AppCommon.escapeHtml(String(s.index))}</span>
                <span class="source-score">相似度 ${(s.score * 100).toFixed(1)}%</span>
            </div>
            <div class="source-name">${AppCommon.escapeHtml(s.source)}</div>
            <div class="source-snippet">${AppCommon.escapeHtml(s.snippet)}</div>
        </div>
    `).join('');

    // 绑定点击展开/收起（默认 4 行截断，展开后显示全文）
    dom.sourcesList.querySelectorAll('.source-item').forEach((el) => {
        el.addEventListener('click', () => el.classList.toggle('expanded'));
    });
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
    state.isStreaming = true;
    // 流式期间按钮变为"停止"态而非禁用：点击可主动中止（走 abort → 保留内容 + 已停止标记）
    setSendBtnStop(true);
    setStreamScroll(true);

    // 显示用户消息
    appendUserMessage(text);

    // 显示助手占位
    const assistantId = appendAssistantPlaceholder();

    // 刷新续流：记录当前 run，页面刷新后可通过 /agent/chat/resume 断点续传
    const runState = {
        sessionId: state.currentSessionId || 'default',
        question: text,
        lastEventId: 0,
        runActive: true,
        gapResuming: false,
    };
    persistRunState(runState, true); // run 开始：立即落盘（刷新续流的起点）

    const { handler, typewriter, isTerminated } = createAgentEventHandler(assistantId, {
        onEventId(id) {
            runState.lastEventId = id;
            persistRunState(runState); // 每帧高频写入：走节流
        },
        onSessionId(sid) {
            runState.sessionId = sid;
            persistRunState(runState);
        },
        onGap() {
            // token 序号跳号：中止当前连接，走断点续流重建气泡内容
            resumeInterruptedRun(assistantId, runState);
        },
        onCancelled() {
            // cancelled 是终态：不再保留续流状态（立即落盘）
            runState.runActive = false;
            persistRunState(runState, true);
        },
        onRetry() {
            if (state.isStreaming) return;
            dom.chatInput.value = text;
            sendMessage();
        },
    });

    const ctrl = new AbortController();
    state.abortController = ctrl;

    // 连接活性监视：超过 30s 无任何字节（含心跳）→ 横幅提示 + 自动续流
    let staleHandled = false;
    const sseOpts = {
        onStale() {
            if (staleHandled) return;
            staleHandled = true;
            showStaleBanner();
            resumeInterruptedRun(assistantId, runState);
        },
    };

    try {
        await api.chatStream(text, state.currentSessionId, handler, ctrl.signal, sseOpts);
        // 用户点"停止"且连接干净关闭（reader.cancel 路径，不抛 AbortError）时在这里收尾
        if (ctrl.signal.aborted && !runState.gapResuming && !isTerminated()) {
            typewriter.finish();
            markMessageStopped(assistantId);
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            console.log('对话已中止');
            // 断帧/断流续流触发的主动中止不覆盖气泡（内容由续流连接重建）
            if (!runState.gapResuming && !isTerminated()) {
                // 用户主动停止：冲刷打字机缓冲，保留已生成内容，尾部加"已停止"标记
                typewriter.finish();
                markMessageStopped(assistantId);
            }
        } else {
            console.error('对话失败:', e);
            if (!runState.gapResuming) {
                // 已有流式内容/工具卡片时内嵌错误提示，否则整覆渲染错误消息
                const el = document.querySelector(`#${assistantId} .message-content`);
                if (el && el.querySelector('.stream-text, .tool-cards')) {
                    el.querySelector('.typing-indicator')?.remove();
                    const err = document.createElement('div');
                    err.className = 'stream-error-card';
                    err.textContent = `❌ 请求失败: ${AppCommon.formatError(e)}`;
                    el.appendChild(err);
                } else {
                    updateAssistantMessage(assistantId, `❌ 请求失败: ${AppCommon.formatError(e)}`, true);
                }
            }
        }
        // 刷新会话列表（即使失败也要更新时间显示）
        loadSessions();
    } finally {
        // 断帧/断流续流接管期间，由 resumeInterruptedRun 负责终态清理
        if (!runState.gapResuming) {
            // run 结束（done/error/abort/异常）：标记不再活跃，刷新页面不再续流
            // 注意：HITL 审批中断也算流结束，审批恢复走的是 /agent/approve 而非续流
            runState.runActive = false;
            persistRunState(runState, true); // 终态：立即落盘
            // 如果触发了审批，保持 streaming 状态直到用户处理（按钮保持禁用，
            // 审批事件处理器中已恢复"发送"态图标）
            if (!state.pendingApproval) {
                state.isStreaming = false;
                setSendBtnStop(false);
                dom.btnSend.disabled = false;
            }
            state.abortController = null;
            setStreamScroll(false);
        }
        autoScroll();
    }
}

// ============================================
// 断帧/断流恢复（序号跳号 / onStale 触发）
// ============================================
// 中止当前连接，从服务端缓冲头部完整回放（lastEventId=0），由同一气泡重建内容
async function resumeInterruptedRun(assistantId, runState) {
    if (runState.gapResuming) return; // 防重入
    runState.gapResuming = true;

    // 中止旧连接；旧处理器因代次作废，其后续事件静默丢弃
    if (state.abortController) state.abortController.abort();

    const { handler } = createAgentEventHandler(assistantId, {
        onEventId(id) {
            runState.lastEventId = id;
            persistRunState(runState); // 每帧高频写入：走节流
        },
        onCancelled() {
            runState.runActive = false;
            persistRunState(runState, true); // 终态：立即落盘
        },
    });

    // 首帧到达时才清空气泡并隐藏横幅：回放以服务端缓冲为准重建内容；
    // 若续流失败（如无缓冲），保留原有部分内容与警告
    let firstFrame = true;
    const wrappedHandler = (event) => {
        if (firstFrame) {
            firstFrame = false;
            hideStaleBanner();
            const el = document.querySelector(`#${assistantId} .message-content`);
            if (el) el.innerHTML = '';
        }
        handler(event);
    };

    state.abortController = new AbortController();
    setStreamScroll(true);
    try {
        await authCtx.sseConnect({
            url: `${API_BASE}/agent/chat/resume`,
            body: { session_id: runState.sessionId },
            lastEventId: 0,
            onEvent: wrappedHandler,
            signal: state.abortController.signal,
        });
    } catch (e) {
        if (e.name !== 'AbortError') {
            console.warn('断流恢复失败:', e.status || e.message);
            appendStatusMessage(`⚠️ 连接恢复失败: ${AppCommon.formatError(e)}`);
        }
    } finally {
        runState.gapResuming = false;
        runState.runActive = false;
        persistRunState(runState, true); // 终态：立即落盘
        hideStaleBanner();
        state.isStreaming = false;
        setSendBtnStop(false);
        dom.btnSend.disabled = false;
        state.abortController = null;
        setStreamScroll(false);
        autoScroll();
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
    // 发送按钮：流式期间为"停止"态，点击分发为中止；否则发送
    dom.btnSend.addEventListener('click', () => {
        if (state.isStreaming) {
            if (state.abortController) state.abortController.abort();
            return;
        }
        sendMessage();
    });

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

    // 会话列表点击：容器级事件委托（renderSessionList 做 diff 复用节点，不再逐项绑定）
    // 删除按钮优先于切换会话
    dom.sessionList.addEventListener('click', (e) => {
        const delBtn = e.target.closest('.session-delete');
        if (delBtn) {
            const item = delBtn.closest('.session-item[data-id]');
            if (item) deleteSession(item.dataset.id);
            return;
        }
        const item = e.target.closest('.session-item[data-id]');
        if (item) switchSession(item.dataset.id);
    });

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

    // 滚动吸底状态跟踪（用户上翻暂停吸底）+ "回到底部"按钮
    dom.chatMessages.addEventListener('scroll', () => {
        _stickToBottom = isNearBottom();
        updateBackToBottom();
        // 滚动到顶附近且有更早历史时自动向前分页渲染
        if (dom.chatMessages.scrollTop < 80 && dom.chatMessages.querySelector('.history-more')) {
            loadEarlierHistory();
        }
    });
    if (dom.backToBottom) {
        dom.backToBottom.addEventListener('click', scrollToBottom);
    }
}

// ============================================
// 认证模块
// ============================================

function showAuthModal() {
    $('authOverlay').style.display = 'flex';
    // 主界面虚化（替代原 .auth-overlay + .app 死选择器：DOM 中 .app 在遮罩之前，兄弟选择器永不匹配）
    document.querySelector('.app')?.classList.add('auth-blurred');
}

function hideAuthModal() {
    $('authOverlay').style.display = 'none';
    document.querySelector('.app')?.classList.remove('auth-blurred');
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
            <span class="user-name">👤 ${AppCommon.escapeHtml(state.user.username)}</span>
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

// 刷新续流：页面加载时检测未完成的 run，通过 /agent/chat/resume 断点续传
async function resumeActiveRun() {
    let run = null;
    try {
        run = JSON.parse(sessionStorage.getItem('agent_run') || 'null');
    } catch {
        run = null;
    }
    if (!run || !run.runActive || !state.token || !run.sessionId) return;

    // 重建对话现场（先清屏，避免与自动恢复的历史消息重复）
    clearChat();
    if (dom.welcomeScreen) dom.welcomeScreen.style.display = 'none';
    if (run.question) appendUserMessage(run.question);
    const assistantId = appendAssistantPlaceholder();

    state.isStreaming = true;
    // 刷新续流期间同样允许主动中止
    setSendBtnStop(true);
    const ctrl = new AbortController();
    state.abortController = ctrl;
    setStreamScroll(true);

    const { handler, typewriter, isTerminated } = createAgentEventHandler(assistantId, {
        onEventId(id) {
            run.lastEventId = id;
            persistRunState(run); // 每帧高频写入：走节流
        },
    });

    try {
        await authCtx.sseConnect({
            url: `${API_BASE}/agent/chat/resume`,
            body: { session_id: run.sessionId },
            lastEventId: run.lastEventId || 0,
            onEvent: handler,
            signal: ctrl.signal,
        });
        // 用户点"停止"且连接干净关闭（不抛 AbortError）时在这里收尾
        if (ctrl.signal.aborted && !isTerminated()) {
            typewriter.finish();
            markMessageStopped(assistantId);
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            // 用户主动停止：保留已生成内容 + "已停止"标记
            if (!isTerminated()) {
                typewriter.finish();
                markMessageStopped(assistantId);
            }
        } else {
            // 404 或失败：静默降级，不影响正常对话
            console.warn('断点续流失败:', e.status || e.message);
            // 清掉无法续上的占位气泡
            document.getElementById(assistantId)?.remove();
        }
    } finally {
        // 无论续流成败，run 痕迹都不再保留（先取消挂起的节流写入，防止尾写复活）
        cancelPersistRunState(run);
        sessionStorage.removeItem('agent_run');
        state.isStreaming = false;
        setSendBtnStop(false);
        dom.btnSend.disabled = false;
        state.abortController = null;
        setStreamScroll(false);
    }
}

function init() {
    bindEvents();
    bindAuthEvents();
    // 认证检查完成后尝试断点续流（无 token 时 resumeActiveRun 内部直接返回）
    checkAuth().finally(() => resumeActiveRun());
    console.log('🔬 材料科学 Agent 前端已加载');
}

document.addEventListener('DOMContentLoaded', init);
