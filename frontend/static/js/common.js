/**
 * 两页公共函数库（index.html app.js / rag.html rag.js 共用）
 * 功能：HTML 转义、Markdown 渲染（DOMPurify 收口 + hljs 按需加载）、
 *       错误友好化、认证上下文（token 刷新 + SSE 连接 401 重试）
 * 纯原生 JS，无构建依赖，挂载到 window.AppCommon
 */
(function (global) {
    'use strict';

    const API_BASE = '/api/v1';

    // ============================================
    // HTML 转义
    // ============================================
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ============================================
    // Markdown 渲染
    // ============================================
    // marked 全局配置：模块初始化执行一次（不必每次渲染重复 setOptions）
    if (global.marked) {
        global.marked.setOptions({
            breaks: true,
            gfm: true,
        });
    }

    // ---- highlight.js 按需加载 ----
    // 检测到代码块时才动态加载（单例 Promise 防重复注入）；加载失败静默降级为纯文本代码块
    let _hljsPromise = null;

    function ensureHljs() {
        if (global.hljs) return Promise.resolve(global.hljs);
        if (!_hljsPromise) {
            _hljsPromise = new Promise((resolve) => {
                const script = document.createElement('script');
                script.src = '/static/vendor/highlight.js/highlight.min.js';
                script.onload = () => resolve(global.hljs);
                script.onerror = () => {
                    console.warn('highlight.js 加载失败，代码块将不高亮');
                    resolve(null);
                };
                document.head.appendChild(script);
            });
        }
        return _hljsPromise;
    }

    // 对容器内未高亮的 pre code 逐个补高亮（直接操作 DOM，替代脆弱的正则替换路径）：
    // 有 language-x class 且 hljs 支持该语言 → highlight(code, {language})；
    // 否则 highlightAuto（无语言代码块也能获得高亮）
    function highlightCodeBlocks(container) {
        if (!container || !container.querySelector('pre code')) return;
        ensureHljs().then((hljs) => {
            if (!hljs) return;
            container.querySelectorAll('pre code:not(.hljs)').forEach((codeEl) => {
                const m = codeEl.className.match(/language-(\w+)/);
                const lang = m && hljs.getLanguage(m[1]) ? m[1] : null;
                try {
                    const result = lang
                        ? hljs.highlight(codeEl.textContent, { language: lang })
                        : hljs.highlightAuto(codeEl.textContent);
                    codeEl.innerHTML = result.value;
                    codeEl.classList.add('hljs');
                } catch (e) {
                    console.warn('代码高亮失败:', e);
                }
            });
        });
    }

    function renderMarkdown(text) {
        if (!text) return '';
        const html = global.marked.parse(text);
        // 所有 Markdown 渲染的 innerHTML 注入点统一经 DOMPurify 消毒（防 XSS）；
        // 代码高亮由 highlightCodeBlocks 在插入 DOM 后异步补齐
        return global.DOMPurify ? global.DOMPurify.sanitize(html) : html;
    }

    // ============================================
    // 错误信息友好化
    // ============================================
    // SSE/REST 错误对象的 message 是后端原始响应体
    // （可能是 HTML 错误页或大段 JSON），不直接上屏；原始体 console.warn 保留
    function formatError(e) {
        if (!e) return '未知错误';
        if (e.friendly) return e.message; // 已是友好文案（如"登录已过期"）
        if (e.status) {
            // 尝试从 JSON 响应体取 detail/message 字段
            let msg = null;
            try {
                const parsed = JSON.parse(e.message);
                const v = parsed?.detail ?? parsed?.message;
                if (typeof v === 'string') msg = v;
            } catch { /* 非 JSON 响应体 */ }
            if (msg && msg.length <= 200) return msg;
            console.warn('后端原始错误响应:', e.message);
            return `服务异常（状态码 ${e.status}）`;
        }
        return '网络异常，请检查连接';
    }

    // 构造带 friendly 标记的错误（formatError 原样上屏，用于代码内生成的友好文案）
    function friendlyError(msg) {
        const err = new Error(msg);
        err.friendly = true;
        return err;
    }

    // ============================================
    // 认证上下文（token 状态由调用页注入，不读全局）
    // ============================================
    // ctx = {
    //   getToken(): string|null          当前 access token
    //   getRefreshToken(): string|null   当前 refresh token
    //   setTokens(access, refresh): void 刷新成功后回写
    //   onAuthFail(): void               刷新失败/最终 401 时的登出处理
    // }
    function createAuthContext(ctx) {
        let refreshingPromise = null; // 并发刷新去重

        async function doRefresh() {
            const res = await fetch(`${API_BASE}/auth/refresh`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: ctx.getRefreshToken() }),
            });
            if (!res.ok) throw new Error('刷新令牌失效');
            const data = await res.json();
            ctx.setTokens(data.access_token, data.refresh_token);
            return data.access_token;
        }

        // 带去重的刷新（多个并发请求共享同一次刷新）
        async function refresh() {
            if (!refreshingPromise) {
                refreshingPromise = doRefresh().finally(() => { refreshingPromise = null; });
            }
            return refreshingPromise;
        }

        // Token 将在 5 分钟内过期时自动刷新
        async function ensureTokenValid() {
            const token = ctx.getToken();
            if (!token || !ctx.getRefreshToken()) return;
            try {
                const payload = JSON.parse(atob(token.split('.')[1]));
                const exp = payload.exp * 1000;
                if (Date.now() + 5 * 60 * 1000 >= exp) {
                    await refresh();
                }
            } catch (e) {
                console.warn('Token 解析失败', e);
            }
        }

        // SSE 连接统一封装：自动携带最新 token；401 时刷新 token 并重试一次
        // （仅一次，防循环），仍失败走 ctx.onAuthFail()
        async function sseConnect(connectOpts, retried = false) {
            try {
                await ensureTokenValid();
                const headers = { ...(connectOpts.headers || {}) };
                const token = ctx.getToken();
                if (token) headers['Authorization'] = `Bearer ${token}`;
                return await StreamClient.connect({ ...connectOpts, headers });
            } catch (e) {
                if (e.status !== 401) throw e;
                // 401：有刷新令牌且未重试过 → 刷新后重试一次
                if (!retried && ctx.getRefreshToken()) {
                    try {
                        await refresh();
                    } catch {
                        ctx.onAuthFail();
                        throw friendlyError('登录已过期，请重新登录');
                    }
                    return sseConnect(connectOpts, true);
                }
                ctx.onAuthFail();
                throw friendlyError('登录已过期，请重新登录');
            }
        }

        return { refresh, ensureTokenValid, sseConnect };
    }

    global.AppCommon = {
        escapeHtml,
        renderMarkdown,
        ensureHljs,
        highlightCodeBlocks,
        formatError,
        friendlyError,
        createAuthContext,
    };
})(window);
