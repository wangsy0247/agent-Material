/**
 * 通用 SSE 流客户端
 * 功能：SSE 帧解析、流读取、打字机渲染、事件合并、断点续流连接
 * 纯原生 JS，无构建依赖，挂载到 window.StreamClient
 */
(function (global) {
    'use strict';

    // ============================================
    // SSE 帧解析
    // ============================================

    // 按 \n\n 切帧，帧内逐行解析 id:/event:/data:（data 可多行，按 SSE 规范用 \n 拼接）
    // 返回 { frames, remaining, heartbeat }；frames 元素为 { id, event, data(JSON.parse 后) }
    // heartbeat 表示本批内容中出现注释行（服务器心跳 ": ping"），供连接活性监视使用
    // data JSON 解析失败的帧 console.warn 跳过
    function parseSSEFrames(buffer) {
        const parts = buffer.split('\n\n');
        const remaining = parts.pop() || '';
        const frames = [];
        let heartbeat = false;

        for (const part of parts) {
            if (!part.trim()) continue;
            let id = null;
            let event = null;
            const dataLines = [];

            for (const rawLine of part.split('\n')) {
                const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
                if (!line) continue; // 空行
                if (line.startsWith(':')) { heartbeat = true; continue; } // 注释行（心跳）
                const colon = line.indexOf(':');
                if (colon === -1) continue;
                const field = line.slice(0, colon);
                let value = line.slice(colon + 1);
                if (value.startsWith(' ')) value = value.slice(1);
                if (field === 'id') {
                    const n = Number(value);
                    id = Number.isNaN(n) ? null : n;
                } else if (field === 'event') {
                    event = value;
                } else if (field === 'data') {
                    dataLines.push(value);
                }
            }
            if (!dataLines.length) continue;

            const raw = dataLines.join('\n');
            let data = null;
            try {
                data = JSON.parse(raw);
            } catch (e) {
                console.warn('SSE frame parse error:', raw, e);
                continue;
            }
            frames.push({ id, event, data });
        }
        return { frames, remaining, heartbeat };
    }

    // ============================================
    // 通用流读取器
    // ============================================

    // 消费 fetch response 的 body，逐帧回调 onFrame({ id, event, data })
    // onHeartbeat：收到 SSE 注释行（心跳帧）时回调，用于上层感知"连接还活着"
    // signal：外部 AbortSignal，中止时取消底层读取并抛 AbortError
    async function readStream(response, { onFrame, onHeartbeat, signal } = {}) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // 中止信号：外部 abort 时取消底层读取（挂起的 read() 会以 done 解除阻塞）
        const onAbort = () => { reader.cancel().catch(() => {}); };
        if (signal) {
            if (signal.aborted) {
                reader.cancel().catch(() => {});
                throw new DOMException('Aborted', 'AbortError');
            }
            signal.addEventListener('abort', onAbort, { once: true });
        }

        try {
            while (true) {
                if (signal && signal.aborted) throw new DOMException('Aborted', 'AbortError');
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const { frames, remaining, heartbeat } = parseSSEFrames(buffer);
                buffer = remaining;
                if (heartbeat && onHeartbeat) onHeartbeat();
                for (const frame of frames) {
                    if (onFrame) onFrame(frame);
                }
            }
            // 冲刷末尾残余（最后一帧可能没有结尾的 \n\n）
            buffer += decoder.decode();
            if (buffer.trim()) {
                const { frames, heartbeat } = parseSSEFrames(buffer + '\n\n');
                if (heartbeat && onHeartbeat) onHeartbeat();
                for (const frame of frames) {
                    if (onFrame) onFrame(frame);
                }
            }
        } finally {
            if (signal) signal.removeEventListener('abort', onAbort);
        }
    }

    // ============================================
    // 打字机（rAF 平滑吐字）
    // ============================================

    // token 追加到内部 buffer；rAF 循环每帧取出 max(8, buffer 的 1/4) 字符
    // （不足 8 且未结束则等待积攒）；onRender 回调语义为增量：onRender(delta, fullText)
    // delta 是本次新吐出的片段，fullText 是已吐出的累计文本（上层一般只需 append delta）
    // buffer 空时不排 rAF（不空转）；距上次 push 超过 100ms 且 buffer 有内容则吐出全部
    // （idle 兜底，避免流停顿时尾部 <8 字符长时间不吐出）
    // finish() 立即 flush 全部并停止循环与定时器，返回完整文本
    function Typewriter(onRender) {
        let buffer = '';
        let rendered = '';
        let finished = false;
        let rafId = null;
        let idleTimer = null;

        const clearIdle = () => {
            if (idleTimer !== null) {
                clearTimeout(idleTimer);
                idleTimer = null;
            }
        };

        // 吐出一个片段并回调
        const emit = (chunk) => {
            rendered += chunk;
            buffer = buffer.slice(chunk.length);
            onRender(chunk, rendered);
        };

        const step = () => {
            rafId = null;
            if (finished) return;
            if (buffer.length >= 8) {
                const take = Math.min(buffer.length, Math.max(8, Math.ceil(buffer.length / 4)));
                emit(buffer.slice(0, take));
            }
            // 只在还有待快吐内容（>=8）时自续；<8 的尾部由 idle 定时器兜底吐出，
            // 避免积攒期 60fps 空转
            if (buffer.length >= 8) rafId = requestAnimationFrame(step);
        };

        // idle 兜底：流停顿 100ms 后吐出尾部积攒（不等 8 字符）
        const armIdle = () => {
            clearIdle();
            idleTimer = setTimeout(() => {
                idleTimer = null;
                if (finished || !buffer) return;
                emit(buffer);
            }, 100);
        };

        return {
            // 追加 token 文本
            push(text) {
                if (finished || !text) return;
                buffer += text;
                armIdle();
                if (rafId === null) rafId = requestAnimationFrame(step);
            },
            // 流结束：立即全量 flush 并停止循环与定时器，返回完整文本
            finish() {
                if (!finished) {
                    finished = true;
                    clearIdle();
                    if (rafId !== null) {
                        cancelAnimationFrame(rafId);
                        rafId = null;
                    }
                    if (buffer) emit(buffer);
                }
                return rendered;
            },
            // 当前全量文本（含尚未吐出的缓冲）
            getText() {
                return rendered + buffer;
            },
        };
    }

    // ============================================
    // 事件合并（throttle）
    // ============================================

    // windowMs 窗口内多次调用合并为数组，窗口结束一次性 fn(mergedArray)
    function throttleMerge(fn, windowMs = 300) {
        let pending = [];
        let timer = null;
        return function (item) {
            pending.push(item);
            if (timer === null) {
                timer = setTimeout(() => {
                    timer = null;
                    const batch = pending;
                    pending = [];
                    if (batch.length) fn(batch);
                }, windowMs);
            }
        };
    }

    // ============================================
    // SSE 连接封装
    // ============================================

    // fetch POST + readStream；自动跟踪 lastEventId（带 Last-Event-ID 请求头）
    // onEvent 透传 { type, content, timestamp, ..., id, index }（data JSON 展开 + 帧 id + token 序号）
    // onStale/staleMs：连接活性监视——任何帧（含心跳注释行）到达都刷新计时，
    // 超过 staleMs（默认 30s）无任何字节则触发 onStale（每次静默期触发一次，随后重新计时）
    // 返回最终的 lastEventId；HTTP 错误抛出带 status 的 Error
    async function connect({ url, body, headers = {}, lastEventId = 0, onEvent, onStale, staleMs = 30000, signal }) {
        const finalHeaders = { 'Content-Type': 'application/json', ...headers };
        if (lastEventId) finalHeaders['Last-Event-ID'] = String(lastEventId);

        const res = await fetch(url, {
            method: 'POST',
            headers: finalHeaders,
            body: typeof body === 'string' ? body : JSON.stringify(body),
            signal,
        });
        if (!res.ok) {
            const err = new Error(await res.text());
            err.status = res.status;
            throw err;
        }

        // 活性监视：帧或心跳到达都重置定时器
        let staleTimer = null;
        const touch = () => {
            if (!onStale) return;
            if (staleTimer !== null) clearTimeout(staleTimer);
            staleTimer = setTimeout(() => {
                staleTimer = null;
                onStale();
                touch(); // 重新计时，持续静默则周期性上报
            }, staleMs);
        };
        touch();

        let lastId = lastEventId;
        try {
            await readStream(res, {
                onFrame(frame) {
                    touch();
                    if (typeof frame.id === 'number') lastId = frame.id;
                    if (onEvent) {
                        const data = frame.data || {};
                        onEvent({ ...data, id: frame.id, index: data.index ?? null });
                    }
                },
                onHeartbeat: touch,
                signal,
            });
        } finally {
            if (staleTimer !== null) clearTimeout(staleTimer);
        }
        return lastId;
    }

    global.StreamClient = { parseSSEFrames, readStream, Typewriter, throttleMerge, connect };
})(window);
