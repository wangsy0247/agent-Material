/**
 * 轻量 Toast 通知与确认对话框组件
 * 功能：右上角滑入通知（info/error，3s 自动消失，可堆叠）、Promise 化确认模态
 * 纯原生 JS，无构建依赖，挂载到 window.Toast
 */
(function (global) {
    'use strict';

    let container = null;

    // 通知容器（右上角堆叠），首次使用时创建
    function ensureContainer() {
        if (container && container.isConnected) return container;
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
        return container;
    }

    function show(msg, type) {
        const box = ensureContainer();
        const el = document.createElement('div');
        el.className = `toast toast-${type}`;
        el.textContent = msg; // textContent 赋值，天然防 XSS
        box.appendChild(el);
        // 下一帧加 show 类触发滑入动画
        requestAnimationFrame(() => el.classList.add('toast-show'));
        setTimeout(() => {
            el.classList.remove('toast-show');
            el.addEventListener('transitionend', () => el.remove(), { once: true });
            // 兜底：transitionend 未触发时定时移除
            setTimeout(() => el.remove(), 500);
        }, 3000);
    }

    // 确认模态：遮罩 + 确认/取消按钮， resolve(true/false)
    function confirmModal(message) {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'toast-confirm-overlay';
            overlay.innerHTML = `
                <div class="toast-confirm-modal">
                    <div class="toast-confirm-msg"></div>
                    <div class="toast-confirm-actions">
                        <button class="toast-confirm-btn toast-confirm-cancel">取消</button>
                        <button class="toast-confirm-btn toast-confirm-ok">确认</button>
                    </div>
                </div>
            `;
            overlay.querySelector('.toast-confirm-msg').textContent = message;
            const done = (val) => {
                overlay.remove();
                resolve(val);
            };
            overlay.querySelector('.toast-confirm-ok').addEventListener('click', () => done(true));
            overlay.querySelector('.toast-confirm-cancel').addEventListener('click', () => done(false));
            // 点击遮罩空白处视为取消
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) done(false);
            });
            document.body.appendChild(overlay);
        });
    }

    global.Toast = {
        info(msg) { show(msg, 'info'); },
        error(msg) { show(msg, 'error'); },
        confirm: confirmModal,
    };
})(window);
