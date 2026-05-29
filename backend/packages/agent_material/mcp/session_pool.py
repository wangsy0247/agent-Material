"""Persistent MCP session pool for stateful tool calls.

When MCP tools are loaded via langchain-mcp-adapters with ``session=None``,
each tool call creates a new MCP session. For stateful servers like Playwright,
this means browser state (opened pages, filled forms) is lost between calls.

This module provides a session pool that maintains persistent MCP sessions,
scoped by ``(server_name, scope_key)`` — typically scope_key is the thread_id —
so that consecutive tool calls share the same session and server-side state.
Sessions are evicted in LRU order when the pool reaches capacity.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any

from mcp import ClientSession

logger = logging.getLogger(__name__)


class MCPSessionPool:
    """Manages persistent MCP sessions scoped by ``(server_name, scope_key)``."""

    MAX_SESSIONS = 256
    SESSION_CLOSE_TIMEOUT = 5.0

    def __init__(self) -> None:
        self._entries: OrderedDict[
            tuple[str, str],
            tuple[ClientSession, asyncio.AbstractEventLoop],
        ] = OrderedDict()
        # Store (cm, loop) so we can close the CM on the loop it was created in.
        self._context_managers: dict[tuple[str, str], tuple[Any, asyncio.AbstractEventLoop]] = {}
        # Dead CM references kept alive to prevent async generator finalizer crashes.
        self._dead_cms: list[tuple[Any, asyncio.AbstractEventLoop]] = []
        # threading.Lock protects the dicts; asyncio.Lock prevents concurrent session creation.
        self._lock = threading.Lock()
        self._async_create_lock = asyncio.Lock()

    async def get_session(
        self,
        server_name: str,
        scope_key: str,
        connection: dict[str, Any],
    ) -> ClientSession:
        """Get or create a persistent MCP session.

        If an existing session was created in a different event loop (e.g.
        the sync-wrapper path), it is closed and replaced with a fresh one
        in the current loop.
        """
        key = (server_name, scope_key)
        current_loop = asyncio.get_running_loop()

        # Fast path: check under thread lock only
        with self._lock:
            if key in self._entries:
                session, loop = self._entries[key]
                if loop is current_loop:
                    self._entries.move_to_end(key)
                    return session

        # Slow path: acquire asyncio.Lock to prevent concurrent creation of the same key.
        # This avoids race conditions where two tasks simultaneously call create_session
        # for the same key, leading to async generator cleanup errors.
        async with self._async_create_lock:
            # Double-check under thread lock after acquiring async lock
            with self._lock:
                if key in self._entries:
                    session, loop = self._entries[key]
                    if loop is current_loop:
                        self._entries.move_to_end(key)
                        return session

            cms_to_close: list[tuple[tuple[str, str], Any, asyncio.AbstractEventLoop]] = []
            with self._lock:
                # Evict LRU entries when at capacity
                while len(self._entries) >= self.MAX_SESSIONS:
                    oldest_key = next(iter(self._entries))
                    cm_tuple = self._context_managers.pop(oldest_key, None)
                    self._entries.pop(oldest_key)
                    if cm_tuple is not None:
                        cms_to_close.append((oldest_key, cm_tuple[0], cm_tuple[1]))

            for close_key, cm, loop in cms_to_close:
                await self._close_cm(close_key, cm, loop)

            from langchain_mcp_adapters.sessions import create_session

            cm = create_session(connection)
            session = await cm.__aenter__()
            await session.initialize()

            with self._lock:
                # Final check: another path (e.g. sync wrapper) may have sneaked in
                if key in self._entries:
                    existing_session, loop = self._entries[key]
                    if loop is current_loop:
                        self._entries.move_to_end(key)
                        # IMPORTANT: close the redundant CM in the CURRENT task (same task
                        # that created it) to avoid anyio "cancel scope in different task" errors.
                        try:
                            await cm.__aexit__(None, None, None)
                        except Exception:
                            logger.warning("Error closing redundant MCP session %s", key, exc_info=True)
                        return existing_session
                    # Different loop — evict the old one and replace with the new.
                    old_cm_tuple = self._context_managers.pop(key, None)
                    self._entries.pop(key)
                    if old_cm_tuple is not None:
                        # Old CM was created in a different task/loop — schedule cleanup carefully
                        asyncio.create_task(self._close_cm(key, old_cm_tuple[0], old_cm_tuple[1]))

                self._entries[key] = (session, current_loop)
                self._context_managers[key] = (cm, current_loop)
                logger.info("Created persistent MCP session for %s/%s", server_name, scope_key)
            return session

    async def _close_cm(
        self,
        key: tuple[str, str],
        cm: Any,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Close a single context manager on the loop it was created in."""
        try:
            current_loop = asyncio.get_running_loop()
            target_loop = loop or current_loop
            if target_loop is not current_loop and not target_loop.is_closed():
                if target_loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        cm.__aexit__(None, None, None), target_loop
                    )
                    future.result(timeout=self.SESSION_CLOSE_TIMEOUT)
                else:
                    target_loop.run_until_complete(cm.__aexit__(None, None, None))
            else:
                await cm.__aexit__(None, None, None)
        except RuntimeError as e:
            # Known limitation: anyio TaskGroup cancel scopes are bound to the
            # creating task. Closing from a different task is unsafe with the
            # current langchain-mcp-adapters/mcp/anyio stack. We downgrade to
            # debug to avoid noisy logs during normal operation.
            if "cancel scope" in str(e).lower() or "asynchronous generator" in str(e).lower():
                logger.debug("Ignored expected anyio cancel-scope error when closing MCP session %s: %s", key, e)
            else:
                logger.warning("Error closing MCP session %s", key, exc_info=True)
        except Exception:
            logger.warning("Error closing MCP session %s", key, exc_info=True)

    async def close_scope(self, scope_key: str) -> None:
        with self._lock:
            keys = [k for k in self._entries if k[1] == scope_key]
            cms = [(k, self._context_managers.pop(k, None)) for k in keys]
            for k in keys:
                self._entries.pop(k, None)
        for key, cm_tuple in cms:
            if cm_tuple is not None:
                await self._close_cm(key, cm_tuple[0], cm_tuple[1])

    async def close_server(self, server_name: str) -> None:
        with self._lock:
            keys = [k for k in self._entries if k[0] == server_name]
            cms = [(k, self._context_managers.pop(k, None)) for k in keys]
            for k in keys:
                self._entries.pop(k, None)
        for key, cm_tuple in cms:
            if cm_tuple is not None:
                await self._close_cm(key, cm_tuple[0], cm_tuple[1])

    async def close_all(self) -> None:
        with self._lock:
            cms = list(self._context_managers.items())
            self._context_managers.clear()
            self._entries.clear()
        for key, (cm, loop) in cms:
            await self._close_cm(key, cm, loop)

    def close_all_sync(self) -> None:
        with self._lock:
            entries = list(self._entries.items())
            cms = dict(self._context_managers)
            self._entries.clear()
            self._context_managers.clear()
            # Keep CM references alive to prevent async generator finalizer crashes.
            self._dead_cms.extend(cms.values())

        for key, (_, loop) in entries:
            cm_tuple = cms.get(key)
            if cm_tuple is None:
                continue
            cm, _ = cm_tuple
            if loop.is_closed():
                continue
            try:
                if loop.is_running():
                    # Use _close_cm instead of raw cm.__aexit__ so that anyio
                    # RuntimeErrors are swallowed inside the task coroutine,
                    # preventing "unhandled exception during asyncio.run() shutdown".
                    future = asyncio.run_coroutine_threadsafe(
                        self._close_cm(key, cm, loop), loop
                    )
                    future.result(timeout=self.SESSION_CLOSE_TIMEOUT)
                else:
                    loop.run_until_complete(self._close_cm(key, cm, loop))
            except BaseException:
                # Swallow everything (including KeyboardInterrupt during shutdown)
                # to avoid noisy unhandled exceptions. Child processes will be
                # reaped by the OS when the process exits.
                logger.debug("Ignored error closing MCP session %s during sync close", key)


_pool: MCPSessionPool | None = None
_pool_lock = threading.Lock()
# Hold stale pools to prevent async generator finalizers from firing
# (they crash with anyio when closed from a different task).
_stale_pools: list[MCPSessionPool] = []


def get_session_pool() -> MCPSessionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = MCPSessionPool()
    return _pool


def reset_session_pool() -> None:
    global _pool, _stale_pools
    if _pool is not None:
        _stale_pools.append(_pool)
    _pool = None
