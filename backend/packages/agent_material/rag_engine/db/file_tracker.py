"""SQLite 文件状态追踪 — 替代扁平 md5.text。

特性：
- 快速路径：mtime + size 不变则跳过 MD5 计算
- 自动检测新增 / 修改 / 删除文件
- 旧 md5.text 自动迁移至 SQLite
"""

import hashlib
import json
import os
import threading

from agent_material.utils.logger import logger


def _compute_md5(file_path: str) -> str | None:
    """计算文件 MD5（4096 字节块）。"""
    try:
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"计算 MD5 失败: {file_path} — {e}")
        return None


class FileTracker:
    """SQLite 文件状态追踪器 — 线程安全。"""

    def __init__(self, db_path: str = "rag_queue.db"):
        from agent_material.utils.path_tools import get_abs_path

        self._db_path = get_abs_path(db_path)
        self._lock = threading.Lock()

        from agent_material.rag_engine.db.schema import init_db

        self._conn = init_db(self._db_path)

    # ── 查询 ──────────────────────────────────────────────

    def get_all_tracked(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_path, md5_hash, file_mtime, file_size, status, last_indexed_at, parent_ids "
                "FROM file_states WHERE status='active'"
            ).fetchall()
        return [
            {
                "file_path": r[0],
                "md5_hash": r[1],
                "file_mtime": r[2],
                "file_size": r[3],
                "status": r[4],
                "last_indexed_at": r[5],
                "parent_ids": json.loads(r[6]) if r[6] else [],
            }
            for r in rows
        ]

    def get_by_path(self, file_path: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM file_states WHERE file_path=?",
                (file_path,),
            ).fetchone()
        if row is None:
            return None
        return {
            "file_path": row[1],
            "md5_hash": row[2],
            "file_mtime": row[3],
            "file_size": row[4],
            "last_indexed_at": row[5],
            "status": row[6],
            "parent_ids": json.loads(row[7]) if row[7] else [],
        }

    def check_md5(self, md5_hash: str) -> bool:
        """检查 MD5 是否已存在于追踪表中。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM file_states WHERE md5_hash=? AND status='active' LIMIT 1",
                (md5_hash,),
            ).fetchone()
        return row is not None

    # ── 变更 ──────────────────────────────────────────────

    def upsert(self, file_path: str, md5_hash: str, mtime: float, size: int):
        with self._lock:
            self._conn.execute(
                """INSERT INTO file_states (file_path, md5_hash, file_mtime, file_size)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(file_path) DO UPDATE SET
                       md5_hash=excluded.md5_hash,
                       file_mtime=excluded.file_mtime,
                       file_size=excluded.file_size,
                       status='active'""",
                (file_path, md5_hash, mtime, size),
            )
            self._conn.commit()

    def record_indexed(self, file_path: str, parent_ids: list[str] | None = None):
        """标记文件已成功索引。"""
        with self._lock:
            if parent_ids is not None:
                self._conn.execute(
                    "UPDATE file_states SET last_indexed_at=datetime('now'), parent_ids=? WHERE file_path=?",
                    (json.dumps(parent_ids, ensure_ascii=False), file_path),
                )
            else:
                self._conn.execute(
                    "UPDATE file_states SET last_indexed_at=datetime('now') WHERE file_path=?",
                    (file_path,),
                )
            self._conn.commit()

    def mark_deleted(self, file_path: str):
        with self._lock:
            self._conn.execute(
                "UPDATE file_states SET status='deleted' WHERE file_path=?",
                (file_path,),
            )
            self._conn.commit()

    def remove_source(self, source: str):
        with self._lock:
            self._conn.execute("DELETE FROM file_states WHERE file_path=?", (source,))
            self._conn.commit()

    # ── 目录扫描 ──────────────────────────────────────────

    def scan_directory(
        self,
        directory: str,
        allow_types: list[str],
        rapid: bool = True,
    ) -> dict[str, list[str]]:
        """扫描目录，返回变更分类。

        Args:
            directory: 扫描目录路径
            allow_types: 允许的文件扩展名列表
            rapid: 启用快速路径（mtime+size 不变则跳过 MD5）

        Returns:
            {"new": [...], "modified": [...], "unchanged": [...], "deleted": [...]}
        """
        from agent_material.utils.path_tools import get_abs_path

        abs_dir = get_abs_path(directory)

        # 获取数据库中的活跃文件记录
        tracked = {t["file_path"]: t for t in self.get_all_tracked()}

        # 扫描磁盘上的文件
        on_disk: dict[str, dict] = {}
        if os.path.isdir(abs_dir):
            for root, _, files in os.walk(abs_dir):
                for f in files:
                    if not any(f.lower().endswith(ext) for ext in allow_types):
                        continue
                    abs_path = os.path.join(root, f)
                    try:
                        stat = os.stat(abs_path)
                        on_disk[abs_path] = {
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                        }
                    except OSError as e:
                        logger.warning(f"无法访问文件 {abs_path}: {e}")

        # 分类
        new: list[str] = []
        modified: list[str] = []
        unchanged: list[str] = []

        for path, info in on_disk.items():
            if path not in tracked:
                # 不在追踪表中 → 新文件
                md5 = _compute_md5(path)
                if md5:
                    self.upsert(path, md5, info["mtime"], info["size"])
                    new.append(path)
                continue

            t = tracked[path]
            if rapid and info["mtime"] == t["file_mtime"] and info["size"] == t["file_size"]:
                # 快速路径：mtime + size 未变，跳过 MD5
                unchanged.append(path)
                continue

            # 计算 MD5 判断是否变化
            md5 = _compute_md5(path)
            if md5 is None:
                unchanged.append(path)  # 计算失败，保守保留
                continue

            if md5 == t["md5_hash"]:
                # 文件未变，更新 mtime（避免下次重复计算）
                self.upsert(path, md5, info["mtime"], info["size"])
                unchanged.append(path)
            else:
                self.upsert(path, md5, info["mtime"], info["size"])
                modified.append(path)

        # 检测删除：在追踪表中但磁盘上不存在
        deleted = [p for p in tracked if p not in on_disk]
        for path in deleted:
            self.mark_deleted(path)

        logger.info(
            f"[FileTracker] 扫描完成 | "
            f"新增 {len(new)} | 修改 {len(modified)} | "
            f"未变 {len(unchanged)} | 已删 {len(deleted)}"
        )
        return {
            "new": new,
            "modified": modified,
            "unchanged": unchanged,
            "deleted": deleted,
        }

    # ── 迁移 ──────────────────────────────────────────────

    def migrate_from_md5_text(self, md5_file: str = "md5.text") -> int:
        """从旧版 md5.text 迁移数据到 SQLite。"""
        from agent_material.utils.path_tools import get_abs_path

        md5_path = get_abs_path(md5_file)
        if not os.path.exists(md5_path):
            logger.info("[FileTracker] 未找到 md5.text，跳过迁移")
            return 0

        imported = 0
        with open(md5_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or len(line) != 32:
                    continue
                # md5.text 只存了哈希，没有源文件路径，需要回查数据库
                # 如果已有活跃文件状态则更新，否则跳过
                with self._lock:
                    self._conn.execute(
                        """INSERT OR IGNORE INTO file_states (file_path, md5_hash, file_mtime, file_size)
                           VALUES (?, ?, 0, 0)""",
                        (f"legacy_{line[:8]}", line),
                    )
                imported += 1

        logger.info(f"[FileTracker] 从 md5.text 迁移了 {imported} 条记录")
        return imported
