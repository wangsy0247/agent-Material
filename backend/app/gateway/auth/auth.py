"""
认证核心模块

提供：
- 密码哈希与验证（bcrypt）
- JWT 生成与解析
- 用户数据库管理（SQLite）
- FastAPI 依赖：get_current_user

配置来源：config/auth_config.yaml
"""

import os
import time
import uuid
import sqlite3
import threading
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from .schemas import TokenPayload, UserInfo
from agent_material.utils.config_handler import get_abs_path
from agent_material.utils.logger import logger

# ---- 配置读取 ----
_auth_config: dict[str, Any] | None = None

def _load_auth_config() -> dict[str, Any]:
    """懒加载认证配置"""
    global _auth_config
    if _auth_config is None:
        import yaml
        cfg_path = get_abs_path("config/auth_config.yaml")
        if Path(cfg_path).exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                _auth_config = yaml.safe_load(f) or {}
        else:
            _auth_config = {}
    return _auth_config


def _get_config_value(key: str, default: Any = None) -> Any:
    """从配置文件或环境变量读取配置"""
    env_map = {
        "secret_key": "AUTH_SECRET_KEY",
        "allow_anonymous": "AUTH_ALLOW_ANONYMOUS",
    }
    env_key = env_map.get(key)
    if env_key and env_key in os.environ:
        val = os.environ[env_key]
        if key == "allow_anonymous":
            return val.lower() in ("true", "1", "yes")
        return val
    cfg = _load_auth_config()
    return cfg.get(key, default)


SECRET_KEY: str = _get_config_value("secret_key", "change-me-in-production")
ALGORITHM: str = _get_config_value("algorithm", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = _get_config_value("access_token_expire_minutes", 60)
REFRESH_TOKEN_EXPIRE_DAYS: int = _get_config_value("refresh_token_expire_days", 7)
ALLOW_ANONYMOUS: bool = _get_config_value("allow_anonymous", False)

# ---- 密码工具 ----
import bcrypt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode()
    return bcrypt.checkpw(plain_password.encode(), hashed_password)


def get_password_hash(password: str) -> str:
    """哈希密码"""
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    return hashed.decode()


# ---- JWT 工具 ----

def create_access_token(user_id: str, username: str) -> str:
    """创建访问令牌"""
    expire = time.time() + ACCESS_TOKEN_EXPIRE_MINUTES * 60
    payload = {
        "sub": user_id,
        "username": username,
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str, username: str) -> str:
    """创建刷新令牌"""
    expire = time.time() + REFRESH_TOKEN_EXPIRE_DAYS * 86400
    payload = {
        "sub": user_id,
        "username": username,
        "type": "refresh",
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenPayload | None:
    """解码并验证 JWT Token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        token_type: str = payload.get("type")
        if user_id is None or username is None or token_type not in ("access", "refresh"):
            return None
        return TokenPayload(
            sub=user_id,
            username=username,
            type=token_type,  # type: ignore[arg-type]
            exp=payload.get("exp"),
        )
    except JWTError:
        return None


# ---- 用户数据库 ----

DEFAULT_DB_PATH: str = get_abs_path("logs/auth_users.db")


class UserStore:
    """用户存储（SQLite）"""

    def __init__(self, db_path: str | None = None):
        self.db_path: str = db_path or DEFAULT_DB_PATH
        self._lock: threading.Lock = threading.Lock()
        self._local: threading.local = threading.local()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """初始化用户表"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    hashed_password TEXT NOT NULL,
                    is_admin INTEGER DEFAULT 0,
                    created_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"
            )
            conn.commit()
            conn.close()
            logger.info(f"[UserStore] 数据库初始化完成: {self.db_path}")

    def create_user(
        self, username: str, password: str, email: str | None = None, is_admin: bool = False
    ) -> UserInfo | None:
        """创建用户，返回用户信息（用户名已存在时返回 None）"""
        user_id = str(uuid.uuid4())
        hashed = get_password_hash(password)
        now = time.time()

        def _insert() -> bool:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO users (user_id, username, email, hashed_password, is_admin, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, username, email, hashed, int(is_admin), now),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

        with self._lock:
            success = _insert()

        if success:
            logger.info(f"[UserStore] 创建用户: {username} ({user_id})")
            return UserInfo(
                user_id=user_id,
                username=username,
                email=email,
                created_at=now,
                is_admin=is_admin,
            )
        return None

    def get_user_by_username(self, username: str) -> dict | None:
        """通过用户名获取用户（包含密码哈希）"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def get_user_by_id(self, user_id: str) -> UserInfo | None:
        """通过用户ID获取用户信息"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return UserInfo(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            created_at=row["created_at"],
            is_admin=bool(row["is_admin"]),
        )

    def verify_user(self, username: str, password: str) -> UserInfo | None:
        """验证用户名密码，成功返回用户信息"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not verify_password(password, user["hashed_password"]):
            return None
        return UserInfo(
            user_id=user["user_id"],
            username=user["username"],
            email=user["email"],
            created_at=user["created_at"],
            is_admin=bool(user["is_admin"]),
        )

    def ensure_default_admin(self) -> None:
        """如果配置了默认管理员且不存在，则自动创建"""
        cfg = _load_auth_config()
        admin_cfg = cfg.get("default_admin", {})
        if not admin_cfg or not admin_cfg.get("username"):
            return
        username = admin_cfg["username"]
        password = admin_cfg["password"]
        email = admin_cfg.get("email")

        existing = self.get_user_by_username(username)
        if existing:
            return

        self.create_user(username, password, email, is_admin=True)
        logger.info(f"[UserStore] 自动创建默认管理员: {username}")


# 全局用户存储实例（懒加载）
_user_store: UserStore | None = None
_user_store_lock: threading.Lock = threading.Lock()


def get_user_store() -> UserStore:
    """获取用户存储单例"""
    global _user_store
    if _user_store is None:
        with _user_store_lock:
            if _user_store is None:
                _user_store = UserStore()
                _user_store.ensure_default_admin()
    return _user_store


# ---- FastAPI 依赖 ----

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserInfo:
    """
    FastAPI 依赖：获取当前认证用户

    - 携带有效 Token → 返回对应用户信息
    - 未携带 Token 且 allow_anonymous=true → 返回匿名用户
    - 未携带 Token 且 allow_anonymous=false → 抛 401
    - Token 无效或过期 → 抛 401
    """
    if credentials is None or not credentials.credentials:
        if ALLOW_ANONYMOUS:
            return UserInfo(
                user_id="anonymous",
                username="anonymous",
                email=None,
                created_at=0.0,
                is_admin=False,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = decode_token(credentials.credentials)
    if token_data is None or token_data.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )

    store = get_user_store()
    user = store.get_user_by_id(token_data.sub)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserInfo | None:
    """
    可选认证依赖：有 Token 则验证，无 Token 返回 None
    """
    if credentials is None or not credentials.credentials:
        return None
    token_data = decode_token(credentials.credentials)
    if token_data is None or token_data.type != "access":
        return None
    store = get_user_store()
    return store.get_user_by_id(token_data.sub)
