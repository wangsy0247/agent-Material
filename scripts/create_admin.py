"""交互式创建首个管理员账号。

脚本不接受命令行明文密码，避免密码进入 shell 历史和进程列表。
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "packages"))

from app.gateway.auth.auth import get_user_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="创建 Agent Material 管理员")
    parser.add_argument("username", help="管理员用户名")
    parser.add_argument("--email", default=None, help="可选邮箱")
    args = parser.parse_args()

    password = os.environ.get("AGENT_BOOTSTRAP_ADMIN_PASSWORD")
    if password is None:
        password = getpass.getpass("管理员密码（至少 12 个字符）: ")
        confirmation = getpass.getpass("再次输入管理员密码: ")
        if password != confirmation:
            print("两次输入的密码不一致", file=sys.stderr)
            return 2

    if len(password) < 12:
        print("管理员密码至少需要 12 个字符", file=sys.stderr)
        return 2

    store = get_user_store()
    if store.get_user_by_username(args.username) is not None:
        print(f"用户已存在，未做修改: {args.username}", file=sys.stderr)
        return 1

    user = store.create_user(
        username=args.username,
        password=password,
        email=args.email,
        is_admin=True,
    )
    if user is None:
        print("管理员创建失败", file=sys.stderr)
        return 1

    print(f"管理员已创建: {user.username} ({user.user_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
