"""
认证模块数据模型 (Pydantic V2)
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Literal


class UserBase(BaseModel):
    """用户基础信息"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr | None = Field(default=None, description="邮箱")


class UserCreate(UserBase):
    """用户注册请求"""
    password: str = Field(..., min_length=6, max_length=128, description="密码")


class UserLogin(BaseModel):
    """用户登录请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class UserInfo(UserBase):
    """用户信息响应"""
    user_id: str = Field(..., description="用户唯一ID")
    created_at: float = Field(..., description="创建时间戳")
    is_admin: bool = Field(default=False, description="是否为管理员")


class TokenResponse(BaseModel):
    """Token 响应"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: Literal["bearer"] = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间（秒）")
    user: UserInfo = Field(..., description="用户信息")


class TokenRefreshRequest(BaseModel):
    """刷新 Token 请求"""
    refresh_token: str = Field(..., description="刷新令牌")


class TokenPayload(BaseModel):
    """JWT Payload 内部模型"""
    sub: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    type: Literal["access", "refresh"] = Field(..., description="令牌类型")
    exp: float | None = Field(default=None, description="过期时间")
