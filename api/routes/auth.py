"""
认证路由：注册、登录、Token 刷新、用户信息
"""

from fastapi import APIRouter, Depends, HTTPException, status
from api.auth_schemas import UserCreate, UserLogin, TokenResponse, TokenRefreshRequest, UserInfo
from api.auth import get_user_store, create_access_token, create_refresh_token, decode_token, get_current_user
from utils.logger import logger

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", response_model=UserInfo, summary="用户注册")
async def register(request: UserCreate):
    """
    注册新用户账号

    **请求示例：**
    ```json
    {
        "username": "zhangsan",
        "password": "password123",
        "email": "zhangsan@example.com"
    }
    ```
    """
    store = get_user_store()
    user = store.create_user(
        username=request.username,
        password=request.password,
        email=request.email,
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已存在",
        )
    logger.info(f"[Auth] 用户注册成功: {request.username}")
    return user


@router.post("/login", response_model=TokenResponse, summary="用户登录")
async def login(request: UserLogin):
    """
    用户登录，获取 Access Token 和 Refresh Token

    **请求示例：**
    ```json
    {
        "username": "zhangsan",
        "password": "password123"
    }
    ```
    """
    store = get_user_store()
    user = store.verify_user(request.username, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user.user_id, user.username)
    refresh_token = create_refresh_token(user.user_id, user.username)

    logger.info(f"[Auth] 用户登录成功: {request.username}")
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=60 * 60,  # 60 minutes in seconds
        user=user,
    )


@router.post("/refresh", response_model=TokenResponse, summary="刷新 Token")
async def refresh(request: TokenRefreshRequest):
    """
    使用 Refresh Token 获取新的 Access Token

    **请求示例：**
    ```json
    {
        "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
    }
    ```
    """
    token_data = decode_token(request.refresh_token)
    if token_data is None or token_data.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌",
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

    access_token = create_access_token(user.user_id, user.username)
    refresh_token = create_refresh_token(user.user_id, user.username)

    logger.info(f"[Auth] Token 刷新成功: {user.username}")
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=60 * 60,
        user=user,
    )


@router.get("/me", response_model=UserInfo, summary="获取当前用户信息")
async def get_me(current_user: UserInfo = Depends(get_current_user)):
    """
    获取当前登录用户的信息
    """
    return current_user
