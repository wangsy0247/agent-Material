#!/bin/bash
# ============================================
# 材料科学 LLM Agent API 启动脚本
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  材料科学 LLM Agent API 服务启动脚本${NC}"
echo -e "${GREEN}============================================${NC}"

# 检查 Python 环境
echo -e "\n${YELLOW}[1/4] 检查 Python 环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到 python 命令${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"

# 检查依赖
echo -e "\n${YELLOW}[2/4] 检查依赖...${NC}"
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}! 未找到 fastapi，正在安装依赖...${NC}"
    pip install fastapi uvicorn sse-starlette python-multipart -q
fi
echo -e "${GREEN}✓ 依赖检查完成${NC}"

# 检查 .env 文件
echo -e "\n${YELLOW}[3/4] 检查环境变量...${NC}"
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ 发现 .env 文件${NC}"
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${YELLOW}! 未找到 .env 文件，请确保已配置环境变量${NC}"
    echo -e "  需要的环境变量: MP_API_KEY, LANGCHAIN_API_KEY 等"
fi

# 启动服务
echo -e "\n${YELLOW}[4/4] 启动服务...${NC}"
echo -e "${GREEN}============================================${NC}"

# 读取端口（环境变量 ）
PORT=${AGENT_API_PORT}
HOST=${AGENT_API_HOST}

echo -e "  ${GREEN}绑定地址: ${HOST}:${PORT}${NC}"

# 根据参数选择启动模式
MODE=${1:-dev}

if [ "$MODE" == "prod" ]; then
    echo -e "${GREEN}启动模式: 生产环境${NC}"
    uvicorn backend.app.gateway.app:app --host ${HOST} --port ${PORT} --workers 4
elif [ "$MODE" == "docker" ]; then
    echo -e "${GREEN}启动模式: Docker${NC}"
    echo -e "${YELLOW}构建 Docker 镜像...${NC}"
    docker build -t agent-material-api .
    echo -e "${YELLOW}启动 Docker 容器...${NC}"
    docker run -d \
        --name agent-material-api \
        -p ${PORT}:${PORT} \
        -e AGENT_API_PORT=${PORT} \
        -e AGENT_API_HOST=0.0.0.0 \
        -v "$(pwd)/.env:/app/.env" \
        -v "$(pwd)/chroma_db:/app/chroma_db" \
        -v "$(pwd)/workspace:/app/workspace" \
        agent-material-api
    echo -e "${GREEN}✓ Docker 容器已启动${NC}"
    echo -e "  API 地址: http://localhost:${PORT}"
    echo -e "  文档地址: http://localhost:${PORT}/docs"
else
    echo -e "${GREEN}启动模式: 开发环境 (热重载)${NC}"
    echo -e "  API 地址: http://localhost:${PORT}"
    echo -e "  文档地址: http://localhost:${PORT}/docs"
    echo -e "  ${YELLOW}按 Ctrl+C 停止服务${NC}\n"
    uvicorn backend.app.gateway.app:app --host ${HOST} --port ${PORT} --reload
fi
