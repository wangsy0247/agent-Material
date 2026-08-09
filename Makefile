SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

.PHONY: help setup run

help: ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make \033[36m%-8s\033[0m %s\n", $$1, $$2}'

setup: ## 初始化：安装依赖、准备 .env、校验 config/、创建运行时目录
	@set -e
	echo "============================================"
	echo "  Agent Material 初始化 (make setup)"
	echo "============================================"

	echo "[1/5] 检查 Python 版本..."
	python -c 'import sys; v = sys.version_info; assert v >= (3, 10), f"需要 Python >= 3.10，当前 {sys.version}"; print(f"  ✓ Python {v.major}.{v.minor}.{v.micro}")'

	echo "[2/5] 安装依赖（editable 模式）..."
	pip install -e backend/packages/agent_material -q
	pip install -e . -q
	echo "  ✓ 依赖安装完成"

	echo "[3/5] 准备环境变量 .env..."
	if [ ! -f .env ]; then
		cp .env.example .env
		echo "  ✓ 已从 .env.example 创建 .env（请编辑填入 DASHSCOPE_API_KEY 等）"
	else
		echo "  ✓ .env 已存在，跳过创建"
	fi
	if ! grep -q '^AUTH_SECRET_KEY=.' .env; then
		[ -n "$$(tail -c 1 .env)" ] && echo >> .env
		echo "AUTH_SECRET_KEY=$$(python -c 'import secrets; print(secrets.token_hex(32))')" >> .env
		echo "  ✓ 已生成随机 AUTH_SECRET_KEY 写入 .env（覆盖 config/auth_config.yaml 中的占位 JWT 密钥）"
	else
		echo "  ✓ .env 中已有 AUTH_SECRET_KEY"
	fi

	echo "[4/5] 校验 config/ 配置文件..."
	python -c "import yaml, glob; files = sorted(glob.glob('config/*.yaml')); assert files, 'config/ 下没有找到 yaml 配置文件'; [yaml.safe_load(open(f, encoding='utf-8')) for f in files]; print('  ✓ %d 个配置文件解析正常' % len(files)); [print('    -', f) for f in files]"
	if grep -q 'admin123' config/auth_config.yaml; then
		echo "  ⚠ 注意：config/auth_config.yaml 中仍是默认管理员 admin/admin123，生产环境请修改 default_admin 段"
	fi

	echo "[5/5] 创建运行时目录..."
	mkdir -p logs workspace data downloads
	echo "  ✓ logs/ workspace/ data/ downloads/ 就绪"

	echo "============================================"
	echo "  初始化完成！下一步："
	echo "    1. 编辑 .env 填入 API 密钥"
	echo "    2. 将知识库 PDF 放入 data/"
	echo "    3. make run 启动服务"
	echo "============================================"

run: ## 启动 API 服务（读取 .env；AGENT_API_RELOAD=true 时热重载）
	@set -e
	if [ ! -f .env ]; then
		echo "错误：未找到 .env，请先执行 make setup"
		exit 1
	fi
	set -a; . ./.env; set +a
	export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
	HOST=$${AGENT_API_HOST:-0.0.0.0}
	PORT=$${AGENT_API_PORT:-8000}
	RELOAD=""
	[ "$${AGENT_API_RELOAD:-true}" = "true" ] && RELOAD="--reload"
	echo "启动服务: http://$$HOST:$$PORT （文档: http://$$HOST:$$PORT/docs）"
	exec uvicorn backend.app.gateway.app:app --host "$$HOST" --port "$$PORT" $$RELOAD
