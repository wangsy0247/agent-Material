#!/bin/bash
  # ============================================
  # WSL 网络诊断与端口转发工具
  # ============================================

  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  RED='\033[0;31m'
  NC='\033[0m'

  WSL_IP=$(ip addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
  PORT=${1:-8080}

  echo -e "${GREEN}============================================${NC}"
  echo -e "${GREEN}  WSL 网络诊断${NC}"
  echo -e "${GREEN}============================================${NC}"
  echo ""
  echo -e "WSL IP: ${YELLOW}${WSL_IP}${NC}"
  echo -e "端口: ${YELLOW}${PORT}${NC}"
  echo ""

  # 检查端口占用
  echo -e "${GREEN}[1] 端口占用检查${NC}"
  if ss -tlnp | grep -q ":${PORT} "; then
      echo -e "${GREEN}✓ 端口 ${PORT} 已被监听${NC}"
      ss -tlnp | grep ":${PORT} "
  else
      echo -e "${RED}✗ 端口 ${PORT} 未被监听${NC}"
      echo "  请先启动服务: bash scripts/start_server.sh"
  fi
  echo ""

  # Windows 访问测试
  echo -e "${GREEN}[2] Windows 访问测试${NC}"
  echo -e "  在 Windows PowerShell 中运行:"
  echo -e "  ${YELLOW}curl http://localhost:${PORT}/api/v1/system/health${NC}"
  echo ""

  # WSL 内部访问测试
  echo -e "${GREEN}[3] WSL 内部访问测试${NC}"
  curl -s http://localhost:${PORT}/api/v1/system/health | python3 -m json.tool 2>/dev/null || echo -e "${RED}
  服务未启动${NC}"
  echo ""

  # Windows IP
  echo -e "${GREEN}[4] Windows 宿主机 IP（局域网访问用）${NC}"
  WINDOWS_IP=$(ip route show | grep -oP '(?<=default via\s)\d+(\.\d+){3}')
  echo -e "  Windows 网关 IP: ${YELLOW}${WINDOWS_IP}${NC}"
  echo ""

  # 端口转发命令
  echo -e "${GREEN}[5] 手动端口转发命令（管理员 PowerShell）${NC}"
  echo -e "  ${YELLOW}netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=${PORT} connectaddr
  ess=${WSL_IP} connectport=${PORT}${NC}"
  echo -e "  ${YELLOW}netsh advfirewall firewall add rule name=\\\"Agent API\\\" dir=in action=allow protocol=t
  cp localport=${PORT}${NC}"
  echo ""

  # 快捷访问链接
  echo -e "${GREEN}[6] 访问链接${NC}"
  echo -e "  WSL 内部:  ${YELLOW}http://localhost:${PORT}${NC}"
  echo -e "  Windows:   ${YELLOW}http://localhost:${PORT}${NC}"
  echo -e "  局域网:    ${YELLOW}http://${WINDOWS_IP}:${PORT}${NC} (需配置转发)"
  echo -e "  API 文档:  ${YELLOW}http://localhost:${PORT}/docs${NC}"