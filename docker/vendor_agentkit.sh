#!/usr/bin/env bash
# 校驗版控中的 agentkit 快照，並準備 build-info.json。
#
#   bash docker/vendor_agentkit.sh
#
# ★ 為什麼要複製而不是繼續用 sys.path 引導
#   src/a2a.py 平常靠 sys.path 指到 ../Agent_Hub_RAG_Frontend 找 agentkit（見該檔開頭），
#   那是「四個專案共用一台機器、各自 venv」的本機開發場景設計的零安裝路徑。
#   Docker build context 進不去 sibling 目錄，容器裡也沒有 ../Agent_Hub_RAG_Frontend
#   那層結構，所以打包成 image 需要一份實體複本。
#
# 正本仍在 Agent_Hub_RAG_Frontend。要更新時明確執行 vendor-pin、審查差異與回歸測試，再提交。
# 每次 build 只校驗固定快照，不會把旁邊 Hub 的最新修改帶進來。
#
# Docker 建置也會校驗快照、程式雜湊和完整 Python 版本鎖；漏準備會建置失敗。
set -euo pipefail

cd "$(dirname "$0")/.."

# Builds now use the reviewed snapshot. Update explicitly with vendor-pin.
PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" -m src.delivery vendor-check
"$PYTHON_BIN" -m src.delivery prepare
