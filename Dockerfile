# Part-rag。build 之前先校驗固定 agentkit 並準備版本資訊：
#   bash docker/vendor_agentkit.sh
# 未準備 build-info.json 或快照不符時建置會失敗，不再依賴旁邊的 Hub 目錄。
#
# ★ 這個檔有兩個 target
#     api     :8000 的問答服務。只裝 requirements-api.txt，實測 537MB。
#     worker  排程（src.schedule）與入庫。api 的全部再加上 pandas /
#             OCR 那組與它的原生函式庫。實測 1.12GB。
#   兩者共用 base stage，所以 worker 不會把 api 的 layer 再存一份 —— 兩個
#   image 都在的機器上，多出來的只有那 590MB 的差額。
#
#   docker compose up --build -d          # 只建 api（scheduler 掛了 profile）
#   docker compose --profile scheduler up --build -d scheduler   # 要排程才建 worker
#   docker build --target api -t partrag .          # 手動建 api
#   docker build --target worker -t partrag-worker .
#
# ★ 不指定 --target 時建到的是 worker（最後一個 stage），也就是**功能較全**的
#   那一個。反過來讓 api 當預設的話，有人拿它去跑 schedule 會在半夜某個 stage
#   才炸在 import pandas / OCR —— 那些 import 是惰性的，不會在啟動時就失敗。
#   寧可預設胖一點，也不要預設在 01:00 才壞。

# ---------------------------------------------------------------------------
# base：兩個 target 共用的作業系統相依與 api 的 Python 套件
# ---------------------------------------------------------------------------
FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 AS base

WORKDIR /app

# libgomp1 / libstdc++6：onnxruntime 的 OpenMP / C++ runtime（embedding 與
#   rapidocr 都靠它），python:3.12-slim 預設沒帶，缺了不是 build 失敗而是
#   import 時才炸。torch 換掉之後這兩個仍然必要 —— onnxruntime 自己就要。
# tzdata：**排程正確性的一部分，不是選配**。容器預設 UTC，沒有時區資料時
#   TZ=Asia/Taipei 會被 libc 當成無效值安靜地退回 UTC ——「每天 01:00 跑」
#   於是變成台北時間早上九點，正好上班時間 OCR 把 CPU 吃滿，而且不會報錯。
#   src/schedule.py 的 _check_tz() 會在啟動時驗這件事，這裡負責讓它驗得過。
#   放在 base 而不是只放 worker：api 的回答裡也會印時間戳。
# curl：HEALTHCHECK 用。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        tzdata \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 這裡原本帶著 `--extra-index-url https://download.pytorch.org/whl/cpu`，理由是
# Linux 的 PyPI 上 torch 預設是 CUDA 版（連帶 nvidia-* 好幾 GB，這台服務沒有
# GPU）。換成 BGE-M3 的 ONNX 之後 torch 整個不在相依裡了，那個
# index 也就沒有作用 —— 留著只會讓下一個人以為 torch 還在。
COPY requirements-api.txt requirements-api-linux.lock ./
RUN pip install --no-cache-dir -r requirements-api.txt -c requirements-api-linux.lock

# ---------------------------------------------------------------------------
# 程式碼。兩個 target 各 COPY 一次而不是放進 base，是為了讓「改一行程式碼」
# 不會使 worker 的 apt / pip layer 失效。
# ---------------------------------------------------------------------------
# vendor/agentkit 和 lock 一起納入版控。AGENTKIT_PATH 指到 /app/vendor，
# 本機與容器使用同一個經校驗的快照；來源 commit 和內容雜湊包含在版本資訊。

# ---------------------------------------------------------------------------
# api：:8000 的問答服務
# ---------------------------------------------------------------------------
FROM base AS api

COPY main.py ./
COPY Dockerfile build-info.json requirements.txt requirements-worker-linux.lock ./
COPY src ./src
COPY vendor ./vendor
ENV AGENTKIT_PATH=/app/vendor
ENV PYTHONUNBUFFERED=1
RUN python -m src.delivery verify-build --target api

EXPOSE 8000

# /readyz 檢查 DB、必要 schema、非空語料與 A2A 掛載，不能接流量時回 503。
# LLM、排程新鮮度與 embedding 暖機不阻擋字面檢索；詳情見 /api/health 和 /a2a/health。
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/readyz || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# worker：排程與入庫。api 的全部 + pandas / OCR 與它的原生函式庫
# ---------------------------------------------------------------------------
FROM base AS worker

# 只有排程才需要的系統套件放在這個 stage，不放 base：
#   api 的 build 因此不會碰到任何只有 worker 才需要的外部套件庫 ——
#   在網路受限的建置環境裡，那種額外的下載是最常讓 build 失敗的地方，
#   不該讓「只想更新 api」的那些次也一起承擔。
# 展示版：公司系統對接（PLM 資料庫的 ODBC driver）已移除。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxcb1 \
        libgl1 \
        libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

# OCR's OpenCV wheel links to XCB/OpenGL/GLib even though no GUI is displayed.
# Verify native dependencies at build time; a missing library must fail the build.

# requirements.txt 開頭是 `-r requirements-api.txt`，那一組在 base 已經裝好，
# 這裡只會補上 pandas / OCR 那幾個。
COPY requirements-api.txt requirements.txt requirements-worker-linux.lock ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements-worker-linux.lock
RUN python -c "import cv2; from rapidocr_onnxruntime import RapidOCR; print('worker native dependencies OK')"

COPY main.py ./
COPY Dockerfile build-info.json ./
COPY src ./src
COPY vendor ./vendor
ENV AGENTKIT_PATH=/app/vendor
ENV PYTHONUNBUFFERED=1
RUN python -m src.delivery verify-build --target worker

# 排程的實際指令由 docker-compose.yml 指定（--daily-at / --budget）。這裡的
# 預設值讓 `docker run partrag-worker` 至少印得出狀態，而不是起一個沒有排程
# 設定的常駐行程。
CMD ["python", "-m", "src.schedule", "--status"]
