#!/usr/bin/env bash
# ===========================================================================
# 伺服器端的部署執行器。**不要在 Windows 上直接跑這支**。
# ===========================================================================
#
#   deploy/apply.sh receive <old_sha> <new_sha>   ← .git/hooks/post-receive 呼叫
#   deploy/apply.sh rebuild                       ← 程式碼沒變但要重建（例：agentkit 換了）
#   deploy/apply.sh rollback [sha]                ← 回到上一個健康的版本
#   deploy/apply.sh status                        ← 現在跑的是哪一版、健不健康
#
# 平常不必手動執行 —— 本機 `bash deploy/deploy.sh deploy` 會 push，push 觸發
# post-receive，post-receive 呼叫這裡。手動執行是給「push 不進來要查原因」用的。
#
# ★ 這支腳本刻意**只碰版控裡的東西**
#   `.env`、`logs/`、named volume 全都不在版控裡，git 不會動它們，這裡
#   也沒有任何一行會動它們。伺服器上手改的 A2A_HUB_PUBLIC_URL、排程抓回來的
#   附件、pg_restore 進去的語料，部署一百次都不會消失。
#   （以前的 tar/scp 流程正好相反：忘記加 --exclude=.env 就會靜靜蓋掉。）
#
# ★ 為什麼 build 不必挑檔案
#   image 內容由 Dockerfile 的 COPY 清單決定，`.dockerignore` 再擋掉一層。
#   git 已經保證 .venv/、__pycache__/ 這些根本不會上到伺服器，
#   所以「打包時漏排除某個目錄」這個錯誤在這條路徑上不存在。

set -euo pipefail

# hook 的環境是精簡過的，docker 常常不在 PATH 裡。
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

STATE_DIR="$REPO/.deploy"
HISTORY="$STATE_DIR/history.tsv"
LOCK="$STATE_DIR/lock"
LOG="$STATE_DIR/apply.log"
mkdir -p "$STATE_DIR"

# 這個專案的規則（本機 plan 用的是同一份）。
# shellcheck source=target.conf
. "$REPO/deploy/target.conf"
# 伺服器專屬、不進版控的設定（例：Agent_Hub_RAG_Frontend 的 FRONTEND_PORT）。
[ -f "$STATE_DIR/deploy.env" ] && . "$STATE_DIR/deploy.env"

say()  { printf '%s\n' "$*"; printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }
step() { printf '\n── %s\n' "$*"; printf '%s ── %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }
die()  { printf '\n✗ %s\n' "$*" >&2; printf '%s ✗ %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; exit 1; }

# docker compose v2 / v1 都吃。
if docker compose version >/dev/null 2>&1; then dc() { docker compose "$@"; }
elif command -v docker-compose >/dev/null 2>&1; then dc() { docker-compose "$@"; }
else die "伺服器上找不到 docker compose"; fi

# --- 同時只准一個部署在跑 -------------------------------------------------
# mkdir 是原子操作，比 flock 可攜（有些精簡系統沒有 util-linux）。
take_lock() {
    if ! mkdir "$LOCK" 2>/dev/null; then
        local owner; owner=$(cat "$LOCK/owner" 2>/dev/null || echo '?')
        die "另一個部署正在進行（$owner）。確定它已經死了就 rm -rf $LOCK"
    fi
    printf 'pid=%s %s\n' "$$" "$(date '+%F %T')" >"$LOCK/owner"
    trap 'rm -rf "$LOCK"' EXIT
}

# --- 把 changed files 分類成「要做什麼」 ----------------------------------
NEED_BUILD=0 NEED_UP=0
FILES_IMAGE='' FILES_MOUNT='' FILES_COMPOSE='' FILES_MIGRATION='' FILES_NOTE='' FILES_IDLE=''

classify() {
    local f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        local matched=0
        if [ -n "$T_IMAGE_RE" ] && printf '%s' "$f" | grep -Eq "$T_IMAGE_RE"; then
            FILES_IMAGE+="  $f"$'\n'; NEED_BUILD=1; matched=1
        fi
        if [ -n "$T_COMPOSE_RE" ] && printf '%s' "$f" | grep -Eq "$T_COMPOSE_RE"; then
            FILES_COMPOSE+="  $f"$'\n'; NEED_UP=1; matched=1
        fi
        if [ -n "$T_MOUNT_RE" ] && printf '%s' "$f" | grep -Eq "$T_MOUNT_RE"; then
            FILES_MOUNT+="  $f"$'\n'; matched=1
        fi
        # migration 跟 note 是「額外提醒」，可以跟上面的分類重疊
        # （migration 既進 image 也要有人決定跑不跑）。
        if [ -n "$T_MIGRATION_RE" ] && printf '%s' "$f" | grep -Eq "$T_MIGRATION_RE"; then
            FILES_MIGRATION+="  $f"$'\n'
        fi
        if [ -n "$T_NOTE_RE" ] && printf '%s' "$f" | grep -Eq "$T_NOTE_RE"; then
            FILES_NOTE+="  $f"$'\n'; matched=1
        fi
        # 這裡要用 if 不能用 `[ ... ] && ...`：那是迴圈的最後一個命令，命中任何
        # 規則時它會回非零，配上 set -e 整支腳本就在這裡無聲結束 ——
        # 症狀是「push 成功、什麼都沒重建、也沒有錯誤訊息」。
        if [ "$matched" = 0 ]; then FILES_IDLE+="  $f"$'\n'; fi
    done
    return 0
}

report_plan() {
    [ -n "$FILES_IMAGE" ]     && { step "要重建 image 的變更"; printf '%s' "$FILES_IMAGE"; }
    [ -n "$FILES_COMPOSE" ]   && { step "compose 設定變更（重新套用）"; printf '%s' "$FILES_COMPOSE"; }
    [ -n "$FILES_MOUNT" ]     && { step "掛載進容器的檔案（換掉就生效，不必重建）"; printf '%s' "$FILES_MOUNT"; }
    [ -n "$FILES_IDLE" ]      && { step "不影響服務的變更（文件等）"; printf '%s' "$FILES_IDLE"; }
    if [ -n "$FILES_NOTE" ]; then
        step "⚠ 設定範本動過"
        printf '%s' "$FILES_NOTE"
        say  "  伺服器的 .env 不在版控裡、這次沒被碰到。比對一下有沒有新的必填欄位。"
    fi
    if [ -n "$FILES_MIGRATION" ]; then
        step "⚠ 資料庫 schema 變更 —— 這支腳本不會自動執行"
        printf '%s' "$FILES_MIGRATION"
        say  "  要套用的話自己看過再跑，例如："
        say  "  ${T_MIGRATION_HINT:-docker compose exec -T db psql -U postgres -d <db> < <檔案>}"
    fi
}

# --- 健康檢查 --------------------------------------------------------------
health_ok() {
    local out i=0
    while [ "$i" -lt "${T_HEALTH_RETRIES:-24}" ]; do
        if out=$(eval "$T_HEALTH_CMD" 2>/dev/null); then
            if [ -z "${T_HEALTH_OK_RE:-}" ] || printf '%s' "$out" | grep -Eq "$T_HEALTH_OK_RE"; then
                say "健康檢查通過（第 $((i+1)) 次）：$(printf '%s' "$out" | head -c 200)"
                return 0
            fi
        fi
        i=$((i+1)); sleep "${T_HEALTH_INTERVAL:-5}"
    done
    say "健康檢查失敗（試了 $i 次）：$(printf '%s' "${out:-無回應}" | head -c 300)"
    return 1
}

# --- 對外 port 有沒有被改掉（在動任何容器之前擋下來）----------------------
# compose 裡的 port 寫成 `${FRONTEND_PORT:-8600}` / `${API_PORT:-8000}` 這種形式，
# 值是從**compose 專案目錄的 .env**與 shell 環境插值進來的。那個變數哪天消失
# （或多出來、或改了值），port 就會安靜地換掉 —— Hub 是唯一對外開放的服務，
# port 一換使用者的書籤全部失效，而容器 log 上一切正常。
#
# 這裡不去要求「一定要設某個變數」（伺服器上跑成什麼樣就是事實），而是比對
# 「這次會用的 port」跟「現在正在跑的 port」，不一樣才擋。
# ★ 這兩支結尾的 `|| true` 不是防禦性裝飾，是**必要的**。
#   管線裡的 grep 在「找不到任何 port」時回 1，而這支腳本開頭是
#   `set -euo pipefail` —— pipefail 讓整條管線跟著回非零，於是
#   `now=$(running_ports)` 這個賦值失敗，set -e 就把整支腳本**無聲結束**。
#
#   下面 check_ports 的「還沒有容器（第一次部署）→ 沒有可比的基準」那個處理
#   因此永遠跑不到：腳本在賦值那一行就死了，連 `[ -z "$now" ]` 都沒機會執行。
#
#   實測會踩到的情境（用一個最小 compose 專案重現過）：
#     · 全新伺服器第一次部署（還沒有任何容器）
#     · 任何 `docker compose down` 之後的部署
#     · **容器已經死掉時的回滾** —— 也就是最需要它能動的時候
#     · compose 裡沒有 published port 的服務（例如新加的 pipeline）
#   症狀正是這支檔案在別處警告過的那種：push 成功、什麼都沒重建、
#   而且沒有任何錯誤訊息。
planned_ports() {
    dc config 2>/dev/null | grep -E '^[[:space:]]*published:' \
        | grep -oE '[0-9]+' | sort -u | tr '\n' ' ' | sed 's/ *$//' || true
}
running_ports() {
    local id out=''
    for id in $(dc ps -q 2>/dev/null); do
        out+=$(docker port "$id" 2>/dev/null)$'\n'
    done
    printf '%s' "$out" | grep -oE ':[0-9]+$' | tr -d ':' | sort -u \
        | tr '\n' ' ' | sed 's/ *$//' || true
}
check_ports() {
    [ "${ALLOW_PORT_CHANGE:-0}" = 1 ] && return 0
    local now planned
    now=$(running_ports)
    planned=$(planned_ports)
    # 還沒有容器（第一次部署）→ 沒有可比的基準。
    if [ -z "$now" ]; then return 0; fi
    # 解析不出來就不要亂擋，但要講一聲 —— 無聲跳過的檢查會被當成「檢查過了」。
    if [ -z "$planned" ]; then
        say "（跳過 port 檢查：docker compose config 解析不出 published port）"
        return 0
    fi
    if [ "$now" != "$planned" ]; then
        die "對外 port 會被改掉：現在是【$now】，這次會變成【$planned】。
    容器還沒動，服務不受影響。
    通常是 compose 插值用的變數變了（.env 或 shell 環境裡的 FRONTEND_PORT /
    API_PORT 之類）—— 先確認是不是你要的。
    確定要換就重跑一次並帶著 ALLOW_PORT_CHANGE=1（寫在 .deploy/deploy.env 裡）。"
    fi
    return 0
}

# ===========================================================================
# image 快照：讓回滾變成「秒級、而且不需要網路」
# ===========================================================================
#
# ★ 原本的問題
#   回滾是 `git reset --hard <舊版>` + `dc up --build`，也就是**重新 build 一次**
#   —— 要跑完整的 pip install，需要連得到 PyPI。這有兩個後果：
#     1. 回滾要好幾分鐘，而那正是服務已經壞掉、大家在等的時候。
#     2. 外網不通就回滾不了。在網路受限的環境裡，這不是假想風險。
#   更糟的是自動回滾（健康檢查沒過時）也走同一條路：新版壞掉 + 網路剛好不通
#   = 服務停在壞的狀態，而腳本會告訴你「回滾後健康檢查還是不過」。
#
# ★ 做法
#   每次 build **之前**，把正在跑的那顆 image 貼一個 `sha-<前12碼>` 標籤。
#   回滾時如果那個標籤還在，就把它 retag 回 compose 用的名字，然後
#   `up -d --no-build --force-recreate` —— 純本機操作，幾秒鐘。
#   標籤不在（例如第一次用這個機制、或已經被輪掉）才退回原本的重建流程。
#
# ★ 為什麼是「貼標籤」而不是 `docker save` 成檔案
#   標籤不佔額外空間：那顆 image 的 layer 本來就在磁碟上，`docker tag` 只是
#   多一個名字。原本 build 完之後舊 image 會變成 dangling（遲早被 prune 掉），
#   貼標籤等於把它從「等著被清掉」變成「留著備用」。
#
# ★ 保留幾個
#   T_IMAGE_KEEP（預設 5）。超過的只刪標籤（`docker rmi repo:tag`），
#   不刪 image 本體 —— 同一顆 image 可能同時掛著別的名字。
# ★ 主服務**和額外服務都要**快照
#   T_EXTRA_SERVICES 裡的服務可能用另一顆 image（Part-RAG 就是這樣：
#   api 539MB、scheduler/worker 1.16GB，不同 target、不同 tag）。
#   只快照主服務的話，回滾會把 api 換回舊版、scheduler 卻還跑著新的壞版本 ——
#   那個「一半舊一半新」的狀態比不回滾更難查。
#   所以快照與回滾都對整組服務做，而且**要嘛全部都有標籤才走快速路徑，
#   要嘛全部重建** —— 不做局部回滾。
IMAGE_KEEP=${T_IMAGE_KEEP:-5}

# `rfq-rag-api:latest` -> `rfq-rag-api`
# 只切最後一段的冒號，而且只在那一段沒有 `/` 時才切 —— 不然
# `registry:5000/img` 這種帶 port 的位址會被切成 `registry`。
repo_of() {
    case "${1##*/}" in
        *:*) printf '%s' "${1%:*}" ;;
        *)   printf '%s' "$1" ;;
    esac
}

# 這次部署會動到哪些服務：主服務 + **已經被建立過**的額外服務。
# 沒建立過的不算 —— bring_up 的額外服務迴圈也是同一個判斷（它不會替人啟用
# 預設暫停的東西），兩邊要一致，不然會去快照一個根本不存在的服務。
deploy_services() {
    printf '%s\n' "$T_SERVICE"
    local spec svc prof
    for spec in ${T_EXTRA_SERVICES:-}; do
        svc=${spec%%:*}; prof=${spec#*:}
        local args=()
        [ -n "$prof" ] && args+=(--profile "$prof")
        if [ -n "$(dc ${args[@]+"${args[@]}"} ps -aq "$svc" 2>/dev/null)" ]; then
            printf '%s\n' "$svc"
        fi
    done
    return 0
}

# compose 現在給某個 service 用的完整 image 名（含 tag）。
# 優先問正在跑的容器；容器不在就讀上次記下來的（每個服務一個檔）。
service_image_ref() {
    local svc=${1:-$T_SERVICE} cid ref f="$STATE_DIR/image-name.${1:-$T_SERVICE}"
    cid=$(dc ps -q "$svc" 2>/dev/null | head -1 || true)
    if [ -n "$cid" ]; then
        ref=$(docker inspect -f '{{.Config.Image}}' "$cid" 2>/dev/null || true)
        if [ -n "$ref" ]; then
            printf '%s\n' "$ref" >"$f"
            printf '%s' "$ref"; return 0
        fi
    fi
    [ -s "$f" ] && { cat "$f"; return 0; }
    return 1
}

# 我們自己記一份「貼過哪些標籤、順序為何」，最新的在最後一行。
#
# ★ 為什麼不直接問 docker 排序
#   第一版是 `docker images --format '{{.Tag}}\t{{.CreatedAt}}' | sort -k2 -r`。
#   那是錯的，而且錯得很難看：CreatedAt 長得像
#   `YYYY-MM-DD HH:MM:SS +0800 CST`，裡面有空白，而 sort 預設就是用空白切欄位
#   —— `-k2` 只拿到日期那一段，同一天貼的標籤全部並列，然後退化成按整行
#   字典序比較。實測結果是**它把剛剛才貼上的那個回滾標籤自己刪掉了**，
#   卻留下幾個名字排比較後面的舊標籤。
#   （而且 retag 不會更新 CreatedAt，所以就算解析對了，那個時間也不是
#     「這個標籤多新」，是「底下那顆 image 多新」—— 本來就問錯問題。）
#
# 一個純文字檔就沒有這些問題：順序是我們寫進去的，最後 N 行就是最近 N 次，
# 剛貼上的那個永遠在最後一行，不可能被自己輪掉。
TAGS_FILE="$STATE_DIR/image-tags"

remember_tag() {
    local tag=$1 tmp="$TAGS_FILE.tmp"
    { grep -vxF "$tag" "$TAGS_FILE" 2>/dev/null || true; } >"$tmp"
    printf '%s\n' "$tag" >>"$tmp"
    mv "$tmp" "$TAGS_FILE"
}

# ★ 輪替是**按 repo 分開算**的。
#   有兩個服務的專案（Part-RAG 的 api + scheduler）每次部署會貼兩個標籤，
#   混在一起算的話 T_IMAGE_KEEP=5 只夠回溯兩次半，而且兩個服務保留的版本
#   還會對不上 —— 回滾需要「同一個 sha 的兩顆 image 都在」才走得了快速路徑。
prune_image_tags() {
    local repo=$1 tag live='' others='' keep tmp="$TAGS_FILE.tmp"
    [ -s "$TAGS_FILE" ] || return 0
    # 先濾掉已經不存在的（有人手動 rmi、或 docker prune 掃掉）——
    # 不然死掉的項目會佔用保留額度，實際留下的比設定的少。
    # 同時把「別的 repo 的」原樣留著，這一輪不動它們。
    while IFS= read -r tag; do
        [ -n "$tag" ] || continue
        docker image inspect "$tag" >/dev/null 2>&1 || continue
        if [ "$(repo_of "$tag")" = "$repo" ]; then live+="$tag"$'\n'
        else others+="$tag"$'\n'; fi
    done <"$TAGS_FILE"

    keep=$(printf '%s' "$live" | tail -n "$IMAGE_KEEP")
    printf '%s' "$live" | head -n -"$IMAGE_KEEP" | while IFS= read -r tag; do
        [ -n "$tag" ] || continue
        # rmi 一個「還有別的名字指著」的標籤只是解除命名，不會刪掉 image 本體；
        # repo:latest 永遠在，所以正在跑的那顆不會被這裡弄掉。
        docker rmi "$tag" >/dev/null 2>&1 && say "  輪掉舊標籤 $tag" || true
    done
    { printf '%s' "$others"; printf '%s\n' "$keep"; } >"$tmp"
    mv "$tmp" "$TAGS_FILE"
    return 0
}

# build 之前呼叫。$1 = 現在跑的是哪個 commit。
# build 之前呼叫。$1 = 現在跑的是哪個 commit。整組服務各自貼標籤。
snapshot_image() {
    local sha=$1 svc ref repo id n=0
    [ -n "$sha" ] || return 0
    while IFS= read -r svc; do
        [ -n "$svc" ] || continue
        ref=$(service_image_ref "$svc") || continue
        repo=$(repo_of "$ref")
        id=$(docker inspect -f '{{.Id}}' "$ref" 2>/dev/null || true)
        [ -n "$id" ] || { say "（找不到 image $ref，$svc 跳過快照）"; continue; }
        if docker tag "$id" "$repo:sha-${sha:0:12}" 2>/dev/null; then
            say "已保留現行 image：$repo:sha-${sha:0:12}（回滾用，不佔額外空間）"
            remember_tag "$repo:sha-${sha:0:12}"
            prune_image_tags "$repo"
            n=$((n+1))
        else
            say "（$repo 標記失敗，這一版的回滾會走重建那條路）"
        fi
    done < <(deploy_services)
    [ "$n" -gt 0 ] || say "（沒有可快照的執行中容器，這一版的回滾會走重建那條路）"
    return 0
}

# 回滾到 $1。整組服務的標籤**都在**才走快速路徑，缺任何一個就整組重建。
#
# ★ 為什麼不做局部回滾
#   少一顆就回滾一半，會留下「api 是舊的、scheduler 是新的」這種狀態 ——
#   兩邊對資料庫的假設可能不一樣，而且沒有任何地方會顯示這件事。
#   寧可多花幾分鐘重建，也不要一個沒人看得出來的混合版本。
rollback_to() {
    local sha=$1 svc ref repo tag ok=1 svcs=''
    git reset --hard "$sha" >/dev/null
    svcs=$(deploy_services)

    while IFS= read -r svc; do
        [ -n "$svc" ] || continue
        ref=$(service_image_ref "$svc") || { ok=0; break; }
        tag="$(repo_of "$ref"):sha-${sha:0:12}"
        if ! docker image inspect "$tag" >/dev/null 2>&1; then
            say "（$svc 沒有 $tag 這個標籤 —— 可能是第一次用這個機制，或已經被輪掉）"
            ok=0; break
        fi
    done <<<"$svcs"

    if [ "$ok" = 1 ]; then
        step "用保留的 image 回滾（不 build、不需要網路）"
        local failed=0
        while IFS= read -r svc; do
            [ -n "$svc" ] || continue
            ref=$(service_image_ref "$svc")
            tag="$(repo_of "$ref"):sha-${sha:0:12}"
            say "  $svc <- $tag"
            docker tag "$tag" "$ref" || { failed=1; break; }
        done <<<"$svcs"
        # 全部 retag 完才一次重建容器 —— 中途失敗的話什麼都還沒動。
        if [ "$failed" = 0 ] && \
           dc up -d --no-build --force-recreate $svcs; then
            return 0
        fi
        say "（用保留的 image 起不來，改走重建）"
    fi

    step "改用重建回滾（需要連得到套件來源）"
    NEED_BUILD=1
    bring_up
}

# --- 真正動手的部分 --------------------------------------------------------
# 這一版跑起來之前，正在服務的是哪個 commit。build 前拿它來貼 image 標籤。
SNAPSHOT_SHA=''

bring_up() {
    if [ "$NEED_BUILD" = 1 ] || [ "$NEED_UP" = 1 ]; then check_ports; fi
    if [ "$NEED_BUILD" = 1 ]; then
        # 一定要在 build 之前 —— build 完舊 image 就變成 dangling，
        # 名字已經被新的搶走了。
        snapshot_image "$SNAPSHOT_SHA"
        if [ -n "${T_PREBUILD:-}" ]; then
            step "build 前準備：$T_PREBUILD"
            eval "$T_PREBUILD" || die "準備步驟失敗，沒有動到正在跑的容器"
        fi
        step "重建並啟動 $T_SERVICE"
        dc up --build -d "$T_SERVICE" || die "build/up 失敗，舊容器還在跑（服務沒有中斷）"
    elif [ "$NEED_UP" = 1 ]; then
        step "重新套用 compose 設定"
        dc up -d "$T_SERVICE" || die "up 失敗"
    else
        step "沒有需要重啟的變更，容器維持原狀"
        return 0
    fi

    # 額外服務：只更新「已經存在」的，不會替人啟用預設暫停的東西。
    local spec svc prof
    for spec in ${T_EXTRA_SERVICES:-}; do
        svc=${spec%%:*}; prof=${spec#*:}
        local args=()
        [ -n "$prof" ] && args+=(--profile "$prof")
        # ${args[@]+...}：空陣列配 set -u 在舊版 bash 會炸成 unbound variable。
        if [ -n "$(dc ${args[@]+"${args[@]}"} ps -aq "$svc" 2>/dev/null)" ]; then
            # ★ --build 不能省。額外服務**可能是另一顆 image**（Part-RAG 就是
            #   這樣：api 539MB、scheduler/worker 1.16GB，不同 target、不同 tag）。
            #   以前兩者共用同一個 tag，主服務那次 build 產出的新 image 一個
            #   `up -d` 就換過去了；現在不會 —— 少了 --build，額外服務會一直
            #   跑著舊程式碼，而它沒有版本號、log 也完全正常。
            #
            #   用兩個服務的最小 compose 專案重現過：部署新版之後主服務換了，
            #   額外服務容器裡還是上一版的內容。更糟的是接下來的 image 快照會
            #   把那顆過期的 image 貼上新版的 sha 標籤 —— **標籤在說謊**，
            #   哪天照著它回滾會回到一個從來不存在的組合。
            #
            #   代價是額外服務有啟用時每次部署多一次 build；但它通常以主服務的
            #   base stage 為底，共用 layer 全部命中快取，實測只多幾十秒。
            step "順帶重建並更新已存在的 $svc（它可能是另一顆 image，見 Dockerfile 的 target）"
            if ! dc ${args[@]+"${args[@]}"} up --build -d "$svc"; then
                record "$(git rev-parse HEAD)" fail "extra-service-failed:$svc"
                die "$svc 更新失敗，部署不完整；請查看 logs 並重試或回滾"
            fi
        else
            say "（$svc 沒有被建立過，跳過 —— 要啟用是另一件事，見 README 的「資料管線與維運設計」一節）"
        fi
    done
}

record() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "${3:-}" >>"$HISTORY"; }

release_ok() {
    [ -n "${T_RELEASE_CMD:-}" ] || return 0
    # Old image snapshots remain usable when rolling back across this feature's introduction.
    if [ ! -f "$REPO/src/delivery.py" ]; then
        say "（舊版沒有發版資訊工具；保留原健康檢查與 image 快照）"
        return 0
    fi
    step "保存實際執行版本與 image 的發版紀錄"
    eval "$T_RELEASE_CMD" || { say "發版紀錄／版本一致性檢查失敗"; return 1; }
}

last_good() {
    [ -f "$HISTORY" ] || return 1
    awk -F'\t' -v cur="$1" '$3=="ok" && $2!=cur {sha=$2} END{if(sha)print sha}' "$HISTORY"
}

# ===========================================================================
MODE=${1:-status}

case "$MODE" in
receive)
    OLD=${2:-}; NEW=${3:-$(git rev-parse HEAD)}
    take_lock
    say "部署 $(basename "$REPO")：$(git log -1 --format='%h %s' "$NEW")"

    if [ -z "$OLD" ] || [ "$OLD" = "0000000000000000000000000000000000000000" ] \
       || ! git cat-file -e "$OLD^{commit}" 2>/dev/null; then
        say "（沒有上一版可比 —— 當成全新部署，完整重建）"
        NEED_BUILD=1; NEED_UP=1
        FILES_IMAGE="  <全部>"$'\n'
    else
        # 用 process substitution 而不是 pipe：pipe 會讓 classify 跑在 subshell 裡，
        # NEED_BUILD 這些旗標就傳不回來了（而且症狀是「什麼都沒重建，也沒報錯」）。
        classify < <(git diff --name-only "$OLD" "$NEW")
        if [ "$NEED_BUILD" = 0 ] && [ "$NEED_UP" = 0 ] && [ -z "$FILES_MOUNT" ]; then
            report_plan
            record "$NEW" ok "no-op"
            say ""; say "✓ 只有文件類變更，檔案已更新，容器不用動。"
            exit 0
        fi
    fi

    report_plan
    PREV=$(last_good "$NEW" || true)
    # 現在正在服務的是 OLD（hook 已經把工作目錄切到 NEW，但容器還沒動）。
    SNAPSHOT_SHA=$OLD
    bring_up

    step "健康檢查"
    if health_ok && release_ok; then
        record "$NEW" ok "$([ "$NEED_BUILD" = 1 ] && echo rebuild || echo reload)"
        # 也把剛上線的這一版標起來 —— 不然「下一次部署之前」這個版本沒有標籤，
        # 要回到它就得重建。標一次的成本是零（只是多一個名字）。
        SNAPSHOT_SHA=$NEW; snapshot_image "$NEW"
        say ""; say "✓ 部署完成：$(git log -1 --format='%h %s')"
    else
        record "$NEW" fail ""
        if [ "${AUTO_ROLLBACK:-1}" = 1 ] && [ -n "$PREV" ]; then
            step "⚠ 新版沒過健康檢查 —— 自動回到上一個健康版本 ${PREV:0:8}"
            # 走 rollback_to：有保留的 image 就秒回，不需要外網。
            # 這裡正是最需要它的地方 —— 服務已經壞了，不該再賭一次 pip install。
            rollback_to "$PREV"
            if health_ok; then
                record "$PREV" ok "rollback"
                die "新版有問題，已經回到 ${PREV:0:8}，服務正常。伺服器現在跑的是舊版。
    先看 docker compose logs --tail=100 $T_SERVICE 找原因，修好後正常 push 即可
    （修正 commit 是壞版本的後代，deploy 分支照樣 fast-forward）。
    如果是用 --amend 改掉壞的那個 commit，push 要加 --force-history。"
            fi
            record "$PREV" fail "rollback-failed"
            die "回滾後健康檢查還是不過 —— 問題可能不在這次的程式碼。
    看 docker compose logs $T_SERVICE 與 $LOG"
        fi
        die "新版沒過健康檢查（沒有可回滾的版本，或 AUTO_ROLLBACK=0）。
    看 docker compose logs --tail=100 $T_SERVICE"
    fi
    ;;

rebuild)
    take_lock
    say "強制重建（程式碼沒變，通常是 agentkit 或基底 image 換了）"
    NEED_BUILD=1
    # 程式碼沒變，所以現在跑的跟等一下要建的是同一個 commit。這裡快照的是
    # 「重建之前那顆 image」—— agentkit 換壞了的話，要回去的就是它。
    SNAPSHOT_SHA=$(git rev-parse HEAD)
    bring_up
    step "健康檢查"
    health_ok && release_ok || die "重建後健康／版本一致性檢查不過，看 docker compose logs --tail=100 $T_SERVICE"
    record "$(git rev-parse HEAD)" ok "forced-rebuild"
    say ""; say "✓ 重建完成：$(git log -1 --format='%h %s')"
    ;;

rollback)
    take_lock
    CUR=$(git rev-parse HEAD)
    TARGET=${2:-$(last_good "$CUR" || true)}
    [ -n "$TARGET" ] || die "history 裡找不到可以回去的版本，指定 sha：deploy/apply.sh rollback <sha>"
    git cat-file -e "$TARGET^{commit}" 2>/dev/null || die "這台機器上沒有 commit $TARGET"
    say "回滾 ${CUR:0:8} → ${TARGET:0:8}：$(git log -1 --format='%h %s' "$TARGET")"
    # 先把現在這顆（壞的那版）也標起來再回滾 —— 不然「回滾之後發現搞錯了、
    # 想回到剛剛那版再查一次 log」就得重建。
    SNAPSHOT_SHA=$CUR; snapshot_image "$CUR"
    rollback_to "$TARGET"
    step "健康檢查"
    if health_ok; then record "$TARGET" ok "manual-rollback"; say ""; say "✓ 已回到 ${TARGET:0:8}"
    else record "$TARGET" fail "manual-rollback"; die "回滾後仍不健康，看 logs"; fi
    ;;

status)
    printf '專案      %s\n' "$(basename "$REPO")"
    printf '版本      %s\n' "$(git log -1 --format='%h %s（%cr）' 2>/dev/null || echo '不是 git repo')"
    if ! git diff --quiet 2>/dev/null; then
        printf '⚠ 伺服器上有未提交的修改：\n'; git diff --stat | sed 's/^/  /'
    fi
    printf '\n容器\n'; dc ps -a 2>/dev/null | sed 's/^/  /'
    printf '\n健康檢查\n'
    if out=$(eval "$T_HEALTH_CMD" 2>&1); then printf '  %s\n' "$(printf '%s' "$out" | head -c 400)"
    else printf '  ✗ 打不通：%s\n' "$(printf '%s' "$out" | head -c 200)"; fi
    if [ -f "$HISTORY" ]; then printf '\n最近幾次部署\n'; tail -5 "$HISTORY" | sed 's/^/  /'; fi
    ;;

*)  die "不認得的模式：$MODE（receive|rebuild|rollback|status）" ;;
esac
