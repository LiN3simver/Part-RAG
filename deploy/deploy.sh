#!/usr/bin/env bash
# ===========================================================================
# 一鍵部署（本機端）。在 Windows 用 **Git Bash** 跑，不是 PowerShell、不是 cmd。
# ===========================================================================
#
#   bash deploy/deploy.sh plan        這次會傳什麼、會做什麼（唯讀，不動伺服器）
#   bash deploy/deploy.sh deploy      真的部署（會先把 plan 印出來讓你確認）
#   bash deploy/deploy.sh status      伺服器現在跑哪一版、健不健康
#   bash deploy/deploy.sh logs [svc]  伺服器上的容器 log
#   bash deploy/deploy.sh rebuild     程式碼沒變但要重建（agentkit 換了之類）
#   bash deploy/deploy.sh rollback    回到上一個健康的版本
#   bash deploy/deploy.sh bootstrap   第一次用這套機制時跑一次（可重複執行）
#
#   常用旗標：-y 不問直接做、--allow-dirty 連未 commit 的修改一起部署
#
# ★ 傳輸方式是 `git push`，不是 tar/scp
#   1. 只傳「這次真的變了的東西」——git 傳的是物件差異，改一支 .py 就是幾 KB。
#   2. **沒進版控的東西永遠不會被上傳**：.venv/（6GB）、__pycache__/、
#      執行期資料、.env、logs/ 全都在 .gitignore 裡，不需要在部署指令上維護一長串
#      --exclude（那串只要漏一個就會出事，而且沒有任何錯誤訊息）。
#   3. 伺服器上「有、但版控裡沒有」的東西不會被碰：.env 裡手加的
#      A2A_HUB_PUBLIC_URL、排程抓回來的 PDF、資料庫，部署再多次都在。
#   4. 伺服器知道自己是哪一版，所以「改了什麼 → 要重建還是只要重啟」是算出來的，
#      不是靠人記得。規則寫在 deploy/target.conf。
#
# ★ 前提：這個 repo 的 .gitignore 就是「哪些檔案該上伺服器」的唯一定義。
#   新增了不該上去的大檔（dump、匯出的 CSV、模型權重）就加進 .gitignore，
#   部署這條路自然就擋掉了。

set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SELF_DIR/.." && pwd)
cd "$PROJECT_DIR"
PROJECT=$(basename "$PROJECT_DIR")

# shellcheck source=target.conf
. "$SELF_DIR/target.conf"

: "${DEPLOY_BRANCH:=deploy}"      # 伺服器上專門用來接收的分支
: "${REMOTE_NAME:=prod}"          # 本地 git remote 的名字
# 連不上時要快點講，不要吊在那裡看起來像在做事。
: "${SSH_OPTS:=-o ConnectTimeout=10}"

ASSUME_YES=0 ALLOW_DIRTY=0 FORCE_HISTORY=0 WITH_DEPS=0
ARGS=()
for a in "$@"; do
    case "$a" in
        -y|--yes)          ASSUME_YES=1 ;;
        --allow-dirty)     ALLOW_DIRTY=1 ;;
        --force-history)   FORCE_HISTORY=1 ;;
        --with-dependents) WITH_DEPS=1 ;;
        -h|--help)         sed -n '2,40p' "$0"; exit 0 ;;
        *)                 ARGS+=("$a") ;;
    esac
done
CMD=${ARGS[0]:-plan}

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    B=$'\033[1m'; DIM=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else B=''; DIM=''; G=''; Y=''; R=''; N=''; fi
say()  { printf '%s\n' "$*"; }
head_() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }
warn() { printf '%s⚠ %s%s\n' "$Y" "$*" "$N"; }
die()  { printf '%s✗ %s%s\n' "$R" "$*" "$N" >&2; exit 1; }

# MSYS_NO_PATHCONV：Git Bash 會把看起來像路徑的參數硬轉成 Windows 路徑，
# 遠端的 ~/apps/... 被轉掉之後 ssh 那頭會 cd 到不存在的目錄（而且錯誤訊息
# 完全不像路徑問題）。所有出遠門的指令都要帶著它。
ssh_run() { MSYS_NO_PATHCONV=1 ssh $SSH_OPTS "$T_HOST" "$*"; }

confirm() {
    [ "$ASSUME_YES" = 1 ] && return 0
    [ -t 0 ] || die "非互動環境請加 -y"
    printf '\n%s%s [y/N] %s' "$B" "$1" "$N"
    local ans; read -r ans
    [[ "$ans" =~ ^[Yy]$ ]] || { say "取消，什麼都沒做。"; exit 1; }
}

# --- 伺服器現在是哪一版 ----------------------------------------------------
# 回傳伺服器目前的 commit；那邊還沒有 repo 就回空字串。
# 連不上（ssh 回 255）跟「沒有 repo」是兩件事，不要混在一起 —— 混了的話網路不通
# 會被講成「第一次部署」，然後你就會對著一台根本沒連上的機器按下 deploy。
server_sha() {
    local out
    out=$(ssh_run "cd $T_REMOTE 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo __NONE__") \
        || die "連不上 $T_HOST（ssh 失敗）。先確認網路與金鑰：ssh $T_HOST"
    out=$(printf '%s' "$out" | tr -d '\r' | tail -n 1)
    [ "$out" = "__NONE__" ] && return 0
    printf '%s' "$out"
}

ensure_bootstrapped() {
    ssh_run "test -x $T_REMOTE/.git/hooks/post-receive" 2>/dev/null \
        || die "伺服器還沒設定過。先跑一次：bash deploy/deploy.sh bootstrap"
}

# --- 本地工作目錄的狀態 ----------------------------------------------------
# 回傳「要部署的 commit」。乾淨就是 HEAD；--allow-dirty 時用 git stash create
# 生一個臨時 commit（不會動到你的工作目錄，也不會進 main 的歷史）。
resolve_push_sha() {
    local untracked; untracked=$(git ls-files --others --exclude-standard | head -5)
    if [ -n "$untracked" ]; then
        warn "以下檔案還沒 git add，**不會**被部署（這通常正是你要的，但確認一下）："
        printf '%s\n' "$untracked" | sed 's/^/    /' >&2
    fi
    if git diff --quiet HEAD 2>/dev/null; then
        git rev-parse HEAD; return
    fi
    if [ "$ALLOW_DIRTY" = 1 ]; then
        local tmp; tmp=$(git stash create) || true
        [ -n "$tmp" ] || die "git stash create 失敗"
        warn "帶著未 commit 的修改部署（$(git diff --name-only HEAD | wc -l) 個檔案）。" >&2
        warn "伺服器上會有一版本機歷史裡沒有的 commit —— 驗完記得回來 commit。" >&2
        printf '%s\n' "$tmp"; return
    fi
    {
        printf '%s\n' "本機有未 commit 的修改："
        git diff --name-only HEAD | sed 's/^/    /'
        printf '%s\n' "先 commit（建議）或加 --allow-dirty 帶著跑。"
        printf '%s\n' "部署的是 commit 不是工作目錄，這樣伺服器上跑的每一版都追得回來。"
    } >&2
    exit 1
}

# --- 分類：跟 deploy/apply.sh 用同一組規則（都來自 target.conf）------------
NEED_BUILD=0 NEED_UP=0 HIT_DEPS=0
show_changes() {
    local files=$1 f
    local image='' mount='' compose='' migration='' note='' idle='' n=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        n=$((n+1)); local m=0
        if [ -n "$T_IMAGE_RE" ] && grep -Eq "$T_IMAGE_RE" <<<"$f"; then image+="    $f"$'\n'; NEED_BUILD=1; m=1; fi
        if [ -n "$T_COMPOSE_RE" ] && grep -Eq "$T_COMPOSE_RE" <<<"$f"; then compose+="    $f"$'\n'; NEED_UP=1; m=1; fi
        if [ -n "$T_MOUNT_RE" ] && grep -Eq "$T_MOUNT_RE" <<<"$f"; then mount+="    $f"$'\n'; m=1; fi
        if [ -n "$T_MIGRATION_RE" ] && grep -Eq "$T_MIGRATION_RE" <<<"$f"; then migration+="    $f"$'\n'; fi
        if [ -n "$T_NOTE_RE" ] && grep -Eq "$T_NOTE_RE" <<<"$f"; then note+="    $f"$'\n'; m=1; fi
        if [ -n "$T_DEPENDENT_RE" ] && grep -Eq "$T_DEPENDENT_RE" <<<"$f"; then HIT_DEPS=1; fi
        # 用 if 不用 `[ ... ] && ...`：後者是迴圈的最後一個命令，命中規則時回非零，
        # set -e 會讓函式在這裡直接結束（畫面上只少了半段報告，不會報錯）。
        if [ "$m" = 0 ]; then idle+="    $f"$'\n'; fi
    done <<<"$files"

    say "共 $n 個檔案有變。"
    [ -n "$image" ]   && { head_ "要重建 image（會 docker compose up --build -d $T_SERVICE）"; printf '%s' "$image"; }
    [ -n "$compose" ] && { head_ "compose 設定變更（會重新套用）"; printf '%s' "$compose"; }
    [ -n "$mount" ]   && { head_ "掛載進容器的檔案（換掉就生效，不必重建）"; printf '%s' "$mount"; }
    [ -n "$idle" ]    && { head_ "不影響服務（文件等）"; printf '%s' "$idle"; }
    [ -n "$note" ]    && { head_ "${Y}⚠ 設定範本動過${N} —— 伺服器的 .env 不會被碰，自己去補新欄位"; printf '%s' "$note"; }
    [ -n "$migration" ] && {
        head_ "${Y}⚠ 資料庫 schema 變更 —— 部署不會自動執行，要自己上去跑${N}"; printf '%s' "$migration"; }

    head_ "結論"
    if [ "$NEED_BUILD" = 1 ]; then say "  重建 image 並換掉容器（舊容器要到新的 build 成功才會被取代）。"
    elif [ "$NEED_UP" = 1 ]; then say "  不必重建，只重新套 compose 設定。"
    elif [ -n "$mount" ];   then say "  只換掛載的檔案，容器完全不用動。"
    else                          say "  容器不用動，只是把檔案更新上去。"; fi
    [ "$HIT_DEPS" = 1 ] && warn "  這次改到別的專案會 COPY 走的東西（$T_DEPENDENT_DIRS 要跟著重建）"
    say "  伺服器的 .env / logs/ / volume / 資料庫${DIM}（都不在版控裡）${N}不會被碰。"
}

plan() {
    local local_sha srv
    local_sha=$(git rev-parse HEAD)
    srv=$(server_sha)
    head_ "$PROJECT → $T_HOST:$T_REMOTE"
    if [ -z "$srv" ]; then
        warn "伺服器上還沒有版本紀錄（第一次部署，或還沒 bootstrap）。"
        say "這次會把整個版控內容送過去並完整重建。"
        NEED_BUILD=1; return
    fi
    say "伺服器  ${srv:0:8}  $(git log -1 --format='%s' "$srv" 2>/dev/null || echo '（本機沒有這個 commit，歷史對不上）')"
    say "本機    ${local_sha:0:8}  $(git log -1 --format='%s' HEAD)"
    if [ "$srv" = "$local_sha" ]; then
        say ""; say "${G}兩邊一樣，沒有東西要部署。${N}"
        say "${DIM}（要強制重建 image 用：bash deploy/deploy.sh rebuild）${N}"
        return
    fi
    git cat-file -e "$srv^{commit}" 2>/dev/null || {
        warn "本機找不到伺服器那個 commit —— 伺服器可能被別人 push 過，或歷史被改寫過。"
        say "先看 bash deploy/deploy.sh status，確認要不要蓋掉它。"
        NEED_BUILD=1; return
    }
    if [ -n "$(git rev-list "$local_sha..$srv" 2>/dev/null)" ]; then
        warn "伺服器有本機沒有的 commit（$(git rev-list --count "$local_sha..$srv") 個）——"
        warn "push 會被 git 擋下來。確定要蓋掉就加 --force-history。"
    fi
    show_changes "$(git diff --name-only "$srv" "$local_sha")"
}

# ===========================================================================
case "$CMD" in

plan) plan ;;

# 不連伺服器，單純問「改了這幾個檔案的話會怎樣」。
# 用來驗 target.conf 的規則有沒有寫對（改了 Dockerfile 的 COPY 清單之後值得跑一次）。
explain)
    [ "${#ARGS[@]}" -gt 1 ] || die "用法：bash deploy/deploy.sh explain <檔案> [檔案...]"
    show_changes "$(printf '%s\n' "${ARGS[@]:1}")"
    ;;

deploy)
    ensure_bootstrapped
    PUSH_SHA=$(resolve_push_sha)
    plan
    [ "$NEED_BUILD" = 0 ] && [ "$NEED_UP" = 0 ] && [ "$(server_sha)" = "$(git rev-parse HEAD)" ] && exit 0
    confirm "照上面做？"
    head_ "推送到伺服器（只傳差異）"
    push_args=(--no-verify)
    [ "$FORCE_HISTORY" = 1 ] && push_args+=(--force)
    # 伺服器的 post-receive 會接手：對齊工作目錄 → 判斷要不要重建 → 健康檢查
    # →（不健康的話）自動回到上一個健康版本。它的輸出會原樣帶著 remote: 印在下面。
    MSYS_NO_PATHCONV=1 git push "${push_args[@]}" "$REMOTE_NAME" "$PUSH_SHA:refs/heads/$DEPLOY_BRANCH" \
        || die "push 或伺服器端部署失敗 —— 上面 remote: 那幾行是伺服器講的。
   服務沒有中斷（build 失敗時舊容器還在跑；健康檢查沒過會自動回滾）。"
    say ""
    say "${G}✓ 完成。${N}驗一下：bash deploy/deploy.sh status"
    if [ "$HIT_DEPS" = 1 ] && [ -n "$T_DEPENDENT_DIRS" ]; then
        head_ "連帶重建"
        say "這次改的東西會被下列專案 COPY 進它們的 image，它們要重建才會拿到新版："
        for d in $T_DEPENDENT_DIRS; do say "    $d"; done
        if [ "$WITH_DEPS" = 1 ] || { [ "$ASSUME_YES" = 0 ] && confirm "現在一起重建？" && true; }; then
            for d in $T_DEPENDENT_DIRS; do
                [ -f "$PROJECT_DIR/$d/deploy/deploy.sh" ] || { warn "$d 還沒裝部署腳本，跳過"; continue; }
                head_ "→ $d"
                (cd "$PROJECT_DIR/$d" && bash deploy/deploy.sh rebuild -y)
            done
        else
            say "${DIM}之後自己跑：cd $T_DEPENDENT_DIRS && bash deploy/deploy.sh rebuild${N}"
        fi
    fi
    ;;

rebuild)
    ensure_bootstrapped
    confirm "在伺服器上強制重建 $PROJECT 的 image（程式碼不變）？"
    ssh_run "cd $T_REMOTE && bash deploy/apply.sh rebuild"
    ;;

rollback)
    ensure_bootstrapped
    head_ "伺服器上的部署紀錄"
    ssh_run "tail -10 $T_REMOTE/.deploy/history.tsv 2>/dev/null || echo '（還沒有紀錄）'"
    confirm "回到上一個${ARGS[1]:+指定的 }健康版本？"
    ssh_run "cd $T_REMOTE && bash deploy/apply.sh rollback ${ARGS[1]:-}"
    ;;

status)
    ssh_run "cd $T_REMOTE && bash deploy/apply.sh status"
    ;;

logs)
    ssh_run "cd $T_REMOTE && docker compose logs --tail=${LOG_LINES:-80} ${ARGS[1]:-$T_SERVICE}"
    ;;

history)
    ssh_run "cat $T_REMOTE/.deploy/history.tsv 2>/dev/null || echo '（還沒有紀錄）'"
    ;;

bootstrap)
    head_ "$PROJECT → $T_HOST:$T_REMOTE"
    say "會做這些事（重複執行安全，不會動到伺服器上的資料）："
    say "  1. 確認伺服器有 git / docker compose"
    say "  2. 在 $T_REMOTE 建立（或沿用既有的）git repo"
    say "  3. 裝上 post-receive hook —— 之後 push 就會自動部署"
    say "  4. 建立 .deploy/（部署紀錄與伺服器專屬設定，不進版控）"
    say "  5. 本地加一個 git remote：$REMOTE_NAME"
    confirm "開始？"

    head_ "1/5 檢查伺服器"
    ssh_run "command -v git >/dev/null || { echo '伺服器沒有 git：sudo apt-get install -y git'; exit 1; }
             docker compose version >/dev/null 2>&1 || docker-compose version >/dev/null 2>&1 \
               || { echo '伺服器沒有 docker compose'; exit 1; }
             echo \"git \$(git --version | awk '{print \$3}') / docker compose ok\"" \
        || die "伺服器前提不滿足（上面那行是伺服器講的）"

    head_ "2/5 準備 repo"
    # 既有部署的目錄裡本來就有 .git（之前 tar 是連 .git 一起打包的），
    # 那就沿用 —— 有共同歷史，第一次 push 只傳差異，不必重傳幾百 MB。
    ssh_run "set -e
             mkdir -p $T_REMOTE && cd $T_REMOTE
             if [ ! -d .git ]; then git init -q; echo '建了新的 repo（第一次 push 會傳完整歷史，會比較久）'
             else echo \"沿用既有 repo（HEAD=\$(git rev-parse --short HEAD 2>/dev/null || echo 無))\"; fi
             git symbolic-ref HEAD refs/heads/main
             git config receive.denyCurrentBranch ignore
             git config core.fileMode false
             # 伺服器上 checkout 出來的 .sh 必須是 LF（帶 CR 的 shell 腳本會噴
             # \$'\\r': command not found，而且看起來完全不像行尾問題）。
             # Linux 的 git 預設就是 false，這行只是不要靠預設值。
             git config core.autocrlf false
             git config gc.auto 0
             mkdir -p .deploy"

    head_ "3/5 安裝 post-receive hook"
    # quoted heredoc：裡面的 $ 原樣送過去，不要在本機展開。
    MSYS_NO_PATHCONV=1 ssh $SSH_OPTS "$T_HOST" "cat > $T_REMOTE/.git/hooks/post-receive && chmod +x $T_REMOTE/.git/hooks/post-receive" <<'HOOK'
#!/usr/bin/env bash
# 由 deploy/deploy.sh bootstrap 安裝。真正的部署邏輯在版控裡的 deploy/apply.sh，
# 這裡只做「把工作目錄對齊到剛收到的 commit」這件 hook 非做不可的事。
set -euo pipefail
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_QUARANTINE_PATH
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

while read -r old new ref; do
    if [ "$ref" != "refs/heads/deploy" ]; then
        echo "（忽略 $ref，只有 refs/heads/deploy 會觸發部署）"
        continue
    fi

    # 有人直接在伺服器上改了版控裡的檔案 —— 不要靜靜蓋掉，也不要卡住部署：
    # 留一份 patch 再對齊。（.env、logs/ 不在版控裡，本來就不受影響。）
    #
    # 前面那個 rev-parse 不能省：全新的 repo 還沒有任何 commit，HEAD 是無效的
    # revision，`git diff HEAD` 會噴 fatal 而且回非零 —— 看起來就像「伺服器上有
    # 未提交的修改」，於是第一次部署每次都會產生一個空的 patch 檔。
    if git rev-parse --verify -q HEAD >/dev/null 2>&1 && ! git diff --quiet HEAD 2>/dev/null; then
        ts=$(date +%Y%m%d-%H%M%S)
        mkdir -p .deploy
        git diff HEAD > ".deploy/server-local-changes-$ts.patch" || true
        echo "⚠ 伺服器上有未提交的修改，已備份成 .deploy/server-local-changes-$ts.patch"
    fi

    git reset --hard "$new" -q

    if [ -f deploy/apply.sh ]; then
        bash deploy/apply.sh receive "$old" "$new"
    else
        echo "⚠ 這個版本裡沒有 deploy/apply.sh：檔案更新了，但容器沒有重建。"
    fi
done
HOOK
    say "hook 裝好了。"

    head_ "4/5 伺服器專屬設定"
    ssh_run "test -f $T_REMOTE/.deploy/deploy.env" 2>/dev/null && say "已存在，不覆蓋。" || {
        ssh_run "cat > $T_REMOTE/.deploy/deploy.env" <<ENVF
# 只在這台伺服器上生效、不進版控的部署設定。deploy/apply.sh 會 source 它。
# 例：Agent_Hub_RAG_Frontend 的 compose 需要 FRONTEND_PORT；或 AUTO_ROLLBACK=0 關掉自動回滾。
${T_DEPLOY_ENV:-}
ENVF
        say "建立 .deploy/deploy.env"
    }

    head_ "5/5 本地 remote"
    if git remote | grep -qx "$REMOTE_NAME"; then
        git remote set-url "$REMOTE_NAME" "$T_HOST:$T_REMOTE"; say "更新 remote $REMOTE_NAME"
    else
        git remote add "$REMOTE_NAME" "$T_HOST:$T_REMOTE"; say "新增 remote $REMOTE_NAME"
    fi

    head_ "${G}bootstrap 完成${N}"
    say "接下來："
    say "  bash deploy/deploy.sh plan      看第一次會傳什麼"
    say "  bash deploy/deploy.sh deploy    真的部署"
    warn "第一次 push 如果伺服器上原本沒有 .git，會傳整個歷史（這個 repo 約 $(git count-objects -vH | awk '/size-pack/{print $2$3}')）。"
    ;;

*) die "不認得的指令：$CMD（plan|deploy|status|logs|rebuild|rollback|history|bootstrap）" ;;
esac
