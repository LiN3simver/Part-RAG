# 部署：git push + 自動判斷 + 秒級回滾

```bash
bash deploy/deploy.sh plan      # 預覽：會傳什麼、伺服器會做什麼（唯讀）
bash deploy/deploy.sh deploy    # 部署
bash deploy/deploy.sh rollback  # 回到上一個健康的版本
```

## 運作方式

傳輸走 **`git push`**，不是 tar/scp：

- **只傳差異**，且**沒進版控的東西永遠不會上傳**（`.env`、`data/`、`.venv/`）——
  `.gitignore` 就是「哪些檔案該上伺服器」的唯一定義，不必維護一串 `--exclude`。
- **伺服器上「有、但版控裡沒有」的東西不會被碰**：`.env`、排程抓回來的 PDF、資料庫內容。
- **「改了什麼 → 要做什麼」是算出來的**：`git diff` 出變更清單，對照 `target.conf` 的規則，
  決定要重建 image、只重套 compose 設定，還是什麼都不用做。

```
push → post-receive hook
        ├ 伺服器上有人手改過版控檔案？→ 先存成 .deploy/*.patch，不會靜靜蓋掉
        ├ 更新到新版（.env / data/ 不受影響）
        └ deploy/apply.sh
            ├ 分類變更 → 決定要不要重建
            ├ docker compose up --build -d
            ├ 健康檢查 /readyz，重試到 2 分鐘
            └ 不健康 → 自動回到上一個健康版本
```

build 失敗或健康檢查沒過時，**舊容器仍在跑**，服務不中斷。

## 回滾不需要重新 build

每次 build 前，先把正在跑的 image 貼上 `sha-<commit>` 標籤保留（只是多一個名字，不佔額外空間）。
回滾時直接 retag 回去、`up -d --no-build` —— 幾秒鐘、不需要連外網。
多個服務（`api`／`scheduler`）一起回滾，避免新舊版本混搭。

## 檔案

| 檔案 | 在哪執行 | 做什麼 |
|---|---|---|
| `deploy/deploy.sh` | 本機 | plan / deploy / status / logs / history / rollback / rebuild / bootstrap / explain |
| `deploy/apply.sh` | 伺服器 | push 後由 hook 自動呼叫 |
| `deploy/target.conf` | 兩邊共用 | 伺服器位置、服務名、「哪些路徑要重建」的規則、健康檢查 |

## 刻意不做的事

- **不自動跑資料庫 migration**：migration 有變只印提醒。DDL 錯一次要靠 dump 才救得回來，必須有人看著。
- **不自動啟用排程**：`scheduler` 預設暫停，部署只更新已建立過的排程容器，不替人決定「開始對外連線並寫語料」。
- **不改 `.env`、不碰 volume**：`.env.example` 動過時只提醒比對。
- **對外 port 會變就中止**：部署前比對現行與新設定的 port，不一致時在動任何容器之前停下。
- **部署的是 commit，不是工作目錄**：有未提交的修改會被擋下，伺服器上跑過的每一版都追得回來。
