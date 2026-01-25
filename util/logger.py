import logging
import os
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


### 建立 logs 資料夾 ###
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_path = log_dir / "run.log"


# 建立 handler（每天換一個檔案）
handler = TimedRotatingFileHandler(
    filename=log_path,
    when="midnight",
    interval=1,
    backupCount=7,  # 保留最近 7 天的 log 檔
    encoding="utf-8",
    utc=False
)
handler.suffix = "%Y-%m-%d"


# 設定 log 格式
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)


# 建立 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# 避免重複輸出到 console（若有其他 handler）
logger.propagate = False