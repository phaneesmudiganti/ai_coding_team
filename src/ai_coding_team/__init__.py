import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> str:
    root = Path(__file__).resolve().parents[2]
    log_path = Path(log_dir) if log_dir else root / "log"
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "ai_coding_team.log"

    logger = logging.getLogger()
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return str(log_file)
