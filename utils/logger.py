import logging
import os
import time
from pathlib import Path

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger writing to console and logs/pipeline.log."""
    global _configured
    if not _configured:
        _configure_root()
        _configured = True
    return logging.getLogger(name)


def _configure_root() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    formatter.converter = time.gmtime  # UTC timestamps

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
