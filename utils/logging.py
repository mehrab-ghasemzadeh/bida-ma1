"""Console + file logging and a small CSV metric writer."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_logger(name: str = "cdhsi", log_file: Optional[str | Path] = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


class CSVLogger:
    """Append-only CSV writer that fixes its header on the first row."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields: Optional[List[str]] = None

    def log(self, row: Dict[str, Any]) -> None:
        if self.fields is None:
            self.fields = list(row.keys())
            with open(self.path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fields).writeheader()
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            writer.writerow(row)
