"""
簡易 CSV logger，將訓練與評估過程中的指標記錄到同一份檔案。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: List[str] = list(fieldnames)
        self._initialized = self.path.exists()

    def log(self, **kwargs) -> None:
        row = {name: kwargs.get(name, "") for name in self.fieldnames}
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)

