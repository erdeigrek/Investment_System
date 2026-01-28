from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass(frozen=True)
class AppPaths:
    root: Path
    data_dir: Path

    @staticmethod
    def from_app_file(app_file: str | Path) -> "AppPaths":
        root = Path(app_file).resolve().parent.parent  # logic/ -> streamlit_app/
        data_dir = root / "data"
        return AppPaths(root=root, data_dir=data_dir)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML powinien mieć strukturę słownika na top-level.")
    return data

def default_progress_path(data_dir: Path) -> Path:
    return data_dir / "progress.json"
