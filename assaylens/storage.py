"""Local filesystem project persistence; intentionally single-user and no authentication."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid
from typing import Any

ROOT = Path(os.getenv("ASSAYLENS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "workspace")))


def new_identifier() -> str:
    return uuid.uuid4().hex


def path(kind: str, identifier: str) -> Path:
    if kind not in ("datasets", "runs", "runs_v2", "custom_runs") or len(identifier) != 32 or not all(c in "0123456789abcdef" for c in identifier):
        raise ValueError("Invalid dataset/run identifier")
    p = ROOT / kind / identifier
    p.mkdir(parents=True, exist_ok=True)
    return p


def existing_path(kind: str, identifier: str) -> Path:
    if kind not in ("datasets", "runs", "runs_v2", "custom_runs") or len(identifier) != 32 or not all(c in "0123456789abcdef" for c in identifier):
        raise FileNotFoundError("Invalid identifier")
    p = ROOT / kind / identifier
    if not p.is_dir():
        raise FileNotFoundError("No such dataset or run")
    return p


def dump_json(p: Path, value: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(p)


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def listing(kind: str) -> list[dict]:
    top = ROOT / kind
    if not top.exists():
        return []
    result = []
    for p in sorted(top.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        meta = p / ("audit.json" if kind == "datasets" else "metrics.json")
        if p.is_dir() and meta.exists():
            result.append({"id": p.name, "created": int(p.stat().st_mtime), **load_json(meta)})
    return result[:100]
