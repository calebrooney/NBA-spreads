"""
Persistence helpers for ML evaluation outputs.

This module writes machine-readable artifacts (JSON) so walk-forward results can be
compared across runs without relying on terminal logs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


def _sanitize(obj: Any) -> Any:
    """
    Recursively convert nested structures into JSON-friendly primitives.

    :param obj: Arbitrary nested structure.
    :return: Sanitized structure.
    """
    if is_dataclass(obj):
        return _sanitize(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    # Last resort: stringify unknown objects (keeps JSON writing robust).
    return str(obj)


def save_results_json(payload: dict[str, Any], out_path: Path) -> Path:
    """
    Write a JSON payload to disk, creating parent directories as needed.

    :param payload: JSON-serializable dict after sanitization.
    :param out_path: Destination file path.
    :return: The resolved output path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize(payload)
    out_path.write_text(json.dumps(sanitized, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out_path


def utc_timestamp_for_filename() -> str:
    """
    Build a filesystem-friendly UTC timestamp string.

    :return: Timestamp like `20260416T123456Z`.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def next_iteration_results_path(out_dir: Path, *, stem: str = "walk_forward_results") -> Path:
    """
    Choose the next `XX_<stem>.json` filename in `out_dir`.

    This supports repeated benchmarking runs where you want stable ordering like:
    `01_walk_forward_results.json`, `02_walk_forward_results.json`, ...

    :param out_dir: Directory where results are written.
    :param stem: Filename stem after the `XX_` prefix.
    :return: Suggested output path (parent dir is not created here).
    """
    out_dir = Path(out_dir)
    pat = re.compile(rf"^(?P<i>\d{{2}})_{re.escape(stem)}\.json$")

    max_i = 0
    if out_dir.exists():
        for p in out_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if not m:
                continue
            try:
                max_i = max(max_i, int(m.group("i")))
            except ValueError:
                continue

    next_i = max_i + 1
    return out_dir / f"{next_i:02d}_{stem}.json"


def results_payload(
    *,
    results: dict[str, pd.DataFrame],
    train_eval_cfg: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a structured JSON payload for a walk-forward evaluation run.

    :param results: Mapping of model/baseline name -> summarized metrics dataframe.
    :param train_eval_cfg: TrainEvalConfig (or compatible dataclass) describing the run.
    :param extra: Optional additional metadata to embed.
    :return: Dict suitable for `save_results_json`.
    """
    meta: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)

    tables: dict[str, Any] = {}
    for name, df in results.items():
        if df.empty:
            tables[name] = {"rows": []}
            continue
        # `fold` may be int or the string "overall" depending on summarize_fold_results.
        tables[name] = {"rows": json.loads(df.to_json(orient="records", date_format="iso"))}

    return {
        "meta": meta,
        "config": _sanitize(train_eval_cfg),
        "results": tables,
    }
