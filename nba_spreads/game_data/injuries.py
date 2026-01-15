## come back to this shit and INJURIEStest.py later

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
import tabula
import nbainjuries.injury as inj

@dataclass
class InjuryFetchConfig:
    cache_dir: Path = Path("data/raw/injuries_pdf")
    timeout: int = 20
    retries: int = 3
    sleep_seconds: int = 2

def fetch_injury_report_df(dt: datetime, cfg: InjuryFetchConfig = InjuryFetchConfig()) -> pd.DataFrame:
    """
    Fetch NBA injury report PDF for timestamp dt using nbainjuries URL generator,
    download locally (reliable), parse with tabula, return one consolidated DataFrame.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # URL + deterministic filename
    url = inj.gen_url(dt)
    fname = url.split("/")[-1]
    pdf_path = cfg.cache_dir / fname

    # Download (with caching)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        last_err = None
        for attempt in range(1, cfg.retries + 1):
            try:
                r = requests.get(url, timeout=cfg.timeout, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                pdf_path.write_bytes(r.content)
                break
            except Exception as e:
                last_err = e
                if attempt < cfg.retries:
                    time.sleep(cfg.sleep_seconds)
                else:
                    raise RuntimeError(f"Failed to download injury report after {cfg.retries} attempts: {url}") from last_err

    # Parse locally
    dfs = tabula.read_pdf(str(pdf_path), pages="all", stream=True)

    # Consolidate
    out = pd.concat([d for d in dfs if d is not None and not d.empty], ignore_index=True)

    # Light cleanup: drop fully-empty columns, strip whitespace in col names
    out.columns = [str(c).strip() for c in out.columns]
    out = out.dropna(axis=1, how="all")

    return out
