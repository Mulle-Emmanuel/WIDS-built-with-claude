"""Frame source for the official AWID3-CSV edition (one row per packet, 50,000-row chunks).

Ground truth comes from the dataset's own `Label` column (the AWID3 authors labelled each packet
with their published Wireshark filters). Multi-valued fields are joined with '-' in this edition,
so only the first token of each field is used.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List

import pandas as pd

from .frames import Frame, anonymize_mac

COLS = ["frame.number", "frame.time_epoch", "frame.len", "wlan.fc.type", "wlan.fc.subtype",
        "wlan.fc.retry", "wlan.ta", "wlan.ra", "Label"]


def chunk_files(directory: str, pattern: str = "*.csv") -> List[Path]:
    """Chunks sorted by their numeric suffix (Deauth_2 before Deauth_10), not alphabetically."""
    files = list(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching '{pattern}' in {directory}")

    def num(p: Path) -> int:
        m = re.search(r"(\d+)\.csv$", p.name)
        if not m:
            raise ValueError(f"chunk name has no number, cannot order: {p.name}")
        return int(m.group(1))

    return sorted(files, key=num)


def _tok(s: str) -> str:
    return s.split("-")[0].strip() if s else ""


class CsvStats:
    rows = skipped_unlabelled = skipped_bad = 0
    labels: dict

    def __init__(self):
        self.labels = {}


def iter_csv_frames(directory: str, pattern: str = "*.csv", normal_label: str = "Normal",
                    anonymize: bool = False, stats: CsvStats | None = None) -> Iterator[Frame]:
    stats = stats if stats is not None else CsvStats()
    normal = normal_label.strip().lower()
    for path in chunk_files(directory, pattern):
        df = pd.read_csv(path, usecols=lambda c: c in COLS, dtype=str, keep_default_na=False,
                         on_bad_lines="skip")
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        df = df[COLS]
        for _num, ts, ln, ty, st, rt, ta, ra, lab in df.itertuples(index=False, name=None):
            stats.rows += 1
            lab = lab.strip()
            if not lab:
                stats.skipped_unlabelled += 1
                continue
            stats.labels[lab] = stats.labels.get(lab, 0) + 1
            try:
                t, ftype, subtype = float(_tok(ts)), int(_tok(ty)), int(_tok(st))
                length = int(float(_tok(ln) or 0))
            except ValueError:
                stats.skipped_bad += 1
                continue
            src, dst = _tok(ta).lower() or None, _tok(ra).lower() or None
            if anonymize:
                src, dst = anonymize_mac(src), anonymize_mac(dst)
            yield Frame(t, ftype, subtype, src, dst, _tok(rt).lower() in ("1", "true"), None, length,
                        lab.lower() != normal)
