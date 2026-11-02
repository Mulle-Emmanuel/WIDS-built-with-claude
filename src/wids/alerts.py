"""Alert output: console, JSON-lines file and SQLite. MAC addresses are evidence, not identity."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RED, YEL, RST = "\033[91m", "\033[93m", "\033[0m"

ALERT_COLS = ["detected_utc", "window_start_utc", "mode", "predicted_class", "confidence", "likely_type",
              "n_frames", "frame_rate", "n_deauth", "deauth_rate", "deauth_unique_targets",
              "top_deauth_src", "top_src", "inference_ms"]


def utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AlertSink:
    def __init__(self, db_path: Optional[str] = None, jsonl_path: Optional[str] = None,
                 console: bool = True, color: bool = True, fresh: bool = False):
        """fresh=True starts new log files, so a demo run never mixes with an earlier run."""
        self.console, self.color = console, color
        if fresh:
            for f in (db_path, jsonl_path):
                if f and Path(f).exists():
                    Path(f).unlink()
        if jsonl_path:
            Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        self.jsonl = open(jsonl_path, "a", buffering=1) if jsonl_path else None
        self.db = None
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(db_path)
            self.db.execute("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, "
                            + ", ".join(f"{c} {'REAL' if c in ('confidence','frame_rate','deauth_rate','inference_ms') else 'TEXT'}"
                                        for c in ALERT_COLS) + ")")
            self.db.execute("CREATE TABLE IF NOT EXISTS predictions (window_start_utc TEXT, mode TEXT, "
                            "probability REAL, predicted_class TEXT, n_frames INTEGER, n_deauth INTEGER)")
            self.db.commit()

    def prediction(self, rec: dict):
        if self.db:
            self.db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                            (rec["window_start_utc"], rec["mode"], rec["probability"], rec["predicted_class"],
                             rec["n_frames"], rec["n_deauth"]))

    def alert(self, rec: dict):
        if self.console:
            c = RED if self.color else ""
            r = RST if self.color else ""
            print(f"{c}[ALERT]{r} {rec['window_start_utc']}  {rec['predicted_class'].upper()}  "
                  f"conf={rec['confidence']:.3f}  {rec['likely_type']}\n"
                  f"        frames/s={rec['frame_rate']:.0f}  deauth/s={rec['deauth_rate']:.0f}  "
                  f"targets={rec['deauth_unique_targets']}  src={rec['top_deauth_src'] or rec['top_src'] or '-'} "
                  f"(source MAC may be spoofed)  [{rec['mode']}]", flush=True)
        if self.jsonl:
            self.jsonl.write(json.dumps(rec) + "\n")
        if self.db:
            self.db.execute(f"INSERT INTO alerts ({','.join(ALERT_COLS)}) VALUES ({','.join('?' * len(ALERT_COLS))})",
                            [rec[c] for c in ALERT_COLS])
            self.db.commit()

    def close(self):
        if self.db:
            self.db.commit()
            self.db.close()
        if self.jsonl:
            self.jsonl.close()
