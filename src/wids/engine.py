"""Glue: frames -> windows -> LSTM -> alerts. Same code for replay and live."""
from __future__ import annotations

import time
from typing import Optional

from .alerts import AlertSink, utc
from .detector import Detector
from .features import Window, WindowAggregator
from .frames import Frame


DEAUTH_EVIDENCE_MIN = 10      # deauth+disassoc frames in the window before the alert text says "deauth flood"


class Engine:
    def __init__(self, detector: Detector, sink: AlertSink, mode: str, consecutive: int = 1):
        self.det, self.sink, self.mode = detector, sink, mode
        self.consecutive = max(1, consecutive)
        self.agg = WindowAggregator(detector.window_s)
        self.streak = 0
        self.stats = {"frames": 0, "windows": 0, "scored": 0, "alerts": 0, "infer_ms_total": 0.0}

    def feed(self, f: Frame):
        self.stats["frames"] += 1
        for w in self.agg.add(f):
            self._on_window(w)

    def tick(self, now: Optional[float] = None):
        for w in self.agg.advance_to(time.time() if now is None else now):
            self._on_window(w)

    def finish(self):
        for w in self.agg.flush():
            self._on_window(w)
        self.sink.close()
        return self.summary()

    def summary(self) -> dict:
        s = dict(self.stats)
        s["mean_inference_ms"] = s["infer_ms_total"] / s["scored"] if s["scored"] else 0.0
        return s

    def _on_window(self, w: Window):
        self.stats["windows"] += 1
        t0 = time.perf_counter()
        p = self.det.push(w)
        ms = (time.perf_counter() - t0) * 1000
        if p is None:
            return
        self.stats["scored"] += 1
        self.stats["infer_ms_total"] += ms
        mal = p >= self.det.threshold
        ev = w.evidence
        base = {"window_start_utc": utc(w.t_start), "mode": self.mode, "probability": p,
                "predicted_class": "malicious" if mal else "normal",
                "n_frames": ev["n_frames"], "n_deauth": ev["n_deauth"]}
        self.sink.prediction(base)
        self.streak = self.streak + 1 if mal else 0
        if mal and self.streak >= self.consecutive:
            self.stats["alerts"] += 1
            ws = self.det.window_s
            dominated = (ev["n_deauth"] + ev["n_disassoc"]) >= DEAUTH_EVIDENCE_MIN
            self.sink.alert({
                "detected_utc": utc(time.time()), "window_start_utc": base["window_start_utc"], "mode": self.mode,
                "predicted_class": "malicious", "confidence": p,
                # descriptive evidence only: the LSTM made the decision above
                "likely_type": "likely deauthentication flood" if dominated else "malicious (type not determined)",
                "n_frames": ev["n_frames"], "frame_rate": ev["n_frames"] / ws, "n_deauth": ev["n_deauth"],
                "deauth_rate": ev["n_deauth"] / ws, "deauth_unique_targets": ev["n_deauth_dst"],
                "top_deauth_src": ev["top_deauth_src"], "top_src": ev["top_src"], "inference_ms": ms})
