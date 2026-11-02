"""Loads the trained LSTM + preprocessing artefacts and scores window sequences."""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np

from .features import FEATURE_NAMES, Window

REQUIRED = ["wids_lstm.keras", "scaler.joblib", "meta.json"]


class ModelArtifactsError(RuntimeError):
    pass


def load_artifacts(models_dir: str):
    d = Path(models_dir)
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        raise ModelArtifactsError(
            f"missing {missing} in '{d}'. Train first: wids train --windows <files> --models-dir {d}")
    import keras
    meta = json.loads((d / "meta.json").read_text())
    if meta["all_features"] != FEATURE_NAMES:
        raise ModelArtifactsError("feature schema in meta.json differs from this code version; "
                                  "retrain the model or check out the matching code")
    return keras.models.load_model(d / "wids_lstm.keras"), joblib.load(d / "scaler.joblib"), meta


class Detector:
    def __init__(self, models_dir: str, threshold: Optional[float] = None):
        self.model, self.scaler, self.meta = load_artifacts(models_dir)
        self.seq_len = int(self.meta["seq_len"])
        self.window_s = float(self.meta["window_s"])
        self.threshold = float(self.meta["threshold"] if threshold is None else threshold)
        self.cols = [FEATURE_NAMES.index(c) for c in self.meta["feature_cols"]]
        self.buf: deque = deque(maxlen=self.seq_len)
        self._last_idx: Optional[int] = None

    def reset(self):
        self.buf.clear()
        self._last_idx = None

    def push(self, w: Window) -> Optional[float]:
        """Add one window; return P(malicious) once `seq_len` consecutive windows are buffered."""
        if self._last_idx is not None and w.idx != self._last_idx + 1:
            self.buf.clear()                       # gap in time: do not stitch unrelated windows
        self._last_idx = w.idx
        self.buf.append(np.asarray(w.features, np.float32)[self.cols])
        if len(self.buf) < self.seq_len:
            return None
        x = self.scaler.transform(np.stack(self.buf)).astype(np.float32)[None]
        return float(self.model(x, training=False).numpy()[0, 0])
