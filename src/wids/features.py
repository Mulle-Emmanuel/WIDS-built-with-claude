"""Per-window feature extraction. One shared implementation for training, replay and live use."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .frames import BROADCAST, Frame

COUNT_FEATURES = ["n_frames", "n_deauth", "n_disassoc", "n_auth", "n_assoc", "n_probe", "n_beacon", "n_action"]

# Order matters: the model input columns are selected from this list by name.
FEATURE_NAMES: List[str] = (
    ["log_" + c for c in COUNT_FEATURES]
    + ["frac_mgmt", "frac_ctrl", "frac_data", "retry_ratio", "deauth_bcast_frac",
       "rssi_mean", "rssi_std", "rssi_available",
       "log_uniq_src", "log_uniq_dst", "log_deauth_uniq_dst", "log_deauth_uniq_src",
       "d_log_frames", "d_log_deauth"]
)
DEAUTH_DERIVED = ["log_n_deauth", "log_n_disassoc", "deauth_bcast_frac", "log_deauth_uniq_dst",
                  "log_deauth_uniq_src", "d_log_deauth"]
EVIDENCE_COLS = ["n_frames", "n_deauth", "n_disassoc", "n_deauth_dst", "top_deauth_src", "top_src"]


def model_features(exclude) -> List[str]:
    return [f for f in FEATURE_NAMES if f not in set(exclude)]


@dataclass
class Window:
    idx: int
    t_start: float
    features: List[float]
    n_attack: int
    label: int
    evidence: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {"window_idx": self.idx, "t_start": self.t_start}
        row.update(dict(zip(FEATURE_NAMES, self.features)))
        row.update({"n_attack": self.n_attack, "label": self.label})
        row.update({"ev_" + k: v for k, v in self.evidence.items()})
        return row


class WindowAggregator:
    """Turns a time-ordered stream of Frames into fixed-length Windows.

    Empty seconds are emitted as all-zero windows, so the LSTM sees silence as silence.
    A gap longer than `max_gap_windows` starts a new segment (rate deltas reset).
    """

    def __init__(self, window_s: float = 1.0, min_attack_frames: int = 3, max_gap_windows: int = 120):
        self.w, self.min_attack, self.max_gap = window_s, min_attack_frames, max_gap_windows
        self.cur: Optional[int] = None
        self.buf: List[Frame] = []
        self._prev_lf = self._prev_ld = 0.0

    def _idx(self, ts: float) -> int:
        return int(math.floor(ts / self.w))

    def add(self, f: Frame) -> List[Window]:
        out: List[Window] = []
        i = self._idx(f.ts)
        if self.cur is None:
            self.cur = i
        i = max(i, self.cur)                      # late frame: attribute to the current window
        if i > self.cur:
            out += self._close_until(i)
        self.buf.append(f)
        return out

    def advance_to(self, ts: float) -> List[Window]:
        """Close windows that ended before wall-clock `ts` (live mode, no traffic case)."""
        if self.cur is None:
            return []
        i = self._idx(ts)
        return self._close_until(i) if i > self.cur else []

    def flush(self) -> List[Window]:
        if self.cur is None or not self.buf:
            return []
        out = [self._emit(self.buf, self.cur)]
        self.buf, self.cur = [], self.cur + 1
        return out

    def _close_until(self, target: int) -> List[Window]:
        out: List[Window] = []
        if target - self.cur > self.max_gap:      # long silence: close current, jump ahead
            if self.buf:
                out.append(self._emit(self.buf, self.cur))
            self.buf, self.cur = [], target
            self._prev_lf = self._prev_ld = 0.0
            return out
        while self.cur < target:
            out.append(self._emit(self.buf, self.cur))
            self.buf = []
            self.cur += 1
        return out

    def _emit(self, frames: List[Frame], idx: int) -> Window:
        n = len(frames)
        c = Counter()
        mgmt = ctrl = data = retry = bcast = n_attack = 0
        srcs, dsts, d_dst, d_src = set(), set(), set(), Counter()
        rssi = []
        for f in frames:
            if f.src:
                srcs.add(f.src)
            if f.dst:
                dsts.add(f.dst)
            retry += f.retry
            n_attack += f.is_attack
            if f.rssi is not None:
                rssi.append(f.rssi)
            if f.ftype == 0:
                mgmt += 1
                s = f.subtype
                if s == 12:
                    c["n_deauth"] += 1
                    d_src[f.src] += 1
                    d_dst.add(f.dst)
                    bcast += f.dst == BROADCAST
                elif s == 10:
                    c["n_disassoc"] += 1
                elif s == 11:
                    c["n_auth"] += 1
                elif s in (0, 1, 2, 3):
                    c["n_assoc"] += 1
                elif s in (4, 5):
                    c["n_probe"] += 1
                elif s == 8:
                    c["n_beacon"] += 1
                elif s == 13:
                    c["n_action"] += 1
            elif f.ftype == 1:
                ctrl += 1
            elif f.ftype == 2:
                data += 1
        c["n_frames"] = n
        nd = c["n_deauth"]
        lf, ld = math.log1p(n), math.log1p(nd)
        feats = {"log_" + k: math.log1p(c[k]) for k in COUNT_FEATURES}
        feats.update({
            "frac_mgmt": mgmt / n if n else 0.0, "frac_ctrl": ctrl / n if n else 0.0,
            "frac_data": data / n if n else 0.0, "retry_ratio": retry / n if n else 0.0,
            "deauth_bcast_frac": bcast / nd if nd else 0.0,
            "rssi_mean": float(np.mean(rssi)) if rssi else 0.0,
            "rssi_std": float(np.std(rssi)) if rssi else 0.0,
            "rssi_available": 1.0 if rssi else 0.0,
            "log_uniq_src": math.log1p(len(srcs)), "log_uniq_dst": math.log1p(len(dsts)),
            "log_deauth_uniq_dst": math.log1p(len(d_dst)), "log_deauth_uniq_src": math.log1p(len(d_src)),
            "d_log_frames": lf - self._prev_lf, "d_log_deauth": ld - self._prev_ld,
        })
        self._prev_lf, self._prev_ld = lf, ld
        top_src = Counter(f.src for f in frames if f.src).most_common(1)
        ev = {"n_frames": n, "n_deauth": nd, "n_disassoc": c["n_disassoc"], "n_deauth_dst": len(d_dst),
              "top_deauth_src": d_src.most_common(1)[0][0] if d_src else "",
              "top_src": top_src[0][0] if top_src else ""}
        return Window(idx, idx * self.w, [feats[k] for k in FEATURE_NAMES], int(n_attack),
                      int(n_attack >= self.min_attack), ev)


def frames_to_windows(frames, window_s=1.0, min_attack_frames=3, max_gap_windows=120) -> List[Window]:
    agg = WindowAggregator(window_s, min_attack_frames, max_gap_windows)
    out: List[Window] = []
    for f in frames:
        out += agg.add(f)
    return out + agg.flush()
