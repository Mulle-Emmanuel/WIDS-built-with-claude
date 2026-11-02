"""Sequence construction and leak-free splitting.

Splitting unit = a *block* of consecutive windows from one capture. Whole blocks go to
train, validation or test, and sequences are only built inside a block, so no window (and
no overlapping sequence) can appear in two splits. Blocks are stratified by whether they
contain any malicious window, so every split sees both classes.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config


class SplitError(ValueError):
    pass


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (n_windows, n_feat) contiguous windows. Returns sequences, labels (of the last window),
    and the positional index of each sequence's last window."""
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]), np.float32), np.empty(0, np.int8), np.empty(0, int)
    idx = np.arange(seq_len - 1, n)
    seqs = np.stack([X[i - seq_len + 1:i + 1] for i in idx]).astype(np.float32)
    return seqs, y[idx].astype(np.int8), idx


def _runs(window_idx: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous runs (start, stop) of consecutive window indices."""
    if len(window_idx) == 0:
        return []
    breaks = np.where(np.diff(window_idx) != 1)[0] + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, len(window_idx)]
    return list(zip(starts, stops))


def assign_blocks(block_has_attack: Dict[tuple, bool], cfg: Config) -> Dict[tuple, str]:
    rng = np.random.default_rng(cfg.seed)
    assignment: Dict[tuple, str] = {}
    for flag in (True, False):
        keys = sorted(k for k, v in block_has_attack.items() if v == flag)
        rng.shuffle(keys)
        n = len(keys)
        n_tr = int(round(cfg.train_frac * n))
        n_va = int(round(cfg.val_frac * n))
        if n >= 3:                                   # guarantee every split gets at least one
            n_va = max(n_va, 1)
            n_tr = min(n_tr, n - n_va - 1)
        for i, k in enumerate(keys):
            assignment[k] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assignment


def split_runs(widx, split):
    """Contiguous (start, stop, split) runs: consecutive window ids AND the same split (None = excluded)."""
    out, start = [], 0
    for i in range(1, len(widx) + 1):
        if i == len(widx) or widx[i] != widx[i - 1] + 1 or split[i] != split[start]:
            if split[start] is not None:
                out.append((start, i, split[start]))
            start = i
    return out


def build_splits(df: pd.DataFrame, feature_cols: List[str], cfg: Config):
    """df: window table (columns: capture, window_idx, label, features...).

    Blocks are assigned to a split; sequences are built over contiguous runs of windows that belong to the
    SAME split, so (a) no sequence ever contains a window from another split, and (b) adjacent blocks of one
    split merge, which keeps the first windows of each block (e.g. an attack onset) as prediction targets."""
    flags = {}
    for cap, g in df.groupby("capture", sort=True):
        g = g.sort_values("window_idx")
        bid = ((g["window_idx"] - g["window_idx"].min()) // cfg.block_windows).to_numpy()
        for b in np.unique(bid):
            part = g[bid == b]
            if len(part) >= cfg.seq_len:
                flags[(cap, int(b))] = bool(part["label"].max() == 1)
    if not flags:
        raise SplitError("no block has enough windows to build a sequence; check seq_len/block_windows")
    assignment = assign_blocks(flags, cfg)

    out = {s_: {"X": [], "y": [], "meta": []} for s_ in ("train", "val", "test")}
    for cap, g in df.groupby("capture", sort=True):
        g = g.sort_values("window_idx")
        base = g["window_idx"].min()
        bid = ((g["window_idx"] - base) // cfg.block_windows).to_numpy()
        split = np.array([assignment.get((cap, int(b))) for b in bid], dtype=object)
        feats = g[feature_cols].to_numpy(np.float32)
        labels = g["label"].to_numpy(np.int8)
        widx = g["window_idx"].to_numpy()
        for a_, e_, s_ in split_runs(widx, split):
            Xs, ys, last = make_sequences(feats[a_:e_], labels[a_:e_], cfg.seq_len)
            if len(ys) == 0:
                continue
            lw = widx[a_:e_][last]
            out[s_]["X"].append(Xs)
            out[s_]["y"].append(ys)
            out[s_]["meta"].append(pd.DataFrame({"capture": cap, "block": (lw - base) // cfg.block_windows,
                                                 "window_idx": lw}))
    result = {}
    n_att = sum(flags.values())
    for s_, d in out.items():
        if not d["X"]:
            raise SplitError(f"split '{s_}' is empty: not enough data or blocks (try a smaller block_windows)")
        y = np.concatenate(d["y"])
        result[s_] = {"X": np.concatenate(d["X"]), "y": y, "meta": pd.concat(d["meta"], ignore_index=True)}
        if len(np.unique(y)) < 2:
            raise SplitError(
                f"split '{s_}' contains only class {int(y[0])}. Need more attack AND normal blocks "
                f"(currently {n_att} attack blocks, {len(flags) - n_att} normal). "
                f"Try a smaller block_windows or more captures.")
    return result


def block_sequences(df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> Dict[tuple, dict]:
    """Sequences per block: {(capture, block): X, y, last (window_idx of last window), lo, hi, has_attack}."""
    out: Dict[tuple, dict] = {}
    for cap, g in df.groupby("capture", sort=True):
        g = g.sort_values("window_idx")
        bid = ((g["window_idx"] - g["window_idx"].min()) // cfg.block_windows).to_numpy()
        for b in np.unique(bid):
            part = g[bid == b]
            feats = part[feature_cols].to_numpy(np.float32)
            labels = part["label"].to_numpy(np.int8)
            widx = part["window_idx"].to_numpy()
            Xs, ys, ls = [], [], []
            for a, e in _runs(widx):
                X_, y_, last = make_sequences(feats[a:e], labels[a:e], cfg.seq_len)
                if len(y_):
                    Xs.append(X_), ys.append(y_), ls.append(widx[a:e][last])
            if Xs:
                out[(cap, int(b))] = {"X": np.concatenate(Xs), "y": np.concatenate(ys), "last": np.concatenate(ls),
                                      "has_attack": bool(labels.max() == 1), "lo": int(widx.min()), "hi": int(widx.max())}
    return out


def assign_folds(block_has_attack: Dict[tuple, bool], k: int, seed: int) -> Dict[tuple, int]:
    """Stratified (attack / no-attack) assignment of whole blocks to k folds."""
    rng = np.random.default_rng(seed)
    folds: Dict[tuple, int] = {}
    for flag in (True, False):
        keys = sorted(b for b, v in block_has_attack.items() if v == flag)
        rng.shuffle(keys)
        for i, b in enumerate(keys):
            folds[b] = i % k
    return folds


def purge_mask(last: np.ndarray, seq_len: int, test_ranges: List[Tuple[int, int]], purge: int) -> np.ndarray:
    """True for sequences that may be used for training/validation: they must not overlap any test
    block, nor come within `purge` windows of it (adjacent windows are strongly correlated)."""
    keep = np.ones(len(last), bool)
    first = last - seq_len + 1
    for lo, hi in test_ranges:
        keep &= ~((last >= lo - purge) & (first <= hi + purge))
    return keep


def sequences_from_table(df: pd.DataFrame, feature_cols: List[str], seq_len: int):
    """All sequences from contiguous runs of a window table (used for the false-positive check)."""
    Xs, ys, metas = [], [], []
    for cap, g in df.groupby("capture", sort=True):
        g = g.sort_values("window_idx")
        feats, labels, widx = g[feature_cols].to_numpy(np.float32), g["label"].to_numpy(np.int8), g["window_idx"].to_numpy()
        for a, e in _runs(widx):
            X_, y_, last = make_sequences(feats[a:e], labels[a:e], seq_len)
            if len(y_):
                Xs.append(X_), ys.append(y_), metas.append(pd.DataFrame({"capture": cap, "window_idx": widx[a:e][last]}))
    if not Xs:
        raise SplitError("no sequences could be built (too few contiguous windows)")
    return np.concatenate(Xs), np.concatenate(ys), pd.concat(metas, ignore_index=True)
