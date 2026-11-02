import numpy as np
import pandas as pd
import pytest

from wids.config import Config
from wids.features import FEATURE_NAMES
from wids.sequences import SplitError, build_splits, make_sequences, split_runs


def table(n_norm=600, n_att=600):
    rng = np.random.default_rng(0)
    rows = []
    for cap, n, att in (("n", n_norm, False), ("a", n_att, True)):
        for i in range(n):
            lab = int(att and (i // 60) % 2 == 1)
            r = {"capture": cap, "window_idx": 1000 + i, "label": lab}
            r.update({f: float(rng.normal() + lab) for f in FEATURE_NAMES})
            rows.append(r)
    return pd.DataFrame(rows)


def test_make_sequences_shapes_and_last_label():
    X = np.arange(30, dtype=np.float32).reshape(15, 2)
    y = np.zeros(15, np.int8); y[12] = 1
    S, ys, idx = make_sequences(X, y, 5)
    assert S.shape == (11, 5, 2) and ys.sum() == 1 and idx[0] == 4
    assert (S[0, -1] == X[4]).all()


def test_splits_have_no_shared_windows_and_both_classes():
    cfg = Config(block_windows=60, seq_len=10)
    sp = build_splits(table(), FEATURE_NAMES, cfg)
    keys = {s: set(zip(d["meta"]["capture"], d["meta"]["block"])) for s, d in sp.items()}
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"]) and not (keys["val"] & keys["test"])
    for d in sp.values():
        assert set(np.unique(d["y"])) == {0, 1}


def test_split_is_deterministic():
    cfg = Config(block_windows=60, seq_len=10)
    a = build_splits(table(), FEATURE_NAMES, cfg)["test"]["meta"]
    b = build_splits(table(), FEATURE_NAMES, cfg)["test"]["meta"]
    assert a.equals(b)


def test_split_error_when_no_attack():
    with pytest.raises(SplitError):
        build_splits(table(n_att=0), FEATURE_NAMES, Config(block_windows=60))


def test_split_runs_break_on_split_change_and_gaps():
    widx = np.r_[np.arange(0, 45), np.arange(45, 75), np.arange(75, 100)]
    split = np.array(["train"] * 45 + ["val"] * 30 + ["train"] * 25, dtype=object)
    assert split_runs(widx, split) == [(0, 45, "train"), (45, 75, "val"), (75, 100, "train")]
    split2 = np.array(["train"] * 100, dtype=object)
    widx2 = np.r_[np.arange(0, 50), np.arange(60, 110)]                 # time gap
    assert split_runs(widx2, split2) == [(0, 50, "train"), (50, 100, "train")]
    split3 = np.array(["train"] * 10 + [None] * 10 + ["val"] * 10, dtype=object)
    assert [r[2] for r in split_runs(np.arange(30), split3)] == ["train", "val"]


def test_contiguous_run_yields_a_just_started_onset_sequence():
    """Attack starts 4 windows into a block. Old per-block code could never score windows 0-8 of a block;
    now a train/train neighbour supplies the context, so a 'just started' sequence exists."""
    cfg = Config(block_windows=30, seq_len=10, train_frac=1.0, val_frac=0.0)
    n = 90
    rows = []
    for i in range(n):
        lab = int(i >= 64)                           # onset at offset 4 of block 2 (windows 60-89)
        r = {"capture": "c", "window_idx": i, "label": lab}
        r.update({f: float(lab) if f == FEATURE_NAMES[0] else 0.0 for f in FEATURE_NAMES})
        rows.append(r)
    df = pd.DataFrame(rows)
    # all three blocks end up in one split: force it by assigning directly through split_runs on the table
    widx = df["window_idx"].to_numpy()
    runs = split_runs(widx, np.array(["train"] * n, dtype=object))
    assert runs == [(0, n, "train")]
    X, y, last = make_sequences(df[FEATURE_NAMES].to_numpy(np.float32), df["label"].to_numpy(np.int8), 10)
    onset = [i for i in range(len(y)) if last[i] == 64][0]
    assert y[onset] == 1 and (X[onset, :-1, 0] == 0).all() and X[onset, -1, 0] == 1
