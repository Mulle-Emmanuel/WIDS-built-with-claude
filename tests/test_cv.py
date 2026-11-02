"""Blocked-CV and false-positive-check plumbing on SYNTHETIC windows (numbers are meaningless)."""
import json
import random

import numpy as np
import pandas as pd
import pytest

from wids.cli import main
from wids.features import frames_to_windows
from wids.frames import parse_dot11_bytes
from wids.sequences import purge_mask
import synth


def windows_table(frames, capture):
    fr = []
    for t, raw in frames:
        f = parse_dot11_bytes(synth.radiotap() + raw, t)
        f.is_attack = f.src == synth.ATTACKER
        fr.append(f)
    df = pd.DataFrame([w.to_row() for w in frames_to_windows(sorted(fr, key=lambda x: x.ts))])
    df.insert(0, "capture", capture)
    return df


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    d = tmp_path_factory.mktemp("cv")
    rng = random.Random(5)
    base = 1_700_000_000
    a = synth.normal_frames(base, 400, rng, fps=15) + synth.flood_frames(base + 220, 170, rng, rate=120)
    windows_table(a, "deauth").to_csv(d / "a.csv", index=False)
    windows_table(synth.normal_frames(base + 50_000, 200, rng, fps=15), "other_normal").to_csv(d / "n.csv", index=False)
    return d


def test_purge_mask_excludes_neighbours():
    last = np.arange(10, 60)                      # sequences ending at windows 10..59, seq_len 10
    keep = purge_mask(last, 10, [(30, 39)], purge=5)
    kept = last[keep]
    # every kept sequence ends >5 windows before the test block, or starts >5 windows after it
    assert all(l < 30 - 5 or l - 9 > 39 + 5 for l in kept)
    assert 24 in kept and 25 not in kept          # last allowed sequence before the purge zone
    assert 54 in kept and 53 not in kept          # first allowed sequence after the purge zone


def test_cv_runs_and_reports(ws):
    main(["cv", "--windows", str(ws / "a.csv"), "--results-dir", str(ws / "res"), "--folds", "3",
          "--epochs", "3", "--lstm-units", "8", "--block-windows", "30"])
    m = json.loads((ws / "res" / "cv_metrics.json").read_text())
    assert len(m["folds"]) == 3
    assert {"lstm", "frame_rate_rule", "deauth_rate_rule", "mgmt_rate_rule"} <= set(m["pooled"])
    assert m["pooled"]["lstm"]["n"] == sum(f["n_test"] for f in m["folds"])
    assert "onset_delays_windows" in m and all("onsets" in f for f in m["folds"])
    assert sum(len(f["onsets"]) for f in m["folds"]) == 1          # one synthetic attack episode
    assert (ws / "res" / "cv_confusion_matrix.png").exists()


def test_cv_refuses_too_few_blocks(ws):
    with pytest.raises(SystemExit):
        main(["cv", "--windows", str(ws / "a.csv"), "--results-dir", str(ws / "res2"), "--folds", "5",
              "--epochs", "1", "--block-windows", "400"])


def test_fpcheck_uses_only_attack_free_windows(ws):
    main(["train", "--windows", str(ws / "a.csv"), "--models-dir", str(ws / "models"), "--results-dir", str(ws / "r3"),
          "--epochs", "3", "--lstm-units", "8", "--block-windows", "30"])
    main(["fpcheck", "--windows", str(ws / "a.csv"), str(ws / "n.csv"), "--models-dir", str(ws / "models"),
          "--results-dir", str(ws / "r3")])
    r = json.loads((ws / "r3" / "fpcheck.json").read_text())
    assert r["normal_sequences"] > 0 and 0 <= r["false_positive_rate"] <= 1
    assert set(r["captures"]) == {"deauth", "other_normal"}
    assert r["flagged_events"] <= r["flagged"] and (ws / "r3" / "fpcheck_flagged.csv").exists()
    assert r["fpr_upper95_independent_segments"] >= r["fpr_upper95_overlapping_sequences"] > 0
    main(["transfer", "--windows", str(ws / "a.csv"), "--models-dir", str(ws / "models"), "--results-dir", str(ws / "r3")])
    t = json.loads((ws / "r3" / "transfer.json").read_text())
    assert set(t["overall"]["confusion_matrix"]) == {"tn", "fp", "fn", "tp"} and "deauth" in t["per_capture"]
    assert {"deauth_rate_rule", "mgmt_rate_rule", "frame_rate_rule"} <= set(t["rules"])
    assert "deauth" in t["rules"]["mgmt_rate_rule"]


def test_threshold_is_not_capped_for_count_scores():
    """Regression: choose_threshold once capped every threshold at 1.0, breaking rule baselines on counts."""
    from wids.evaluate import choose_threshold
    y = np.array([0] * 50 + [1] * 50)
    scores = np.r_[np.random.default_rng(0).integers(1000, 1600, 50), np.random.default_rng(1).integers(1700, 2600, 50)]
    t = choose_threshold(y, scores, 0.02)
    assert t > 1000                                   # not capped at 1.0
    assert (scores[y == 1] >= t).all()                # all attacks caught
    assert (scores[y == 0] >= t).mean() <= 0.02       # FPR within budget


def test_threshold_is_lowest_within_fpr_budget_not_weakest_attack():
    """Regression: the threshold must not be pinned to the weakest validation attack score."""
    from wids.evaluate import choose_threshold
    rng = np.random.default_rng(0)
    normal = rng.uniform(0, 0.1, 100)
    attack = np.r_[0.9995, rng.uniform(0.9999, 1.0, 20)]
    y = np.r_[np.zeros(100), np.ones(21)]
    t = choose_threshold(y, np.r_[normal, attack], 0.02)
    assert 0 < t < 0.11                               # near the normal scores, far below 0.9995
    assert (np.r_[normal, attack][y == 0] >= t).mean() <= 0.02


def test_threshold_sits_mid_gap_when_fp_budget_is_zero():
    """Regression for the real failure: ~40 validation normals => 1 FP = 2.5% > 2% budget => zero allowed FPs.
    The cut-off must be mid-gap, not pinned to the weakest validation attack."""
    from wids.evaluate import choose_threshold
    rng = np.random.default_rng(0)
    normal = rng.uniform(0, 0.003, 40)
    attack = np.r_[0.9995, rng.uniform(0.9996, 1.0, 20)]
    y = np.r_[np.zeros(40), np.ones(21)]
    scores = np.r_[normal, attack]
    t = choose_threshold(y, scores, 0.02)
    assert 0.3 < t < 0.7
    assert (scores[y == 1] >= t).all() and not (scores[y == 0] >= t).any()
    # a held-out attack that scores a bit lower than every validation attack is still caught
    assert 0.98 >= t


def test_exclude_deauth_preset_drops_columns():
    from wids.cli import _cfg_from_args
    from argparse import Namespace
    cfg = _cfg_from_args(Namespace(exclude_features="deauth", include_rssi=False))
    assert "log_n_deauth" in cfg.exclude_features and "rssi_mean" in cfg.exclude_features


def test_every_block_window_is_a_test_target(ws):
    """Test sequences borrow context from the purge zone, so a block's first windows are scored too."""
    import json
    m = json.loads((ws / "res" / "cv_metrics.json").read_text())
    df = pd.read_csv(ws / "a.csv")
    # old behaviour lost 9 targets in EVERY block; now only the capture's first 9 windows (no context) are unscored
    assert m["pooled"]["lstm"]["n"] == len(df) - 9


def test_duplicate_capture_under_two_ids_is_rejected(ws, tmp_path):
    from wids.train import load_windows
    d = pd.read_csv(ws / "a.csv")
    d.to_csv(tmp_path / "x.csv", index=False)
    d.assign(capture="deauth_again").to_csv(tmp_path / "y.csv", index=False)
    with pytest.raises(ValueError, match="identical windows"):
        load_windows([str(tmp_path / "x.csv"), str(tmp_path / "y.csv")])
    with pytest.raises(ValueError, match="duplicate windows"):
        load_windows([str(tmp_path / "x.csv"), str(tmp_path / "x.csv")])


def test_rule_scores_are_raw_per_second_rates():
    from wids.evaluate import rule_score
    cols = ["log_n_frames", "frac_mgmt", "log_n_deauth"]
    X = np.zeros((2, 3, 3), np.float32)
    X[:, -1, 0] = np.log1p([1000, 200]); X[:, -1, 1] = [0.1, 0.5]; X[:, -1, 2] = np.log1p([7, 0])
    assert rule_score("frame_rate_rule", X, cols) == pytest.approx([1000, 200], rel=1e-4)
    assert rule_score("mgmt_rate_rule", X, cols) == pytest.approx([100, 100], rel=1e-4)
    assert rule_score("deauth_rate_rule", X, cols) == pytest.approx([7, 0], abs=1e-3)
