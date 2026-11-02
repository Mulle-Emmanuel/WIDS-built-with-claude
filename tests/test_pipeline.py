"""Integration test on SYNTHETIC traffic: checks the plumbing (extract -> train -> save -> load ->
replay -> alert log). The metrics it produces are meaningless and must never be reported."""
import json
import sqlite3
import subprocess
import sys

import pytest

from wids.cli import main
from wids.config import Config
from wids.detector import Detector, ModelArtifactsError
import synth


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    d = tmp_path_factory.mktemp("ws")
    n, a = synth.make_captures(str(d))
    main(["extract", "--pcap", n, "--out", str(d / "n.csv"), "--capture-id", "normal", "--backend", "python"])
    main(["extract", "--pcap", a, "--out", str(d / "a.csv"), "--capture-id", "attack", "--backend", "python",
          "--attack-macs", synth.ATTACKER])
    main(["train", "--windows", str(d / "n.csv"), str(d / "a.csv"), "--models-dir", str(d / "models"),
          "--results-dir", str(d / "results"), "--epochs", "6", "--block-windows", "40", "--lstm-units", "8"])
    return d, a


def test_artifacts_and_results_written(workspace):
    d, _ = workspace
    for f in ("wids_lstm.keras", "scaler.joblib", "meta.json"):
        assert (d / "models" / f).exists()
    for f in ("metrics.json", "confusion_matrix.png", "training_curves.png"):
        assert (d / "results" / f).exists()
    m = json.loads((d / "results" / "metrics.json").read_text())
    assert set(m["test_lstm"]["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
    assert m["baseline_comparison_only"] is not None


def test_replay_raises_alerts_and_logs(workspace):
    d, attack = workspace
    db, jl = d / "alerts.db", d / "alerts.jsonl"
    main(["replay", "--pcap", attack, "--models-dir", str(d / "models"), "--db", str(db), "--jsonl", str(jl)])
    rows = sqlite3.connect(db).execute("SELECT predicted_class, confidence, top_deauth_src FROM alerts").fetchall()
    assert rows, "flood replay produced no alerts"
    assert all(r[0] == "malicious" and 0 <= r[1] <= 1 for r in rows)
    assert any(r[2] == synth.ATTACKER for r in rows)
    assert json.loads(jl.read_text().splitlines()[0])["mode"] == "REPLAY"


def test_normal_replay_mostly_quiet(workspace):
    d, _ = workspace
    n = str(d / "normal.pcap")
    db = d / "n_alerts.db"
    main(["replay", "--pcap", n, "--models-dir", str(d / "models"), "--db", str(db), "--jsonl", str(d / "n.jsonl")])
    alerts = sqlite3.connect(db).execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    scored = sqlite3.connect(db).execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert alerts < 0.2 * scored


def test_missing_artifacts_error(tmp_path):
    with pytest.raises(ModelArtifactsError):
        Detector(str(tmp_path))


def test_non_wifi_capture_rejected(tmp_path):
    p = tmp_path / "eth.pcap"
    synth.write_pcap(str(p), [(1.0, b"\x00" * 60)], linktype=1)     # Ethernet linktype
    with pytest.raises(SystemExit):
        main(["extract", "--pcap", str(p), "--out", str(tmp_path / "x.csv"), "--backend", "python"])


def test_training_features_match_inference_features(workspace):
    """Windows extracted for training and windows built during replay must be identical."""
    import pandas as pd
    from wids.features import FEATURE_NAMES, frames_to_windows
    from wids.frames import iter_pcap
    d, attack = workspace
    csv = pd.read_csv(d / "a.csv")
    ws = frames_to_windows(iter_pcap(attack, [synth.ATTACKER]))
    rep = pd.DataFrame([w.to_row() for w in ws])
    assert len(csv) == len(rep)
    assert (csv[FEATURE_NAMES].to_numpy() == pytest.approx(rep[FEATURE_NAMES].to_numpy()))


def test_alert_text_uses_absolute_deauth_count(workspace):
    d, attack = workspace
    jl = d / "alerts_tag.jsonl"
    main(["replay", "--pcap", attack, "--models-dir", str(d / "models"), "--db", str(d / "tag.db"), "--jsonl", str(jl)])
    recs = [json.loads(x) for x in jl.read_text().splitlines()]
    assert any(r["likely_type"] == "likely deauthentication flood" for r in recs)


def test_replay_starts_fresh_logs_unless_append(workspace):
    d, attack = workspace
    db, jl = d / "fresh.db", d / "fresh.jsonl"
    args = ["replay", "--pcap", attack, "--models-dir", str(d / "models"), "--db", str(db), "--jsonl", str(jl)]
    main(args)
    n1 = sqlite3.connect(db).execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    main(args)
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == n1
    main(args + ["--append"])
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 2 * n1
