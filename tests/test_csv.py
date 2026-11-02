"""CSV (AWID3-CSV style) path must give the same windows as the pcap path on the same traffic.
Synthetic data: plumbing check only."""
import random

import pandas as pd
import pytest

from wids.csvsrc import chunk_files
from wids.cli import main
from wids.features import FEATURE_NAMES, frames_to_windows
from wids.frames import iter_pcap, parse_dot11_bytes
import synth


def write_chunks(d, frames, rows_per_chunk=250):
    """Mimic AWID3-CSV: extra columns, '-' joined multi-values, Label column, numbered chunks."""
    recs = []
    for i, (t, raw) in enumerate(sorted(frames, key=lambda x: x[0]), 1):
        f = parse_dot11_bytes(synth.radiotap(-40) + raw, t)
        lab = "Deauth" if f.src == synth.ATTACKER else "Normal"
        recs.append({"frame.encap_type": 23, "frame.len": f.length, "frame.number": i, "frame.time_epoch": f"{t:.6f}",
                     "radiotap.dbm_antsignal": "-34-37-34", "wlan.fc.type": f.ftype, "wlan.fc.retry": int(f.retry),
                     "wlan.fc.subtype": f.subtype, "wlan.ra": f.dst, "wlan.ta": f.src or "", "Label": lab})
    df = pd.DataFrame(recs)
    for k, a in enumerate(range(0, len(df), rows_per_chunk)):
        df.iloc[a:a + rows_per_chunk].to_csv(d / f"Deauth_{k}.csv", index=False)
    return len(range(0, len(df), rows_per_chunk))


def test_chunks_sorted_numerically(tmp_path):
    for n in (0, 2, 10, 1):
        (tmp_path / f"Deauth_{n}.csv").write_text("x\n")
    assert [p.name for p in chunk_files(str(tmp_path))] == ["Deauth_0.csv", "Deauth_1.csv", "Deauth_2.csv", "Deauth_10.csv"]


def test_csv_windows_equal_pcap_windows(tmp_path):
    rng = random.Random(3)
    frames = synth.normal_frames(1_700_000_000, 60, rng) + synth.flood_frames(1_700_000_030, 20, rng)
    n_chunks = write_chunks(tmp_path, frames)
    assert n_chunks > 10                                  # exercises Deauth_10+ ordering
    synth.write_pcap(str(tmp_path / "t.pcap"), frames)
    main(["extract-csv", "--csv-dir", str(tmp_path), "--pattern", "Deauth_*.csv", "--capture-id", "x",
          "--out", str(tmp_path / "w.csv")])
    csv = pd.read_csv(tmp_path / "w.csv")
    ref = pd.DataFrame([w.to_row() for w in frames_to_windows(iter_pcap(str(tmp_path / "t.pcap"), [synth.ATTACKER]))])
    assert len(csv) == len(ref)
    cols = [c for c in FEATURE_NAMES if not c.startswith("rssi")]
    assert csv[cols].to_numpy() == pytest.approx(ref[cols].to_numpy())
    assert (csv["label"].to_numpy() == ref["label"].to_numpy()).all()


def test_wrong_normal_label_is_an_error(tmp_path):
    write_chunks(tmp_path, synth.normal_frames(1_700_000_000, 5, random.Random(1)))
    with pytest.raises(SystemExit):
        main(["extract-csv", "--csv-dir", str(tmp_path), "--capture-id", "x", "--normal-label", "Benign",
              "--out", str(tmp_path / "w.csv")])


def test_replay_csv_raises_alerts_with_trained_model(tmp_path):
    """Synthetic: train a tiny model on windows, then replay the CSV chunks through the detector."""
    import sqlite3
    rng = random.Random(11)
    frames = synth.normal_frames(1_700_000_000, 240, rng, fps=15) + synth.flood_frames(1_700_000_120, 100, rng, rate=120)
    csvdir = tmp_path / "csv"
    csvdir.mkdir()
    write_chunks(csvdir, frames, rows_per_chunk=1000)
    main(["extract-csv", "--csv-dir", str(csvdir), "--pattern", "Deauth_*.csv", "--capture-id", "x",
          "--out", str(tmp_path / "w.csv")])
    main(["train", "--windows", str(tmp_path / "w.csv"), "--models-dir", str(tmp_path / "m"),
          "--results-dir", str(tmp_path / "r"), "--epochs", "4", "--lstm-units", "8", "--block-windows", "30"])
    db = tmp_path / "a.db"
    main(["replay-csv", "--csv-dir", str(csvdir), "--pattern", "Deauth_*.csv", "--models-dir", str(tmp_path / "m"),
          "--db", str(db), "--jsonl", str(tmp_path / "a.jsonl")])
    n = sqlite3.connect(db).execute("SELECT COUNT(*) FROM alerts WHERE mode='REPLAY-CSV'").fetchone()[0]
    assert n > 0
