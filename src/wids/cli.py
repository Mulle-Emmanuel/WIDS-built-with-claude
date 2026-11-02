"""Command-line interface:  wids extract | train | evaluate | replay | live | check"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .config import Config
from .features import WindowAggregator
from .frames import (CaptureToolError, DLT_IEEE802_11_RADIO, NoWifiFramesError, iter_pcap, iter_tshark,
                     parse_dot11_bytes)


def _macs(s):
    return [m.strip().lower() for m in s.split(",") if m.strip()] if s else []


# ------------------------------------------------------------------------------- extract
def cmd_extract(a):
    cfg = Config()
    macs = _macs(a.attack_macs)
    if a.backend == "tshark":
        frames = iter_tshark(a.pcap, a.attack_filter, macs, a.anonymize)
    else:
        if a.attack_filter:
            sys.exit("--attack-filter needs --backend tshark (display filters are a Wireshark feature)")
        frames = iter_pcap(a.pcap, macs, a.anonymize)
    agg = WindowAggregator(a.window_s, a.min_attack_frames, cfg.max_gap_windows)
    rows = []
    for f in frames:
        rows += [w.to_row() for w in agg.add(f)]
    rows += [w.to_row() for w in agg.flush()]
    if not rows:
        sys.exit("no windows produced: empty capture?")
    df = pd.DataFrame(rows)
    df.insert(0, "capture", a.capture_id or Path(a.pcap).stem)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{a.out}: {len(df)} windows, {int(df['label'].sum())} malicious "
          f"({df['label'].mean():.1%}); backend={a.backend}")
    if df["label"].sum() == 0 and (a.attack_filter or macs):
        print("WARNING: an attack filter/MAC list was given but no window was labelled malicious. "
              "Check the filter against the capture before training.")


def cmd_extract_csv(a):
    from .csvsrc import CsvStats, iter_csv_frames
    st = CsvStats()
    agg = WindowAggregator(a.window_s, a.min_attack_frames, Config().max_gap_windows)
    rows = []
    for f in iter_csv_frames(a.csv_dir, a.pattern, a.normal_label, a.anonymize, st):
        rows += [w.to_row() for w in agg.add(f)]
    rows += [w.to_row() for w in agg.flush()]
    print(f"rows read={st.rows}  skipped(unlabelled)={st.skipped_unlabelled}  skipped(bad)={st.skipped_bad}")
    print(f"label values seen: {st.labels}")
    if not rows:
        sys.exit("no windows produced")
    if a.normal_label not in st.labels:
        sys.exit(f"normal label '{a.normal_label}' never appears; pass --normal-label with the right value")
    df = pd.DataFrame(rows)
    df.insert(0, "capture", a.capture_id)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{a.out}: {len(df)} windows, {int(df['label'].sum())} malicious ({df['label'].mean():.1%}); backend=csv")


# --------------------------------------------------------------------------- train/eval
def _cfg_from_args(a) -> Config:
    cfg = Config()
    if getattr(a, "config", None):
        cfg = Config.from_dict(json.loads(Path(a.config).read_text()))
    for k in ("epochs", "seq_len", "block_windows", "seed", "lstm_units"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    if getattr(a, "include_rssi", False):
        cfg.exclude_features = []
    ex = getattr(a, "exclude_features", None)
    if ex:
        from .features import DEAUTH_DERIVED, FEATURE_NAMES
        names = []
        for tok in ex.split(","):
            names += DEAUTH_DERIVED if tok.strip() == "deauth" else [tok.strip()]
        bad = [n for n in names if n not in FEATURE_NAMES]
        if bad:
            sys.exit(f"unknown feature(s) {bad}; use 'deauth' or names from features.FEATURE_NAMES")
        cfg.exclude_features = sorted(set(cfg.exclude_features) | set(names))
    return cfg


def cmd_train(a):
    from .train import run_training
    run_training(a.windows, a.models_dir, a.results_dir, _cfg_from_args(a))


def cmd_cv(a):
    from .cv import run_cv
    cfg = _cfg_from_args(a)
    if a.block_windows is None:
        cfg.block_windows = 30
    run_cv(a.windows, a.results_dir, cfg, a.folds)


def cmd_fpcheck(a):
    from .cv import run_fpcheck
    run_fpcheck(a.windows, a.models_dir, a.results_dir)


def cmd_transfer(a):
    from .cv import run_transfer
    run_transfer(a.windows, a.models_dir, a.results_dir)


def cmd_evaluate(a):
    from .train import run_evaluation
    run_evaluation(a.windows, a.models_dir, a.results_dir)


# ------------------------------------------------------------------------ replay / live
def _make_engine(a, mode):
    from .alerts import AlertSink
    from .detector import Detector
    from .engine import Engine
    det = Detector(a.models_dir, a.threshold)
    sink = AlertSink(a.db, a.jsonl, fresh=not a.append)
    return Engine(det, sink, mode, a.consecutive or int(det.meta["config"].get("consecutive", 1)))


def _print_summary(s):
    print(f"\nframes={s['frames']}  windows={s['windows']}  scored={s['scored']}  alerts={s['alerts']}  "
          f"mean inference={s['mean_inference_ms']:.1f} ms/window")


def cmd_replay(a):
    eng = _make_engine(a, "REPLAY")
    print(f"*** REPLAY of {a.pcap}: recorded traffic, NOT live monitoring ***")
    first_ts = wall0 = None
    try:
        for f in iter_pcap(a.pcap, anonymize=a.anonymize):
            if a.speed > 0:
                first_ts = f.ts if first_ts is None else first_ts
                wall0 = time.time() if wall0 is None else wall0
                delay = wall0 + (f.ts - first_ts) / a.speed - time.time()
                if delay > 0:
                    time.sleep(delay)
            eng.feed(f)
    except KeyboardInterrupt:
        print("\ninterrupted")
    _print_summary(eng.finish())


def cmd_replay_csv(a):
    """Replay AWID3-CSV rows (real recorded frames, ground-truth labels ignored) through the detector."""
    from .csvsrc import iter_csv_frames
    eng = _make_engine(a, "REPLAY-CSV")
    print(f"*** REPLAY of AWID3-CSV rows from {a.csv_dir} ({a.pattern}): recorded traffic, NOT live monitoring ***")
    first_ts = wall0 = None
    try:
        for f in iter_csv_frames(a.csv_dir, a.pattern, a.normal_label, a.anonymize):
            if a.speed > 0:
                first_ts = f.ts if first_ts is None else first_ts
                wall0 = time.time() if wall0 is None else wall0
                delay = wall0 + (f.ts - first_ts) / a.speed - time.time()
                if delay > 0:
                    time.sleep(delay)
            eng.feed(f)
    except KeyboardInterrupt:
        print("\ninterrupted")
    _print_summary(eng.finish())


def check_monitor(iface: str):
    if shutil.which("iw") is None:
        print("warning: `iw` not found, cannot verify monitor mode", file=sys.stderr)
        return
    r = subprocess.run(["iw", "dev", iface, "info"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"interface '{iface}' not found: {r.stderr.strip()}")
    if "type monitor" not in r.stdout:
        sys.exit(f"'{iface}' is not in monitor mode. Run:\n  nmcli device set {iface} managed no\n"
                 f"  sudo ip link set {iface} down && sudo iw dev {iface} set type monitor\n"
                 f"  sudo ip link set {iface} up && sudo iw dev {iface} set channel <N>")


def cmd_live(a):
    check_monitor(a.iface)
    eng = _make_engine(a, "LIVE")
    try:
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        s.bind((a.iface, 0))
    except PermissionError:
        sys.exit("raw capture needs root or CAP_NET_RAW: run with sudo (activate the venv's python explicitly).")
    s.settimeout(0.2)
    print(f"*** LIVE monitoring on {a.iface} (Ctrl+C to stop) ***")
    end = time.time() + a.duration if a.duration else None
    try:
        while end is None or time.time() < end:
            try:
                raw = s.recv(65535)
            except socket.timeout:
                eng.tick()
                continue
            f = parse_dot11_bytes(raw, time.time(), DLT_IEEE802_11_RADIO)
            if f is not None:
                eng.feed(f)
            eng.tick()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        s.close()
    _print_summary(eng.finish())


def cmd_check(a):
    import importlib
    ok = True
    for m in ("numpy", "pandas", "sklearn", "scapy", "keras", "joblib", "matplotlib"):
        try:
            importlib.import_module(m)
            print(f"  ok   {m}")
        except ImportError:
            print(f"  MISSING {m}")
            ok = False
    print("  ok   tshark" if shutil.which("tshark") else "  MISSING tshark (needed for `extract --backend tshark`)")
    print("  ok   iw" if shutil.which("iw") else "  MISSING iw (needed for live mode checks)")
    sys.exit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="wids", description="LSTM-based wireless intrusion detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="pcap -> per-window feature table (CSV)")
    e.add_argument("--pcap", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--capture-id")
    e.add_argument("--backend", choices=["tshark", "python"], default="tshark")
    e.add_argument("--attack-filter", help="Wireshark display filter selecting ground-truth attack frames")
    e.add_argument("--attack-macs", help="comma-separated transmitter MACs whose frames are attack frames")
    e.add_argument("--window-s", type=float, default=1.0)
    e.add_argument("--min-attack-frames", type=int, default=3)
    e.add_argument("--anonymize", action="store_true")
    e.set_defaults(fn=cmd_extract)

    x = sub.add_parser("extract-csv", help="AWID3-CSV chunks -> per-window feature table")
    x.add_argument("--csv-dir", required=True)
    x.add_argument("--pattern", default="*.csv")
    x.add_argument("--capture-id", required=True)
    x.add_argument("--normal-label", default="Normal")
    x.add_argument("--out", required=True)
    x.add_argument("--window-s", type=float, default=1.0)
    x.add_argument("--min-attack-frames", type=int, default=3)
    x.add_argument("--anonymize", action="store_true")
    x.set_defaults(fn=cmd_extract_csv)

    t = sub.add_parser("train", help="train the LSTM once and save artefacts")
    t.add_argument("--windows", nargs="+", required=True)
    t.add_argument("--models-dir", default="models")
    t.add_argument("--results-dir", default="results")
    t.add_argument("--config")
    t.add_argument("--epochs", type=int)
    t.add_argument("--seq-len", type=int)
    t.add_argument("--block-windows", type=int)
    t.add_argument("--lstm-units", type=int)
    t.add_argument("--seed", type=int)
    t.add_argument("--include-rssi", action="store_true", help="ablation: feed RSSI features to the model")
    t.add_argument("--exclude-features", help="comma list of features to drop; 'deauth' = all deauth-derived ones")
    t.set_defaults(fn=cmd_train)

    cv = sub.add_parser("cv", help="blocked cross-validation with purge gap (headline evaluation)")
    cv.add_argument("--windows", nargs="+", required=True)
    cv.add_argument("--results-dir", default="results")
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--config")
    cv.add_argument("--epochs", type=int)
    cv.add_argument("--seq-len", type=int)
    cv.add_argument("--block-windows", type=int, help="default 30 for cv")
    cv.add_argument("--lstm-units", type=int)
    cv.add_argument("--seed", type=int)
    cv.add_argument("--include-rssi", action="store_true")
    cv.add_argument("--exclude-features", help="comma list of features to drop; 'deauth' = all deauth-derived ones")
    cv.set_defaults(fn=cmd_cv)

    fp = sub.add_parser("fpcheck", help="false-positive check of the saved model on attack-free windows")
    fp.add_argument("--windows", nargs="+", required=True)
    fp.add_argument("--models-dir", default="models")
    fp.add_argument("--results-dir", default="results")
    fp.set_defaults(fn=cmd_fpcheck)

    tr = sub.add_parser("transfer", help="score a DIFFERENT attack's windows with the saved model")
    tr.add_argument("--windows", nargs="+", required=True)
    tr.add_argument("--models-dir", default="models")
    tr.add_argument("--results-dir", default="results")
    tr.set_defaults(fn=cmd_transfer)

    v = sub.add_parser("evaluate", help="re-evaluate the saved model on the test split (no retraining)")
    v.add_argument("--windows", nargs="+", required=True)
    v.add_argument("--models-dir", default="models")
    v.add_argument("--results-dir", default="results")
    v.set_defaults(fn=cmd_evaluate)

    for name, fn in (("replay", cmd_replay), ("live", cmd_live)):
        r = sub.add_parser(name, help="replay a pcap through the detector" if name == "replay"
                           else "live monitor-mode detection")
        if name == "replay":
            r.add_argument("--pcap", required=True)
            r.add_argument("--speed", type=float, default=0, help="0 = as fast as possible, 1 = real time")
        else:
            r.add_argument("--iface", required=True)
            r.add_argument("--duration", type=float)
        r.add_argument("--models-dir", default="models")
        r.add_argument("--db", default="results/alerts.db")
        r.add_argument("--jsonl", default="results/alerts.jsonl")
        r.add_argument("--threshold", type=float)
        r.add_argument("--consecutive", type=int)
        r.add_argument("--anonymize", action="store_true")
        r.add_argument("--append", action="store_true", help="keep existing alert logs (default: start fresh)")
        r.set_defaults(fn=fn)

    rc = sub.add_parser("replay-csv", help="replay AWID3-CSV chunks through the detector (demo without a pcap)")
    rc.add_argument("--csv-dir", required=True)
    rc.add_argument("--pattern", default="*.csv", help="e.g. 'Deauth_2[0-5].csv' for chunks 20-25")
    rc.add_argument("--normal-label", default="Normal")
    rc.add_argument("--speed", type=float, default=0, help="0 = as fast as possible, 1 = real time")
    rc.add_argument("--models-dir", default="models")
    rc.add_argument("--db", default="results/alerts.db")
    rc.add_argument("--jsonl", default="results/alerts.jsonl")
    rc.add_argument("--threshold", type=float)
    rc.add_argument("--consecutive", type=int)
    rc.add_argument("--anonymize", action="store_true")
    rc.add_argument("--append", action="store_true", help="keep existing alert logs (default: start fresh)")
    rc.set_defaults(fn=cmd_replay_csv)

    c = sub.add_parser("check", help="verify the environment")
    c.set_defaults(fn=cmd_check)

    try:                                   # `wids ... | head` must not end in a traceback
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass
    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, NoWifiFramesError, CaptureToolError, ValueError, RuntimeError) as ex:
        sys.exit(f"error: {ex}")


if __name__ == "__main__":
    main()
