"""Train once, save artefacts. Also usable to re-evaluate the saved model on the same test split."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import __version__
from .config import Config
from .evaluate import (available_rules, choose_threshold, compute_metrics, plot_confusion, plot_history,
                       plot_roc, rule_score, save_json)
from .features import FEATURE_NAMES, model_features
from .sequences import build_splits


def load_windows(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"window file not found: {p} (run `wids extract` first)")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    if df.duplicated(["capture", "window_idx"]).any():
        raise ValueError("duplicate windows inside one capture id (same file listed twice?)")
    if "ev_n_frames" in df.columns:                  # the same capture extracted twice under two ids would leak
        import hashlib
        sig = {}
        for cap, g in df.groupby("capture"):
            g = g.sort_values("window_idx")
            h = hashlib.md5(g[["window_idx", "ev_n_frames"]].to_numpy().astype("int64").tobytes()).hexdigest()
            sig.setdefault(h, []).append(cap)
        dups = [v for v in sig.values() if len(v) > 1]
        if dups:
            raise ValueError(f"captures with identical windows (same pcap extracted twice?): {dups}. "
                             f"Remove one of each pair; duplicates would leak data between splits.")
    missing = [c for c in FEATURE_NAMES + ["capture", "window_idx", "label"] if c not in df.columns]
    if missing:
        raise ValueError(f"window files are missing columns: {missing}")
    return df


def class_distribution(df: pd.DataFrame, splits: dict) -> dict:
    out = {"windows_total": int(len(df)),
           "windows_by_capture": {c: {"n": int(len(g)), "malicious": int(g["label"].sum())}
                                  for c, g in df.groupby("capture")},
           "sequences": {}}
    for s, d in splits.items():
        n1 = int(d["y"].sum())
        out["sequences"][s] = {"n": int(len(d["y"])), "malicious": n1, "normal": int(len(d["y"]) - n1)}
    return out


def _predict(model, scaler, X, bs=1024):
    n, t, f = X.shape
    Xs = scaler.transform(X.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
    return model.predict(Xs, batch_size=bs, verbose=0).ravel(), Xs


def run_training(window_files: List[str], models_dir: str, results_dir: str, cfg: Config) -> dict:
    import keras
    from .model import build_model

    keras.utils.set_random_seed(cfg.seed)
    models_dir, results_dir = Path(models_dir), Path(results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_windows(window_files)
    nz = df[df["window_idx"] > 0]
    ws = float((nz["t_start"] / nz["window_idx"]).median())
    if abs(ws - cfg.window_s) > 1e-6:
        raise ValueError(f"window files were extracted with window_s={ws}, config says {cfg.window_s}")
    cols = model_features(cfg.exclude_features)
    splits = build_splits(df, cols, cfg)
    dist = class_distribution(df, splits)
    print(json.dumps(dist["sequences"], indent=2))

    Xtr, ytr = splits["train"]["X"], splits["train"]["y"]
    n, t, f = Xtr.shape
    scaler = StandardScaler().fit(Xtr.reshape(-1, f))          # fitted on training data only

    def scale(X):
        return scaler.transform(X.reshape(-1, f)).reshape(len(X), t, f).astype(np.float32)

    Xtr_s, Xva_s = scale(Xtr), scale(splits["val"]["X"])
    yva = splits["val"]["y"]
    n1 = max(int(ytr.sum()), 1)
    cw = {0: len(ytr) / (2 * max(len(ytr) - n1, 1)), 1: len(ytr) / (2 * n1)}

    model = build_model(t, f, cfg)
    hist = model.fit(Xtr_s, ytr, validation_data=(Xva_s, yva), epochs=cfg.epochs, batch_size=cfg.batch_size,
                     class_weight=cw, verbose=2,
                     callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                                              restore_best_weights=True)])

    pv = model.predict(Xva_s, verbose=0).ravel()
    thr = choose_threshold(yva, pv, cfg.target_fpr)           # chosen on validation, never on test
    pt, _ = _predict(model, scaler, splits["test"]["X"])
    yte = splits["test"]["y"]
    test = compute_metrics(yte, pt, thr)

    baselines = {}                                   # comparison only; thresholds stored for `wids transfer`
    for name in available_rules(cols):
        t = choose_threshold(yva, rule_score(name, splits["val"]["X"], cols), cfg.target_fpr)
        baselines[name] = {"threshold": t,
                           "test": compute_metrics(yte, rule_score(name, splits["test"]["X"], cols), t)}

    model.save(models_dir / "wids_lstm.keras")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    meta = {"version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
            "all_features": FEATURE_NAMES, "feature_cols": cols, "window_s": cfg.window_s,
            "seq_len": cfg.seq_len, "threshold": thr, "config": cfg.to_dict(),
            "train_captures": sorted(df["capture"].unique().tolist()),
            "baselines": {n: {"threshold": b["threshold"]} for n, b in baselines.items()},
            "python": platform.python_version(), "keras": keras.__version__}
    save_json(meta, models_dir / "meta.json")

    metrics = {"test_lstm": test, "baseline_comparison_only": baselines, "class_distribution": dist,
               "epochs_run": len(hist.history["loss"]), "seed": cfg.seed,
               "note": "Measured on held-out blocks of the SAME captures used for training. "
                       "This is not evidence of generalisation to other environments."}
    save_json(metrics, results_dir / "metrics.json")
    plot_confusion(test["confusion_matrix"], results_dir / "confusion_matrix.png")
    plot_history(hist.history, results_dir / "training_curves.png")
    if test["roc_auc"] is not None:
        plot_roc(yte, pt, test["roc_auc"], results_dir / "roc_curve.png")
    pd.DataFrame({"capture": splits["test"]["meta"]["capture"], "window_idx": splits["test"]["meta"]["window_idx"],
                  "y_true": yte, "prob": pt}).to_csv(results_dir / "test_predictions.csv", index=False)
    print(json.dumps({k: test[k] for k in ("accuracy", "precision", "recall", "f1", "false_positive_rate")}, indent=2))
    return metrics


def run_evaluation(window_files: List[str], models_dir: str, results_dir: str) -> dict:
    """Re-evaluate the SAVED model on the test split (rebuilt deterministically). No retraining."""
    import keras
    from .detector import load_artifacts

    model, scaler, meta = load_artifacts(models_dir)
    cfg = Config.from_dict(meta["config"])
    df = load_windows(window_files)
    splits = build_splits(df, meta["feature_cols"], cfg)
    pt, _ = _predict(model, scaler, splits["test"]["X"])
    yte = splits["test"]["y"]
    res = compute_metrics(yte, pt, meta["threshold"])
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    save_json(res, Path(results_dir) / "metrics_reeval.json")
    plot_confusion(res["confusion_matrix"], Path(results_dir) / "confusion_matrix_reeval.png")
    print(json.dumps(res, indent=2))
    return res
