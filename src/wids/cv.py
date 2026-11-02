"""Blocked cross-validation with a purge gap, and a false-positive check on unseen normal traffic.

Why: the AWID3 Deauth CSV holds ONE attack episode. A single train/val/test split would put one or two
attack blocks in the test set. Here every block is tested once by a model that never saw it (nor any
window within `purge` windows of it), and results are pooled over all folds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config
from .evaluate import (available_rules, choose_threshold, compute_metrics, metrics_from_pred, plot_confusion,
                       rule_score, save_json)
from .features import model_features
from .sequences import (SplitError, _runs, assign_folds, block_sequences, make_sequences, purge_mask,
                        sequences_from_table)
from .train import load_windows


def fit_model(Xtr, ytr, Xva, yva, cfg: Config):
    import keras
    from keras.callbacks import EarlyStopping
    from .model import build_model

    keras.utils.set_random_seed(cfg.seed)
    n, t, f = Xtr.shape
    scaler = StandardScaler().fit(Xtr.reshape(-1, f))

    def sc(X):
        return scaler.transform(X.reshape(-1, f)).reshape(len(X), t, f).astype(np.float32)

    n1 = max(int(ytr.sum()), 1)
    cw = {0: n / (2 * max(n - n1, 1)), 1: n / (2 * n1)}
    model = build_model(t, f, cfg)
    model.fit(sc(Xtr), ytr, validation_data=(sc(Xva), yva), epochs=cfg.epochs, batch_size=cfg.batch_size,
              class_weight=cw, verbose=0, callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                                                    restore_best_weights=True)])
    thr = choose_threshold(yva, model.predict(sc(Xva), verbose=0).ravel(), cfg.target_fpr)
    return model, scaler, thr, sc


def run_cv(window_files: List[str], results_dir: str, cfg: Config, k: int = 5, purge: int | None = None) -> dict:
    purge = cfg.seq_len if purge is None else max(purge, cfg.seq_len - 1)   # must cover the context windows
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df = load_windows(window_files)
    cols = model_features(cfg.exclude_features)
    blocks = block_sequences(df, cols, cfg)
    flags = {b: d["has_attack"] for b, d in blocks.items()}
    n_att, n_nor = sum(flags.values()), len(flags) - sum(flags.values())
    print(f"{len(blocks)} blocks of {cfg.block_windows} windows: {n_att} with attack, {n_nor} without")
    if n_att < max(k, 3) or n_nor < max(k, 3):
        raise SplitError(f"need at least {max(k, 3)} attack and normal blocks, have {n_att}/{n_nor}; "
                         f"lower --block-windows or --folds")
    folds = assign_folds(flags, k, cfg.seed)
    tab = {cap: g.sort_values("window_idx").reset_index(drop=True) for cap, g in df.groupby("capture")}

    def test_block(key):
        """Every window of the block is a prediction target, including its first seq_len-1 windows, whose
        sequences borrow context from the preceding purge zone (never seen in training)."""
        cap, lo, hi = key[0], blocks[key]["lo"], blocks[key]["hi"]
        g = tab[cap]
        sub = g[(g.window_idx >= lo - cfg.seq_len + 1) & (g.window_idx <= hi)]
        feats, labels, widx = sub[cols].to_numpy(np.float32), sub["label"].to_numpy(np.int8), sub["window_idx"].to_numpy()
        Xs, ys, ls = [], [], []
        for a, e in _runs(widx):
            X_, y_, last = make_sequences(feats[a:e], labels[a:e], cfg.seq_len)
            if len(y_):
                lw = widx[a:e][last]
                m = lw >= lo
                Xs.append(X_[m]), ys.append(y_[m]), ls.append(lw[m])
        if not Xs:
            return None
        lab = dict(zip(g["window_idx"].tolist(), g["label"].tolist()))
        onsets = [w for w in range(lo, hi + 1) if lab.get(w) == 1 and lab.get(w - 1) == 0]
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ls), onsets
    rng = np.random.default_rng(cfg.seed)
    rule_names = available_rules(cols)
    pred_rows = []
    rows, pool = [], {m: {"y": [], "pred": [], "prob": []} for m in ["lstm", *rule_names]}

    for fold in range(k):
        test_keys = sorted(b for b in blocks if folds[b] == fold)
        rest = [b for b in blocks if folds[b] != fold]
        ranges = {}
        for b in test_keys:
            ranges.setdefault(b[0], []).append((blocks[b]["lo"], blocks[b]["hi"]))
        att, nor = sorted(b for b in rest if flags[b]), sorted(b for b in rest if not flags[b])
        rng.shuffle(att), rng.shuffle(nor)
        val_keys = set(att[:max(1, round(0.2 * len(att)))] + nor[:max(1, round(0.2 * len(nor)))])

        def gather(keys):
            Xs, ys = [], []
            for b in keys:
                d = blocks[b]
                m = purge_mask(d["last"], cfg.seq_len, ranges.get(b[0], []), purge)
                if m.any():
                    Xs.append(d["X"][m]), ys.append(d["y"][m])
            return np.concatenate(Xs), np.concatenate(ys)

        def train_sequences(keys):
            """Sequences over contiguous runs of training-block windows (adjacent train blocks merge, so attack
            onsets are training targets), excluding the purge zone around every test block."""
            Xs, ys = [], []
            for cap, g in tab.items():
                ids = [b[1] for b in keys if b[0] == cap]
                if not ids:
                    continue
                w = g["window_idx"].to_numpy()
                bid = (w - w.min()) // cfg.block_windows
                allowed = np.isin(bid, ids)
                for lo_t, hi_t in ranges.get(cap, []):
                    allowed &= ~((w >= lo_t - purge) & (w <= hi_t + purge))
                sub = g[allowed]
                feats = sub[cols].to_numpy(np.float32)
                labels = sub["label"].to_numpy(np.int8)
                widx = sub["window_idx"].to_numpy()
                for a_, e_ in _runs(widx):
                    X_, y_, _ = make_sequences(feats[a_:e_], labels[a_:e_], cfg.seq_len)
                    if len(y_):
                        Xs.append(X_), ys.append(y_)
            return np.concatenate(Xs), np.concatenate(ys)

        Xtr, ytr = train_sequences([b for b in rest if b not in val_keys])
        Xva, yva = gather(sorted(val_keys))
        tb = {b: test_block(b) for b in test_keys}
        tb = {b: v for b, v in tb.items() if v is not None}
        Xte = np.concatenate([v[0] for v in tb.values()])
        yte = np.concatenate([v[1] for v in tb.values()])
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            raise SplitError(f"fold {fold}: train/val lost a class after purging; use fewer folds or smaller purge")
        model, scaler, thr, sc = fit_model(Xtr, ytr, Xva, yva, cfg)
        p = model.predict(sc(Xte), verbose=0).ravel()
        m = compute_metrics(yte, p, thr)
        pred_rows.append(pd.DataFrame({"fold": fold, "y_true": yte, "prob": p}))
        pool["lstm"]["y"].append(yte), pool["lstm"]["prob"].append(p), pool["lstm"]["pred"].append((p >= thr).astype(int))
        onsets, off = [], 0
        for b, (Xb, yb, lb, ons) in tb.items():
            flagged = dict(zip(lb.tolist(), (p[off:off + len(yb)] >= thr).tolist()))
            off += len(yb)
            for w in ons:
                hit = [x for x in range(w, lb.max() + 1) if flagged.get(x)]
                onsets.append({"block": list(b), "onset_window_idx": int(w),
                               "detection_delay_windows": (hit[0] - w) if hit else None})
        rec = {"fold": fold, "onsets": onsets, "test_blocks": [list(b) for b in test_keys], "n_train": int(len(ytr)), "n_val": int(len(yva)),
               "n_test": int(len(yte)), "test_attack_sequences": int(yte.sum()), "lstm": m, "baselines": {}}
        for name in rule_names:
            sv, st = rule_score(name, Xva, cols), rule_score(name, Xte, cols)
            t_ = choose_threshold(yva, sv, cfg.target_fpr)
            rec["baselines"][name] = compute_metrics(yte, st, t_)
            pool[name]["y"].append(yte), pool[name]["prob"].append(st), pool[name]["pred"].append((st >= t_).astype(int))
        rows.append(rec)
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        rl = {k_: round(v_["threshold"], 2) for k_, v_ in rec["baselines"].items()}
        print(f"fold {fold}: test seqs={len(yte)} (attack {int(yte.sum())})  recall={m['recall']:.3f}  "
              f"FPR={m['false_positive_rate']:.3f}  AUC={auc}  thr={thr:.4g}\n         "
              f"onset delay(windows)={[o['detection_delay_windows'] for o in onsets] or '-'}  "
              f"max normal prob={p[yte == 0].max():.4g}  min attack prob={p[yte == 1].min():.4g}  rule thresholds={rl}")

    pooled = {name: metrics_from_pred(np.concatenate(d["y"]), np.concatenate(d["pred"]), np.concatenate(d["prob"]))
              for name, d in pool.items() if d["y"]}
    rec = [r["lstm"]["recall"] for r in rows]
    fpr = [r["lstm"]["false_positive_rate"] for r in rows]
    out = {"k": k, "purge_windows": purge, "block_windows": cfg.block_windows, "config": cfg.to_dict(),
           "pooled": pooled, "folds": rows,
           "onset_delays_windows": [o["detection_delay_windows"] for r in rows for o in r["onsets"]],
           "lstm_fold_spread": {"recall_min": min(rec), "recall_max": max(rec), "fpr_min": min(fpr), "fpr_max": max(fpr)},
           "note": "All attack data comes from ONE attack episode in ONE capture session. Folds are blocks of that "
                   "episode, so this measures within-session performance, not generalisation to other networks "
                   "or attacks. Pooled AUC mixes five models."}
    save_json(out, Path(results_dir) / "cv_metrics.json")
    pd.concat(pred_rows, ignore_index=True).to_csv(Path(results_dir) / "cv_predictions.csv", index=False)
    plot_confusion(pooled["lstm"]["confusion_matrix"], Path(results_dir) / "cv_confusion_matrix.png",
                   f"LSTM, pooled over {k} blocked folds")
    print(json.dumps({n: {x: round(v[x], 4) for x in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "roc_auc")
                          if v[x] is not None}
                      for n, v in pooled.items()}, indent=2))
    return out


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    r = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return float(min(1.0, (c + r) / d))


def _score(model, scaler, X):
    n, t, f = X.shape
    return model.predict(scaler.transform(X.reshape(-1, f)).reshape(n, t, f).astype(np.float32), verbose=0).ravel()


def _runs_of_flags(meta: pd.DataFrame, flag: np.ndarray):
    """Contiguous flagged stretches (one alert 'event' each): same capture, consecutive window ids."""
    runs, cur = [], None
    for cap, w, f in zip(meta["capture"], meta["window_idx"], flag):
        if f and cur and cur["capture"] == cap and w == cur["end"] + 1:
            cur["end"], cur["len"] = int(w), cur["len"] + 1
        elif f:
            if cur:
                runs.append(cur)
            cur = {"capture": cap, "start": int(w), "end": int(w), "len": 1}
        else:
            if cur:
                runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    return runs


def run_fpcheck(window_files: List[str], models_dir: str, results_dir: str) -> dict:
    """Score windows that contain NO labelled attack (the pre-attack part of each capture) with the saved model."""
    from .detector import load_artifacts
    model, scaler, meta = load_artifacts(models_dir)
    cfg = Config.from_dict(meta["config"])
    df = load_windows(window_files)
    parts = []
    for cap, g in df.groupby("capture"):
        m = g[g["label"] == 1]
        parts.append(g[g["window_idx"] < m["window_idx"].min()] if len(m) else g)
    clean = pd.concat(parts)
    X, y, mt = sequences_from_table(clean, meta["feature_cols"], cfg.seq_len)
    assert y.sum() == 0
    n = len(y)
    p = _score(model, scaler, X)
    flag = p >= meta["threshold"]
    k = int(flag.sum())
    runs = _runs_of_flags(mt, flag)
    n_eff = max(1, n // cfg.seq_len)                       # overlapping sequences share 9 of 10 windows
    per = {}
    for c in sorted(mt["capture"].unique()):
        mk = (mt["capture"] == c).to_numpy()
        per[c] = {"sequences": int(mk.sum()), "flagged": int(flag[mk].sum()),
                  "flagged_events": sum(1 for r in runs if r["capture"] == c)}
    res = {"captures": per, "normal_sequences": int(n), "approx_independent_segments": int(n_eff),
           "flagged": k, "flagged_events": len(runs), "false_positive_rate": float(flag.mean()),
           "fpr_upper95_overlapping_sequences": _wilson_upper(k, n),
           "fpr_upper95_independent_segments": _wilson_upper(int(np.ceil(k / cfg.seq_len)), n_eff),
           "longest_flagged_run_windows": max([r["len"] for r in runs], default=0),
           "normal_traffic_minutes": float(n * cfg.window_s / 60),
           "false_alert_events_per_hour": float(len(runs) / (n * cfg.window_s / 3600)),
           "threshold": meta["threshold"],
           "note": "Windows before each capture's first labelled attack frame. Labels only cover frames matched by "
                   "the AWID3 filters, so unlabelled attack-related frames may still be present. Same lab as training."}
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    save_json(res, Path(results_dir) / "fpcheck.json")
    ev = clean[["capture", "window_idx"] + [c for c in ("ev_n_frames", "ev_n_deauth", "ev_n_disassoc") if c in clean.columns]]
    fl = mt.assign(prob=p)[flag].merge(ev, on=["capture", "window_idx"], how="left")
    fl.to_csv(Path(results_dir) / "fpcheck_flagged.csv", index=False)
    print(json.dumps(res, indent=2))
    print(f"flagged windows with evidence -> {Path(results_dir) / 'fpcheck_flagged.csv'}")
    return res


def run_transfer(window_files: List[str], models_dir: str, results_dir: str) -> dict:
    """Score ALL windows of captures of DIFFERENT attacks with the saved model (trained on deauth only), next to
    the comparison-only rules whose thresholds were fixed on the training capture's validation set."""
    from .detector import load_artifacts
    model, scaler, meta = load_artifacts(models_dir)
    cfg = Config.from_dict(meta["config"])
    cols = meta["feature_cols"]
    df = load_windows(window_files)
    X, y, mt = sequences_from_table(df, cols, cfg.seq_len)
    p = _score(model, scaler, X)
    rules = {n: (b["threshold"], rule_score(n, X, cols)) for n, b in meta.get("baselines", {}).items()
             if n in available_rules(cols)}
    res = {"trained_on": meta["train_captures"], "scored_captures": sorted(df["capture"].unique().tolist()),
           "overall": compute_metrics(y, p, meta["threshold"]), "per_capture": {}, "rules": {n: {} for n in rules}}
    for c in res["scored_captures"]:
        m = (mt["capture"] == c).to_numpy()
        res["per_capture"][c] = compute_metrics(y[m], p[m], meta["threshold"])
        for n, (t, sc) in rules.items():
            res["rules"][n][c] = compute_metrics(y[m], sc[m], t)
    res["note"] = ("Transfer to attacks the model was NOT trained on. Recall is the share of that attack's windows "
                   "flagged by the deauth-trained model; it is not a claim of general attack detection.")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    save_json(res, Path(results_dir) / "transfer.json")
    names = list(rules)
    print(f"{'capture':22s} {'attack':>6s} | {'LSTM rec':>8s} {'FPR':>6s} {'AUC':>6s} | "
          + " ".join(f"{n.replace('_rule', '')[:11]:>11s}" for n in names) + "   (rule recall @FPR)")
    for c, m in res["per_capture"].items():
        cm = m["confusion_matrix"]
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        rr = " ".join(f"{res['rules'][n][c]['recall']:5.3f}@{res['rules'][n][c]['false_positive_rate']:.2f}" for n in names)
        print(f"{c:22s} {cm['tp'] + cm['fn']:6d} | {m['recall']:8.3f} {m['false_positive_rate']:6.3f} {auc:>6s} | {rr}")
    if not names:
        print("(no rule thresholds in meta.json: retrain with this version to get the rule comparison)")
    return res
