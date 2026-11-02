"""Metrics, plots and the comparison-only baseline. All numbers come from this code."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

CLASSES = ["Normal", "Malicious"]


def choose_threshold(y, prob, target_fpr: float) -> float:
    """Max recall subject to FPR <= target_fpr on this (validation) set, with the cut-off placed in the
    MIDDLE of the gap between the highest unflagged score and the lowest flagged score.

    History: v1 took the highest threshold reaching max recall and v2 the lowest threshold within budget.
    On a small validation set (budget of zero false positives) both pinned the cut-off to the weakest
    validation attack score, right at the edge of the gap. The midpoint is the standard max-margin choice.
    `>= threshold` means flagged. Works for probabilities and for raw rule scores (counts)."""
    prob = np.asarray(prob, float)
    fpr, tpr, thr = roc_curve(y, prob, drop_intermediate=False)
    ok = np.where(fpr <= target_fpr)[0]
    i = ok[-1] if len(ok) else 0
    t = thr[i]
    if not np.isfinite(t):                      # roc_curve's first point is +inf: flag nothing
        return float(prob.max()) + 1e-6
    if i + 1 < len(thr):
        t = (t + thr[i + 1]) / 2.0              # strictly inside the gap: same flags on this set
    return float(t)


def metrics_from_pred(y, pred, prob=None, threshold=None) -> dict:
    y, pred = np.asarray(y), np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, prob)) if (prob is not None and len(np.unique(y)) == 2) else None
    return {
        "threshold": None if threshold is None else float(threshold), "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compute_metrics(y, prob, threshold: float) -> dict:
    return metrics_from_pred(y, (np.asarray(prob) >= threshold).astype(int), prob, threshold)


def plot_confusion(cm: dict, path: Path, title="Confusion matrix (test set)"):
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(m, cmap="Blues")
    rows = m.sum(1, keepdims=True).clip(min=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}\n({m[i, j] / rows[i, 0]:.1%} of actual)", ha="center", va="center",
                    fontsize=16, color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASSES, fontsize=15)
    ax.set_yticks([0, 1], CLASSES, fontsize=15, rotation=90, va="center")
    ax.set_xlabel("Predicted class", fontsize=16)
    ax.set_ylabel("Actual class", fontsize=16)
    ax.set_title(title, fontsize=17)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_history(hist: dict, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(hist["loss"], label="train", lw=2)
    ax[0].plot(hist["val_loss"], label="validation", lw=2)
    ax[0].set_title("Loss", fontsize=16)
    ax[1].plot(hist["recall"], label="train", lw=2)
    ax[1].plot(hist["val_recall"], label="validation", lw=2)
    ax[1].set_title("Recall (malicious)", fontsize=16)
    for a in ax:
        a.set_xlabel("Epoch", fontsize=14)
        a.tick_params(labelsize=13)
        a.legend(fontsize=13)
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(y, prob, auc, path: Path):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, lw=2.5, label=f"LSTM (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate", fontsize=15)
    ax.set_ylabel("True positive rate (recall)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=14)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


RULE_FEATURES = {"deauth_rate_rule": ["log_n_deauth"], "mgmt_rate_rule": ["frac_mgmt", "log_n_frames"],
                 "frame_rate_rule": ["log_n_frames"]}


def available_rules(cols) -> list:
    return [r for r, need in RULE_FEATURES.items() if all(f in cols for f in need)]


def rule_score(name: str, X, cols) -> np.ndarray:
    """COMPARISON ONLY. Raw per-second rate in the LATEST window of each (unscaled) sequence.
    A rule flags a sequence when its score >= a threshold tuned on the validation set at the LSTM's FPR target.
    Rules never run inside the detector."""
    g = lambda f: X[:, -1, cols.index(f)]
    if name == "deauth_rate_rule":
        return np.expm1(g("log_n_deauth"))
    if name == "mgmt_rate_rule":
        return g("frac_mgmt") * np.expm1(g("log_n_frames"))
    if name == "frame_rate_rule":
        return np.expm1(g("log_n_frames"))
    raise KeyError(name)


def save_json(obj, path: Path):
    Path(path).write_text(json.dumps(obj, indent=2))
