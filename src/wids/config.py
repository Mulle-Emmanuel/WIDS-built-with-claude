"""Central configuration. Every value here is saved into models/meta.json at training time,
and the detector reads it back, so training and inference cannot silently disagree."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # --- windowing / sequences ---
    window_s: float = 1.0          # length of one observation window (seconds)
    seq_len: int = 10              # windows per LSTM input sequence (10 s of context)
    max_gap_windows: int = 120     # larger time gaps in a capture start a new segment
    # --- labelling ---
    min_attack_frames: int = 3     # attack frames needed in a window to label it malicious
    # --- splitting (block-based, no window shared between splits) ---
    block_windows: int = 120       # windows per block (2 min at 1 s)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # --- model ---
    lstm_units: int = 64
    dense_units: int = 32
    dropout: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 40
    batch_size: int = 64
    patience: int = 6
    seed: int = 42
    # --- operating point ---
    target_fpr: float = 0.02       # threshold = max recall with validation FPR <= this
    consecutive: int = 1           # consecutive malicious windows required to raise an alert
    # RSSI depends on antenna position, so it is excluded by default to avoid learning
    # capture-location artefacts. Remove from this list to run the ablation.
    exclude_features: list = field(default_factory=lambda: ["rssi_mean", "rssi_std", "rssi_available"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
