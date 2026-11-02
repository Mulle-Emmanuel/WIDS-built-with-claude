from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .config import Config


def build_model(seq_len: int, n_features: int, cfg: Config):
    import keras
    from keras import layers

    m = keras.Sequential([
        layers.Input((seq_len, n_features)),
        layers.LSTM(cfg.lstm_units),
        layers.Dropout(cfg.dropout),
        layers.Dense(cfg.dense_units, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="wids_lstm")
    m.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate), loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc"), keras.metrics.Recall(name="recall"),
                       keras.metrics.Precision(name="precision")])
    return m
