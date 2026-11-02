# LSTM-based Wireless Intrusion Detection System (WIDS)

Detects deauthentication-flood activity in 802.11 traffic. An LSTM reads a 10-second sequence of
1-second traffic windows and outputs P(malicious). Train once, save, then load for replay or live use.

**Status:** code and tests complete on synthetic traffic only. No real dataset has been processed yet,
so **no performance results exist**. `results/` is empty until you run the steps below.

## Setup (Fedora)
```bash
sudo dnf install wireshark-cli iw python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
wids check            # every line should say ok
pytest -q             # takes ~3 min; the tshark-parity test only runs if tshark is installed
```

## 1. Get data (see data/README.md)
AWID3 needs registration. Do not commit pcaps (they contain third-party MACs).

## 2. Extract windows (one command per pcap)
```bash
# attack capture: label ground truth with the per-attack Wireshark filter published on the AWID3 page
wids extract --pcap data/raw/Deauth.pcap --capture-id deauth --attack-filter '<AWID3 Deauth filter>' \
     --out data/windows/deauth.csv --anonymize
wids extract --pcap data/raw/Normal.pcap --capture-id normal --out data/windows/normal.csv --anonymize
```
Check the printed "malicious" count. If an attack filter was given and it is 0, the filter is wrong.
Label rule: a window is malicious if it holds >= 3 ground-truth attack frames. Labels never come from
deauth counts, so the model cannot just learn a threshold.

### 2b. Or extract from the official AWID3-CSV edition
```bash
wids extract-csv --csv-dir ~/Downloads/archive/CSV/1.Deauth --pattern 'Deauth_*.csv' \
     --capture-id deauth --normal-label Normal --out data/windows/deauth.csv --anonymize
```
Chunks are ordered by their number (Deauth_2 before Deauth_10). Ground truth is the dataset's own `Label`
column. The command prints the label values it saw and stops if `--normal-label` never appears.
CSV data cannot be replayed through the live parser, so the replay demo still needs a pcap.

### 2c. Headline evaluation: blocked cross-validation (use this for reported numbers)
```bash
wids cv --windows data/windows/deauth.csv --results-dir results --folds 5
```
Whole 30-window blocks are tested once by a model that never saw them or any window within `seq_len`
windows of them. Output: `results/cv_metrics.json` (per-fold + pooled LSTM, and three comparison-only
rule baselines: deauth rate, management-frame rate and total frame rate), `results/cv_confusion_matrix.png`, `results/cv_predictions.csv`.
Every window of a test block is a prediction target (the first 9 borrow context from the purge zone, which no
training sequence ever contained), and the detection delay after each attack onset is reported.

## 3. Train (once) and evaluate
`wids train` builds the model that `replay`/`live` load. Its own single-split test numbers are NOT
the headline when attack data is one episode: report the cross-validation instead.
```bash
wids train --windows data/windows/*.csv --models-dir models --results-dir results
wids evaluate --windows data/windows/*.csv      # re-scores the saved model, no retraining
```
Outputs: `models/{wids_lstm.keras,scaler.joblib,meta.json}`, `results/metrics.json`,
`confusion_matrix.png`, `training_curves.png`, `roc_curve.png`, `test_predictions.csv`.
The decision threshold is picked on the validation set (max recall at <= 2% FPR), never on test.
`metrics.json` also holds a one-rule deauth-rate baseline, for comparison only; it never runs in the detector.

### 3b. False-positive check on attack-free windows from other captures
```bash
wids extract-csv --csv-dir '<CSV>/2.Disas' --pattern 'Disas_*.csv' --capture-id disas --out data/windows/disas.csv --anonymize
wids fpcheck --windows data/windows/disas.csv --models-dir models
```
Only windows before the first attack frame of each capture are scored. Measures false positives, not recall.

### 4a. Demo without a pcap: replay AWID3-CSV rows
```bash
wids replay-csv --csv-dir <CSV>/1.Deauth --pattern 'Deauth_2[0-3].csv' --models-dir models --anonymize
```
Chunks 20-23 cover the end of the normal period and the start of the attack. Mode is shown as REPLAY-CSV.

### 3c. Transfer to a different attack (not part of the headline result)
```bash
wids transfer --windows data/windows/disas.csv --models-dir models
```
Scores every window of other attacks' captures with the deauth-trained model, next to the three comparison-only rules
(thresholds fixed on the training capture). Retrain with this version first: the thresholds live in `models/meta.json`.

## 4. Detect
```bash
wids replay --pcap demo/attack.pcap                 # prints REPLAY banner: recorded traffic
sudo .venv/bin/wids live --iface wlo1 --duration 60 # LIVE banner; interface must be in monitor mode
```
Alerts go to the console, `results/alerts.jsonl` and `results/alerts.db` (SQLite: `alerts`, `predictions`).
Monitor mode:
```bash
nmcli device set wlo1 managed no && sudo ip link set wlo1 down
sudo iw dev wlo1 set type monitor && sudo ip link set wlo1 up && sudo iw dev wlo1 set channel 11
```

## Design notes
* Split: whole 2-minute blocks go to train/val/test (stratified by attack presence); sequences are built
  inside a block, so no window or overlapping sequence crosses splits.
* RSSI features are excluded by default (position artefact). `--include-rssi` runs the ablation.
* Live and replay use the same Python parser; extraction may use tshark. `tests/test_tshark_parse.py`
  compares them when tshark is installed.
* Source MACs are evidence, not identity (spoofable).

## Known limitations
* Test data comes from the same captures as training (different blocks): in-distribution only.
* AWID3 attacks ran with PMF enabled: the detector sees attack attempts, not necessarily successful disconnects.
* Live throughput on a busy channel is untested. The Python parser must keep up with the frame rate.
* Binary classifier: it cannot name the attack type. "likely deauthentication flood" is descriptive evidence only.
