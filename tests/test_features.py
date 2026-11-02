import math

from wids.features import FEATURE_NAMES, WindowAggregator, frames_to_windows, model_features
from wids.frames import BROADCAST, Frame, anonymize_mac, parse_dot11_bytes
import synth


def F(ts, ftype=0, st=8, src="aa:aa:aa:aa:aa:01", dst=BROADCAST, retry=False, rssi=None, attack=False):
    return Frame(ts, ftype, st, src, dst, retry, rssi, 100, attack)


def test_counts_and_fractions():
    w = frames_to_windows([F(10.1, 0, 12), F(10.2, 0, 12, dst="aa:aa:aa:aa:aa:02"), F(10.3, 1, 11),
                           F(10.4, 2, 0, retry=True), F(10.5, 0, 8)])[0]
    d = dict(zip(FEATURE_NAMES, w.features))
    assert math.isclose(d["log_n_frames"], math.log1p(5))
    assert math.isclose(d["log_n_deauth"], math.log1p(2))
    assert math.isclose(d["deauth_bcast_frac"], 0.5)
    assert math.isclose(d["frac_mgmt"], 3 / 5) and math.isclose(d["frac_ctrl"], 1 / 5)
    assert math.isclose(d["retry_ratio"], 1 / 5)
    assert w.evidence["n_deauth"] == 2


def test_silent_seconds_become_zero_windows():
    ws = frames_to_windows([F(10.5), F(13.5)])
    assert [w.idx for w in ws] == [10, 11, 12, 13]
    assert ws[1].evidence["n_frames"] == 0 and ws[1].features[0] == 0.0


def test_delta_features():
    ws = frames_to_windows([F(0.5)] + [F(1.5, 0, 12)] * 20)
    d = dict(zip(FEATURE_NAMES, ws[1].features))
    assert d["d_log_deauth"] > 2


def test_labels_come_from_ground_truth_not_deauth_count():
    legit = [F(5.1 + i * .01, 0, 12) for i in range(50)]                # 50 deauth frames, NOT attack
    attack = [F(6.1 + i * .01, 0, 12, attack=True) for i in range(5)]   # 5 attack frames
    ws = frames_to_windows(legit + attack)
    assert ws[0].label == 0 and ws[1].label == 1


def test_missing_rssi_flagged():
    d = dict(zip(FEATURE_NAMES, frames_to_windows([F(1.0)])[0].features))
    assert d["rssi_available"] == 0 and d["rssi_mean"] == 0
    d = dict(zip(FEATURE_NAMES, frames_to_windows([F(1.0, rssi=-50.0), F(1.1, rssi=-60.0)])[0].features))
    assert d["rssi_available"] == 1 and math.isclose(d["rssi_mean"], -55)


def test_long_gap_starts_new_segment():
    agg = WindowAggregator(max_gap_windows=10)
    ws = agg.add(F(0.5)) + agg.add(F(1000.5)) + agg.flush()
    assert len(ws) == 2


def test_feature_order_and_exclusion():
    cols = model_features(["rssi_mean", "rssi_std", "rssi_available"])
    assert all(c in FEATURE_NAMES for c in cols) and "rssi_mean" not in cols
    assert cols == [c for c in FEATURE_NAMES if c in cols]


def test_anonymize_keeps_broadcast():
    assert anonymize_mac(BROADCAST) == BROADCAST
    a = anonymize_mac("aa:bb:cc:dd:ee:ff")
    assert a == anonymize_mac("aa:bb:cc:dd:ee:ff") and a.startswith("anon-")


def test_parse_dot11_bytes_roundtrip():
    raw = synth.radiotap(-48) + synth.mgmt(12, BROADCAST, synth.ATTACKER, synth.AP1, retry=True)
    f = parse_dot11_bytes(raw, 1.0)
    assert (f.ftype, f.subtype, f.src, f.dst, f.retry, f.rssi) == (0, 12, synth.ATTACKER, BROADCAST, True, -48.0)
    f = parse_dot11_bytes(synth.radiotap() + synth.ack(synth.AP1), 1.0)       # ACK has no transmitter
    assert f.ftype == 1 and f.src is None
    assert parse_dot11_bytes(b"\x00\x01", 1.0) is None                        # malformed: no crash
