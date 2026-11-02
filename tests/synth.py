"""SYNTHETIC 802.11 traffic generator for TESTS ONLY.

Output of this module must never be used to report model performance: it exists to check that
the code runs and that training/inference features agree, nothing more.
"""
import random
import struct

ATTACKER = "de:ad:be:ef:00:01"
AP1, AP2 = "02:00:00:00:00:a1", "02:00:00:00:00:a2"
STAS = [f"02:00:00:00:01:{i:02x}" for i in range(1, 9)]
BCAST = "ff:ff:ff:ff:ff:ff"


def _b(mac):
    return bytes(int(x, 16) for x in mac.split(":"))


def radiotap(rssi=None):
    if rssi is None:
        return struct.pack("<BBHI", 0, 0, 8, 0)
    return struct.pack("<BBHIb", 0, 0, 9, 1 << 5, rssi)


def mgmt(subtype, dst, src, bssid, retry=False, body=b""):
    fc = bytes([(subtype << 4), 0x08 if retry else 0])
    return fc + b"\x00\x00" + _b(dst) + _b(src) + _b(bssid) + b"\x00\x00" + body


def data(dst, src, bssid, retry=False):
    fc = bytes([0x08, 0x01 | (0x08 if retry else 0)])
    return fc + b"\x00\x00" + _b(dst) + _b(src) + _b(bssid) + b"\x00\x00" + b"\x00" * 40


def rts(dst, src):
    return bytes([0xB4, 0]) + b"\x00\x00" + _b(dst) + _b(src)


def ack(dst):
    return bytes([0xD4, 0]) + b"\x00\x00" + _b(dst)


def normal_frames(t0, seconds, rng, fps=40, rssi=True):
    out = []
    for s in range(seconds):
        for _ in range(rng.randint(fps // 2, fps * 2)):
            t = t0 + s + rng.random()
            sta, ap = rng.choice(STAS), rng.choice([AP1, AP2])
            r = rng.random()
            if r < 0.40:
                f = data(ap, sta, ap, retry=rng.random() < 0.1)
            elif r < 0.55:
                f = mgmt(8, BCAST, ap, ap)
            elif r < 0.70:
                f = rts(ap, sta)
            elif r < 0.80:
                f = ack(sta)
            elif r < 0.90:
                f = mgmt(4, BCAST, sta, BCAST)
            else:
                f = mgmt(13, ap, sta, ap)
            out.append((t, f))
        if rng.random() < 0.03:                       # occasional legitimate deauth
            out.append((t0 + s + rng.random(), mgmt(12, rng.choice(STAS), AP1, AP1, body=b"\x03\x00")))
    return out


def flood_frames(t0, seconds, rng, rate=150):
    out = []
    for s in range(seconds):
        for _ in range(rate):
            out.append((t0 + s + rng.random(), mgmt(12, BCAST, ATTACKER, AP1, body=b"\x07\x00")))
    return out


def write_pcap(path, frames, with_rssi=True, linktype=127):
    rng = random.Random(1)
    frames = sorted(frames, key=lambda x: x[0])
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype))
        for t, f in frames:
            pkt = (radiotap(-rng.randint(40, 70) if with_rssi else None) if linktype == 127 else b"") + f
            fh.write(struct.pack("<IIII", int(t), int((t % 1) * 1e6), len(pkt), len(pkt)) + pkt)


def make_captures(dirpath, normal_s=420, attack_s=360, flood_from=60, seed=7):
    rng = random.Random(seed)
    base = 1_700_000_000.0
    n_path, a_path = f"{dirpath}/normal.pcap", f"{dirpath}/attack.pcap"
    write_pcap(n_path, normal_frames(base, normal_s, rng))
    a = normal_frames(base + 10_000, attack_s, rng)
    a += flood_frames(base + 10_000 + flood_from, attack_s - flood_from - 30, rng)
    write_pcap(a_path, a)
    return n_path, a_path
