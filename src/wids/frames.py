"""Frame records and the two ways of producing them.

* Python parser (`parse_dot11_bytes`): used for live capture and PCAP replay, so both go
  through the identical code path the model was deployed with.
* TShark (`iter_tshark`): used only for bulk extraction of large datasets (fast).
  A consistency test compares both parsers on the same capture.
"""
from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

BROADCAST = "ff:ff:ff:ff:ff:ff"
DLT_IEEE802_11 = 105
DLT_IEEE802_11_RADIO = 127


class NoWifiFramesError(RuntimeError):
    """The capture contains no 802.11 frames (e.g. interface was not in monitor mode)."""


class CaptureToolError(RuntimeError):
    """A required external tool (tshark) is missing or failed."""


@dataclass(slots=True)
class Frame:
    ts: float
    ftype: int                 # 0 mgmt, 1 control, 2 data, 3 extension
    subtype: int
    src: Optional[str]         # transmitter address (addr2); may be spoofed
    dst: Optional[str]         # receiver address (addr1)
    retry: bool
    rssi: Optional[float]      # dBm, None when the driver does not report it
    length: int
    is_attack: bool = False    # ground-truth label (training data only)


def _mac(b: bytes) -> str:
    return ":".join(f"{x:02x}" for x in b)


def anonymize_mac(mac: Optional[str], salt: str = "wids") -> Optional[str]:
    """Stable pseudonym for a MAC. Broadcast is kept so broadcast-deauth stays visible."""
    if mac is None or mac == BROADCAST:
        return mac
    return "anon-" + hashlib.sha256((salt + mac).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- radiotap + 802.11 parser
def _radiotap_rssi(raw: bytes) -> tuple:
    """Return (radiotap_length, dBm antenna signal or None). Handles extended presence words."""
    if len(raw) < 8:
        raise ValueError("short radiotap header")
    rt_len = struct.unpack_from("<H", raw, 2)[0]
    present = struct.unpack_from("<I", raw, 4)[0]
    off, word = 8, present
    while word & (1 << 31):                      # skip extended presence bitmaps
        if off + 4 > len(raw):
            return rt_len, None
        word = struct.unpack_from("<I", raw, off)[0]
        off += 4
    if not present & (1 << 5):
        return rt_len, None
    for bit, size, align in ((0, 8, 8), (1, 1, 1), (2, 1, 1), (3, 4, 2), (4, 2, 2)):
        if present & (1 << bit):
            off = (off + align - 1) & ~(align - 1)
            off += size
    if off >= rt_len or off >= len(raw):
        return rt_len, None
    return rt_len, float(struct.unpack_from("<b", raw, off)[0])


def parse_dot11_bytes(raw: bytes, ts: float, linktype: int = DLT_IEEE802_11_RADIO) -> Optional[Frame]:
    """Parse one captured frame. Returns None for frames too short to read.
    Malformed packets never raise: a bad packet must not stop a live monitor."""
    if linktype not in (DLT_IEEE802_11, DLT_IEEE802_11_RADIO):
        raise NoWifiFramesError(f"link type {linktype} is not 802.11")
    try:
        rssi = None
        if linktype == DLT_IEEE802_11_RADIO:
            rt_len, rssi = _radiotap_rssi(raw)
            body = raw[rt_len:]
        else:
            body = raw
        if len(body) < 10:
            return None
        fc0, fc1 = body[0], body[1]
        if fc0 & 0x03 != 0:                       # unknown protocol version
            return None
        ftype, subtype = (fc0 >> 2) & 0x3, (fc0 >> 4) & 0xF
        dst = _mac(body[4:10])
        src = _mac(body[10:16]) if len(body) >= 16 else None
        return Frame(ts, ftype, subtype, src, dst, bool(fc1 & 0x08), rssi, len(raw))
    except Exception:
        return None


def iter_pcap(path: str, attack_macs: Iterable[str] = (), anonymize: bool = False) -> Iterator[Frame]:
    """Stream frames from a pcap/pcapng via the Python parser (replay path)."""
    from scapy.layers.dot11 import Dot11, RadioTap
    from scapy.utils import PcapReader

    attack = {m.lower() for m in attack_macs}
    seen_wifi = total = 0
    try:
        reader = PcapReader(path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise CaptureToolError(f"cannot read capture {path}: {e}") from e
    with reader:
        for pkt in reader:
            total += 1
            if isinstance(pkt, RadioTap):
                lt = DLT_IEEE802_11_RADIO
            elif isinstance(pkt, Dot11):
                lt = DLT_IEEE802_11
            else:
                continue
            f = parse_dot11_bytes(bytes(pkt.original), float(pkt.time), lt)
            if f is None:
                continue
            seen_wifi += 1
            f.is_attack = f.src in attack if attack else False
            if anonymize:
                f.src, f.dst = anonymize_mac(f.src), anonymize_mac(f.dst)
            yield f
    if total and not seen_wifi:
        raise NoWifiFramesError(
            f"{path}: {total} packets but no 802.11 frames. Was the interface in monitor mode?")


# ------------------------------------------------------------------------- TShark front-end
TSHARK_FIELDS = ["frame.number", "frame.time_epoch", "wlan.fc.type", "wlan.fc.subtype",
                 "wlan.ta", "wlan.ra", "wlan.fc.retry", "wlan_radio.signal_dbm",
                 "radiotap.dbm_antsignal", "frame.len"]


def _first(s: str) -> str:
    return s.split(",")[0].strip() if s else ""


def parse_tshark_line(line: str, attack_numbers: Optional[set] = None,
                      attack_macs: frozenset = frozenset(), anonymize: bool = False) -> Optional[Frame]:
    parts = line.rstrip("\n").split("\t")
    parts += [""] * (len(TSHARK_FIELDS) - len(parts))
    num, ts, ftype, subtype, ta, ra, retry, sig1, sig2, length = (_first(p) for p in parts[:10])
    if ftype == "" or subtype == "" or ts == "":
        return None
    try:
        sig = sig1 or sig2
        src, dst = (ta.lower() or None), (ra.lower() or None)
        attack = bool((attack_numbers and int(num) in attack_numbers) or (src in attack_macs))
        if anonymize:
            src, dst = anonymize_mac(src), anonymize_mac(dst)
        return Frame(float(ts), int(ftype), int(subtype), src, dst,
                     retry.lower() in ("1", "true"), float(sig) if sig else None,
                     int(length or 0), attack)
    except ValueError:
        return None


def iter_tshark(path: str, attack_filter: Optional[str] = None, attack_macs: Iterable[str] = (),
                anonymize: bool = False) -> Iterator[Frame]:
    """Stream frames from a pcap with tshark. `attack_filter` is a Wireshark display filter that
    selects ground-truth attack frames (e.g. the per-attack filter published with AWID3)."""
    tshark = shutil.which("tshark")
    if tshark is None:
        raise CaptureToolError("tshark not found. Install it: sudo dnf install wireshark-cli")
    attack_numbers = None
    if attack_filter:
        r = subprocess.run([tshark, "-r", path, "-Y", attack_filter, "-T", "fields", "-e", "frame.number"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise CaptureToolError(f"tshark attack filter failed: {r.stderr.strip()[:300]}")
        attack_numbers = {int(x) for x in r.stdout.split() if x.isdigit()}
    cmd = [tshark, "-r", path, "-T", "fields", "-E", "separator=/t"]
    for f in TSHARK_FIELDS:
        cmd += ["-e", f]
    macs = frozenset(m.lower() for m in attack_macs)
    n_lines = n_frames = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
        for line in p.stdout:
            n_lines += 1
            fr = parse_tshark_line(line, attack_numbers, macs, anonymize)
            if fr is not None:
                n_frames += 1
                yield fr
        err = p.stderr.read()
    if p.returncode not in (0, None):
        raise CaptureToolError(f"tshark failed on {path}: {err.strip()[:300]}")
    if n_lines and not n_frames:
        raise NoWifiFramesError(f"{path}: no 802.11 frames decoded. Was the interface in monitor mode?")
