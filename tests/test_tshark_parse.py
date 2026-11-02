import shutil
import subprocess

import pytest

from wids.frames import iter_pcap, iter_tshark, parse_tshark_line
import synth


def test_parse_line_deauth():
    line = "12\t1700000000.5\t0\t12\tde:ad:be:ef:00:01\tff:ff:ff:ff:ff:ff\t0\t-50\t\t60\n"
    f = parse_tshark_line(line, attack_numbers={12})
    assert (f.ftype, f.subtype, f.rssi, f.is_attack) == (0, 12, -50.0, True)


def test_parse_line_no_wlan_fields_is_skipped():
    assert parse_tshark_line("1\t1700000000.5\t\t\t\t\t\t\t\t60\n") is None


def test_parse_line_retry_true_and_antsignal_fallback():
    f = parse_tshark_line("1\t5.0\t2\t0\taa:aa:aa:aa:aa:01\taa:aa:aa:aa:aa:02\ttrue\t\t-61\t80")
    assert f.retry and f.rssi == -61.0


@pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark not installed")
def test_python_and_tshark_parsers_agree(tmp_path):
    n, _ = synth.make_captures(tmp_path, normal_s=20, attack_s=20, flood_from=5)
    a = list(iter_pcap(n))
    b = list(iter_tshark(n))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.ftype, x.subtype, x.src, x.dst, x.retry) == (y.ftype, y.subtype, y.src, y.dst, y.retry)
