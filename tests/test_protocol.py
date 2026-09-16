"""Tests for the public ECG serial protocol parser."""

import sys
import unittest
from pathlib import Path


HOST_DIR = Path(__file__).resolve().parents[1] / "host"
sys.path.insert(0, str(HOST_DIR))

from protocol import (  # noqa: E402
    DAT_ECG_HR,
    DAT_ECG_LEAD,
    DAT_ECG_WAVE,
    MODULE_ECG,
    ProtocolParser,
    unpack_ecg_hr,
    unpack_ecg_lead,
    unpack_ecg_wave,
)


def pack_frame(module_id: int, second_id: int, payload: bytes) -> bytes:
    """Build one frame using the course device's packed-byte convention."""
    if len(payload) > 6:
        raise ValueError("payload must fit in six bytes")

    data = payload.ljust(6, b"\x00")
    data_head = 0
    packed_second_id = second_id & 0x7F
    if second_id & 0x80:
        data_head |= 0x01

    packed_data = []
    for index, value in enumerate(data):
        if value & 0x80:
            data_head |= 1 << (index + 1)
        packed_data.append((value & 0x7F) | 0x80)

    raw = [
        module_id,
        data_head | 0x80,
        packed_second_id | 0x80,
        *packed_data,
    ]
    checksum = (sum(raw) & 0x7F) | 0x80
    return bytes([*raw, checksum])


def parse(raw: bytes):
    parser = ProtocolParser()
    frame = None
    for value in raw:
        frame = parser.feed(value) or frame
    return frame


class ProtocolParserTests(unittest.TestCase):
    def test_wave_value_round_trip(self):
        frame = parse(pack_frame(MODULE_ECG, DAT_ECG_WAVE, b"\x12\xB4"))
        self.assertIsNotNone(frame)
        self.assertEqual(unpack_ecg_wave(frame), 0x12B4)

    def test_lead_status_round_trip(self):
        frame = parse(pack_frame(MODULE_ECG, DAT_ECG_LEAD, b"\x01"))
        self.assertIsNotNone(frame)
        self.assertEqual(unpack_ecg_lead(frame), 1)

    def test_heart_rate_round_trip(self):
        frame = parse(pack_frame(MODULE_ECG, DAT_ECG_HR, b"\x00\x48"))
        self.assertIsNotNone(frame)
        self.assertEqual(unpack_ecg_hr(frame), 72)

    def test_bad_checksum_is_rejected(self):
        raw = bytearray(pack_frame(MODULE_ECG, DAT_ECG_WAVE, b"\x01\x02"))
        raw[-1] ^= 0x01
        self.assertIsNone(parse(bytes(raw)))

    def test_new_module_byte_resynchronizes_parser(self):
        raw = pack_frame(MODULE_ECG, DAT_ECG_HR, b"\x00\x3C")
        frame = parse(bytes([0x01, 0x10]) + raw[1:])
        self.assertIsNotNone(frame)
        self.assertEqual(frame.module_id, MODULE_ECG)
        self.assertEqual(unpack_ecg_hr(frame), 60)


if __name__ == "__main__":
    unittest.main()
