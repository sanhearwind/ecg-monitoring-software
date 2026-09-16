"""
ECG Monitor - Communication Protocol Parser
=============================================
10-byte frame format:
  [ModuleID(1B)] [DataHead(1B)] [SecondID(1B)] [Data0..5(6B)] [CheckSum(1B)]

- ModuleID:  0x00~0x7F (MSB=0, marks frame start)
- DataHead:  stores MSBs of Data[0..5] in its lower 6 bits (bit0=DataID MSB, bit1=Data0 MSB, ...)
- SecondID, Data[0..5], CheckSum: MSB always = 1 (0x80~0xFF)
- CheckSum:  (ModuleID + DataHead + SecondID + Data[0..5]) & 0x7F, then |= 0x80
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import struct

# Module IDs
MODULE_SYS  = 0x01
MODULE_ECG  = 0x10
MODULE_RESP = 0x11
MODULE_TEMP = 0x12
MODULE_SPO2 = 0x13
MODULE_NBP  = 0x14
MODULE_WAVE = 0x71

# ECG Second IDs
DAT_ECG_WAVE = 0x02
DAT_ECG_LEAD = 0x03
DAT_ECG_HR   = 0x04


@dataclass
class PackFrame:
    """Parsed protocol frame."""
    module_id: int = 0
    second_id: int = 0
    data: bytes = b'\x00' * 6
    checksum: int = 0


class ProtocolParser:
    """State machine parser for the 10-byte ECG protocol."""

    STATE_IDLE = 0
    STATE_GOT_MODULE_ID = 1

    def __init__(self):
        self.reset()

    def reset(self):
        self._state = self.STATE_IDLE
        self._buf: list[int] = []
        self._module_id: int = 0

    def feed(self, byte_val: int) -> Optional[PackFrame]:
        """Feed one byte, return a complete frame if parsed successfully."""
        if self._state == self.STATE_IDLE:
            if byte_val < 0x80:
                # Valid module ID
                self._module_id = byte_val
                self._buf = [byte_val]
                self._state = self.STATE_GOT_MODULE_ID
        elif self._state == self.STATE_GOT_MODULE_ID:
            if byte_val >= 0x80:
                self._buf.append(byte_val)
                if len(self._buf) == 10:
                    frame = self._try_unpack(self._buf)
                    self._state = self.STATE_IDLE
                    self._buf = []
                    return frame
            else:
                # Unexpected module ID, restart
                self._module_id = byte_val
                self._buf = [byte_val]
        return None

    def _try_unpack(self, raw: list[int]) -> Optional[PackFrame]:
        """Unpack a 10-byte raw buffer, verify checksum."""
        module_id = raw[0]
        data_head = raw[1]
        second_id = raw[2]
        data_bytes = list(raw[3:9])
        recv_checksum = raw[9]

        # Calculate checksum
        checksum = module_id + data_head + second_id
        for b in data_bytes:
            checksum += b
        checksum &= 0x7F

        if (recv_checksum & 0x7F) != checksum:
            return None  # Checksum mismatch

        # Restore MSBs from data_head
        # data_head bit0 = second_id MSB, bit1 = data[0] MSB, ..., bit6 = data[5] MSB
        second_id = (second_id & 0x7F) | ((data_head & 0x01) << 7)
        data_head >>= 1

        restored_data = bytearray(6)
        for i in range(6):
            restored_data[i] = (data_bytes[i] & 0x7F) | ((data_head & 0x01) << 7)
            data_head >>= 1

        return PackFrame(
            module_id=module_id,
            second_id=second_id,
            data=bytes(restored_data),
            checksum=recv_checksum,
        )


def unpack_ecg_wave(frame: PackFrame) -> int:
    """Extract u16 wave value from ECG wave frame."""
    return (frame.data[0] << 8) | frame.data[1]


def unpack_ecg_lead(frame: PackFrame) -> int:
    """Extract lead status (0=on, 1=off)."""
    return frame.data[0]


def unpack_ecg_hr(frame: PackFrame) -> int:
    """Extract heart rate (u16)."""
    return (frame.data[0] << 8) | frame.data[1]
