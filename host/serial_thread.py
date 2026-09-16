"""
ECG Monitor - Serial Communication Thread
==========================================
Runs in a background QThread, receives bytes from serial port,
parses protocol frames, and emits Qt signals for the GUI.
"""

import serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, pyqtSignal
from protocol import (
    ProtocolParser, PackFrame,
    MODULE_ECG, DAT_ECG_WAVE, DAT_ECG_LEAD, DAT_ECG_HR,
    unpack_ecg_wave, unpack_ecg_lead, unpack_ecg_hr,
)


def list_serial_ports() -> list[str]:
    """Return a list of available serial port names."""
    return [p.device for p in serial.tools.list_ports.comports()]


class SerialThread(QThread):
    """Background thread for serial port reading."""

    # Signals
    wave_received = pyqtSignal(float)       # ECG wave value (float for plot)
    lead_status = pyqtSignal(int)           # 0=on, 1=off
    heart_rate = pyqtSignal(int)            # bpm
    raw_frame = pyqtSignal(str)             # hex string of raw frame
    connection_changed = pyqtSignal(bool)   # True=connected
    error_occurred = pyqtSignal(str)        # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._serial: serial.Serial | None = None
        self._running = False
        self._parser = ProtocolParser()

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def open_port(self, port: str, baudrate: int = 115200) -> bool:
        """Open the serial port."""
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                stopbits=serial.STOPBITS_ONE,
                parity=serial.PARITY_NONE,
                timeout=0.05,
            )
            self._parser.reset()
            self.connection_changed.emit(True)
            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            return False

    def close_port(self):
        """Close the serial port."""
        self._running = False
        serial_port = self._serial
        if serial_port and serial_port.is_open:
            serial_port.close()
        if self.isRunning():
            self.wait(1000)
        self._serial = None
        self.connection_changed.emit(False)

    def send_frame(self, module_id: int, second_id: int, data: bytes = b'\x00' * 6):
        """Pack and send a 10-byte frame."""
        if not self.is_connected:
            return
        raw = self._pack(module_id, second_id, data)
        try:
            self._serial.write(raw)
        except Exception as e:
            self.error_occurred.emit(f"Send error: {e}")

    def run(self):
        """Main thread loop: read bytes and parse frames."""
        self._running = True
        while self._running and self.is_connected:
            try:
                serial_port = self._serial
                if serial_port is None:
                    break
                raw = serial_port.read(64)
                if not raw:
                    continue
                for b in raw:
                    frame = self._parser.feed(b)
                    if frame:
                        self._handle_frame(frame)
            except serial.SerialException:
                self._running = False
                self.error_occurred.emit("Serial port disconnected")
                self._serial = None
                self.connection_changed.emit(False)
            except Exception as e:
                if self._running:
                    self.error_occurred.emit(str(e))

    def _handle_frame(self, frame: PackFrame):
        """Dispatch parsed frame to appropriate signal."""
        hex_str = ' '.join(f'{b:02X}' for b in [frame.module_id, frame.second_id] + list(frame.data))
        self.raw_frame.emit(hex_str)

        if frame.module_id == MODULE_ECG:
            if frame.second_id == DAT_ECG_WAVE:
                val = unpack_ecg_wave(frame)
                self.wave_received.emit(float(val))
            elif frame.second_id == DAT_ECG_LEAD:
                status = unpack_ecg_lead(frame)
                self.lead_status.emit(status)
            elif frame.second_id == DAT_ECG_HR:
                hr = unpack_ecg_hr(frame)
                self.heart_rate.emit(hr)

    @staticmethod
    def _pack(module_id: int, second_id: int, data: bytes) -> bytes:
        """Pack a 10-byte frame (same logic as STM32 PackWithCheckSum)."""
        assert module_id < 0x80
        assert len(data) == 6

        raw = bytearray(10)
        raw[0] = module_id
        raw[2] = second_id
        for i in range(6):
            raw[3 + i] = data[i]

        # Build data_head and set MSBs
        data_head = 0
        for i in range(8, 1, -1):
            data_head <<= 1
            data_head |= (raw[i] & 0x80) >> 7
            raw[i] = raw[i] | 0x80

        raw[1] = data_head | 0x80

        # Checksum
        checksum = 0
        for i in range(9):
            checksum += raw[i]
        raw[9] = (checksum & 0x7F) | 0x80

        return bytes(raw)
