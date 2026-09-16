# Architecture

## Runtime flow

```text
STM32 serial bytes
        |
        v
SerialThread (QThread)
        |
        v
ProtocolParser.feed()
        |
        +--> checksum-verified PackFrame
                    |
                    +--> waveform / lead / heart-rate signals
                                      |
                                      v
                              ECGMainWindow
                       plot + status + CSV recorder
                                      |
                                      +--> ArrhythmiaDetector
                                      +--> offline CSV playback
```

## Components

| Component | Responsibility |
|---|---|
| `host/main.py` | Creates `QApplication`, applies the default font and opens the main window. |
| `host/main_window.py` | Builds the UI, plots the waveform, displays device status, records CSV data, replays saved data and presents analysis results. |
| `host/serial_thread.py` | Discovers ports, manages `pyserial` in a background thread and emits parsed device messages. |
| `host/protocol.py` | Reconstructs checksum-verified 10-byte frames and decodes ECG waveform, lead and heart-rate payloads. |
| `host/arrhythmia_detector.py` | Applies educational filtering, R-peak/RR analysis and rule-based rhythm classification. |

## Live acquisition

1. The user chooses a serial port and baud rate.
2. `SerialThread` opens the port and reads bytes away from the UI thread.
3. Each byte is passed to `ProtocolParser.feed()`.
4. A frame is emitted only after the parser reconstructs ten bytes and verifies
   the checksum.
5. ECG module messages are decoded and emitted as Qt signals.
6. `ECGMainWindow` updates the waveform and status panels and optionally writes
   timestamped values to CSV.

## Offline playback

The main window loads a CSV recording, schedules samples at the selected playback
speed and sends them through the same display and analysis path used for live
data. Playback is a demonstration and debugging path; it is not a substitute for
hardware validation.
