# Host application

Run from the repository root after installing `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python host\main.py
```

This directory is the independently developed replacement host, not the legacy
PyQt5 template supplied with the course.

| File | Purpose |
|---|---|
| `main.py` | Application entry point. |
| `main_window.py` | UI, plotting, recording, playback and application state. |
| `serial_thread.py` | Background serial-port communication. |
| `protocol.py` | Ten-byte frame parser and ECG payload helpers. |
| `arrhythmia_detector.py` | Educational filtering and rule-based rhythm analysis. |
