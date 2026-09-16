"""
ECG Monitor - Main Window
==========================
PyQt5 main window with ECG waveform display, heart rate, lead status,
serial port control, and data logging.
"""

import sys
import os
import time
import csv
from collections import deque

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QToolBar, QAction, QComboBox, QPushButton, QLabel, QGroupBox,
    QFrame,
    QTextEdit, QSplitter, QStatusBar, QMessageBox, QFileDialog,
    QApplication, QSizePolicy, QScrollArea, QDialog,
)
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QFont, QColor, QIcon, QFontMetrics

import pyqtgraph as pg
import numpy as np

from serial_thread import SerialThread, list_serial_ports
from arrhythmia_detector import ArrhythmiaDetector, ArrhythmiaType, DetectionResult


# ─── Styling ────────────────────────────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #2c3440;
    color: #f8fafc;
    font-size: 12pt;
}
QLabel {
    background-color: transparent;
}
QToolBar#topToolbar {
    background-color: #3a4350;
    border: 1px solid #566170;
    border-radius: 8px;
    spacing: 4px;
    padding: 6px;
}
QToolBar#topToolbar::separator {
    background-color: #697586;
    width: 1px;
    margin: 4px 5px;
}
QLabel#appTitle {
    color: #ffffff;
    font-size: 13pt;
    font-weight: 700;
    padding-right: 10px;
}
QLabel#toolbarLabel {
    color: #dbeafe;
    padding-left: 4px;
}
QFrame#plotPanel {
    background-color: #374151;
    border: 1px solid #64748b;
    border-radius: 8px;
}
QFrame#infoCard {
    background-color: #3a4350;
    border: 1px solid #64748b;
    border-radius: 8px;
}
QLabel#panelTitle {
    color: #ffffff;
    font-size: 13pt;
    font-weight: 700;
}
QLabel#cardTitle {
    color: #ffffff;
    font-size: 12pt;
    font-weight: 700;
}
QLabel#panelMeta {
    color: #bbf7d0;
    font-size: 11pt;
    font-weight: 600;
}
QGroupBox {
    background-color: #3a4350;
    border: 1px solid #64748b;
    border-radius: 8px;
    margin-top: 12px;
    padding: 22px 14px 14px 14px;
    font-weight: bold;
    color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #dbeafe;
    font-size: 11pt;
    font-weight: 600;
}
QPushButton {
    background-color: #4b5563;
    border: 1px solid #7b8797;
    border-radius: 6px;
    padding: 7px 10px;
    color: #ffffff;
    font-weight: 600;
    font-size: 11pt;
}
QPushButton:hover {
    background-color: #5b6778;
    border-color: #93a4b8;
}
QPushButton:pressed {
    background-color: #374151;
}
QPushButton:disabled {
    background-color: #3b424f;
    border-color: #596273;
    color: #aebaca;
}
QPushButton#connectBtn {
    background-color: #15803d;
    border-color: #4ade80;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#connectBtn:hover {
    background-color: #16a34a;
}
QPushButton#disconnectBtn {
    background-color: #b91c1c;
    border-color: #f87171;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#disconnectBtn:hover {
    background-color: #dc2626;
}
QPushButton#recordingBtn {
    background-color: #dc2626;
    border-color: #fca5a5;
    color: #ffffff;
}
QComboBox {
    background-color: #f8fafc;
    border: 1px solid #93a4b8;
    border-radius: 6px;
    padding: 6px 6px;
    color: #111827;
    min-height: 22px;
    font-size: 11pt;
}
QComboBox QAbstractItemView {
    background-color: #f8fafc;
    color: #111827;
    selection-background-color: #2563eb;
}
QTextEdit {
    background-color: #25303b;
    border: 1px solid #64748b;
    border-radius: 6px;
    color: #d1fae5;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 15px;
    padding: 8px;
}
QLabel#hrValue {
    font-weight: bold;
    color: #fde047;
    padding-top: 4px;
}
QLabel#leadOn {
    font-weight: bold;
    color: #86efac;
}
QLabel#leadOff {
    font-weight: bold;
    color: #fca5a5;
}
QLabel#statName {
    color: #dbeafe;
    font-size: 11pt;
}
QLabel#statValue {
    color: #ffffff;
    font-size: 12pt;
    font-weight: 700;
}
QStatusBar {
    background-color: #2c3440;
    color: #e2e8f0;
    border-top: 1px solid #64748b;
    font-size: 11pt;
}
QSplitter::handle {
    background-color: #2c3440;
}
"""


class ECGMainWindow(QMainWindow):
    """Main application window."""

    WAVE_BUF_SIZE = 2000  # Number of points to display

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Monitor - 教学原型（非医疗设备）")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # Data buffers
        self._wave_buf = deque([0.0] * self.WAVE_BUF_SIZE, maxlen=self.WAVE_BUF_SIZE)
        self._logging = False
        self._log_file = None
        self._log_writer = None
        self._frame_count = 0
        self._total_frames = 0
        self._last_wave_value_update = 0.0
        self._adaptive_labels: dict[QLabel, dict[str, object]] = {}
        self._adaptive_controls: dict[object, dict[str, object]] = {}
        self._adaptive_base_styles: dict[object, str] = {}
        self._adaptive_font_update_pending = False
        self._updating_adaptive_fonts = False
        self._compact_toolbar = False
        self._start_time = 0.0

        # Arrhythmia detector
        self._arrhythmia_detector = ArrhythmiaDetector(fs=500)
        self._last_detection_result: DetectionResult | None = None

        # Data playback state
        self._playback_data: list[float] = []
        self._playback_index = 0
        self._playback_samples_per_tick = 1
        self._playback_playing = False
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        # Serial thread
        self._serial_thread = SerialThread(self)
        self._serial_thread.wave_received.connect(self._on_wave)
        self._serial_thread.lead_status.connect(self._on_lead)
        self._serial_thread.heart_rate.connect(self._on_hr)
        self._serial_thread.raw_frame.connect(self._on_raw_frame)
        self._serial_thread.connection_changed.connect(self._on_connection_changed)
        self._serial_thread.error_occurred.connect(self._on_error)

        # Debounced adaptive font update timer (must exist before _setup_adaptive_labels)
        self._adaptive_font_timer = QTimer(self)
        self._adaptive_font_timer.setSingleShot(True)
        self._adaptive_font_timer.timeout.connect(self._update_adaptive_fonts)

        self._setup_ui()
        self._setup_adaptive_labels()
        self._refresh_ports()

        # Plot refresh timer
        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._update_waveform)
        self._plot_timer.start(16)

        # FPS timer
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

    # ─── UI Setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # ── Toolbar area ──
        toolbar = QToolBar()
        toolbar.setObjectName("topToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._title_label = QLabel("")

        # Port selector
        self._port_combo = QComboBox()
        self._port_combo.setMinimumWidth(108)
        self._port_combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self._port_label = QLabel("串口")
        self._port_label.setObjectName("toolbarLabel")
        self._port_label.setToolTip("串口")
        toolbar.addWidget(self._port_combo)

        # Refresh button
        self._refresh_btn = QPushButton("🔄")
        self._refresh_btn.setMinimumSize(38, 36)
        self._refresh_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._refresh_btn.setToolTip("刷新串口列表")
        self._refresh_btn.clicked.connect(self._refresh_ports)
        toolbar.addWidget(self._refresh_btn)

        # Baud rate
        self._baud_combo = QComboBox()
        self._baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
        self._baud_combo.setCurrentText("115200")
        self._baud_combo.setMinimumWidth(106)
        self._baud_combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self._baud_label = QLabel("波特率")
        self._baud_label.setObjectName("toolbarLabel")
        self._baud_label.setToolTip("波特率")
        toolbar.addWidget(self._baud_combo)

        toolbar.addSeparator()

        # Connect / Disconnect
        self._connect_btn = QPushButton("连接")
        self._connect_btn.setObjectName("connectBtn")
        self._connect_btn.clicked.connect(self._on_connect)
        toolbar.addWidget(self._connect_btn)

        self._disconnect_btn = QPushButton("断开")
        self._disconnect_btn.setObjectName("disconnectBtn")
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._disconnect_btn.setEnabled(False)
        toolbar.addWidget(self._disconnect_btn)

        toolbar.addSeparator()

        # Log control
        self._log_btn = QPushButton("⏺ 记录")
        self._log_btn.setToolTip("开始/停止记录ECG数据")
        self._log_btn.clicked.connect(self._toggle_logging)
        toolbar.addWidget(self._log_btn)

        # Load data for playback
        self._load_btn = QPushButton("加载")
        self._load_btn.setToolTip("加载CSV数据文件，模拟实时心电播放")
        self._load_btn.clicked.connect(self._on_load_data)
        toolbar.addWidget(self._load_btn)

        # Playback control
        self._play_btn = QPushButton("播放")
        self._play_btn.setToolTip("播放/暂停加载的数据")
        self._play_btn.clicked.connect(self._on_play_toggle)
        self._play_btn.setEnabled(False)
        toolbar.addWidget(self._play_btn)

        # Playback speed
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self._speed_combo.setCurrentText("1x")
        self._speed_combo.setToolTip("播放速度")
        self._speed_combo.setMinimumWidth(72)
        self._speed_combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        toolbar.addWidget(self._speed_combo)

        # Clear
        self._clear_btn = QPushButton("清除")
        self._clear_btn.setToolTip("清除波形并重置检测状态")
        self._clear_btn.clicked.connect(self._clear_waveform)
        toolbar.addWidget(self._clear_btn)

        self._log_view_btn = QPushButton("日志")
        self._log_view_btn.setToolTip("查看通信日志")
        self._log_view_btn.clicked.connect(self._show_log_dialog)
        toolbar.addWidget(self._log_view_btn)
        self._apply_toolbar_density()

        # ── Main content ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter)

        # Left: waveform plot
        plot_widget = self._create_plot_widget()
        splitter.addWidget(plot_widget)

        # Right: info panel
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        # Use proportional sizes so the right panel stays usable at minimum width.
        total_width = max(self.width() - 44, 1100 - 44)
        left_width = int(total_width * 0.65)
        right_width = total_width - left_width
        splitter.setSizes([left_width, right_width])

        # ── Status bar ──
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._fps_label = QLabel("0/s")
        self._fps_label.setFixedWidth(72)
        self._fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._fps_label.setStyleSheet("font-size: 11pt; color: #e2e8f0;")
        self._status_bar.addPermanentWidget(self._fps_label)

        self.setStyleSheet(DARK_STYLE)

    def _create_plot_widget(self) -> QWidget:
        """Create the ECG waveform plot area."""
        widget = QFrame()
        widget.setObjectName("plotPanel")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("ECG Waveform")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        header.addStretch()
        sample_label = QLabel("500 Hz")
        sample_label.setObjectName("panelMeta")
        header.addWidget(sample_label)
        layout.addLayout(header)

        # pyqtgraph plot
        pg.setConfigOptions(antialias=True)
        self._plot = pg.PlotWidget()
        self._plot.setBackground('#080a0d')
        self._plot.showGrid(x=True, y=True, alpha=0.22)
        self._plot.setLabel('left', 'ADC', color='#94a3b8', size='14pt')
        self._plot.setLabel('bottom', 'Sample', color='#94a3b8', size='14pt')
        self._plot.setYRange(0, 5000)
        self._plot.setXRange(0, self.WAVE_BUF_SIZE)
        self._plot.setMenuEnabled(False)

        # ECG curve (green)
        pen = pg.mkPen(color=(34, 197, 94), width=1.8)
        self._curve = self._plot.plot(pen=pen)
        self._curve.setData([], [])

        # Grid lines styled like ECG paper
        for axis_name in ('left', 'bottom'):
            axis = self._plot.getAxis(axis_name)
            axis.setPen('#303642')
            axis.setTextPen('#94a3b8')
            axis.setStyle(tickFont=QFont("Microsoft YaHei", 10))

        layout.addWidget(self._plot)
        return widget

    def _create_info_card(self, title: str, minimum_height: int) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("infoCard")
        card.setMinimumHeight(minimum_height)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setMinimumHeight(22)
        layout.addWidget(title_label)
        return card, layout

    def _create_right_panel(self) -> QWidget:
        """Create the right-side info panel."""
        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setMinimumHeight(0)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(6)

        # ── Heart Rate ──
        self._hr_group, hr_layout = self._create_info_card("心率", 104)
        hr_layout.setSpacing(0)
        self._hr_label = QLabel("--")
        self._hr_label.setObjectName("hrValue")
        self._hr_label.setAlignment(Qt.AlignCenter)
        self._hr_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hr_layout.addWidget(self._hr_label, 1)
        hr_unit = QLabel("bpm")
        hr_unit.setAlignment(Qt.AlignCenter)
        hr_unit.setStyleSheet("color: #dbeafe; font-size: 12pt; font-weight: 600;")
        hr_layout.addWidget(hr_unit)
        layout.addWidget(self._hr_group, 4)

        # ── Lead Status ──
        self._lead_group, lead_layout = self._create_info_card("导联状态", 82)
        lead_layout.setSpacing(2)
        self._lead_label = QLabel("未连接")
        self._lead_label.setObjectName("leadOff")
        self._lead_label.setAlignment(Qt.AlignCenter)
        self._lead_label.setMinimumHeight(38)
        self._lead_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lead_layout.addWidget(self._lead_label, 1)
        layout.addWidget(self._lead_group, 2)

        # ── Experimental rhythm analysis ──
        self._arrhythmia_group, arrhythmia_layout = self._create_info_card("实验性节律分析", 96)
        arrhythmia_layout.setSpacing(5)
        self._arrhythmia_label = QLabel("等待数据...")
        self._arrhythmia_label.setAlignment(Qt.AlignCenter)
        self._arrhythmia_label.setMinimumHeight(36)
        self._arrhythmia_label.setStyleSheet("color: #7dd3fc;")
        self._arrhythmia_label.setWordWrap(True)
        self._arrhythmia_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        arrhythmia_layout.addWidget(self._arrhythmia_label, 1)
        self._arrhythmia_detail_label = QLabel("")
        self._arrhythmia_detail_label.setAlignment(Qt.AlignCenter)
        self._arrhythmia_detail_label.setStyleSheet("color: #dbeafe;")
        self._arrhythmia_detail_label.setWordWrap(True)
        self._arrhythmia_detail_label.setTextFormat(Qt.PlainText)
        self._arrhythmia_detail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        arrhythmia_layout.addWidget(self._arrhythmia_detail_label)
        layout.addWidget(self._arrhythmia_group, 3)

        # ── Connection Info ──
        self._info_group, info_card_layout = self._create_info_card("连接信息", 132)
        info_layout = QGridLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setHorizontalSpacing(18)
        info_layout.setVerticalSpacing(4)
        status_name = QLabel("状态")
        status_name.setObjectName("statName")
        status_name.setMinimumWidth(82)
        status_name.setAlignment(Qt.AlignVCenter)
        info_layout.addWidget(status_name, 0, 0)
        self._conn_label = QLabel("未连接")
        self._conn_label.setObjectName("statValue")
        self._conn_label.setStyleSheet("color: #fca5a5;")
        self._conn_label.setAlignment(Qt.AlignVCenter)
        self._conn_label.setMinimumHeight(24)
        self._conn_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        info_layout.addWidget(self._conn_label, 0, 1)
        frame_name = QLabel("帧计数")
        frame_name.setObjectName("statName")
        frame_name.setMinimumWidth(82)
        frame_name.setAlignment(Qt.AlignVCenter)
        info_layout.addWidget(frame_name, 1, 0)
        self._frame_cnt_label = QLabel("0")
        self._frame_cnt_label.setObjectName("statValue")
        self._frame_cnt_label.setAlignment(Qt.AlignVCenter)
        self._frame_cnt_label.setMinimumHeight(24)
        self._frame_cnt_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        info_layout.addWidget(self._frame_cnt_label, 1, 1)
        uptime_name = QLabel("运行时间")
        uptime_name.setObjectName("statName")
        uptime_name.setMinimumWidth(82)
        uptime_name.setAlignment(Qt.AlignVCenter)
        info_layout.addWidget(uptime_name, 2, 0)
        self._uptime_label = QLabel("00:00:00")
        self._uptime_label.setObjectName("statValue")
        self._uptime_label.setAlignment(Qt.AlignVCenter)
        self._uptime_label.setMinimumHeight(24)
        self._uptime_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        info_layout.addWidget(self._uptime_label, 2, 1)
        info_layout.setColumnStretch(0, 0)
        info_layout.setColumnStretch(1, 1)
        for row in range(3):
            info_layout.setRowMinimumHeight(row, 26)
        info_card_layout.addLayout(info_layout)
        layout.addWidget(self._info_group, 3)

        # ── Wave value ──
        self._wave_group, wave_layout = self._create_info_card("当前波形值", 84)
        wave_layout.setAlignment(Qt.AlignCenter)
        self._wave_val_label = QLabel("0")
        self._wave_val_label.setAlignment(Qt.AlignCenter)
        self._wave_val_label.setStyleSheet("color: #7dd3fc;")
        self._wave_val_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        wave_layout.addWidget(self._wave_val_label, 1)
        layout.addWidget(self._wave_group, 3)

        # Keep the log buffer for diagnostics without crowding the monitor panel.
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.hide()
        return panel

    def _setup_adaptive_labels(self):
        """Register labels that should resize their fonts to fit available space."""
        self._adaptive_labels = {
            self._hr_label: {"min": 20, "max": 74, "bold": True, "sample": "000", "container": self._hr_group, "height_ratio": 0.48},
            self._lead_label: {"min": 12, "max": 28, "bold": True, "sample": "导联脱落", "container": self._lead_group, "height_ratio": 0.44},
            self._arrhythmia_label: {"min": 12, "max": 32, "bold": True, "wrap": True, "sample": "室性心动过速", "container": self._arrhythmia_group, "height_ratio": 0.40},
            self._arrhythmia_detail_label: {"min": 10, "max": 20, "wrap": True, "sample": "心率: 000 bpm    RR变异: 00.0%\n算法评分: 100%", "container": self._arrhythmia_group, "height_ratio": 0.26},
            self._wave_val_label: {"min": 14, "max": 44, "bold": True, "sample": "5000", "container": self._wave_group, "height_ratio": 0.58},
        }

        self._adaptive_controls = {}

        for label in self._adaptive_labels:
            self._adaptive_base_styles[label] = label.styleSheet()
            label.installEventFilter(self)
            container = self._adaptive_labels[label].get("container")
            if container is not None:
                container.installEventFilter(self)
        for control in self._adaptive_controls:
            self._adaptive_base_styles[control] = control.styleSheet()
            control.installEventFilter(self)

        self._schedule_adaptive_font_update()

    def _schedule_adaptive_font_update(self):
        if self._updating_adaptive_fonts:
            return
        self._adaptive_font_update_pending = True
        self._adaptive_font_timer.start(50)

    def _update_adaptive_fonts(self):
        if self._updating_adaptive_fonts:
            return

        self._adaptive_font_update_pending = False
        self._updating_adaptive_fonts = True
        try:
            for label, spec in self._adaptive_labels.items():
                self._fit_label_font(label, spec)
            for control, spec in self._adaptive_controls.items():
                self._fit_control_font(control, spec)
        finally:
            self._updating_adaptive_fonts = False

    def _fit_label_font(self, label: QLabel, spec: dict[str, object]):
        stored_text = label.property("fullText")
        if stored_text is not None and label.text() != stored_text:
            label.setText(stored_text)

        container = spec.get("container")
        if container is not None:
            width = max(1, container.contentsRect().width() - 28)
            height_ratio = float(spec.get("height_ratio", 1.0))
            height = max(1, int((container.contentsRect().height() - 46) * height_ratio))
        else:
            width = max(1, label.contentsRect().width() - 8)
            height = max(1, label.contentsRect().height() - 6)
        if width <= 1 or height <= 1:
            return

        min_size = int(spec.get("min", 10))
        max_size = int(spec.get("max", 24))
        actual_text = label.property("fullText") or label.text() or " "
        text = str(spec.get("sample") or actual_text)
        wrap = bool(spec.get("wrap", False))
        bold = bool(spec.get("bold", False))

        best_size = min_size
        for size in range(min_size, max_size + 1):
            test_font = QFont(label.font())
            test_font.setPointSize(size)
            test_font.setBold(bold)
            metrics = QFontMetrics(test_font)

            if wrap:
                rect = metrics.boundingRect(0, 0, width, height, Qt.TextWordWrap | Qt.AlignCenter, text)
            else:
                rect = metrics.boundingRect(text)

            if rect.width() <= width and rect.height() <= int(height * 1.18):
                best_size = size
            else:
                break

        current_font = label.font()
        if current_font.pointSize() != best_size or current_font.bold() != bold:
            current_font.setPointSize(best_size)
            current_font.setBold(bold)
            label.setFont(current_font)
        self._apply_adaptive_font_style(label, best_size, bold)
        self._apply_label_overflow_policy(label, str(actual_text), best_size, width, wrap)

    def _apply_label_overflow_policy(self, label: QLabel, text: str, point_size: int, width: int, wrap: bool):
        label.setToolTip("" if text == label.text() else text)
        if wrap:
            label.setWordWrap(True)
            return

        font = QFont(label.font())
        font.setPointSize(point_size)
        metrics = QFontMetrics(font)
        available_width = max(1, width - 4)
        if metrics.horizontalAdvance(text) > available_width:
            label.setToolTip(text)
            label.setText(metrics.elidedText(text, Qt.ElideRight, available_width))

    def _fit_control_font(self, control, spec: dict[str, object]):
        width = control.width()
        height = max(int(spec.get("height", control.sizeHint().height())), control.minimumHeight(), 32)
        min_width = int(spec.get("min_width", control.minimumWidth() or 60))
        control.setMinimumHeight(height)
        control.setMinimumWidth(min_width)
        if width <= 1:
            width = max(control.sizeHint().width(), min_width)

        text = str(spec.get("sample") or getattr(control, "currentText", lambda: "")() or getattr(control, "text", lambda: "")() or " ")
        min_size = int(spec.get("min", 10))
        max_size = int(spec.get("max", 18))
        best_size = min_size
        available_width = max(min_width - 18, width - 18)
        available_height = max(20, height - 12)

        for size in range(min_size, max_size + 1):
            test_font = QFont(control.font())
            test_font.setPointSize(size)
            metrics = QFontMetrics(test_font)
            rect = metrics.boundingRect(text)
            if rect.width() <= available_width and rect.height() <= available_height:
                best_size = size
            else:
                break

        current_font = control.font()
        if current_font.pointSize() != best_size:
            current_font.setPointSize(best_size)
            control.setFont(current_font)
        self._apply_adaptive_font_style(control, best_size, bool(spec.get("bold", False)))
        if hasattr(control, "setToolTip") and not control.toolTip():
            control.setToolTip(text)

    def _apply_adaptive_font_style(self, widget, point_size: int, bold: bool):
        base_style = self._adaptive_base_styles.get(widget, widget.styleSheet()).strip()
        extra = f"font-size: {point_size}pt;"
        if bold:
            extra += " font-weight: 700;"
        next_style = f"{base_style} {extra}".strip()
        if widget.styleSheet() != next_style:
            widget.setStyleSheet(next_style)

    def _set_adaptive_base_style(self, widget, style: str):
        if self._adaptive_base_styles.get(widget) == style:
            return
        self._adaptive_base_styles[widget] = style
        if widget in self._adaptive_labels:
            spec = self._adaptive_labels[widget]
            point_size = widget.font().pointSize()
            if point_size <= 0:
                point_size = int(spec.get("min", 10))
            self._apply_adaptive_font_style(widget, point_size, bool(spec.get("bold", False)))
        else:
            widget.setStyleSheet(style)

    def _set_adaptive_text(self, label: QLabel, text: str, refit: bool = True):
        if label.property("fullText") == text and label.text() == text:
            return
        label.setProperty("fullText", text)
        label.setText(text)

    def _set_stat_text(self, label: QLabel, text: str):
        if label.text() != text:
            label.setText(text)

    def eventFilter(self, obj, event):
        watched_events = (QEvent.Resize, QEvent.Show)
        if obj in self._adaptive_labels and event.type() in watched_events:
            self._schedule_adaptive_font_update()
        if obj in self._adaptive_controls and event.type() in watched_events:
            self._schedule_adaptive_font_update()
        if event.type() in watched_events:
            for spec in self._adaptive_labels.values():
                if obj is spec.get("container"):
                    self._schedule_adaptive_font_update()
                    break
        return super().eventFilter(obj, event)

    def _apply_toolbar_density(self, force: bool = False):
        compact = self.width() < 1180
        if compact == self._compact_toolbar and not force:
            return
        self._compact_toolbar = compact

        if compact:
            self._log_btn.setText("停止" if self._logging else "记录")
            self._load_btn.setText("加载")
            self._play_btn.setText("暂停" if self._playback_playing else "播放")
            self._clear_btn.setText("清除")
            self._port_combo.setMinimumWidth(90)
            self._baud_combo.setMinimumWidth(90)
            self._speed_combo.setMinimumWidth(58)
        else:
            self._log_btn.setText("⏹ 停止" if self._logging else "⏺ 记录")
            self._load_btn.setText("加载")
            self._play_btn.setText("暂停" if self._playback_playing else "播放")
            self._clear_btn.setText("清除")
            self._port_combo.setMinimumWidth(120)
            self._baud_combo.setMinimumWidth(120)
            self._speed_combo.setMinimumWidth(82)

    # ─── Port Management ───────────────────────────────────────────────────────

    def _refresh_ports(self):
        self._port_combo.clear()
        ports = list_serial_ports()
        self._port_combo.addItems(ports)
        if not ports:
            self._port_combo.addItem("无可用串口")

    def _on_connect(self):
        port = self._port_combo.currentText()
        if "无" in port:
            QMessageBox.warning(self, "警告", "未检测到可用串口，请检查设备连接。")
            return
        baud = int(self._baud_combo.currentText())
        if self._serial_thread.isRunning():
            self._append_log("[WARN] 串口线程已在运行，请先断开")
            return
        if self._serial_thread.open_port(port, baud):
            self._serial_thread.start()
            self._start_time = time.time()
            self._frame_count = 0
            self._total_frames = 0
            self._arrhythmia_detector.reset()
            self._set_adaptive_text(self._arrhythmia_label, "正在学习...")
            self._set_adaptive_text(self._arrhythmia_detail_label, "")
            self._append_log(f"[INFO] 已连接 {port} @ {baud}")

    def _on_disconnect(self):
        self._serial_thread.close_port()
        self._append_log("[INFO] 已断开连接")

    def _on_connection_changed(self, connected: bool):
        self._connect_btn.setEnabled(not connected)
        self._disconnect_btn.setEnabled(connected)
        self._port_combo.setEnabled(not connected)
        self._baud_combo.setEnabled(not connected)
        if connected:
            self._set_stat_text(self._conn_label, "已连接")
            if self._conn_label.styleSheet() != "color: #86efac;":
                self._conn_label.setStyleSheet("color: #86efac;")
        else:
            self._set_stat_text(self._conn_label, "未连接")
            if self._conn_label.styleSheet() != "color: #fca5a5;":
                self._conn_label.setStyleSheet("color: #fca5a5;")
            self._start_time = 0.0
            self._frame_count = 0
            self._set_stat_text(self._uptime_label, "00:00:00")
            if self._logging:
                self._toggle_logging()

    # ─── Data Handlers ─────────────────────────────────────────────────────────

    def _on_wave(self, value: float):
        """Receive one ECG wave sample."""
        self._process_wave_sample(value)

        # Log to CSV
        if self._logging and self._log_writer:
            elapsed = time.time() - self._start_time
            self._log_writer.writerow([f"{elapsed:.3f}", "WAVE", f"{value:.0f}"])

    def _process_wave_sample(self, value: float):
        """Update waveform state and feed the arrhythmia detector."""
        self._wave_buf.append(value)
        self._frame_count += 1
        self._total_frames += 1
        now = time.monotonic()
        if now - self._last_wave_value_update >= 0.12:
            self._last_wave_value_update = now
            self._set_adaptive_text(self._wave_val_label, f"{value:.0f}", refit=False)

        # Feed to arrhythmia detector
        result = self._arrhythmia_detector.feed_sample(value)
        if result is not None:
            self._last_detection_result = result
            self._update_arrhythmia_display(result)

    def _update_arrhythmia_display(self, result: DetectionResult):
        """Update arrhythmia detection display."""
        # Map arrhythmia type to display text and color
        type_config = {
            ArrhythmiaType.NORMAL: ("✅ 正常窦性心律", "#86efac"),
            ArrhythmiaType.TACHYCARDIA: ("⚠️ 心动过速", "#fde047"),
            ArrhythmiaType.BRADYCARDIA: ("⚠️ 心动过缓", "#fde047"),
            ArrhythmiaType.ATRIAL_FIBRILLATION: ("🔴 房颤", "#fca5a5"),
            ArrhythmiaType.VENTRICULAR_TACHYCARDIA: ("🔴 室性心动过速", "#fca5a5"),
            ArrhythmiaType.PVC: ("⚠️ 室性早搏", "#fde047"),
            ArrhythmiaType.UNKNOWN: ("❓ 未知", "#dbeafe"),
        }

        text, color = type_config.get(result.arrhythmia_type, ("❓ 未知", "#dbeafe"))
        self._set_adaptive_text(self._arrhythmia_label, text)
        self._set_adaptive_base_style(self._arrhythmia_label, f"color: {color};")
        if result.heart_rate > 0:
            self._update_heart_rate_card(int(round(result.heart_rate)))

        # Build detail text
        detail_line = f"心率: {result.heart_rate:.0f} bpm    RR变异: {result.rr_cv*100:.1f}%"
        confidence_line = ""
        if result.confidence > 0:
            confidence_line = f"算法评分: {result.confidence*100:.0f}%"
        self._set_adaptive_text(self._arrhythmia_detail_label, "\n".join(part for part in (detail_line, confidence_line) if part))

    def _on_lead(self, status: int):
        """Receive lead status update."""
        if status == 0:
            self._set_adaptive_text(self._lead_label, "✅ 导联正常")
            self._lead_label.setObjectName("leadOn")
        else:
            self._set_adaptive_text(self._lead_label, "❌ 导联脱落")
            self._lead_label.setObjectName("leadOff")
        self._lead_label.style().unpolish(self._lead_label)
        self._lead_label.style().polish(self._lead_label)

        if self._logging and self._log_writer:
            elapsed = time.time() - self._start_time
            self._log_writer.writerow([f"{elapsed:.3f}", "LEAD", str(status)])

    def _on_hr(self, bpm: int):
        """Receive heart rate update."""
        self._update_heart_rate_card(bpm)

        if self._logging and self._log_writer:
            elapsed = time.time() - self._start_time
            self._log_writer.writerow([f"{elapsed:.3f}", "HR", str(bpm)])

    def _update_heart_rate_card(self, bpm: int):
        self._set_adaptive_text(self._hr_label, str(bpm))
        if bpm < 50 or bpm > 120:
            self._set_adaptive_base_style(self._hr_label, "color: #fca5a5;")
        else:
            self._set_adaptive_base_style(self._hr_label, "color: #fde047;")

    def _on_raw_frame(self, hex_str: str):
        """Receive raw frame data for logging."""
        self._append_log(hex_str, max_lines=200)

    def _on_error(self, msg: str):
        self._append_log(f"[ERROR] {msg}")

    # ─── Data Playback ──────────────────────────────────────────────────────────

    def _on_load_data(self):
        """Load CSV data file for playback."""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载ECG数据", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            timestamps = []
            values = []
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                first = True
                for row in reader:
                    if not row:
                        continue
                    # Skip header row if columns are textual labels
                    if first and any(col.lower() in ("timestamp", "type", "value") for col in row):
                        first = False
                        continue
                    first = False
                    if len(row) >= 3 and row[1].upper() == 'WAVE':
                        try:
                            timestamps.append(float(row[0]))
                            values.append(float(row[2]))
                        except ValueError:
                            continue

            if not values:
                QMessageBox.warning(self, "警告", "未找到有效的 WAVE 数据。")
                return

            self._playback_data = values
            self._playback_index = 0
            self._playback_playing = False
            self._play_btn.setEnabled(True)
            self._play_btn.setText("播放")
            self._apply_toolbar_density(force=True)

            # Reset detector and UI
            self._clear_waveform()
            self._set_adaptive_text(self._arrhythmia_label, "已加载数据，点击播放")

            filename = os.path.basename(path)
            self._append_log(f"[INFO] 已加载 {filename} ({len(values)} 个采样点)")

            # Update status bar
            duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            self._status_bar.showMessage(f"已加载: {filename} | 时长: {duration:.1f}s | 采样点: {len(values)}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载文件失败: {e}")

    def _on_play_toggle(self):
        """Toggle playback state."""
        if not self._playback_data:
            return

        if self._playback_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start data playback."""
        if self._playback_index >= len(self._playback_data):
            self._clear_waveform()

        self._playback_playing = True
        self._play_btn.setText("暂停")
        self._load_btn.setEnabled(False)
        self._apply_toolbar_density(force=True)

        # Get speed multiplier and compute samples per tick.
        # Use a fixed 16 ms timer interval (≈60 Hz) so playback speed is accurate
        # regardless of Windows timer resolution limitations.
        speed_text = self._speed_combo.currentText()
        speed = float(speed_text.replace('x', ''))
        interval_ms = 16
        self._playback_samples_per_tick = max(1, int(round(speed * 500 * interval_ms / 1000)))
        self._playback_timer.start(interval_ms)

    def _stop_playback(self):
        """Stop data playback."""
        self._playback_playing = False
        self._playback_timer.stop()
        self._play_btn.setText("播放")
        self._load_btn.setEnabled(True)
        self._apply_toolbar_density(force=True)

    def _on_playback_tick(self):
        """Handle playback timer tick - feed one sample."""
        if self._playback_index >= len(self._playback_data):
            self._stop_playback()
            self._set_adaptive_text(self._arrhythmia_label, "✅ 播放完成")
            return

        samples_to_process = min(
            self._playback_samples_per_tick,
            len(self._playback_data) - self._playback_index,
        )
        for _ in range(samples_to_process):
            value = self._playback_data[self._playback_index]
            self._playback_index += 1
            self._process_wave_sample(value)

        # Update progress
        progress = self._playback_index / len(self._playback_data) * 100
        remaining = len(self._playback_data) - self._playback_index
        self._status_bar.showMessage(
            f"播放中: {self._playback_index}/{len(self._playback_data)} ({progress:.0f}%) | 剩余: {remaining} 点"
        )

    # ─── Plot Update ───────────────────────────────────────────────────────────

    def _update_waveform(self):
        data = np.array(self._wave_buf)
        self._curve.setData(np.arange(len(data)), data)

    def _clear_waveform(self):
        self._wave_buf.clear()
        self._wave_buf.extend([0.0] * self.WAVE_BUF_SIZE)
        self._frame_count = 0
        self._total_frames = 0
        self._set_stat_text(self._frame_cnt_label, "0")
        self._set_adaptive_text(self._wave_val_label, "0")
        self._arrhythmia_detector.reset()
        self._set_adaptive_text(self._arrhythmia_label, "等待数据...")
        self._set_adaptive_text(self._arrhythmia_detail_label, "")
        # Reset playback state
        if self._playback_playing:
            self._stop_playback()
        self._playback_index = 0

    # ─── FPS Counter ───────────────────────────────────────────────────────────

    def _update_fps(self):
        fps = self._frame_count
        self._frame_count = 0
        fps_text = f"{fps}/s"
        if self._fps_label.text() != fps_text:
            self._fps_label.setText(fps_text)
        self._set_stat_text(self._frame_cnt_label, str(self._total_frames))

        if self._start_time > 0:
            elapsed = int(time.time() - self._start_time)
            h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
            self._set_stat_text(self._uptime_label, f"{h:02d}:{m:02d}:{s:02d}")

    # ─── Data Logging ──────────────────────────────────────────────────────────

    def _toggle_logging(self):
        if not self._logging:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存 ECG 数据", f"ecg_data_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv)"
            )
            if not path:
                return
            self._log_file = open(path, 'w', newline='', encoding='utf-8')
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(["timestamp", "type", "value"])
            self._logging = True
            self._log_btn.setText("停止")
            self._log_btn.setToolTip("停止记录ECG数据")
            self._log_btn.setObjectName("recordingBtn")
            self._log_btn.style().unpolish(self._log_btn)
            self._log_btn.style().polish(self._log_btn)
            self._apply_toolbar_density(force=True)
            self._append_log(f"[INFO] 开始记录数据到 {path}")
        else:
            self._logging = False
            if self._log_file:
                self._log_file.close()
                self._log_file = None
                self._log_writer = None
            self._log_btn.setText("记录")
            self._log_btn.setToolTip("开始记录ECG数据")
            self._log_btn.setObjectName("")
            self._log_btn.style().unpolish(self._log_btn)
            self._log_btn.style().polish(self._log_btn)
            self._apply_toolbar_density(force=True)
            self._append_log("[INFO] 停止记录数据")

    # ─── Log Area ──────────────────────────────────────────────────────────────

    def _append_log(self, text: str, max_lines: int = 500):
        self._log_text.append(text)
        # Limit lines
        doc = self._log_text.document()
        if doc.blockCount() > max_lines:
            cursor = self._log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, doc.blockCount() - max_lines)
            cursor.removeSelectedText()

    def _show_log_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("通信日志")
        dialog.resize(720, 420)

        layout = QVBoxLayout(dialog)
        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setPlainText(self._log_text.toPlainText())
        layout.addWidget(log_view)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()

    # ─── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._plot_timer.stop()
        self._fps_timer.stop()
        self._playback_timer.stop()
        self._adaptive_font_timer.stop()
        self._serial_thread.close_port()
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_toolbar_density()
        self._schedule_adaptive_font_update()
