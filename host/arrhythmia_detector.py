"""
ECG Arrhythmia Detector
========================
Educational real-time rhythm-analysis module for ECG Monitor.

This rule-based prototype has not been clinically validated. Its labels are
software demonstrations only and must not be used for diagnosis, monitoring or
treatment decisions.

Detection Features:
  - R-peak detection using bandpass filtering + threshold
  - Heart rate calculation (tachycardia >100 bpm, bradycardia <60 bpm)
  - RR interval variability analysis (irregular rhythm detection)
  - QRS width analysis (wide QRS >120ms indicates ventricular origin)

Displayed Rhythm Labels:
  1. Normal Sinus Rhythm (正常窦性心律)
  2. Tachycardia (心动过速) - HR > 100 bpm
  3. Bradycardia (心动过缓) - HR < 60 bpm
  4. Atrial Fibrillation (房颤) - irregular rhythm, narrow QRS
  5. Ventricular Tachycardia (室速) - regular tachycardia, wide QRS
  6. PVC (室性早搏) - premature beat with wide QRS

Algorithm:
  - Sliding window of 10 seconds for analysis
  - Real-time R-peak detection using Pan-Tompkins-like algorithm
  - Classification based on HR, RR variability, and QRS width
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ArrhythmiaType(Enum):
    """Arrhythmia classification types."""
    NORMAL = "正常窦性心律"
    TACHYCARDIA = "心动过速"
    BRADYCARDIA = "心动过缓"
    ATRIAL_FIBRILLATION = "房颤"
    VENTRICULAR_TACHYCARDIA = "室性心动过速"
    PVC = "室性早搏"
    UNKNOWN = "未知"


@dataclass
class DetectionResult:
    """Result of arrhythmia detection."""
    arrhythmia_type: ArrhythmiaType
    heart_rate: float
    rr_interval_mean: float
    rr_interval_std: float
    rr_cv: float  # Coefficient of variation
    qrs_width_mean: float
    confidence: float  # 0.0 to 1.0
    description: str
    # HRV metrics
    sdnn: float = 0.0    # RR间期标准差 (ms)
    rmssd: float = 0.0   # 相邻RR差值均方根 (ms)
    pnn50: float = 0.0   # 相邻RR差值>50ms的百分比 (%)
    shannon_entropy: float = 0.0  # RR间期Shannon熵


class ArrhythmiaDetector:
    """Real-time arrhythmia detector."""

    def __init__(self, fs: int = 500, window_size: int = 10):
        """
        Initialize the detector.

        Args:
            fs: Sampling frequency in Hz (default 500 Hz from ECG data)
            window_size: Analysis window size in seconds (default 10s)
        """
        self.fs = fs
        self.window_size = window_size
        self.window_samples = fs * window_size

        # Data buffers
        self._ecg_buffer = deque(maxlen=self.window_samples)
        self._r_peaks = deque(maxlen=100)  # Store R-peak positions
        self._rr_intervals = deque(maxlen=50)  # Store RR intervals
        self._qrs_widths = deque(maxlen=10)  # Store QRS width estimates (ms)

        # Throttle expensive detection to avoid CPU/memory spikes at 500 Hz.
        # Run R-peak detection at ~20 Hz (every 25 samples @ fs=500).
        self._detect_interval = max(1, fs // 20)
        self._last_detect_sample = 0
        self._last_qrs_width_sample = 0

        # Cache recent filtered/preprocessed data for reuse by analysis routines.
        self._cached_filtered: np.ndarray | None = None
        self._cached_ma: np.ndarray | None = None
        self._cached_preprocessed_2s: np.ndarray | None = None
        self._cached_sample_count = 0

        # Filter coefficients (bandpass 5-15 Hz for QRS detection)
        self._init_filters()

        # State variables
        self._last_r_peak = 0
        self._sample_count = 0
        self._threshold = 0
        self._learning_phase = True
        self._learning_count = 0
        self._warmup_count = 0  # Samples after calibration before analysis
        self._warmup_samples = fs  # 1 second warmup after calibration

        # Dual threshold state (Pan-Tompkins style)
        self._signal_peak = 0  # Running estimate of signal peak level
        self._noise_peak = 0   # Running estimate of noise peak level

    def _init_filters(self):
        """Initialize digital filters."""
        from scipy.signal import butter, iirnotch, lfilter_zi

        nyq = self.fs / 2

        # 0.5 Hz highpass filter (baseline wander removal)
        self._b_hp, self._a_hp = butter(2, 0.5 / nyq, btype='high')
        self._zi_hp = lfilter_zi(self._b_hp, self._a_hp) * 0

        # 50 Hz notch filter (power line interference removal)
        self._b_notch, self._a_notch = iirnotch(50.0, 30.0, self.fs)
        self._zi_notch = lfilter_zi(self._b_notch, self._a_notch) * 0

        # Bandpass filter for QRS detection (5-15 Hz)
        self._b_bp, self._a_bp = butter(4, [5/nyq, 15/nyq], btype='band')
        self._zi_bp = lfilter_zi(self._b_bp, self._a_bp) * 0

    def reset(self):
        """Reset detector state."""
        self._ecg_buffer.clear()
        self._r_peaks.clear()
        self._rr_intervals.clear()
        self._qrs_widths.clear()
        self._last_r_peak = 0
        self._sample_count = 0
        self._last_detect_sample = 0
        self._last_qrs_width_sample = 0
        self._threshold = 0
        self._learning_phase = True
        self._learning_count = 0
        self._warmup_count = 0
        self._signal_peak = 0
        self._noise_peak = 0
        self._cached_filtered = None
        self._cached_ma = None
        self._cached_preprocessed_2s = None
        self._cached_sample_count = 0
        # Reset filter states
        self._zi_hp = self._zi_hp * 0
        self._zi_notch = self._zi_notch * 0
        self._zi_bp = self._zi_bp * 0

    def feed_sample(self, value: float) -> Optional[DetectionResult]:
        """
        Feed one ECG sample to the detector.

        Args:
            value: ECG ADC value (typical range 1800-2200)

        Returns:
            DetectionResult if analysis is ready, None otherwise
        """
        self._ecg_buffer.append(value)
        self._sample_count += 1

        # Learning phase: collect initial data for threshold calibration
        if self._learning_phase:
            self._learning_count += 1
            if self._learning_count >= self.fs * 3:  # 3 seconds for learning
                self._learning_phase = False
                self._calibrate_threshold()
            return None

        # Warmup phase: let filters settle after calibration, skip analysis
        if self._warmup_count < self._warmup_samples:
            self._warmup_count += 1
            self._detect_r_peaks()  # Still detect to update thresholds
            # Clear any R-peaks detected during warmup (filter transients)
            if self._warmup_count >= self._warmup_samples:
                self._r_peaks.clear()
                self._rr_intervals.clear()
                self._qrs_widths.clear()
                self._last_r_peak = 0
                self._last_detect_sample = self._sample_count
            return None

        # Throttle expensive detection to avoid CPU/memory spikes at sample rate.
        if self._sample_count - self._last_detect_sample >= self._detect_interval:
            self._last_detect_sample = self._sample_count
            # Detect R-peaks in the latest segment
            self._detect_r_peaks()

            # Analyze when we have enough data (need ≥5 RR intervals for PVC detection)
            if len(self._r_peaks) >= 6 and len(self._rr_intervals) >= 5:
                return self._analyze()

        return None

    def _calibrate_threshold(self):
        """Calibrate detection threshold from initial data.

        Must match the processing chain used in _detect_r_peaks:
        full_filter -> differential -> squared -> moving_average

        The detector analyzes overlapping one-second windows. Filtering here must
        not mutate the streaming IIR state, otherwise the same samples would be
        fed through the filters repeatedly as windows overlap.
        """
        if len(self._ecg_buffer) < self.fs:
            return

        full_data = np.array(self._ecg_buffer)
        filtered = self._apply_full_filter_window(full_data)[-self.fs:]

        # Differential filter (same as _detect_r_peaks)
        diff_kernel = np.array([-1, -2, 0, 2, 1]) / 8.0
        differentiated = np.convolve(filtered, diff_kernel, mode='same')
        squared = differentiated ** 2

        # Moving average smoothing
        window = int(0.15 * self.fs)
        ma = np.convolve(squared, np.ones(window)/window, mode='same')

        # Set threshold based on signal statistics
        self._threshold = np.mean(ma) + 0.5 * np.std(ma)
        self._signal_peak = self._threshold
        self._noise_peak = np.mean(ma)

    def _detect_r_peaks(self):
        """Detect R-peaks using Pan-Tompkins-like algorithm with differential filter."""
        if len(self._ecg_buffer) < self.fs:
            return

        # Get the latest second of data
        data = np.array(list(self._ecg_buffer)[-self.fs:])

        # Preprocess once for QRS width measurement (highpass + notch, no bandpass).
        preprocessed = self._apply_preprocessing_window(data)

        # Full filter chain: highpass -> notch -> bandpass
        filtered = self._apply_bandpass_window(preprocessed)

        # 5-point differential filter (Pan-Tompkins step)
        # Emphasizes QRS slope, suppresses T-wave
        diff_kernel = np.array([-1, -2, 0, 2, 1]) / 8.0
        differentiated = np.convolve(filtered, diff_kernel, mode='same')

        # Square and smooth
        squared = differentiated ** 2
        window = int(0.15 * self.fs)  # 150ms window
        ma = np.convolve(squared, np.ones(window)/window, mode='same')

        # Cache filtered/MA data for cheap confidence estimation later.
        self._cached_filtered = filtered
        self._cached_ma = ma
        self._cached_sample_count = self._sample_count

        # Minimum distance between beats (300ms = max 200 bpm)
        min_distance = int(0.3 * self.fs)

        # Dual threshold mechanism (Pan-Tompkins style)
        # SPKI = signal peak level, NPKI = noise peak level
        # Threshold1 = NPKI + 0.25 * (SPKI - NPKI)
        # Threshold2 = 0.5 * Threshold1 (for back-search)
        if self._signal_peak == 0:
            self._signal_peak = np.mean(ma) + 1.5 * np.std(ma)
            self._noise_peak = np.mean(ma)

        threshold1 = self._noise_peak + 0.25 * (self._signal_peak - self._noise_peak)

        # Simple peak detection
        for i in range(min_distance, len(ma) - min_distance):
            if ma[i] > threshold1:
                # Check if it's a local maximum
                if ma[i] > ma[i-1] and ma[i] > ma[i+1]:
                    # Convert MA peak position to global sample position.
                    # The MA peak is robust to QRS polarity; refining to raw max
                    # can fail on inverted QRS complexes, so we keep the MA peak.
                    global_pos = self._sample_count - self.fs + i

                    # Check minimum distance from last R-peak
                    if global_pos - self._last_r_peak >= min_distance:
                        self._r_peaks.append(global_pos)

                        # Calculate RR interval
                        if self._last_r_peak > 0:
                            rr = (global_pos - self._last_r_peak) / self.fs
                            self._rr_intervals.append(rr)

                        self._last_r_peak = global_pos

                        # Estimate QRS width for this beat using the preprocessed signal.
                        qrs_width = self._measure_qrs_width_at(preprocessed, i)
                        if qrs_width is not None:
                            self._qrs_widths.append(qrs_width)

                        # Update signal peak level (exponential moving average)
                        self._signal_peak = 0.875 * self._signal_peak + 0.125 * ma[i]
            else:
                # Below threshold -> update noise peak level
                if ma[i] > 0:
                    self._noise_peak = 0.875 * self._noise_peak + 0.125 * ma[i]

    def _measure_qrs_width_at(self, preprocessed: np.ndarray, local_peak_idx: int) -> float | None:
        """Estimate QRS width (ms) at a given local peak index using half-amplitude threshold."""
        # Use a wider search window (±120 ms) to accommodate wide QRS complexes.
        search_half = int(0.12 * self.fs)
        start = max(0, local_peak_idx - search_half)
        end = min(len(preprocessed), local_peak_idx + search_half + 1)
        segment = preprocessed[start:end]
        peak_idx = local_peak_idx - start

        if len(segment) < 5 or peak_idx < 2 or peak_idx >= len(segment) - 2:
            return None

        peak_amp = segment[peak_idx]

        # Estimate baseline robustly as the median of the segment, excluding a
        # small region around the peak. Edge-only baseline can be wrong when the
        # QRS is wide and window edges still sit on the QRS complex.
        exclude_half = int(0.04 * self.fs)
        exclude_start = max(0, peak_idx - exclude_half)
        exclude_end = min(len(segment), peak_idx + exclude_half + 1)
        baseline_samples = np.concatenate([segment[:exclude_start], segment[exclude_end:]])
        if baseline_samples.size == 0:
            return None
        baseline = float(np.median(baseline_samples))

        peak_height = peak_amp - baseline
        if abs(peak_height) < 10:  # Too small to measure
            return None

        # Half-amplitude relative to baseline
        half_amp = baseline + 0.5 * peak_height

        # Search left from peak for QRS onset
        qrs_onset = peak_idx
        for j in range(peak_idx, -1, -1):
            # Use <= or >= depending on peak polarity
            if (peak_height > 0 and segment[j] <= half_amp) or (peak_height < 0 and segment[j] >= half_amp):
                qrs_onset = j
                break

        # Search right from peak for QRS offset
        qrs_offset = peak_idx
        for j in range(peak_idx, len(segment)):
            if (peak_height > 0 and segment[j] <= half_amp) or (peak_height < 0 and segment[j] >= half_amp):
                qrs_offset = j
                break

        width_ms = (qrs_offset - qrs_onset) / self.fs * 1000
        if 20 < width_ms < 300:  # Sanity check
            return float(width_ms)
        return None

    def _apply_preprocessing(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing chain: highpass -> notch."""
        from scipy.signal import lfilter

        # Highpass: remove baseline wander
        filtered, self._zi_hp = lfilter(self._b_hp, self._a_hp, data, zi=self._zi_hp)
        # Notch: remove 50Hz power line interference
        filtered, self._zi_notch = lfilter(self._b_notch, self._a_notch, filtered, zi=self._zi_notch)
        return filtered

    def _apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter (assumes data already preprocessed)."""
        from scipy.signal import lfilter

        filtered, self._zi_bp = lfilter(self._b_bp, self._a_bp, data, zi=self._zi_bp)
        return filtered

    def _apply_full_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply full filter chain: highpass -> notch -> bandpass."""
        preprocessed = self._apply_preprocessing(data)
        return self._apply_bandpass(preprocessed)

    def _filter_window(self, b: np.ndarray, a: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Filter a standalone analysis window without mutating streaming state."""
        from scipy.signal import lfilter, lfilter_zi

        data = np.asarray(data, dtype=float)
        if data.size == 0:
            return data
        zi = lfilter_zi(b, a) * float(data[0])
        filtered, _ = lfilter(b, a, data, zi=zi)
        return filtered

    def _apply_preprocessing_window(self, data: np.ndarray) -> np.ndarray:
        """Apply highpass + notch to an analysis window without state side effects."""
        filtered = self._filter_window(self._b_hp, self._a_hp, data)
        return self._filter_window(self._b_notch, self._a_notch, filtered)

    def _apply_bandpass_window(self, data: np.ndarray) -> np.ndarray:
        """Apply QRS bandpass to an analysis window without state side effects."""
        return self._filter_window(self._b_bp, self._a_bp, data)

    def _apply_full_filter_window(self, data: np.ndarray) -> np.ndarray:
        """Apply full detection filter chain without mutating persistent filter state."""
        preprocessed = self._apply_preprocessing_window(data)
        return self._apply_bandpass_window(preprocessed)

    def _calc_shannon_entropy(self, rr_intervals: np.ndarray, n_bins: int = 20) -> float:
        """
        Calculate Shannon entropy of RR interval distribution.

        Higher entropy indicates more irregular rhythm (e.g., atrial fibrillation).
        Uses histogram-based probability estimation.

        Args:
            rr_intervals: Array of RR intervals in seconds
            n_bins: Number of histogram bins

        Returns:
            Shannon entropy value (nats)
        """
        if len(rr_intervals) < 3:
            return 0.0

        # Build histogram
        hist, _ = np.histogram(rr_intervals, bins=n_bins, density=True)
        # Normalize to probability
        total = hist.sum()
        if total == 0:
            return 0.0
        prob = hist / total

        # Shannon entropy: -Σ p*log(p), filter out zero bins
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log(prob)))

    def _calc_hrv(self, rr_intervals: np.ndarray) -> tuple[float, float, float]:
        """
        Calculate Heart Rate Variability metrics.

        Args:
            rr_intervals: Array of RR intervals in seconds

        Returns:
            Tuple of (SDNN, RMSSD, pNN50) in ms/ms/%
        """
        if len(rr_intervals) < 2:
            return 0.0, 0.0, 0.0

        rr_ms = rr_intervals * 1000  # Convert to ms

        # SDNN: standard deviation of RR intervals
        sdnn = float(np.std(rr_ms))

        # RMSSD: root mean square of successive differences
        diff = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(diff ** 2)))

        # pNN50: percentage of successive RR intervals differing > 50ms
        pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100)

        return sdnn, rmssd, pnn50

    def _estimate_confidence(self, base_confidence: float) -> float:
        """
        Dynamically estimate detection confidence.

        Factors:
          - R-peak detection consistency (expected vs actual count)
          - Signal quality (SNR from cached detection data)
          - Base confidence from classification rules

        Returns:
            Confidence value clamped to [0.0, 1.0]
        """
        if len(self._rr_intervals) < 2:
            return base_confidence

        rr_mean = np.mean(list(self._rr_intervals))

        # Factor 1: Detection consistency
        # Expected R-peaks in the window based on heart rate
        expected_count = self.window_size / rr_mean if rr_mean > 0 else 0
        actual_count = len(self._r_peaks)
        if expected_count > 0:
            consistency = min(actual_count / expected_count, 1.0)
        else:
            consistency = 0.5

        # Factor 2: Signal quality (SNR) using cached filtered/MA data.
        quality = 0.5
        if self._cached_filtered is not None and self._cached_filtered.size > 0:
            squared = self._cached_filtered ** 2
            peak_power = np.max(squared)
            mean_power = np.mean(squared)
            if mean_power > 0:
                snr = peak_power / mean_power
                # Normalize: SNR of 10 -> quality=1.0, SNR of 1 -> quality=0
                quality = min(max((snr - 1) / 9, 0), 1.0)

        # Weighted combination
        confidence = base_confidence * 0.6 + consistency * 0.2 + quality * 0.2
        return float(np.clip(confidence, 0.0, 1.0))

    def _analyze(self) -> DetectionResult:
        """Analyze collected data and classify arrhythmia."""
        # Calculate heart rate
        rr_intervals = np.array(self._rr_intervals)
        hr = 60.0 / rr_intervals.mean() if len(rr_intervals) > 0 else 0

        # RR interval statistics
        rr_mean = rr_intervals.mean() if len(rr_intervals) > 0 else 0
        rr_std = rr_intervals.std() if len(rr_intervals) > 0 else 0
        rr_cv = rr_std / rr_mean if rr_mean > 0 else 0

        # QRS width estimation
        qrs_width = self._estimate_qrs_width()

        # HRV metrics
        sdnn, rmssd, pnn50 = self._calc_hrv(rr_intervals)

        # Shannon entropy of RR intervals
        shannon_ent = self._calc_shannon_entropy(rr_intervals)

        # Classification
        arrhythmia_type, base_confidence, description = self._classify(
            hr, rr_cv, qrs_width, rr_intervals, shannon_ent
        )

        # Dynamic confidence based on signal quality and detection consistency
        confidence = self._estimate_confidence(base_confidence)

        return DetectionResult(
            arrhythmia_type=arrhythmia_type,
            heart_rate=hr,
            rr_interval_mean=rr_mean * 1000,  # Convert to ms
            rr_interval_std=rr_std * 1000,
            rr_cv=rr_cv,
            qrs_width_mean=qrs_width,
            confidence=confidence,
            description=description,
            sdnn=sdnn,
            rmssd=rmssd,
            pnn50=pnn50,
            shannon_entropy=shannon_ent,
        )

    def _estimate_qrs_width(self) -> float:
        """
        Return the median QRS width from recent beat-by-beat measurements.

        QRS widths are measured incrementally inside _detect_r_peaks as new
        R-peaks are discovered, avoiding the expensive recomputation of the
        full preprocessed window on every analysis call.
        """
        if not self._qrs_widths:
            return 80  # Default if measurement fails
        return float(np.median(self._qrs_widths))

    def _classify(
        self, hr: float, rr_cv: float, qrs_width: float,
        rr_intervals: np.ndarray, shannon_entropy: float = 0.0
    ) -> tuple[ArrhythmiaType, float, str]:
        """
        Classify arrhythmia based on features.

        Classification priority:
          1. PVC detection (isolated premature beats with compensatory pause)
          2. Normal sinus rhythm
          3. Ventricular tachycardia (regular tachy + wide QRS)
          4. Atrial fibrillation (sustained irregularity)
          5. Sinus tachycardia / Bradycardia
          6. Unknown

        PVC is checked first because isolated premature beats can cause
        elevated CV/entropy, which would otherwise trigger AF incorrectly.

        Args:
            hr: Heart rate in bpm
            rr_cv: Coefficient of variation of RR intervals
            qrs_width: Mean QRS width in ms
            rr_intervals: Array of RR intervals in seconds
            shannon_entropy: Shannon entropy of RR distribution

        Returns:
            Tuple of (arrhythmia_type, confidence, description)
        """
        # ── Step 1: PVC detection (before AF, since PVC causes irregularity) ──
        # PVC vs AF discrimination using local regularity analysis:
        #   PVC: isolated premature beats → MOST beats are regular,
        #        with occasional short-long pairs
        #   AF:  sustained irregularity → MOST beats are irregular
        #
        # Method: Compute CV of each sliding window of 3 consecutive RR intervals.
        #   - If majority of windows have low CV (< 0.10): mostly regular → PVC
        #   - If majority of windows have high CV (>= 0.10): mostly irregular → AF
        if len(rr_intervals) >= 5:
            # Local CV: CV of each sliding window of 3 consecutive RR intervals
            local_cvs = []
            for i in range(len(rr_intervals) - 2):
                window = rr_intervals[i:i+3]
                w_mean = np.mean(window)
                if w_mean > 0:
                    local_cvs.append(np.std(window) / w_mean)

            if len(local_cvs) > 0:
                local_cvs = np.array(local_cvs)
                # Fraction of windows that are "regular" (CV < 0.10)
                regular_fraction = np.sum(local_cvs < 0.10) / len(local_cvs)

                # PVC: mostly regular with some irregular windows
                # AF: mostly irregular
                if regular_fraction >= 0.5:
                    # Mostly regular rhythm → check for isolated PVCs
                    rr_diff_signed = np.diff(rr_intervals)
                    n_short_long = 0
                    for i in range(len(rr_diff_signed) - 1):
                        if rr_diff_signed[i] < -0.1 and rr_diff_signed[i + 1] > 0.1:
                            n_short_long += 1

                    if n_short_long >= 1:
                        has_wide_qrs = qrs_width >= 120
                        details = []
                        if has_wide_qrs:
                            details.append(f"QRS增宽({qrs_width:.0f}ms)")
                        else:
                            details.append(f"QRS正常({qrs_width:.0f}ms)")
                        conf = 0.8 if has_wide_qrs else 0.75
                        return ArrhythmiaType.PVC, conf, \
                            f"检测到早搏({', '.join(details)})"

        # ── Step 2: Regularity assessment ──
        # Shannon entropy supplements CV for borderline cases.
        # CV >= 0.15 is the primary indicator.
        # For CV in [0.12, 0.15), high entropy (>= 2.0) confirms irregularity.
        is_irregular = rr_cv >= 0.15 or (rr_cv >= 0.12 and shannon_entropy >= 2.0)

        # ── Step 3: Normal sinus rhythm ──
        if 60 <= hr <= 100 and not is_irregular and qrs_width < 120:
            return ArrhythmiaType.NORMAL, 0.9, "心率正常，节律规则，QRS波正常"

        # ── Step 4: Ventricular tachycardia (regular tachy + wide QRS) ──
        if hr > 100 and not is_irregular and qrs_width >= 120:
            return ArrhythmiaType.VENTRICULAR_TACHYCARDIA, 0.85, \
                f"心动过速(HR={hr:.0f})，节律规则，QRS波增宽({qrs_width:.0f}ms)"

        # ── Step 5: Atrial fibrillation (sustained irregularity) ──
        # AF requires SUSTAINED irregularity, not just a few outlier beats
        if is_irregular:
            # Additional AF check: irregularity should be sustained
            # (high entropy confirms it's not just 1-2 outlier beats)
            if hr > 100:
                return ArrhythmiaType.ATRIAL_FIBRILLATION, 0.8, \
                    f"心动过速(HR={hr:.0f})，节律不规则(CV={rr_cv*100:.1f}%, 熵={shannon_entropy:.2f})"
            else:
                return ArrhythmiaType.ATRIAL_FIBRILLATION, 0.75, \
                    f"节律不规则(CV={rr_cv*100:.1f}%, 熵={shannon_entropy:.2f})，疑似房颤"

        # ── Step 6: Sinus tachycardia ──
        if hr > 100:
            return ArrhythmiaType.TACHYCARDIA, 0.85, \
                f"心动过速(HR={hr:.0f})，节律规则"

        # ── Step 7: Bradycardia ──
        if hr < 60:
            return ArrhythmiaType.BRADYCARDIA, 0.85, \
                f"心动过缓(HR={hr:.0f})"

        return ArrhythmiaType.UNKNOWN, 0.5, "心律特征不明确"

    def get_current_hr(self) -> float:
        """Get current heart rate estimate."""
        if len(self._rr_intervals) == 0:
            return 0
        return 60.0 / np.mean(list(self._rr_intervals))

    def get_rr_intervals(self) -> list:
        """Get list of RR intervals."""
        return list(self._rr_intervals)

    def is_ready(self) -> bool:
        """Check if detector has enough data for analysis."""
        return not self._learning_phase and len(self._r_peaks) >= 3


def analyze_ecg_file(filename: str, fs: int = 500) -> DetectionResult:
    """
    Analyze an ECG CSV file for arrhythmia detection.

    Args:
        filename: Path to CSV file with timestamp,type,value format
        fs: Sampling frequency (default 500 Hz)

    Returns:
        DetectionResult with analysis results
    """
    import csv

    # Load data
    timestamps = []
    values = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3 and row[1].upper() == 'WAVE':
                try:
                    timestamps.append(float(row[0]))
                    values.append(float(row[2]))
                except ValueError:
                    continue

    if len(values) == 0:
        return DetectionResult(
            arrhythmia_type=ArrhythmiaType.UNKNOWN,
            heart_rate=0,
            rr_interval_mean=0,
            rr_interval_std=0,
            rr_cv=0,
            qrs_width_mean=0,
            confidence=0,
            description="No data"
        )

    # Create detector and process all samples
    detector = ArrhythmiaDetector(fs=fs)
    result = None

    for value in values:
        r = detector.feed_sample(value)
        if r is not None:
            result = r

    # If not enough data in streaming mode, analyze full file
    if result is None:
        # Direct analysis of full file
        timestamps = np.array(timestamps)
        values = np.array(values)

        # R-peak detection with improved preprocessing
        from scipy.signal import butter, iirnotch, filtfilt, find_peaks

        nyq = fs / 2

        # Highpass: remove baseline wander
        b_hp, a_hp = butter(2, 0.5 / nyq, btype='high')
        filtered = filtfilt(b_hp, a_hp, values.astype(float))

        # Notch: remove 50Hz power line interference
        b_notch, a_notch = iirnotch(50.0, 30.0, fs)
        filtered = filtfilt(b_notch, a_notch, filtered)

        # Bandpass: 5-15 Hz for QRS detection
        b, a = butter(4, [5/nyq, 15/nyq], btype='band')
        bp_filtered = filtfilt(b, a, filtered)

        # Differential filter (Pan-Tompkins)
        diff_kernel = np.array([-1, -2, 0, 2, 1]) / 8.0
        differentiated = np.convolve(bp_filtered, diff_kernel, mode='same')

        squared = differentiated ** 2
        window = int(0.15 * fs)
        ma = np.convolve(squared, np.ones(window)/window, mode='same')

        threshold = np.mean(ma) + 0.5 * np.std(ma)
        min_distance = int(0.3 * fs)
        peaks, _ = find_peaks(ma, height=threshold, distance=min_distance)

        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            hr = 60.0 / rr_intervals.mean()
            rr_mean = rr_intervals.mean()
            rr_std = rr_intervals.std()
            rr_cv = rr_std / rr_mean

            # QRS width estimation (half-amplitude method on preprocessed signal)
            qrs_widths = []
            for peak in peaks:
                start = max(0, peak - int(0.08 * fs))
                end = min(len(filtered), peak + int(0.08 * fs))
                segment = filtered[start:end]
                peak_idx = peak - start
                peak_amp = segment[peak_idx]
                min_amp = min(segment[0], segment[-1])
                peak_height = peak_amp - min_amp
                if peak_height > 10:
                    half_amp = min_amp + 0.5 * peak_height
                    # Search left for onset
                    onset = peak_idx
                    for j in range(peak_idx, -1, -1):
                        if segment[j] <= half_amp:
                            onset = j
                            break
                    # Search right for offset
                    offset = peak_idx
                    for j in range(peak_idx, len(segment)):
                        if segment[j] <= half_amp:
                            offset = j
                            break
                    width = (offset - onset) / fs * 1000
                    if 20 < width < 300:
                        qrs_widths.append(width)

            qrs_width = float(np.median(qrs_widths)) if qrs_widths else 80

            # HRV metrics
            sdnn, rmssd, pnn50 = detector._calc_hrv(rr_intervals)
            shannon_ent = detector._calc_shannon_entropy(rr_intervals)

            arrhythmia_type, confidence, description = detector._classify(
                hr, rr_cv, qrs_width, rr_intervals, shannon_ent
            )

            result = DetectionResult(
                arrhythmia_type=arrhythmia_type,
                heart_rate=hr,
                rr_interval_mean=rr_mean * 1000,
                rr_interval_std=rr_std * 1000,
                rr_cv=rr_cv,
                qrs_width_mean=qrs_width,
                confidence=confidence,
                description=description,
                sdnn=sdnn,
                rmssd=rmssd,
                pnn50=pnn50,
                shannon_entropy=shannon_ent,
            )

    return result


if __name__ == "__main__":
    # Test with sample data
    import sys
    import os

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default test files
        test_dir = os.path.join(os.path.dirname(__file__), 'dist')
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
            if files:
                filename = os.path.join(test_dir, files[0])
            else:
                print("No CSV files found in dist/")
                sys.exit(1)
        else:
            print("Usage: python arrhythmia_detector.py <ecg_file.csv>")
            sys.exit(1)

    print(f"Analyzing: {filename}")
    result = analyze_ecg_file(filename)

    print("\n=== Analysis Results ===")
    print(f"Arrhythmia Type: {result.arrhythmia_type.value}")
    print(f"Heart Rate: {result.heart_rate:.1f} bpm")
    print(f"RR Interval: {result.rr_interval_mean:.1f} ± {result.rr_interval_std:.1f} ms")
    print(f"RR CV: {result.rr_cv*100:.1f}%")
    print(f"QRS Width: {result.qrs_width_mean:.1f} ms")
    print(f"Confidence: {result.confidence*100:.0f}%")
    print(f"Description: {result.description}")
    print(f"\n--- HRV Metrics ---")
    print(f"SDNN: {result.sdnn:.1f} ms")
    print(f"RMSSD: {result.rmssd:.1f} ms")
    print(f"pNN50: {result.pnn50:.1f}%")
    print(f"Shannon Entropy: {result.shannon_entropy:.3f}")
