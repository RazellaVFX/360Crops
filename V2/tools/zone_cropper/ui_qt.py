from __future__ import annotations

import json
import queue
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, QPoint, QRect, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOptionSlider,
)
from qfluentwidgets import SpinBox, Theme, setTheme
from qfluentwidgets.common.config import isDarkTheme
from qfluentwidgets.common.style_sheet import themeColor

from .config import ProgressInfo, SequenceJobConfig, ZoneLayoutConfig
from .engine import format_eta, list_sequence_frames, process_sequence, render_layout_overlay
from .layout import generate_zone_layout
from .ui_zone_cropper_qt import Ui_MainWindow


class CacheTimelineSlider(QSlider):
    def __init__(self, parent=None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setStyleSheet(
            "QSlider::handle:horizontal { background: transparent; border: none; }"
        )
        self._total_frames: int = 0
        self._range_start: int = 0
        self._range_count: int = 0
        self._cached_indices: set[int] = set()
        self._bar_rect = QRect()
        self._bucket_colors: list[QColor] = []
        self._bucket_width: float = 1.0
        self._frames_per_bucket: int = 1

    def set_frame_range(self, range_start: int, range_count: int) -> None:
        self._range_start = max(0, int(range_start))
        self._range_count = max(0, int(range_count))
        self._total_frames = self._range_count
        self.setRange(0, max(0, self._range_count - 1))
        self._rebuild_bucket_colors()
        self.update()

    def set_total_frames(self, total_frames: int) -> None:
        self.set_frame_range(0, total_frames)

    def clear_cache(self) -> None:
        self._cached_indices.clear()
        self._rebuild_bucket_colors()
        self.update()

    def mark_cached(self, absolute_index: int) -> None:
        idx = int(absolute_index)
        if idx < self._range_start or idx >= self._range_start + self._range_count:
            return
        if idx in self._cached_indices:
            return
        self._cached_indices.add(idx)
        if not self._bucket_colors:
            self._rebuild_bucket_colors()
        else:
            self._update_bucket_for_frame(idx)
        self.update()

    def _bucket_color_for_range(self, rel_start: int, rel_end: int) -> QColor:
        cached = self._cached_indices
        blue = QColor(70, 130, 255, 180)
        green = QColor(60, 210, 90, 200)
        if rel_end <= rel_start:
            return blue
        cached_count = sum(
            1
            for rel in range(rel_start, rel_end)
            if (self._range_start + rel) in cached
        )
        ratio = cached_count / max(1, rel_end - rel_start)
        r = int(blue.red() + (green.red() - blue.red()) * ratio)
        g = int(blue.green() + (green.green() - blue.green()) * ratio)
        b = int(blue.blue() + (green.blue() - blue.blue()) * ratio)
        a = int(blue.alpha() + (green.alpha() - blue.alpha()) * ratio)
        return QColor(r, g, b, a)

    def _update_bucket_for_frame(self, absolute_index: int) -> None:
        if not self._bucket_colors or self._bar_rect.width() <= 0:
            return
        rel = absolute_index - self._range_start
        if rel < 0 or rel >= self._range_count:
            return
        bucket_count = len(self._bucket_colors)
        bucket = min(bucket_count - 1, rel // self._frames_per_bucket)
        rel_start = bucket * self._frames_per_bucket
        rel_end = min(self._range_count, rel_start + self._frames_per_bucket)
        self._bucket_colors[bucket] = self._bucket_color_for_range(rel_start, rel_end)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild_bucket_colors()

    def _rebuild_bucket_colors(self) -> None:
        self._bucket_colors = []
        self._bar_rect = QRect()
        if self._range_count <= 0:
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self
        )
        if groove.width() <= 1 or groove.height() <= 1:
            return

        bar = groove.adjusted(1, max(1, groove.height() // 4), -1, -max(1, groove.height() // 4))
        if bar.width() <= 0 or bar.height() <= 0:
            return

        self._bar_rect = bar
        frame_count = self._range_count
        if frame_count <= 0:
            return
        bucket_count = max(1, min(frame_count, bar.width()))
        frames_per_bucket = (frame_count + bucket_count - 1) // bucket_count
        self._bucket_width = max(1.0, bar.width() / bucket_count)
        self._frames_per_bucket = frames_per_bucket

        colors: list[QColor] = []
        for bucket in range(bucket_count):
            start = bucket * frames_per_bucket
            end = min(frame_count, start + frames_per_bucket)
            colors.append(self._bucket_color_for_range(start, end))
        self._bucket_colors = colors

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        if self._bucket_colors and self._bar_rect.width() > 0:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            bar = self._bar_rect
            bw = self._bucket_width
            for bucket, color in enumerate(self._bucket_colors):
                x0 = int(bar.x() + bucket * bw)
                x1 = int(bar.x() + (bucket + 1) * bw)
                if x1 <= x0:
                    x1 = x0 + 1
                painter.fillRect(x0, bar.y(), x1 - x0, bar.height(), color)
        self._paint_slider_handle(painter)

    def _paint_slider_handle(self, painter: QPainter) -> None:
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
        )
        if handle_rect.width() <= 0 or handle_rect.height() <= 0:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QColor(0, 0, 0, 180 if isDarkTheme() else 50))
        painter.setBrush(QColor(69, 69, 69) if isDarkTheme() else QColor(255, 255, 255))
        painter.drawEllipse(handle_rect.adjusted(1, 1, -1, -1))

        inner = themeColor()
        inner.setAlpha(255)
        inner_radius = 4 if self.isSliderDown() else 6
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(inner)
        center = handle_rect.center()
        painter.drawEllipse(QPoint(center.x(), center.y()), inner_radius, inner_radius)


class ZoneCropperQtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._install_cache_timeline_slider()
        self._install_frame_range_timeline_row()

        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self.event_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.frames_cache: list[Path] = []
        self._frames_cache_input_dir: Optional[Path] = None
        self.preview_image: Optional[Image.Image] = None
        self.preview_pixmap: Optional[QPixmap] = None
        self.preview_overlay_rgba: Optional[Image.Image] = None
        self.preview_frame_cache: "OrderedDict[Path, Image.Image]" = OrderedDict()
        self.preview_composited_pixmap_cache: "OrderedDict[int, QPixmap]" = OrderedDict()
        self.preview_cached_indices: set[int] = set()
        self.preview_zone_count: int = 0
        self._syncing_preview_controls: bool = False
        self._syncing_frame_range: bool = False
        self._preview_scrubbing: bool = False
        self._pending_preview_idx: Optional[int] = None
        self._live_preview_timer: Optional[QTimer] = None
        self._preview_scrub_timer: Optional[QTimer] = None
        self._state_save_timer: Optional[QTimer] = None
        self._speed_samples: deque[tuple[float, int]] = deque()
        self._last_speed_update_ts: Optional[float] = None
        self._last_crops_per_sec: Optional[float] = None
        self._last_progress_max_steps: int = 0
        self._last_preview_target_size: Optional[tuple[int, int]] = None
        self._last_preview_source_key: Optional[int] = None
        self._last_preview_smooth: Optional[bool] = None

        self._step_mode_widgets = [
            self.ui.yawStepSpin,
            self.ui.pitchStepSpin,
            self.ui.maxPitchStepSpin,
            self.ui.includePolesStepCheck,
            self.ui.allowPoleCenterStepCheck,
        ]
        self._count_mode_widgets = [
            self.ui.targetZonesSpin,
            self.ui.overlapSpin,
            self.ui.maxPitchCountSpin,
            self.ui.includePolesCountCheck,
            self.ui.allowPoleCenterCountCheck,
        ]

        self._configure_widgets()
        self._wire_signals()
        loaded = self._load_state()
        if not loaded:
            self._reset_interface_layout(notify=False)
        self._update_mode_block_state()
        QTimer.singleShot(0, self._preview_if_input_ready)
        self._append_log("Ready.")

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_events)
        self.poll_timer.start(100)

        self._state_save_timer = QTimer(self)
        self._state_save_timer.setSingleShot(True)
        self._state_save_timer.timeout.connect(self._save_state_now)

        self._live_preview_timer = QTimer(self)
        self._live_preview_timer.setSingleShot(True)
        self._live_preview_timer.timeout.connect(lambda: self._preview_layout_internal(False, False))

        self._preview_scrub_timer = QTimer(self)
        self._preview_scrub_timer.setSingleShot(True)
        self._preview_scrub_timer.timeout.connect(self._apply_pending_preview_frame)

    def _install_cache_timeline_slider(self) -> None:
        old_slider = self.ui.previewTimelineSlider
        parent = old_slider.parentWidget()
        new_slider = CacheTimelineSlider(parent)
        new_slider.setObjectName(old_slider.objectName())
        new_slider.setRange(old_slider.minimum(), old_slider.maximum())
        new_slider.setValue(old_slider.value())
        new_slider.setEnabled(old_slider.isEnabled())
        new_slider.setToolTip(old_slider.toolTip())
        layout = self.ui.layoutPreview
        layout.replaceWidget(old_slider, new_slider)
        old_slider.deleteLater()
        self.ui.previewTimelineSlider = new_slider

    def _install_frame_range_timeline_row(self) -> None:
        layout = self.ui.layoutPreview
        slider = self.ui.previewTimelineSlider
        slider_idx = layout.indexOf(slider)
        layout.removeWidget(slider)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        start_spin = SpinBox(self.ui.previewBlock)
        start_spin.setObjectName("frameRangeStartSpin")
        start_spin.setMinimumWidth(80)
        start_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)

        end_spin = SpinBox(self.ui.previewBlock)
        end_spin.setObjectName("frameRangeEndSpin")
        end_spin.setMinimumWidth(80)
        end_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)

        row.addWidget(start_spin)
        row.addWidget(slider, 1)
        row.addWidget(end_spin)
        layout.insertLayout(slider_idx, row)

        self.ui.frameRangeStartSpin = start_spin
        self.ui.frameRangeEndSpin = end_spin

    def _sequence_last_index(self) -> int:
        return max(0, len(self.frames_cache) - 1)

    def _frame_range(self) -> tuple[int, int]:
        start = int(self.ui.frameRangeStartSpin.value())
        end = int(self.ui.frameRangeEndSpin.value())
        return start, end

    def _configure_frame_range_spins(self) -> None:
        last_idx = self._sequence_last_index()
        has_frames = bool(self.frames_cache)
        max_idx = last_idx if has_frames else 0
        prev_end_max = self.ui.frameRangeEndSpin.maximum()

        self._syncing_frame_range = True
        try:
            self.ui.frameRangeStartSpin.blockSignals(True)
            self.ui.frameRangeEndSpin.blockSignals(True)
            self.ui.frameRangeStartSpin.setRange(0, max_idx)
            self.ui.frameRangeEndSpin.setRange(0, max_idx)
            if has_frames:
                start = max(0, min(int(self.ui.frameRangeStartSpin.value()), max_idx))
                end = max(start, min(int(self.ui.frameRangeEndSpin.value()), max_idx))
                if end == 0 and start == 0 and max_idx > 0 and prev_end_max == 0:
                    end = max_idx
            else:
                start = 0
                end = 0
            self.ui.frameRangeStartSpin.setValue(start)
            self.ui.frameRangeEndSpin.setValue(end)
            self.ui.frameRangeStartSpin.setMaximum(end)
            self.ui.frameRangeEndSpin.setMinimum(start)
        finally:
            self.ui.frameRangeStartSpin.blockSignals(False)
            self.ui.frameRangeEndSpin.blockSignals(False)
            self._syncing_frame_range = False

    def _apply_frame_range(self) -> None:
        self._configure_frame_range_spins()
        self._sync_preview_controls()
        self._schedule_state_save()

    def _on_frame_range_start_changed(self, value: int) -> None:
        if self._syncing_frame_range:
            return
        self._syncing_frame_range = True
        try:
            end = int(self.ui.frameRangeEndSpin.value())
            if value > end:
                self.ui.frameRangeEndSpin.setValue(int(value))
            self.ui.frameRangeEndSpin.setMinimum(int(value))
            self.ui.frameRangeStartSpin.setMaximum(int(self.ui.frameRangeEndSpin.value()))
        finally:
            self._syncing_frame_range = False
        self._apply_frame_range()

    def _on_frame_range_end_changed(self, value: int) -> None:
        if self._syncing_frame_range:
            return
        self._syncing_frame_range = True
        try:
            start = int(self.ui.frameRangeStartSpin.value())
            if value < start:
                self.ui.frameRangeStartSpin.setValue(int(value))
            self.ui.frameRangeStartSpin.setMaximum(int(value))
            self.ui.frameRangeEndSpin.setMinimum(int(self.ui.frameRangeStartSpin.value()))
        finally:
            self._syncing_frame_range = False
        self._apply_frame_range()

    def _configure_widgets(self) -> None:
        # Ensure block 5 can grow horizontally without artificial cap.
        self.ui.renderBlock.setMaximumWidth(16777215)
        self.ui.renderBlock.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.ui.gridImport.setColumnStretch(1, 1)
        self.ui.gridImport.setColumnStretch(4, 1)
        self.ui.gridCrop.setColumnStretch(6, 1)
        self.ui.gridStep.setColumnStretch(7, 1)
        self.ui.gridCount.setColumnStretch(7, 1)
        self.ui.gridRender.setColumnStretch(8, 1)
        self.ui.gridRender.setRowStretch(5, 1)
        self.ui.mainSplitter.setStretchFactor(0, 0)
        self.ui.mainSplitter.setStretchFactor(1, 0)
        self.ui.mainSplitter.setStretchFactor(2, 0)
        self.ui.mainSplitter.setStretchFactor(3, 2)
        self.ui.mainSplitter.setStretchFactor(4, 3)
        self.ui.modeRowSplitter.setStretchFactor(0, 1)
        self.ui.modeRowSplitter.setStretchFactor(1, 1)

        self.ui.cropWSpin.setRange(1, 16384)
        self.ui.cropWSpin.setValue(640)
        self.ui.cropWSpin.setSingleStep(5)
        self.ui.cropHSpin.setRange(1, 16384)
        self.ui.cropHSpin.setValue(640)
        self.ui.cropHSpin.setSingleStep(5)
        self.ui.fovXSpin.setRange(1.0, 179.0)
        self.ui.fovXSpin.setValue(90.0)
        self.ui.fovXSpin.setDecimals(2)
        self.ui.fovXSpin.setSingleStep(5.0)

        self.ui.yawStepSpin.setRange(1.0, 180.0)
        self.ui.yawStepSpin.setValue(30.0)
        self.ui.yawStepSpin.setSingleStep(5.0)
        self.ui.pitchStepSpin.setRange(1.0, 180.0)
        self.ui.pitchStepSpin.setValue(20.0)
        self.ui.pitchStepSpin.setSingleStep(5.0)
        self.ui.maxPitchStepSpin.setRange(1.0, 90.0)
        self.ui.maxPitchStepSpin.setValue(85.0)
        self.ui.maxPitchStepSpin.setSingleStep(5.0)
        self.ui.maxPitchCountSpin.setRange(1.0, 90.0)
        self.ui.maxPitchCountSpin.setValue(85.0)
        self.ui.maxPitchCountSpin.setSingleStep(5.0)
        self.ui.targetZonesSpin.setRange(1, 2000)
        self.ui.targetZonesSpin.setValue(24)
        self.ui.targetZonesSpin.setSingleStep(1)
        self.ui.overlapSpin.setRange(0.0, 0.999)
        self.ui.overlapSpin.setDecimals(3)
        self.ui.overlapSpin.setSingleStep(0.05)
        self.ui.overlapSpin.setValue(0.2)

        self.ui.previewIndexSpin.setRange(0, 99999999)
        self.ui.previewIndexSpin.setValue(0)
        self.ui.frameRangeStartSpin.setRange(0, 0)
        self.ui.frameRangeEndSpin.setRange(0, 0)
        self.ui.frameRangeStartSpin.setSingleStep(1)
        self.ui.frameRangeEndSpin.setSingleStep(1)
        self.ui.frameRangeStartSpin.setValue(0)
        self.ui.frameRangeEndSpin.setValue(0)
        self.ui.previewTimelineSlider.setRange(0, 0)
        self.ui.previewTimelineSlider.setValue(0)
        self.ui.previewTimelineSlider.set_frame_range(0, 0)

        self.ui.chunkSizeSpin.setRange(1, 2048)
        self.ui.chunkSizeSpin.setValue(8)
        self.ui.interpModeCombo.addItems(["nearest", "bilinear", "bicubic"])
        self.ui.interpModeCombo.setCurrentText("bilinear")
        self.ui.imageExtCombo.clear()
        self.ui.imageExtCombo.addItems(["jpg", "png"])
        self.ui.imageExtCombo.setCurrentText("jpg")
        self.ui.imageExtCombo.setMinimumWidth(100)
        self.ui.compressionSpin.setRange(0.0, 1.0)
        self.ui.compressionSpin.setDecimals(3)
        self.ui.compressionSpin.setSingleStep(0.05)
        self.ui.compressionSpin.setValue(0.5)
        self.ui.writerThreadsSpin.setRange(1, 512)
        self.ui.writerThreadsSpin.setValue(4)
        self.ui.gpuZoneBatchSpin.setRange(1, 512)
        self.ui.gpuZoneBatchSpin.setValue(8)
        self.ui.progressBar.setRange(0, 1)
        self.ui.progressBar.setValue(0)
        self.ui.logTextEdit.setReadOnly(True)
        self.ui.stopButton.setEnabled(False)

        self.ui.stepModeRadio.setVisible(True)
        self.ui.countModeRadio.setVisible(True)
        self.mode_buttons = QButtonGroup(self)
        self.mode_buttons.setExclusive(True)
        self.mode_buttons.addButton(self.ui.stepModeRadio)
        self.mode_buttons.addButton(self.ui.countModeRadio)
        self.encoder_buttons = QButtonGroup(self)
        self.encoder_buttons.setExclusive(True)
        self.encoder_buttons.addButton(self.ui.encoderGpuRadio)
        self.encoder_buttons.addButton(self.ui.encoderImagecodecsRadio)
        self.encoder_buttons.addButton(self.ui.encoderClassicRadio)
        self.ui.countModeRadio.setChecked(True)
        self.ui.encoderGpuRadio.setChecked(True)
        self.ui.useComputeGpuCheck.setChecked(True)
        self.ui.includePolesStepCheck.setChecked(True)
        self.ui.includePolesCountCheck.setChecked(True)
        self._sync_encoder_controls_for_ext()
        self._apply_fixed_field_widths()
        self._apply_block_minimum_sizes()

    def _apply_fixed_field_widths(self) -> None:
        controls = [
            self.ui.inputDirEdit,
            self.ui.outputDirEdit,
            self.ui.previewIndexSpin,
            self.ui.cropWSpin,
            self.ui.cropHSpin,
            self.ui.fovXSpin,
            self.ui.yawStepSpin,
            self.ui.pitchStepSpin,
            self.ui.maxPitchStepSpin,
            self.ui.targetZonesSpin,
            self.ui.overlapSpin,
            self.ui.maxPitchCountSpin,
            self.ui.interpModeCombo,
            self.ui.imageExtCombo,
            self.ui.chunkSizeSpin,
            self.ui.compressionSpin,
            self.ui.writerThreadsSpin,
            self.ui.gpuZoneBatchSpin,
            self.ui.browseInputButton,
            self.ui.browseOutputButton,
            self.ui.loadSequenceButton,
            self.ui.previewButton,
            self.ui.runBatchButton,
            self.ui.stopButton,
            self.ui.resetButton,
            self.ui.resetInterfaceButton,
            self.ui.saveDefaultButton,
        ]

        # Keep native control heights/styles from qfluentwidgets and only
        # enforce minimum width to avoid clipped content in tight layouts.
        for widget in controls:
            widget.ensurePolished()
            hinted_w = max(widget.sizeHint().width(), widget.minimumSizeHint().width())
            widget.setMinimumWidth(hinted_w)

    def _apply_block_minimum_sizes(self) -> None:
        # Qt best practice: rely on minimumSizeHint()/size policies so controls
        # remain usable and do not clip when the window is shrunk.
        blocks = [
            self.ui.importBlock,
            self.ui.cropBlock,
            self.ui.modeRowSplitter,
            self.ui.previewBlock,
            self.ui.renderBlock,
        ]

        for block in blocks:
            block.ensurePolished()
            hinted_h = max(block.minimumSizeHint().height(), block.minimumHeight())
            block.setMinimumHeight(hinted_h)

        # Mode splitter should fit both sub-blocks comfortably.
        mode_h = max(
            self.ui.modeRowSplitter.minimumHeight(),
            self.ui.stepBlock.minimumSizeHint().height(),
            self.ui.countBlock.minimumSizeHint().height(),
        )
        self.ui.modeRowSplitter.setMinimumHeight(mode_h)

        # Sub-block minimum widths based on real content.
        self.ui.stepBlock.setMinimumWidth(max(self.ui.stepBlock.minimumWidth(), self.ui.stepBlock.minimumSizeHint().width()))
        self.ui.countBlock.setMinimumWidth(
            max(self.ui.countBlock.minimumWidth(), self.ui.countBlock.minimumSizeHint().width())
        )

        handle_w = max(1, int(self.ui.mainSplitter.handleWidth()))
        total_min_h = sum(block.minimumHeight() for block in blocks) + handle_w * (len(blocks) - 1)
        total_min_w = max(
            self.ui.importBlock.minimumSizeHint().width(),
            self.ui.cropBlock.minimumSizeHint().width(),
            self.ui.modeRowSplitter.minimumSizeHint().width(),
            self.ui.previewBlock.minimumSizeHint().width(),
            self.ui.renderBlock.minimumSizeHint().width(),
        )

        # Include layout/frame slack so edges are not clipped.
        self.setMinimumSize(total_min_w + 24, total_min_h + 24)

    def _wire_signals(self) -> None:
        self.ui.browseInputButton.clicked.connect(self._browse_input)
        self.ui.browseOutputButton.clicked.connect(self._browse_output)
        self.ui.loadSequenceButton.clicked.connect(self._load_sequence)
        self.ui.previewButton.clicked.connect(self._preview_layout)
        self.ui.runBatchButton.clicked.connect(self._run_batch)
        self.ui.stopButton.clicked.connect(self._stop_batch)
        self.ui.resetButton.clicked.connect(self._reset_render_settings)
        self.ui.resetInterfaceButton.clicked.connect(self._reset_interface_layout)
        self.ui.saveDefaultButton.clicked.connect(self._save_current_layout_as_default)
        self.ui.stepModeRadio.toggled.connect(self._update_mode_block_state)
        self.ui.countModeRadio.toggled.connect(self._update_mode_block_state)
        self.ui.imageExtCombo.currentTextChanged.connect(self._on_image_ext_changed)
        self.ui.previewIndexSpin.valueChanged.connect(self._on_preview_index_spin_changed)
        self.ui.previewTimelineSlider.valueChanged.connect(self._on_preview_timeline_changed)
        self.ui.previewTimelineSlider.sliderPressed.connect(self._on_preview_timeline_pressed)
        self.ui.previewTimelineSlider.sliderReleased.connect(self._on_preview_timeline_released)
        self.ui.frameRangeStartSpin.valueChanged.connect(self._on_frame_range_start_changed)
        self.ui.frameRangeEndSpin.valueChanged.connect(self._on_frame_range_end_changed)
        self.ui.previewLabel.installEventFilter(self)
        self.ui.inputDirEdit.editingFinished.connect(self._on_input_dir_editing_finished)

        state_signals = (
            self.ui.inputDirEdit.textChanged,
            self.ui.outputDirEdit.textChanged,
            self.ui.cropWSpin.valueChanged,
            self.ui.cropHSpin.valueChanged,
            self.ui.fovXSpin.valueChanged,
            self.ui.yawStepSpin.valueChanged,
            self.ui.pitchStepSpin.valueChanged,
            self.ui.maxPitchStepSpin.valueChanged,
            self.ui.maxPitchCountSpin.valueChanged,
            self.ui.targetZonesSpin.valueChanged,
            self.ui.overlapSpin.valueChanged,
            self.ui.previewIndexSpin.valueChanged,
            self.ui.frameRangeStartSpin.valueChanged,
            self.ui.frameRangeEndSpin.valueChanged,
            self.ui.showBordersCheck.stateChanged,
            self.ui.livePreviewCheck.stateChanged,
            self.ui.interpModeCombo.currentTextChanged,
            self.ui.imageExtCombo.currentTextChanged,
            self.ui.encoderGpuRadio.toggled,
            self.ui.encoderImagecodecsRadio.toggled,
            self.ui.encoderClassicRadio.toggled,
            self.ui.useComputeGpuCheck.stateChanged,
            self.ui.chunkSizeSpin.valueChanged,
            self.ui.compressionSpin.valueChanged,
            self.ui.writerThreadsSpin.valueChanged,
            self.ui.gpuZoneBatchSpin.valueChanged,
            self.ui.includePolesStepCheck.stateChanged,
            self.ui.allowPoleCenterStepCheck.stateChanged,
            self.ui.includePolesCountCheck.stateChanged,
            self.ui.allowPoleCenterCountCheck.stateChanged,
        )
        for signal in state_signals:
            signal.connect(self._schedule_state_save)

        live_preview_signals = (
            self.ui.inputDirEdit.textChanged,
            self.ui.outputDirEdit.textChanged,
            self.ui.cropWSpin.valueChanged,
            self.ui.cropHSpin.valueChanged,
            self.ui.fovXSpin.valueChanged,
            self.ui.yawStepSpin.valueChanged,
            self.ui.pitchStepSpin.valueChanged,
            self.ui.maxPitchStepSpin.valueChanged,
            self.ui.maxPitchCountSpin.valueChanged,
            self.ui.targetZonesSpin.valueChanged,
            self.ui.overlapSpin.valueChanged,
            self.ui.showBordersCheck.stateChanged,
            self.ui.livePreviewCheck.stateChanged,
            self.ui.includePolesStepCheck.stateChanged,
            self.ui.allowPoleCenterStepCheck.stateChanged,
            self.ui.includePolesCountCheck.stateChanged,
            self.ui.allowPoleCenterCountCheck.stateChanged,
        )
        for signal in live_preview_signals:
            signal.connect(self._schedule_live_preview)

        self.ui.mainSplitter.splitterMoved.connect(lambda *_: self._schedule_state_save())
        self.ui.modeRowSplitter.splitterMoved.connect(lambda *_: self._schedule_state_save())

    def _default_main_splitter_sizes(self) -> list[int]:
        splitter = self.ui.mainSplitter
        count = splitter.count()
        widgets = [splitter.widget(i) for i in range(count)]
        min_heights: list[int] = []
        for w in widgets:
            if w is None:
                min_heights.append(40)
                continue
            w.ensurePolished()
            min_heights.append(max(w.minimumHeight(), w.minimumSizeHint().height()))

        handle_w = max(1, splitter.handleWidth())
        occupied_by_handles = handle_w * max(0, count - 1)
        available = splitter.height() - occupied_by_handles
        if available <= 0:
            available = sum(min_heights)

        sizes = list(min_heights)
        extra = max(0, available - sum(min_heights))
        if extra > 0 and len(sizes) >= 5:
            preview_extra = int(extra * 0.45)
            render_extra = extra - preview_extra
            sizes[3] += preview_extra
            sizes[4] += render_extra
        elif extra > 0 and sizes:
            sizes[-1] += extra
        return sizes

    def _default_layout_path(self) -> Path:
        return Path(__file__).resolve().with_name("ui_defaults_qt.json")

    def _load_saved_default_layout(self) -> Optional[dict[str, list[int]]]:
        p = self._default_layout_path()
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        main = payload.get("main")
        mode = payload.get("mode")
        if not isinstance(main, list) or not isinstance(mode, list):
            return None
        out = {"main": [int(x) for x in main], "mode": [int(x) for x in mode]}
        window_size = payload.get("window_size")
        if isinstance(window_size, list) and len(window_size) >= 2:
            out["window_size"] = [int(window_size[0]), int(window_size[1])]
        return out

    def _save_current_layout_as_default(self) -> None:
        payload = self._splitter_sizes()
        try:
            self._default_layout_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._append_log("Saved current interface layout as defaults.")
            QMessageBox.information(self, "Save as default", "Default interface layout saved.")
        except Exception as exc:
            QMessageBox.critical(self, "Save as default", f"Failed to save defaults:\n{exc}")

    def _reset_interface_layout(self, notify: bool = True) -> None:
        saved_defaults = self._load_saved_default_layout()
        if saved_defaults is not None:
            self.ui.mainSplitter.setSizes(saved_defaults["main"])
            self.ui.modeRowSplitter.setSizes(saved_defaults["mode"])
            ws = saved_defaults.get("window_size")
            if isinstance(ws, list) and len(ws) >= 2:
                self.resize(int(ws[0]), int(ws[1]))
        else:
            self.ui.mainSplitter.setSizes(self._default_main_splitter_sizes())
            self.ui.modeRowSplitter.setSizes([1, 1])
        self._schedule_state_save()
        self._append_log("Interface layout reset to defaults.")
        if notify:
            QMessageBox.information(self, "Reset interface", "Interface block sizes reset to defaults.")

    def eventFilter(self, obj, event):  # type: ignore[override]
        if (
            obj is self.ui.previewLabel
            and self.preview_pixmap is not None
            and event.type() == QEvent.Type.Resize
        ):
            self._refresh_preview_display()
        return super().eventFilter(obj, event)

    def _state_path(self) -> Path:
        return Path(__file__).resolve().with_name("ui_state_qt.json")

    def _schedule_state_save(self) -> None:
        if self._state_save_timer is not None:
            self._state_save_timer.start(250)

    def _splitter_sizes(self) -> dict[str, list[int]]:
        return {
            "main": list(self.ui.mainSplitter.sizes()),
            "mode": list(self.ui.modeRowSplitter.sizes()),
            "window_size": [int(self.width()), int(self.height())],
        }

    def _save_state_now(self) -> None:
        backend = self._selected_encoder_backend()
        use_imagecodecs = backend == "imagecodecs"
        payload = {
            "input_dir": self.ui.inputDirEdit.text(),
            "output_dir": self.ui.outputDirEdit.text(),
            "crop_w": self.ui.cropWSpin.value(),
            "crop_h": self.ui.cropHSpin.value(),
            "fov_x": self.ui.fovXSpin.value(),
            "mode": "step_mode" if self.ui.stepModeRadio.isChecked() else "count_mode",
            "target_zones": self.ui.targetZonesSpin.value(),
            "overlap": self.ui.overlapSpin.value(),
            "yaw_step": self.ui.yawStepSpin.value(),
            "pitch_step": self.ui.pitchStepSpin.value(),
            "max_pitch_step": self.ui.maxPitchStepSpin.value(),
            "max_pitch_count": self.ui.maxPitchCountSpin.value(),
            "chunk_size": self.ui.chunkSizeSpin.value(),
            "image_ext": self._normalized_image_ext(),
            "compression_strength": self.ui.compressionSpin.value(),
            "encoder_backend": backend,
            "use_imagecodecs": use_imagecodecs,
            "writer_threads": self.ui.writerThreadsSpin.value(),
            "gpu_zone_batch": self.ui.gpuZoneBatchSpin.value(),
            "preview_index": self.ui.previewIndexSpin.value(),
            "frame_range_start": self.ui.frameRangeStartSpin.value(),
            "frame_range_end": self.ui.frameRangeEndSpin.value(),
            "include_poles_step": self.ui.includePolesStepCheck.isChecked(),
            "allow_pole_center_step": self.ui.allowPoleCenterStepCheck.isChecked(),
            "include_poles_count": self.ui.includePolesCountCheck.isChecked(),
            "allow_pole_center_count": self.ui.allowPoleCenterCountCheck.isChecked(),
            "use_torch_gpu": self.ui.useComputeGpuCheck.isChecked(),
            "show_borders": self.ui.showBordersCheck.isChecked(),
            "live_preview": self.ui.livePreviewCheck.isChecked(),
            "interp_mode": self.ui.interpModeCombo.currentText(),
            "splitter_sizes": self._splitter_sizes(),
        }
        try:
            self._state_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_state(self) -> bool:
        p = self._state_path()
        if not p.exists():
            return False
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False

        self.ui.inputDirEdit.setText(str(payload.get("input_dir", self.ui.inputDirEdit.text())))
        self.ui.outputDirEdit.setText(str(payload.get("output_dir", self.ui.outputDirEdit.text())))
        self.ui.cropWSpin.setValue(int(payload.get("crop_w", self.ui.cropWSpin.value())))
        self.ui.cropHSpin.setValue(int(payload.get("crop_h", self.ui.cropHSpin.value())))
        self.ui.fovXSpin.setValue(float(payload.get("fov_x", self.ui.fovXSpin.value())))
        mode = str(payload.get("mode", "count_mode"))
        self.ui.stepModeRadio.setChecked(mode == "step_mode")
        self.ui.countModeRadio.setChecked(mode != "step_mode")
        self.ui.targetZonesSpin.setValue(int(payload.get("target_zones", self.ui.targetZonesSpin.value())))
        self.ui.overlapSpin.setValue(float(payload.get("overlap", self.ui.overlapSpin.value())))
        self.ui.yawStepSpin.setValue(float(payload.get("yaw_step", self.ui.yawStepSpin.value())))
        self.ui.pitchStepSpin.setValue(float(payload.get("pitch_step", self.ui.pitchStepSpin.value())))
        self.ui.maxPitchStepSpin.setValue(float(payload.get("max_pitch_step", self.ui.maxPitchStepSpin.value())))
        self.ui.maxPitchCountSpin.setValue(float(payload.get("max_pitch_count", self.ui.maxPitchCountSpin.value())))
        self.ui.chunkSizeSpin.setValue(int(payload.get("chunk_size", self.ui.chunkSizeSpin.value())))
        saved_ext = str(payload.get("image_ext", self._normalized_image_ext())).strip().lower()
        if saved_ext.startswith("."):
            saved_ext = saved_ext[1:]
        if saved_ext not in {"jpg", "png"}:
            saved_ext = "jpg"
        self.ui.imageExtCombo.setCurrentText(saved_ext)
        self.ui.compressionSpin.setValue(float(payload.get("compression_strength", self.ui.compressionSpin.value())))
        saved_backend = str(payload.get("encoder_backend", "")).strip().lower()
        if saved_backend not in {"gpu", "imagecodecs", "classic"}:
            if bool(payload.get("use_imagecodecs", False)):
                saved_backend = "imagecodecs"
            else:
                saved_backend = "classic"
        self._set_encoder_backend(saved_backend)
        self.ui.useComputeGpuCheck.setChecked(bool(payload.get("use_torch_gpu", self.ui.useComputeGpuCheck.isChecked())))
        self.ui.writerThreadsSpin.setValue(int(payload.get("writer_threads", self.ui.writerThreadsSpin.value())))
        self.ui.gpuZoneBatchSpin.setValue(int(payload.get("gpu_zone_batch", self.ui.gpuZoneBatchSpin.value())))
        self.ui.previewIndexSpin.setValue(int(payload.get("preview_index", self.ui.previewIndexSpin.value())))
        if "frame_range_start" in payload or "frame_range_end" in payload:
            self.ui.frameRangeStartSpin.setValue(int(payload.get("frame_range_start", self.ui.frameRangeStartSpin.value())))
            self.ui.frameRangeEndSpin.setValue(int(payload.get("frame_range_end", self.ui.frameRangeEndSpin.value())))
        self.ui.includePolesStepCheck.setChecked(bool(payload.get("include_poles_step", self.ui.includePolesStepCheck.isChecked())))
        self.ui.allowPoleCenterStepCheck.setChecked(
            bool(payload.get("allow_pole_center_step", self.ui.allowPoleCenterStepCheck.isChecked()))
        )
        self.ui.includePolesCountCheck.setChecked(
            bool(payload.get("include_poles_count", self.ui.includePolesCountCheck.isChecked()))
        )
        self.ui.allowPoleCenterCountCheck.setChecked(
            bool(payload.get("allow_pole_center_count", self.ui.allowPoleCenterCountCheck.isChecked()))
        )
        self._sync_encoder_controls_for_ext()
        self.ui.showBordersCheck.setChecked(bool(payload.get("show_borders", self.ui.showBordersCheck.isChecked())))
        self.ui.livePreviewCheck.setChecked(bool(payload.get("live_preview", self.ui.livePreviewCheck.isChecked())))
        self.ui.interpModeCombo.setCurrentText(str(payload.get("interp_mode", self.ui.interpModeCombo.currentText())))

        splitter_sizes = payload.get("splitter_sizes", {})
        if isinstance(splitter_sizes, dict):
            main_sizes = splitter_sizes.get("main")
            mode_sizes = splitter_sizes.get("mode")
            window_size = splitter_sizes.get("window_size")
            if isinstance(window_size, list) and len(window_size) >= 2:
                self.resize(int(window_size[0]), int(window_size[1]))
            if isinstance(main_sizes, list) and main_sizes:
                QTimer.singleShot(50, lambda: self.ui.mainSplitter.setSizes([int(x) for x in main_sizes]))
            if isinstance(mode_sizes, list) and mode_sizes:
                QTimer.singleShot(50, lambda: self.ui.modeRowSplitter.setSizes([int(x) for x in mode_sizes]))
        return True

    def _append_log(self, text: str) -> None:
        self.ui.logTextEdit.append(text)

    def _update_mode_block_state(self) -> None:
        is_step = self.ui.stepModeRadio.isChecked()
        for widget in self._step_mode_widgets:
            widget.setEnabled(is_step)
        for widget in self._count_mode_widgets:
            widget.setEnabled(not is_step)
        self._schedule_live_preview()

    def _schedule_live_preview(self) -> None:
        if not self.ui.livePreviewCheck.isChecked() or self._live_preview_timer is None:
            return
        self._live_preview_timer.start(300)

    def _normalized_image_ext(self) -> str:
        text = self.ui.imageExtCombo.currentText().strip().lower()
        if text.startswith("."):
            text = text[1:]
        if text not in {"jpg", "png"}:
            text = "jpg"
        return f".{text}"

    def _selected_encoder_backend(self) -> str:
        if self.ui.encoderGpuRadio.isChecked():
            return "gpu"
        if self.ui.encoderClassicRadio.isChecked():
            return "classic"
        return "imagecodecs"

    def _set_encoder_backend(self, backend: str) -> None:
        if backend == "gpu":
            self.ui.encoderGpuRadio.setChecked(True)
        elif backend == "classic":
            self.ui.encoderClassicRadio.setChecked(True)
        else:
            self.ui.encoderImagecodecsRadio.setChecked(True)

    def _sync_encoder_controls_for_ext(self) -> None:
        is_png = self._normalized_image_ext() == ".png"
        self.ui.encoderGpuRadio.setEnabled(not is_png)
        if is_png and not self.ui.encoderImagecodecsRadio.isChecked():
            self.ui.encoderImagecodecsRadio.setChecked(True)

    def _on_image_ext_changed(self, _value: str) -> None:
        self._sync_encoder_controls_for_ext()

    def _clear_preview_caches(self) -> None:
        self.preview_frame_cache.clear()
        self.preview_composited_pixmap_cache.clear()
        self.preview_cached_indices.clear()
        self._last_preview_target_size = None
        self._last_preview_source_key = None
        self._last_preview_smooth = None
        self.ui.previewTimelineSlider.clear_cache()

    def _on_preview_timeline_pressed(self) -> None:
        self._preview_scrubbing = True

    def _on_preview_timeline_released(self) -> None:
        self._preview_scrubbing = False
        if self._preview_scrub_timer is not None and self._preview_scrub_timer.isActive():
            self._preview_scrub_timer.stop()
            self._apply_pending_preview_frame()
        else:
            self._refresh_preview_display(smooth=True)

    def _apply_pending_preview_frame(self) -> None:
        if self._pending_preview_idx is None:
            return
        self._refresh_preview_frame_only()

    def _get_composited_pixmap(self, idx: int, frame_rgb: Image.Image) -> QPixmap:
        cached = self.preview_composited_pixmap_cache.get(idx)
        if cached is not None and not cached.isNull():
            self.preview_composited_pixmap_cache.move_to_end(idx)
            return cached
        if self.preview_overlay_rgba is None:
            raise ValueError("Preview overlay is not ready.")
        frame_rgba = frame_rgb.convert("RGBA")
        composited = Image.alpha_composite(frame_rgba, self.preview_overlay_rgba).convert("RGB")
        pixmap = QPixmap.fromImage(ImageQt(composited))
        self.preview_composited_pixmap_cache[idx] = pixmap
        while len(self.preview_composited_pixmap_cache) > 16:
            self.preview_composited_pixmap_cache.popitem(last=False)
        return pixmap

    def _browse_input(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose input folder")
        if chosen:
            self.ui.inputDirEdit.setText(chosen)
            self.frames_cache = []
            self._frames_cache_input_dir = None
            self._clear_preview_caches()
            self.ui.previewTimelineSlider.set_frame_range(0, 0)
            self._configure_frame_range_spins()
            self.preview_overlay_rgba = None
            self._preview_if_input_ready()

    def _browse_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if chosen:
            self.ui.outputDirEdit.setText(chosen)

    def _load_sequence(self) -> None:
        input_dir = Path(self.ui.inputDirEdit.text().strip())
        if not input_dir.exists():
            QMessageBox.critical(self, "Error", "Input folder not found.")
            return
        self.frames_cache = list_sequence_frames(input_dir)
        self._frames_cache_input_dir = input_dir
        self._clear_preview_caches()
        last_idx = self._sequence_last_index()
        self.ui.frameRangeStartSpin.setValue(0)
        self.ui.frameRangeEndSpin.setValue(last_idx)
        self.ui.frameCountLabel.setText(f"Frames: {len(self.frames_cache)}")
        self._sync_preview_controls()
        self._append_log(f"Loaded {len(self.frames_cache)} frames from {input_dir}")
        self._schedule_live_preview()

    def _sync_preview_controls(self) -> None:
        self._configure_frame_range_spins()
        start, end = self._frame_range()
        range_count = max(0, end - start + 1) if self.frames_cache else 0
        rel_max = max(0, range_count - 1)

        if self.frames_cache:
            current_abs = max(start, min(int(self.ui.previewIndexSpin.value()), end))
        else:
            current_abs = 0
        rel_idx = current_abs - start if self.frames_cache else 0

        self._syncing_preview_controls = True
        try:
            self.ui.previewIndexSpin.blockSignals(True)
            self.ui.previewIndexSpin.setRange(0, self._sequence_last_index())
            self.ui.previewIndexSpin.setValue(current_abs)
            self.ui.previewIndexSpin.blockSignals(False)
            self.ui.previewTimelineSlider.blockSignals(True)
            self.ui.previewTimelineSlider.set_frame_range(start, range_count)
            self.ui.previewTimelineSlider.setRange(0, rel_max)
            self.ui.previewTimelineSlider.setValue(rel_idx)
            self.ui.previewTimelineSlider.blockSignals(False)
        finally:
            self._syncing_preview_controls = False

    def _on_preview_index_spin_changed(self, value: int) -> None:
        if self._syncing_preview_controls:
            return
        start, end = self._frame_range()
        abs_idx = max(start, min(int(value), end))
        rel_idx = abs_idx - start
        self._syncing_preview_controls = True
        try:
            if abs_idx != int(value):
                self.ui.previewIndexSpin.blockSignals(True)
                self.ui.previewIndexSpin.setValue(abs_idx)
                self.ui.previewIndexSpin.blockSignals(False)
            self.ui.previewTimelineSlider.blockSignals(True)
            self.ui.previewTimelineSlider.setValue(rel_idx)
            self.ui.previewTimelineSlider.blockSignals(False)
        finally:
            self._syncing_preview_controls = False
        self._refresh_preview_frame_only()

    def _on_preview_timeline_changed(self, value: int) -> None:
        if self._syncing_preview_controls:
            return
        start, _end = self._frame_range()
        abs_idx = start + int(value)
        self._syncing_preview_controls = True
        try:
            self.ui.previewIndexSpin.blockSignals(True)
            self.ui.previewIndexSpin.setValue(abs_idx)
            self.ui.previewIndexSpin.blockSignals(False)
        finally:
            self._syncing_preview_controls = False
        self._pending_preview_idx = abs_idx
        if self._preview_scrubbing:
            if self._preview_scrub_timer is not None:
                self._preview_scrub_timer.start(33)
        else:
            self._refresh_preview_frame_only()

    def _load_preview_frame_rgb(self, frame_path: Path) -> Image.Image:
        cached = self.preview_frame_cache.get(frame_path)
        if cached is not None:
            self.preview_frame_cache.move_to_end(frame_path)
            return cached
        with Image.open(frame_path) as img:
            frame_rgb = img.convert("RGB")
        self.preview_frame_cache[frame_path] = frame_rgb
        while len(self.preview_frame_cache) > 12:
            self.preview_frame_cache.popitem(last=False)
        return frame_rgb

    def _preview_absolute_index(self) -> int:
        start, end = self._frame_range()
        if not self.frames_cache:
            return 0
        return max(start, min(int(self.ui.previewIndexSpin.value()), end))

    def _refresh_preview_frame_only(self) -> None:
        if not self.frames_cache or self.preview_overlay_rgba is None:
            return
        idx = self._preview_absolute_index()
        frame_path = self.frames_cache[idx]
        frame_rgb = self._load_preview_frame_rgb(frame_path)
        self.preview_pixmap = self._get_composited_pixmap(idx, frame_rgb)
        if idx not in self.preview_cached_indices:
            self.preview_cached_indices.add(idx)
            self.ui.previewTimelineSlider.mark_cached(idx)
        self._refresh_preview_display()
        self.ui.zoneCountLabel.setText(f"Zones: {self.preview_zone_count}")

    def _build_layout_config(self) -> ZoneLayoutConfig:
        is_step = self.ui.stepModeRadio.isChecked()
        include_poles = self.ui.includePolesStepCheck.isChecked() if is_step else self.ui.includePolesCountCheck.isChecked()
        allow_pole_center = (
            self.ui.allowPoleCenterStepCheck.isChecked() if is_step else self.ui.allowPoleCenterCountCheck.isChecked()
        )
        max_pitch = self.ui.maxPitchStepSpin.value() if is_step else self.ui.maxPitchCountSpin.value()
        return ZoneLayoutConfig(
            mode="step_mode" if is_step else "count_mode",
            overlap=float(self.ui.overlapSpin.value()),
            target_zones=int(self.ui.targetZonesSpin.value()),
            yaw_step_deg=float(self.ui.yawStepSpin.value()),
            pitch_step_deg=float(self.ui.pitchStepSpin.value()),
            include_poles=include_poles,
            allow_pole_center=allow_pole_center,
            max_pitch_deg=float(max_pitch),
        )

    def _build_job_config(self) -> SequenceJobConfig:
        frame_start, frame_end = self._frame_range()
        return SequenceJobConfig(
            input_dir=Path(self.ui.inputDirEdit.text().strip()),
            output_dir=Path(self.ui.outputDirEdit.text().strip()),
            crop_width=int(self.ui.cropWSpin.value()),
            crop_height=int(self.ui.cropHSpin.value()),
            fov_x_deg=float(self.ui.fovXSpin.value()),
            chunk_size=int(self.ui.chunkSizeSpin.value()),
            mode=self.ui.interpModeCombo.currentText().strip(),
            image_ext=self._normalized_image_ext(),
            compression_strength=float(self.ui.compressionSpin.value()),
            use_imagecodecs=self._selected_encoder_backend() == "imagecodecs",
            writer_threads=int(self.ui.writerThreadsSpin.value()),
            use_torch_gpu=self.ui.useComputeGpuCheck.isChecked(),
            gpu_zone_batch_size=int(self.ui.gpuZoneBatchSpin.value()),
            use_native_torch_backend=False,
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            layout=self._build_layout_config(),
        )

    def _preview_layout(self) -> None:
        self._preview_layout_internal(show_errors=True, log_success=True)

    def _on_input_dir_editing_finished(self) -> None:
        self.frames_cache = []
        self._frames_cache_input_dir = None
        self._clear_preview_caches()
        self.preview_overlay_rgba = None
        self.ui.previewTimelineSlider.set_frame_range(0, 0)
        self._configure_frame_range_spins()
        self._preview_if_input_ready()

    def _preview_if_input_ready(self) -> None:
        input_text = self.ui.inputDirEdit.text().strip()
        if not input_text:
            return
        input_dir = Path(input_text)
        if not input_dir.exists():
            return
        self._preview_layout_internal(show_errors=False, log_success=False)

    def _preview_layout_internal(self, show_errors: bool, log_success: bool) -> None:
        try:
            input_dir = Path(self.ui.inputDirEdit.text().strip())
            if not self.frames_cache or self._frames_cache_input_dir != input_dir:
                if not input_dir.exists():
                    raise ValueError("Input folder not found.")
                self.frames_cache = list_sequence_frames(input_dir)
                self._frames_cache_input_dir = input_dir
                self._clear_preview_caches()
                self.ui.frameCountLabel.setText(f"Frames: {len(self.frames_cache)}")
                self._sync_preview_controls()
            if not self.frames_cache:
                raise ValueError("No frames to preview.")

            cfg = self._build_job_config()
            cfg.validate()
            idx = self._preview_absolute_index()
            frame = self.frames_cache[idx]
            frame_rgb = self._load_preview_frame_rgb(frame)
            zones = generate_zone_layout(
                layout=cfg.layout,
                fov_x_deg=cfg.fov_x_deg,
                crop_width=cfg.crop_width,
                crop_height=cfg.crop_height,
            )
            overlay = render_layout_overlay(
                (int(frame_rgb.width), int(frame_rgb.height)),
                zones,
                show_borders=self.ui.showBordersCheck.isChecked(),
                fov_x_deg=cfg.fov_x_deg,
                crop_width=cfg.crop_width,
                crop_height=cfg.crop_height,
            )
            self.preview_composited_pixmap_cache.clear()
            self._last_preview_target_size = None
            self._last_preview_source_key = None
            self._last_preview_smooth = None
            self.preview_overlay_rgba = overlay
            self.preview_zone_count = len(zones)
            self.preview_pixmap = self._get_composited_pixmap(idx, frame_rgb)
            self.preview_cached_indices.add(idx)
            self.ui.previewTimelineSlider.mark_cached(idx)
            self._refresh_preview_display()
            self.ui.zoneCountLabel.setText(f"Zones: {self.preview_zone_count}")
            if log_success:
                self._append_log(f"Previewed frame {frame.name}; zones={len(zones)}")
        except Exception as exc:
            if show_errors:
                QMessageBox.critical(self, "Preview error", str(exc))

    def _refresh_preview_display(self, *, smooth: Optional[bool] = None) -> None:
        if self.preview_pixmap is None:
            return
        label_size = self.ui.previewLabel.size()
        if label_size.width() <= 1 or label_size.height() <= 1:
            return
        if smooth is None:
            smooth = not self._preview_scrubbing
        target_size = (int(label_size.width()), int(label_size.height()))
        source_key = int(self.preview_pixmap.cacheKey())
        if (
            self._last_preview_target_size == target_size
            and self._last_preview_source_key == source_key
            and self._last_preview_smooth == smooth
        ):
            return
        transform = (
            Qt.TransformationMode.SmoothTransformation
            if smooth
            else Qt.TransformationMode.FastTransformation
        )
        shown = self.preview_pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            transform,
        )
        self.ui.previewLabel.setPixmap(shown)
        self.ui.previewLabel.setText("")
        self._last_preview_target_size = target_size
        self._last_preview_source_key = source_key
        self._last_preview_smooth = smooth

    def _run_batch(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            QMessageBox.warning(self, "Busy", "Batch processing is already running.")
            return
        output_text = self.ui.outputDirEdit.text().strip()
        if not output_text:
            QMessageBox.critical(self, "Invalid settings", "Output folder is required before rendering.")
            return
        try:
            cfg = self._build_job_config()
            cfg.validate()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        self.stop_event = threading.Event()
        self._speed_samples.clear()
        self._last_speed_update_ts = None
        self._last_crops_per_sec = None
        self._last_progress_max_steps = 0
        self.ui.statusLabel.setText("Running...")
        self.ui.runBatchButton.setEnabled(False)
        self.ui.stopButton.setEnabled(True)
        self.ui.progressBar.setRange(0, 1)
        self.ui.progressBar.setValue(0)

        def worker() -> None:
            try:
                zones, written = process_sequence(
                    cfg,
                    stop_event=self.stop_event,
                    progress_cb=lambda p: self.event_queue.put(("progress", p)),
                )
                if self.stop_event and self.stop_event.is_set():
                    self.event_queue.put(("stopped", (len(zones), written)))
                else:
                    self.event_queue.put(("done", (len(zones), written)))
            except Exception as exc:
                self.event_queue.put(("error", str(exc)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        self._append_log(f"Started batch: input={cfg.input_dir} output={cfg.output_dir}")

    def _reset_render_settings(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            QMessageBox.warning(self, "Busy", "Stop the current batch before resetting render settings.")
            return
        confirmed = QMessageBox.question(
            self,
            "Reset render settings",
            "Reset render settings to default values shown in the preset?",
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        self.ui.interpModeCombo.setCurrentText("bilinear")
        self.ui.imageExtCombo.setCurrentText("jpg")
        self.ui.chunkSizeSpin.setValue(8)
        self._set_encoder_backend("gpu")
        self.ui.useComputeGpuCheck.setChecked(True)
        self._sync_encoder_controls_for_ext()
        self.ui.compressionSpin.setValue(0.0)
        self.ui.writerThreadsSpin.setValue(128)
        self.ui.gpuZoneBatchSpin.setValue(1)
        self._append_log("Render settings reset to defaults.")

    def _stop_batch(self) -> None:
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
            self._append_log("Stop requested.")
            self.ui.statusLabel.setText("Stopping...")

    def _poll_events(self) -> None:
        # Drain the queue completely first, accumulating speed samples for all
        # progress events but deferring widget repaints.  Calling setText /
        # setValue inside a tight loop would fire one Qt repaint (and one GPU
        # compositing command) per zone event, which visibly competes with
        # CUDA during rendering.  Instead we apply a single UI refresh at the
        # end of each 100 ms poll cycle.
        last_progress: Optional[ProgressInfo] = None
        terminal_event: Optional[tuple[str, object]] = None

        while True:
            try:
                event_type, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "progress":
                info = payload
                assert isinstance(info, ProgressInfo)
                cur_step = (info.frame_index - 1) * info.total_zones + info.zone_index
                now = time.perf_counter()
                self._speed_samples.append((now, cur_step))
                window_size = max(2, info.total_zones)
                while len(self._speed_samples) > window_size:
                    self._speed_samples.popleft()

                if self._last_speed_update_ts is None or (now - self._last_speed_update_ts) >= 1.0:
                    if len(self._speed_samples) >= 2:
                        t0, s0 = self._speed_samples[0]
                        t1, s1 = self._speed_samples[-1]
                        dt = max(t1 - t0, 1e-6)
                        ds = max(s1 - s0, 0)
                        self._last_crops_per_sec = ds / dt
                    self._last_speed_update_ts = now

                last_progress = info
            else:
                terminal_event = (event_type, payload)

        # Single UI refresh for all progress events accumulated this cycle.
        if last_progress is not None:
            info = last_progress
            max_steps = max(info.total_frames * info.total_zones, 1)
            cur_step = (info.frame_index - 1) * info.total_zones + info.zone_index
            if max_steps != self._last_progress_max_steps:
                self.ui.progressBar.setRange(0, max_steps)
                self._last_progress_max_steps = max_steps
            self.ui.progressBar.setValue(cur_step)
            self.ui.frameCountLabel.setText(f"Frames: {info.frame_index}/{info.total_frames}")
            self.ui.zoneCountLabel.setText(f"Zones: {info.zone_index}/{info.total_zones}")
            if self._last_crops_per_sec is not None:
                self.ui.statusLabel.setText(f"Rendering | ETA {format_eta(info.eta_seconds)} | {self._last_crops_per_sec:.1f} crops/s")
            else:
                self.ui.statusLabel.setText(f"Rendering | ETA {format_eta(info.eta_seconds)}")

        if terminal_event is None:
            return
        event_type, payload = terminal_event
        if event_type == "done":
            zones, written = payload
            self.ui.statusLabel.setText("Done")
            self._append_log(f"Done. Zones per frame={zones}; total crops={written}")
            self.ui.runBatchButton.setEnabled(True)
            self.ui.stopButton.setEnabled(False)

        elif event_type == "stopped":
            zones, written = payload
            self.ui.statusLabel.setText("Stopped")
            self.ui.progressBar.setRange(0, 1)
            self.ui.progressBar.setValue(0)
            self.ui.frameCountLabel.setText("Frames: 0")
            self.ui.zoneCountLabel.setText("Zones: 0")
            self._append_log(f"Stopped. Zones per frame={zones}; crops written={written}")
            self.ui.runBatchButton.setEnabled(True)
            self.ui.stopButton.setEnabled(False)

        elif event_type == "error":
            self.ui.statusLabel.setText("Failed")
            self._append_log(f"Error: {payload}")
            QMessageBox.critical(self, "Batch error", str(payload))
            self.ui.runBatchButton.setEnabled(True)
            self.ui.stopButton.setEnabled(False)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_state_now()
        if self.stop_event:
            self.stop_event.set()
        super().closeEvent(event)


def run_app() -> None:
    app = QApplication.instance()
    own_app = False
    if app is None:
        app = QApplication([])
        own_app = True
    setTheme(Theme.DARK)
    window = ZoneCropperQtWindow()
    window.show()
    if own_app:
        app.exec()


if __name__ == "__main__":
    run_app()
