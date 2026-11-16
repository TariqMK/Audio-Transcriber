"""The queue's building blocks: a file row with its waveform, the empty drop view,
and small helpers (an eliding label, the model popup)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QRectF, QSize, Qt, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QPainter, QPainterPath
from PyQt6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QVBoxLayout, QWidget)

from verbatica import theme

# One easing curve for everything that moves
EASE = QEasingCurve.Type.OutCubic


def format_duration(seconds: float) -> str:
    """48:12, or 1:31:40 for an hour or more."""
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_taken(seconds: float) -> str:
    """How long a file took: 48 s, 2 min 12 s."""
    if seconds < 1:
        return "under 1 s"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, secs = divmod(seconds, 60)
    return f"{minutes} min {secs} s" if secs else f"{minutes} min"


class ElidedLabel(QLabel):
    """A one-line label that shortens long text with an ellipsis instead of widening the window."""

    def __init__(self, text: str = "", mode=Qt.TextElideMode.ElideMiddle, parent=None):
        super().__init__(parent)
        self._full = text
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        super().setText(text)

    def setText(self, text: str):
        self._full = text
        self._elide()

    def text(self) -> str:
        return self._full

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 2, hint.height())

    def minimumSizeHint(self):
        return QSize(40, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        super().setText(self.fontMetrics().elidedText(self._full, self._mode, max(self.width(), 40)))


class ModelPopup(QComboBox):
    """The model menu, with a chevron drawn in the theme's icon color."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tokens = theme.current()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        color = self.tokens.muted if self.isEnabled() else self.tokens.line_strong
        theme.icon("chevron", color, 12).paint(painter, self.width() - 20, (self.height() - 12) // 2, 12, 12)

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(hint.width() + 8, hint.height())


class Waveform(QWidget):
    """A file's real waveform. Bars already written are ink, the rest are faint, and the blue
    caret from the app icon marks the point reached. Silence the app skips stays flat."""

    BAR = 2
    GAP = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peaks: list = []
        self.fraction = 0.0
        self.state = "waiting"
        self.tokens = theme.current()
        self.setMinimumWidth(80)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(400)
        self._animation.setEasingCurve(EASE)
        self._animation.valueChanged.connect(self._set_fraction)

    def set_peaks(self, peaks):
        self.peaks = list(peaks or [])
        self.update()

    def set_state(self, state: str):
        self.state = state
        if state == "done":
            self._animation.stop()
            self.fraction = 1.0
        self.update()

    def set_fraction(self, fraction: float, animate: bool = True):
        fraction = max(0.0, min(1.0, fraction))
        self._animation.stop()
        if not animate or not self.isVisible():
            self._set_fraction(fraction)
            return
        self._animation.setStartValue(self.fraction)
        self._animation.setEndValue(fraction)
        self._animation.start()

    def _set_fraction(self, value):
        self.fraction = float(value)
        self.update()

    def bars(self, count: int) -> list:
        """Merge the peaks into `count` bars, keeping the loudest value of each group."""
        if not self.peaks or count <= 0:
            return []
        n = len(self.peaks)
        return [max(self.peaks[int(i * n / count):max(int((i + 1) * n / count), int(i * n / count) + 1)])
                for i in range(count)]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        t = self.tokens
        width, height = self.width(), self.height()
        mid = height / 2
        step = self.BAR + self.GAP
        values = self.bars(width // step)

        if not values:  # waiting or failed before decoding: a faint baseline
            painter.setBrush(theme.qcolor(t.wave_todo))
            painter.drawRoundedRect(QRectF(0, mid - 0.5, width, 1), 0.5, 0.5)
            return

        done_color = theme.qcolor(t.wave_done)
        if self.state == "done":
            done_color = theme.qcolor(t.text_2)
        todo_color = theme.qcolor(t.wave_todo)
        cut = self.fraction * len(values)
        for i, value in enumerate(values):
            bar_height = max(2.0, value * height)
            painter.setBrush(done_color if i < cut else todo_color)
            painter.drawRoundedRect(QRectF(i * step, mid - bar_height / 2, self.BAR, bar_height), 1, 1)

        if self.state == "active":
            x = min(cut * step, width - 2)
            painter.setBrush(theme.qcolor(t.caret))
            painter.drawRoundedRect(QRectF(x, 0, 2, height), 1, 1)


class StatusIcon(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)

    def show_icon(self, name: str, color: str):
        self.setPixmap(theme.icon(name, color).pixmap(16, 16))


class FileRow(QFrame):
    """One file in the queue: its state icon, name and length, waveform, result and one action."""

    remove_requested = pyqtSignal(object)  # self
    reveal_requested = pyqtSignal(object)  # self

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.state = "waiting"
        self.output_file = ""
        self.duration = 0.0
        self.tokens = theme.current()
        self.setObjectName("row")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(12)

        self.status_icon = StatusIcon()
        layout.addWidget(self.status_icon)

        names = QVBoxLayout()
        names.setSpacing(0)
        names.setContentsMargins(0, 0, 0, 0)
        self.name_label = ElidedLabel(Path(path).name)
        self.name_label.setStyleSheet("font-weight: 500;")
        self.name_label.setToolTip(path)
        self.sub_label = QLabel("Waiting")
        self.sub_label.setObjectName("meta")
        self.sub_label.setFont(theme.figures(self.sub_label.font()))
        names.addStretch()
        names.addWidget(self.name_label)
        names.addWidget(self.sub_label)
        names.addStretch()
        name_box = QWidget()
        name_box.setLayout(names)
        name_box.setFixedWidth(220)
        layout.addWidget(name_box)

        self.waveform = Waveform()
        layout.addWidget(self.waveform, 1)

        self.result_label = QLabel("")
        self.result_label.setObjectName("ok")
        self.result_label.setFont(theme.figures(self.result_label.font()))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.result_label.setFixedWidth(112)
        layout.addWidget(self.result_label)

        self.action_btn = QPushButton()
        self.action_btn.setObjectName("icon")
        self.action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.action_btn.clicked.connect(self._action)
        keep_space = self.action_btn.sizePolicy()
        keep_space.setRetainSizeWhenHidden(True)  # so every waveform ends at the same place
        self.action_btn.setSizePolicy(keep_space)
        layout.addWidget(self.action_btn)

        self.set_state("waiting")

    # --- State ---------------------------------------------------------------

    def set_state(self, state: str, message: str = ""):
        self.state = state
        t = self.tokens
        self.setProperty("active", state == "active")
        self.waveform.set_state(state)
        self.result_label.setToolTip("")
        if state == "waiting":
            self.status_icon.show_icon("file", t.muted)
            self.sub_label.setText(format_duration(self.duration) if self.duration else "Waiting")
            self.result_label.setText("")
            self._set_action("x", "Remove from queue")
        elif state == "active":
            self.status_icon.show_icon("file", t.text_2)
            self.sub_label.setText("Reading audio")
            self.result_label.setText("")
            self._set_action(None)
        elif state == "done":
            self.status_icon.show_icon("check", t.ok)
            self.result_label.setStyleSheet("")
            self._set_action("reveal", self._reveal_label())
        elif state == "failed":
            self.status_icon.show_icon("alert", t.bad)
            self.sub_label.setText(format_duration(self.duration) if self.duration else "Not read")
            self.result_label.setText("Failed")
            self.result_label.setStyleSheet(f"color: {t.bad};")
            self.result_label.setToolTip(message)
            self.name_label.setToolTip(f"{self.path}\n\n{message}")
            self._set_action("x", "Remove from queue")
        elif state == "stopped":
            self.status_icon.show_icon("stop", t.muted)
            self.sub_label.setText(format_duration(self.duration) if self.duration else "Waiting")
            self.result_label.setText("Stopped")
            self.waveform.set_fraction(0, animate=False)
            self._set_action("x", "Remove from queue")
        self.style().unpolish(self)
        self.style().polish(self)

    def retheme(self, tokens):
        """Repaint icons and colors for a new appearance without touching the row's text."""
        self.tokens = tokens
        self.waveform.tokens = tokens
        icons = {"waiting": ("file", tokens.muted), "active": ("file", tokens.text_2), "done": ("check", tokens.ok),
                 "failed": ("alert", tokens.bad), "stopped": ("stop", tokens.muted)}
        self.status_icon.show_icon(*icons[self.state])
        self.result_label.setStyleSheet(f"color: {tokens.bad};" if self.state == "failed" else "")
        if not self.action_btn.icon().isNull():
            self.action_btn.setIcon(theme.icon("reveal" if self.state == "done" else "x", tokens.muted))
        self.waveform.update()

    def set_audio(self, peaks, duration: float):
        self.duration = duration
        self.waveform.set_peaks(peaks)
        self.sub_label.setText(format_duration(duration))

    def set_progress(self, fraction: float):
        self.waveform.set_fraction(fraction)
        self.result_label.setText(f"{int(fraction * 100)}%")

    def set_done(self, output_base: str, formats, seconds_taken: float, words: int):
        self.output_file = f"{output_base}{formats[0]}" if formats else output_base
        self.result_label.setText(f"{words:,} words")
        self.result_label.setToolTip(f"Took {format_taken(seconds_taken)}")
        self.set_state("done")
        if Path(output_base).name != Path(self.path).stem:  # saved as "talk (2)" to keep an earlier result
            self.sub_label.setText(f"{format_duration(self.duration)}  ·  saved as {Path(output_base).name}"
                                   if self.duration else f"Saved as {Path(output_base).name}")

    def set_busy(self, busy: bool):
        """While a batch runs, waiting files can't be removed."""
        if self.state in ("waiting", "failed", "stopped"):
            self.action_btn.setVisible(not busy)

    def _reveal_label(self) -> str:
        import sys
        place = "Finder" if sys.platform == "darwin" else "Explorer" if sys.platform == "win32" else "folder"
        return f"Show in {place}"

    def _set_action(self, icon_name, tooltip: str = ""):
        if icon_name is None:
            self.action_btn.setVisible(False)
            return
        self.action_btn.setVisible(True)
        self.action_btn.setIcon(theme.icon(icon_name, self.tokens.muted))
        self.action_btn.setIconSize(QSize(16, 16))
        self.action_btn.setToolTip(tooltip)
        self.action_btn.setAccessibleName(f"{tooltip}: {Path(self.path).name}")

    def _action(self):
        if self.state == "done":
            self.reveal_requested.emit(self)
        else:
            self.remove_requested.emit(self)


class Glyph(QWidget):
    """The app icon's idea at rest: a short waveform and the blue caret, for the empty queue."""

    HEIGHTS = (12, 24, 16, 32, 20, 8)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tokens = theme.current()
        self.setFixedSize(52, 40)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme.qcolor(self.tokens.wave_todo))
        x = 0
        for h in self.HEIGHTS:
            path = QPainterPath()
            path.addRoundedRect(QRectF(x, (40 - h) / 2, 4, h), 2, 2)
            painter.drawPath(path)
            x += 7
        painter.setBrush(theme.qcolor(self.tokens.caret))
        painter.drawRoundedRect(QRectF(x + 4, 0, 2, 40), 1, 1)
