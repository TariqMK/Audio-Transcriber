"""Colors, type and icons for the window, in a light and a dark version that follow the
system appearance. Every color in the app comes from the tokens below."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PyQt6.QtSvg import QSvgRenderer


@dataclass(frozen=True)
class Tokens:
    dark: bool
    window: str  # window background
    surface: str  # the queue card
    raised: str  # buttons, popups, chips that are on
    raised_hover: str
    raised_pressed: str
    field: str  # path wells in Settings
    line: str  # hairlines
    line_strong: str  # control outlines
    text: str  # primary text
    text_2: str  # secondary text
    muted: str  # meta and icons; at least 4.5:1 on window and surface
    accent: str  # Transcribe button fill; white text on it is 5.1:1
    accent_hover: str
    accent_pressed: str
    accent_ink: str  # accent used as text or outline on the window background
    caret: str  # the text cursor from the app icon: progress and the drop target
    wave_done: str  # waveform bars already written
    wave_todo: str  # waveform bars still to come
    ok: str
    bad: str


DARK = Tokens(
    dark=True,
    window="#161617", surface="#1c1c1e", raised="#2a2a2d", raised_hover="#323236", raised_pressed="#252528",
    field="#121213", line="rgba(237, 237, 237, 20)", line_strong="rgba(237, 237, 237, 36)",
    text="#ededed", text_2="#a1a1a6", muted="#85858b",
    accent="#1f6fc0", accent_hover="#2479cf", accent_pressed="#1a62ab", accent_ink="#4a9be8",
    caret="#2a82da", wave_done="#ededed", wave_todo="rgba(237, 237, 237, 40)",
    ok="#4cc38a", bad="#f07075",
)

LIGHT = Tokens(
    dark=False,
    window="#f5f5f7", surface="#ffffff", raised="#ffffff", raised_hover="#f0f0f2", raised_pressed="#e6e6e9",
    field="#ffffff", line="rgba(29, 29, 31, 26)", line_strong="rgba(29, 29, 31, 46)",
    text="#1d1d1f", text_2="#5c5c60", muted="#6e6e73",
    accent="#1f6fc0", accent_hover="#2479cf", accent_pressed="#1a62ab", accent_ink="#1f6fc0",
    caret="#2a82da", wave_done="#1d1d1f", wave_todo="rgba(29, 29, 31, 46)",
    ok="#1a7f4b", bad="#cf2f35",
)


def qcolor(value: str) -> QColor:
    """QColor from a token, including the 'rgba(r, g, b, a)' form used in the stylesheet."""
    if value.startswith("rgba("):
        r, g, b, a = (int(part) for part in value[5:-1].split(","))
        return QColor(r, g, b, a)
    return QColor(value)


def system_is_dark() -> bool:
    hints = QGuiApplication.styleHints()
    return hints.colorScheme() == Qt.ColorScheme.Dark


_applied: Tokens | None = None


def current(dark: bool | None = None) -> Tokens:
    """Tokens for the given appearance; with none given, the ones last applied, else the system's."""
    if dark is None and _applied is not None:
        return _applied
    return DARK if (system_is_dark() if dark is None else dark) else LIGHT


def figures(font: QFont) -> QFont:
    """The same font with tabular figures, so times and counts line up and don't jitter."""
    font = QFont(font)
    try:
        font.setFeature(QFont.Tag("tnum"), 1)
    except (AttributeError, TypeError):  # Qt before 6.7
        pass
    return font


def palette(t: Tokens) -> QPalette:
    """Palette for native parts the stylesheet doesn't reach (menus, dialogs, scroll bars)."""
    p = QPalette()
    roles = QPalette.ColorRole
    for role, value in (
        (roles.Window, t.window), (roles.WindowText, t.text), (roles.Base, t.surface),
        (roles.AlternateBase, t.raised), (roles.ToolTipBase, t.raised), (roles.ToolTipText, t.text),
        (roles.Text, t.text), (roles.Button, t.raised), (roles.ButtonText, t.text),
        (roles.BrightText, t.bad), (roles.Link, t.accent_ink), (roles.Highlight, t.accent),
        (roles.HighlightedText, "#ffffff"), (roles.PlaceholderText, t.muted), (roles.Mid, t.raised_hover),
    ):
        p.setColor(role, qcolor(value))
    p.setColor(QPalette.ColorGroup.Disabled, roles.ButtonText, qcolor(t.muted))
    p.setColor(QPalette.ColorGroup.Disabled, roles.Text, qcolor(t.muted))
    p.setColor(QPalette.ColorGroup.Disabled, roles.WindowText, qcolor(t.muted))
    return p


# Spacing follows a 4 px grid. Controls are 28 px tall with a 6 px radius, the queue card 10 px.
STYLESHEET = """
QMainWindow, QDialog#settings {{ background: {window}; }}
QWidget {{ color: {text}; }}
QToolTip {{ background: {raised}; color: {text}; border: 1px solid {line_strong}; padding: 4px 8px; border-radius: 6px; }}

QLabel#title {{ font-size: 15px; font-weight: 600; }}
QLabel#meta, QLabel#hint {{ color: {muted}; font-size: 12px; }}
QLabel#secondary {{ color: {text_2}; }}
QLabel#formLabel {{ color: {text_2}; }}
QLabel#caption {{ color: {muted}; font-size: 12px; }}
QLabel#emptyTitle {{ font-size: 15px; font-weight: 500; }}
QLabel#emptyBody {{ color: {text_2}; }}
QLabel#ok {{ color: {text_2}; font-size: 12px; }}
QLabel#stateDot[state="ready"] {{ background: {ok}; border-radius: 3px; }}
QLabel#stateDot[state="idle"] {{ background: {muted}; border-radius: 3px; }}
QLabel#stateDot[state="busy"] {{ background: {caret}; border-radius: 3px; }}
QLabel#stateDot[state="bad"] {{ background: {bad}; border-radius: 3px; }}
QLabel#updateNote {{ color: {accent_ink}; font-size: 12px; }}

QFrame#card {{ background: {surface}; border: 1px solid {line}; border-radius: 10px; }}
QFrame#card[dropping="true"] {{ border: 1px solid {caret}; }}
QFrame#emptyCard {{ background: transparent; border: 1px dashed {line_strong}; border-radius: 10px; }}
QFrame#emptyCard[dropping="true"] {{ border: 1px solid {caret}; background: {surface}; }}
QFrame#row {{ background: transparent; border: none; border-top: 1px solid {line}; }}
QFrame#row[first="true"] {{ border-top: none; }}
QFrame#row[active="true"] {{ background: {row_active}; }}
QFrame#footer {{ background: {window}; border: none; border-top: 1px solid {line}; }}
QFrame#rule {{ background: {line}; border: none; max-height: 1px; min-height: 1px; }}
QScrollArea, QScrollArea > QWidget > QWidget#rows {{ background: transparent; border: none; }}

QPushButton {{
    background: {raised}; color: {text}; border: 1px solid {line_strong}; border-radius: 6px;
    min-height: 26px; max-height: 26px; padding: 0 12px; font-weight: 500;
}}
QPushButton:hover {{ background: {raised_hover}; }}
QPushButton:pressed {{ background: {raised_pressed}; }}
QPushButton:disabled {{ color: {muted}; background: {window}; border-color: {line}; }}
QPushButton:focus {{ border: 1px solid {caret}; }}

QPushButton#primary {{ background: {accent}; color: #ffffff; border: 1px solid {accent}; padding: 0 16px; }}
QPushButton#primary:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton#primary:pressed {{ background: {accent_pressed}; border-color: {accent_pressed}; }}
QPushButton#primary:disabled {{ background: {raised}; color: {muted}; border-color: {line}; }}
QPushButton#primary:focus {{ border: 1px solid {focus_on_accent}; }}

QPushButton#quiet {{ background: transparent; border: 1px solid transparent; color: {text_2}; padding: 0 8px; }}
QPushButton#quiet:hover {{ background: {raised_hover}; color: {text}; }}
QPushButton#quiet:pressed {{ background: {raised_pressed}; }}
QPushButton#quiet:disabled {{ color: {muted}; background: transparent; }}
QPushButton#quiet:focus {{ border: 1px solid {caret}; }}

QPushButton#icon {{ background: transparent; border: 1px solid transparent; padding: 0; min-width: 26px; max-width: 26px; }}
QPushButton#icon:hover {{ background: {raised_hover}; }}
QPushButton#icon:focus {{ border: 1px solid {caret}; }}

QPushButton#chip {{
    background: transparent; color: {text_2}; border: 1px solid {line_strong}; border-radius: 6px;
    min-height: 22px; max-height: 22px; padding: 0 10px 0 6px; font-size: 12px; font-weight: 400;
}}
QPushButton#chip:checked {{ background: {raised}; color: {text}; }}
QPushButton#chip:hover {{ background: {raised_hover}; }}
QPushButton#chip:focus {{ border: 1px solid {caret}; }}
QPushButton#chip:disabled {{ color: {muted}; background: transparent; border-color: {line}; }}
QPushButton#chip:checked:disabled {{ color: {text_2}; background: {raised}; border-color: {line_strong}; }}

QComboBox {{
    background: {raised}; color: {text}; border: 1px solid {line_strong}; border-radius: 6px;
    min-height: 26px; max-height: 26px; padding: 0 8px 0 10px;
}}
QComboBox:hover {{ background: {raised_hover}; }}
QComboBox:focus {{ border: 1px solid {caret}; }}
QComboBox:disabled {{ color: {muted}; background: {window}; border-color: {line}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ image: none; width: 0; }}
QComboBox QAbstractItemView {{
    background: {raised}; color: {text}; border: 1px solid {line_strong}; border-radius: 6px;
    padding: 4px; outline: none; selection-background-color: {accent}; selection-color: #ffffff;
}}

QFrame#well {{ background: {field}; border: 1px solid {line_strong}; border-radius: 6px; }}

QProgressBar#thin {{ background: {wave_todo}; border: none; border-radius: 1px; max-height: 2px; min-height: 2px; }}
QProgressBar#thin::chunk {{ background: {caret}; border-radius: 1px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {line_strong}; border-radius: 3px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

QMessageBox QPushButton {{ min-width: 72px; }}
"""


def stylesheet(t: Tokens) -> str:
    values = dict(t.__dict__)
    values["row_active"] = "#202023" if t.dark else "#f7f8fa"
    values["focus_on_accent"] = "#9cc8f2" if t.dark else "#0f3f73"
    return STYLESHEET.format(**values)


def apply(app, dark: bool | None = None) -> Tokens:
    """Style the whole app for the system appearance (or the one given) and return its tokens."""
    global _applied
    t = DARK if (system_is_dark() if dark is None else dark) else LIGHT
    _applied = t
    app.setStyle("Fusion")
    app.setPalette(palette(t))
    app.setStyleSheet(stylesheet(t))
    return t


# --- Icons -------------------------------------------------------------------
# Line icons on a 16 px grid with a 1.5 px stroke, drawn in the color of the text they sit next to.

_ICONS = {
    "plus": '<path d="M8 3.5v9M3.5 8h9"/>',
    "check": '<path d="M3.5 8.5l3 3 6-7"/>',
    "check-small": '<path d="M4 8.3l2.6 2.6L12 5.3"/>',
    "x": '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
    "folder": '<path d="M2 4.5C2 3.7 2.7 3 3.5 3h3L8 4.5h4.5c.8 0 1.5.7 1.5 1.5v5.5c0 .8-.7 1.5-1.5 1.5h-9C2.7 13 2 12.3 2 11.5z"/>',
    "reveal": '<path d="M7 3H3.5C2.7 3 2 3.7 2 4.5v8c0 .8.7 1.5 1.5 1.5h8c.8 0 1.5-.7 1.5-1.5V9"/><path d="M9.5 2H14v4.5M14 2L7.5 8.5"/>',
    "file": '<path d="M4 2h5l3 3v8.5c0 .3-.2.5-.5.5h-7a.5.5 0 0 1-.5-.5z"/><path d="M9 2v3h3"/>',
    "alert": '<circle cx="8" cy="8" r="6"/><path d="M8 5v3.5M8 11h.01"/>',
    "stop": '<circle cx="8" cy="8" r="6"/><path d="M6 6h4v4H6z"/>',
    "chevron": '<path d="M4.5 6.5L8 10l3.5-3.5"/>',
}


def icon(name: str, color: str, size: int = 16, ratio: float = 2.0) -> QIcon:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" '
           f'stroke="{qcolor(color).name()}" stroke-opacity="{qcolor(color).alphaF():.3f}" stroke-width="1.5" '
           f'stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    pixmap = QPixmap(int(size * ratio), int(size * ratio))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size * ratio, size * ratio))
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return QIcon(pixmap)
