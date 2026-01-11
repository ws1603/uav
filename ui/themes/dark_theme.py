# ui/themes/dark_theme.py

def get_dark_stylesheet() -> str:
    """返回深色主题样式表"""
    return """
    QMainWindow {
        background-color: #1e1e1e;
    }

    QWidget {
        background-color: #2d2d2d;
        color: #d4d4d4;
        font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
        font-size: 13px;
    }

    QMenuBar {
        background-color: #2d2d2d;
        color: #d4d4d4;
        border-bottom: 1px solid #3d3d3d;
    }

    QMenuBar::item:selected {
        background-color: #3d3d3d;
    }

    QMenu {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
    }

    QMenu::item:selected {
        background-color: #094771;
    }

    QToolBar {
        background-color: #2d2d2d;
        border: none;
        spacing: 5px;
        padding: 5px;
    }

    QPushButton {
        background-color: #0e639c;
        color: white;
        border: none;
        padding: 6px 16px;
        border-radius: 3px;
        min-width: 60px;
    }

    QPushButton:hover {
        background-color: #1177bb;
    }

    QPushButton:pressed {
        background-color: #094771;
    }

    QPushButton:disabled {
        background-color: #3d3d3d;
        color: #808080;
    }

    QGroupBox {
        border: 1px solid #3d3d3d;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 10px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #3c3c3c;
        border: 1px solid #3d3d3d;
        border-radius: 3px;
        padding: 4px 8px;
        min-height: 20px;
    }

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border: 1px solid #0e639c;
    }

    QComboBox::drop-down {
        border: none;
        width: 20px;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #d4d4d4;
        margin-right: 5px;
    }

    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        selection-background-color: #094771;
    }

    QSlider::groove:horizontal {
        height: 6px;
        background-color: #3c3c3c;
        border-radius: 3px;
    }

    QSlider::handle:horizontal {
        width: 16px;
        height: 16px;
        margin: -5px 0;
        background-color: #0e639c;
        border-radius: 8px;
    }

    QSlider::handle:horizontal:hover {
        background-color: #1177bb;
    }

    QProgressBar {
        border: 1px solid #3d3d3d;
        border-radius: 3px;
        text-align: center;
        background-color: #3c3c3c;
    }

    QProgressBar::chunk {
        background-color: #0e639c;
        border-radius: 2px;
    }

    QTabWidget::pane {
        border: 1px solid #3d3d3d;
        border-radius: 3px;
    }

    QTabBar::tab {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        padding: 8px 16px;
        margin-right: 2px;
    }

    QTabBar::tab:selected {
        background-color: #094771;
        border-bottom: none;
    }

    QTabBar::tab:hover:!selected {
        background-color: #3d3d3d;
    }

    QDockWidget {
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }

    QDockWidget::title {
        background-color: #2d2d2d;
        padding: 6px;
        border-bottom: 1px solid #3d3d3d;
    }

    QScrollBar:vertical {
        background-color: #2d2d2d;
        width: 12px;
        margin: 0;
    }

    QScrollBar::handle:vertical {
        background-color: #5a5a5a;
        min-height: 30px;
        border-radius: 6px;
        margin: 2px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #6a6a6a;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }

    QScrollBar:horizontal {
        background-color: #2d2d2d;
        height: 12px;
        margin: 0;
    }

    QScrollBar::handle:horizontal {
        background-color: #5a5a5a;
        min-width: 30px;
        border-radius: 6px;
        margin: 2px;
    }

    QTextEdit, QPlainTextEdit {
        background-color: #1e1e1e;
        border: 1px solid #3d3d3d;
        border-radius: 3px;
    }

    QStatusBar {
        background-color: #007acc;
        color: white;
    }

    QStatusBar::item {
        border: none;
    }

    QLabel {
        background-color: transparent;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 1px solid #3d3d3d;
        border-radius: 3px;
        background-color: #3c3c3c;
    }

    QCheckBox::indicator:checked {
        background-color: #0e639c;
        image: none;
    }

    QCheckBox::indicator:checked::after {
        content: "✓";
    }

    QToolTip {
        background-color: #2d2d2d;
        color: #d4d4d4;
        border: 1px solid #3d3d3d;
        padding: 4px;
    }

    QSplitter::handle {
        background-color: #3d3d3d;
    }

    QSplitter::handle:horizontal {
        width: 2px;
    }

    QSplitter::handle:vertical {
        height: 2px;
    }
    """