DARK_STYLE = """
QWidget {
    background-color: #2b2b2b;
    color: #dddddd;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 13px;
}
QSplitter::handle { background: #222; width: 4px; }
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    color: #888;
    font-size: 11px;
}
QPushButton {
    background-color: #3c3f41;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 5px 10px;
}
QPushButton:hover    { background-color: #4c5052; }
QPushButton:pressed  { background-color: #1e4070; }
QPushButton:disabled { color: #555; border-color: #3a3a3a; }
QProgressBar {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    text-align: center;
    background: #1a1a1a;
    height: 14px;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #2d5a8e;
    border-radius: 3px;
}
QSpinBox, QDoubleSpinBox {
    background-color: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 2px 4px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #2d5a8e; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #555;
    border-radius: 2px;
    background: #1a1a1a;
}
QCheckBox::indicator:checked { background: #2d5a8e; border-color: #2d5a8e; }
QCheckBox::indicator:hover   { border-color: #888; }
"""
