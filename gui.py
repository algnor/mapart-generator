import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QFileDialog, QProgressBar,
    QGroupBox, QSpinBox, QDoubleSpinBox, QSizePolicy, QSplitter,
    QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter

from solver import solve_strip
from renderer import render_strip
from export import export_sponge
from color import BLOCK_NAMES


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    arr  = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    arr  = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def crop_to_tiles(img: Image.Image, cols: int, rows: int) -> Image.Image:
    """Centre-crop img to exactly cols*128 x rows*128, scaling to cover."""
    tw, th = cols * 128, rows * 128
    iw, ih = img.size
    scale  = max(tw / iw, th / ih)
    new_w  = int(iw * scale)
    new_h  = int(ih * scale)
    img    = img.resize((new_w, new_h), Image.LANCZOS)
    left   = (new_w - tw) // 2
    top    = (new_h - th) // 2
    return img.crop((left, top, left + tw, top + th))


# ─────────────────────────────────────────────
#  Zoomable image viewer
# ─────────────────────────────────────────────

class ZoomableLabel(QWidget):
    def __init__(self, placeholder=""):
        super().__init__()
        self.pixmap      = None
        self.zoom        = 1.0
        self.offset      = QPointF(0, 0)
        self._drag_start        = None
        self._drag_offset_start = None
        self._placeholder       = placeholder
        self.setMinimumSize(256, 256)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")

    def set_pixmap(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.fit()
        self.update()

    def fit(self):
        if not self.pixmap:
            return
        sw = self.width()  / self.pixmap.width()
        sh = self.height() / self.pixmap.height()
        self.zoom   = min(sw, sh)
        self.offset = QPointF(
            (self.width()  - self.pixmap.width()  * self.zoom) / 2,
            (self.height() - self.pixmap.height() * self.zoom) / 2,
        )

    def paintEvent(self, _):
        p = QPainter(self)
        if not self.pixmap:
            p.setPen(Qt.gray)
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 2.0)
        w = self.pixmap.width()  * self.zoom
        h = self.pixmap.height() * self.zoom
        p.drawPixmap(
            QRectF(self.offset.x(), self.offset.y(), w, h),
            self.pixmap,
            QRectF(self.pixmap.rect()),
        )

    def wheelEvent(self, e):
        if not self.pixmap:
            return
        factor   = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        cursor   = QPointF(e.pos())
        new_zoom = max(0.2, min(64.0, self.zoom * factor))
        self.offset = cursor - (cursor - self.offset) * (new_zoom / self.zoom)
        self.zoom   = new_zoom
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_start        = QPointF(e.pos())
            self._drag_offset_start = QPointF(self.offset)

    def mouseMoveEvent(self, e):
        if self._drag_start is not None:
            self.offset = self._drag_offset_start + QPointF(e.pos()) - self._drag_start
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag_start = None

    def mouseDoubleClickEvent(self, e):
        self.fit()
        self.update()

    def resizeEvent(self, e):
        self.fit()


# ─────────────────────────────────────────────
#  Drop zone
# ─────────────────────────────────────────────

class DropZone(QLabel):
    image_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drop image here\nor click to browse")
        self.setMinimumHeight(60)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                border-radius: 4px;
                color: #888;
                font-size: 13px;
                padding: 8px;
            }
            QLabel:hover { border-color: #2d5a8e; color: #aaa; }
        """)

    def dragEnterEvent(self, e):
        e.accept() if e.mimeData().hasUrls() else e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.image_dropped.emit(urls[0].toLocalFile())

    def mousePressEvent(self, e):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.image_dropped.emit(path)


# ─────────────────────────────────────────────
#  Worker thread
# ─────────────────────────────────────────────

class GenerateWorker(QThread):
    progress = pyqtSignal(int, int, int, float)   # tc, tr, col, cost
    finished = pyqtSignal(object, object, object) # rendered, blocks, heights
    error    = pyqtSignal(str)

    def __init__(self, tiles, height_penalty, dither_strength):
        super().__init__()
        self.tiles          = tiles
        self.height_penalty = height_penalty
        self.dither_strength = dither_strength

    def run(self):
        try:
            map_cols = len(self.tiles)
            map_rows = len(self.tiles[0])

            rendered = [[None] * map_rows for _ in range(map_cols)]
            blocks   = [[None] * map_rows for _ in range(map_cols)]
            heights  = [[None] * map_rows for _ in range(map_cols)]

            for tc in range(map_cols):
                for tr in range(map_rows):
                    pixels       = self.tiles[tc][tr]
                    rimg         = np.zeros((128, 128, 3), dtype=np.uint8)
                    tile_blocks  = []
                    tile_heights = []

                    for col in range(128):
                        strip      = pixels[:, col, :]
                        path, cost = solve_strip(
                            strip,
                            height_penalty  = self.height_penalty,
                            dither_strength = self.dither_strength,
                        )
                        tile_blocks.append([p[0] for p in path])
                        tile_heights.append([p[1] for p in path])
                        rimg[:, col] = render_strip(path)
                        self.progress.emit(tc, tr, col + 1, cost)

                    rendered[tc][tr] = rimg
                    blocks[tc][tr]   = tile_blocks
                    heights[tc][tr]  = tile_heights

            self.finished.emit(rendered, blocks, heights)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minecraft Map Art Generator")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(DARK_STYLE)

        self.source_image   = None
        self.tiles          = None
        self.rendered       = None
        self.full_rendered  = None
        self.result_blocks  = None
        self.result_heights = None
        self.worker         = None
        self.total_cols     = 0
        self._done_cols     = 0

        self._build_ui()

    def _build_ui(self):
        central  = QWidget()
        self.setCentralWidget(central)
        root     = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── controls (left) ──────────────────
        controls = QWidget()
        controls.setFixedWidth(220)
        cl = QVBoxLayout(controls)
        cl.setSpacing(8)
        cl.setContentsMargins(0, 0, 0, 0)
        root.addWidget(controls)

        self.drop_zone = DropZone()
        self.drop_zone.image_dropped.connect(self.load_image)
        cl.addWidget(self.drop_zone)

        # map grid
        grid_group  = QGroupBox("Map Grid")
        grid_layout = QGridLayout(grid_group)
        grid_layout.setSpacing(4)
        grid_layout.addWidget(QLabel("Maps wide"), 0, 0)
        self.maps_w = QSpinBox()
        self.maps_w.setRange(1, 16)
        self.maps_w.setValue(1)
        self.maps_w.valueChanged.connect(self.update_input_preview)
        grid_layout.addWidget(self.maps_w, 0, 1)
        grid_layout.addWidget(QLabel("Maps tall"), 1, 0)
        self.maps_h = QSpinBox()
        self.maps_h.setRange(1, 16)
        self.maps_h.setValue(1)
        self.maps_h.valueChanged.connect(self.update_input_preview)
        grid_layout.addWidget(self.maps_h, 1, 1)
        cl.addWidget(grid_group)

        # solver settings
        solver_group  = QGroupBox("Solver")
        solver_layout = QGridLayout(solver_group)
        solver_layout.setSpacing(4)

        solver_layout.addWidget(QLabel("Height Penalty"), 0, 0)
        self.hp_spin = QDoubleSpinBox()
        self.hp_spin.setRange(0.0, 20.0)
        self.hp_spin.setSingleStep(0.5)
        self.hp_spin.setValue(2.0)
        self.hp_spin.setToolTip("Higher = flatter, lower = more height variation")
        solver_layout.addWidget(self.hp_spin, 0, 1)

        solver_layout.addWidget(QLabel("Max Height Diff"), 1, 0)
        self.mh_spin = QSpinBox()
        self.mh_spin.setRange(0, 16)
        self.mh_spin.setValue(4)
        solver_layout.addWidget(self.mh_spin, 1, 1)

        solver_layout.addWidget(QLabel("Beam Width"), 2, 0)
        self.bw_spin = QSpinBox()
        self.bw_spin.setRange(1, 256)
        self.bw_spin.setValue(16)
        self.bw_spin.setToolTip("Higher = better quality, slower")
        solver_layout.addWidget(self.bw_spin, 2, 1)

        solver_layout.addWidget(QLabel("Dither Strength"), 3, 0)
        self.dither_spin = QDoubleSpinBox()
        self.dither_spin.setRange(0.0, 1.0)
        self.dither_spin.setSingleStep(0.05)
        self.dither_spin.setValue(0.3)
        self.dither_spin.setToolTip("0 = off, 0.5 = full. Lower reduces artifacts")
        solver_layout.addWidget(self.dither_spin, 3, 1)

        self.flip_h = QCheckBox("Flip Horizontal")
        self.flip_v = QCheckBox("Flip Vertical")
        solver_layout.addWidget(self.flip_h, 4, 0, 1, 2)
        solver_layout.addWidget(self.flip_v, 5, 0, 1, 2)

        cl.addWidget(solver_group)

        # generate
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setEnabled(False)
        self.generate_btn.setMinimumHeight(36)
        self.generate_btn.clicked.connect(self.generate)
        cl.addWidget(self.generate_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        cl.addWidget(self.progress_bar)

        self.status_label = QLabel("Load an image to begin")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        cl.addWidget(self.status_label)

        # export
        export_group  = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(4)

        self.save_preview_btn = QPushButton("Save Preview PNG")
        self.save_preview_btn.setEnabled(False)
        self.save_preview_btn.clicked.connect(self.save_preview)
        export_layout.addWidget(self.save_preview_btn)

        self.save_schem_btn = QPushButton("Save Schematic(s)")
        self.save_schem_btn.setEnabled(False)
        self.save_schem_btn.clicked.connect(self.save_schematics)
        export_layout.addWidget(self.save_schem_btn)

        cl.addWidget(export_group)
        cl.addStretch()

        # ── viewers (right) ──────────────────
        view_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(view_splitter, stretch=1)

        in_group  = QGroupBox("Input  (scroll=zoom, drag=pan, dblclick=fit)")
        in_layout = QVBoxLayout(in_group)
        in_layout.setContentsMargins(4, 16, 4, 4)
        self.input_viewer = ZoomableLabel("Drop an image to begin")
        in_layout.addWidget(self.input_viewer)
        view_splitter.addWidget(in_group)

        out_group  = QGroupBox("Preview  (scroll=zoom, drag=pan, dblclick=fit)")
        out_layout = QVBoxLayout(out_group)
        out_layout.setContentsMargins(4, 16, 4, 4)
        self.output_viewer = ZoomableLabel("Preview will appear here")
        out_layout.addWidget(self.output_viewer)
        view_splitter.addWidget(out_group)

        view_splitter.setSizes([500, 500])

    # ─────────────────────────────────────────
    #  Slots
    # ─────────────────────────────────────────

    def load_image(self, path: str):
        self.source_image = Image.open(path).convert("RGB")
        self.drop_zone.setText(os.path.basename(path))
        self.update_input_preview()
        self.generate_btn.setEnabled(True)
        self.result_blocks  = None
        self.result_heights = None
        self.save_preview_btn.setEnabled(False)
        self.save_schem_btn.setEnabled(False)
        self.status_label.setText("Image loaded — ready to generate")

    def update_input_preview(self):
        if self.source_image is None:
            return
        mw = self.maps_w.value()
        mh = self.maps_h.value()
        cropped = crop_to_tiles(self.source_image, mw, mh)
        self.input_viewer.set_pixmap(pil_to_qpixmap(cropped))
        self.status_label.setText(f"{mw*128}×{mh*128} px  ({mw}×{mh} maps)")

    def generate(self):
        if self.source_image is None:
            return

        mw = self.maps_w.value()
        mh = self.maps_h.value()

        import config
        config.MAX_HEIGHT_DIFF = self.mh_spin.value()
        config.BEAM_WIDTH      = self.bw_spin.value()

        cropped = crop_to_tiles(self.source_image, mw, mh)
        if self.flip_h.isChecked():
            cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v.isChecked():
            cropped = cropped.transpose(Image.FLIP_TOP_BOTTOM)

        arr = np.array(cropped, dtype=np.float32)

        # split into tiles[tc][tr]
        tiles = []
        for tc in range(mw):
            col_tiles = []
            for tr in range(mh):
                x0 = tc * 128
                y0 = tr * 128
                col_tiles.append(arr[y0:y0+128, x0:x0+128, :])
            tiles.append(col_tiles)

        self.tiles      = tiles
        self.total_cols = mw * mh * 128
        self._done_cols = 0

        self.generate_btn.setEnabled(False)
        self.save_preview_btn.setEnabled(False)
        self.save_schem_btn.setEnabled(False)
        self.progress_bar.setRange(0, self.total_cols)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating...")

        self.worker = GenerateWorker(
            tiles,
            height_penalty  = self.hp_spin.value(),
            dither_strength = self.dither_spin.value(),
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, tc: int, tr: int, col: int, cost: float):
        self._done_cols += 1
        self.progress_bar.setValue(self._done_cols)
        self.status_label.setText(
            f"Tile ({tc},{tr})  col {col}/128\ncost={cost:.1f}"
        )

    def on_finished(self, rendered, blocks, heights):
        self.rendered       = rendered
        self.result_blocks  = blocks
        self.result_heights = heights

        mw = self.maps_w.value()
        mh = self.maps_h.value()

        full = np.zeros((mh * 128, mw * 128, 3), dtype=np.uint8)
        for tc in range(mw):
            for tr in range(mh):
                x0 = tc * 128
                y0 = tr * 128
                full[y0:y0+128, x0:x0+128] = rendered[tc][tr]

        self.full_rendered = full
        self.output_viewer.set_pixmap(ndarray_to_qpixmap(full))

        self.generate_btn.setEnabled(True)
        self.save_preview_btn.setEnabled(True)
        self.save_schem_btn.setEnabled(True)
        self.progress_bar.setValue(self.total_cols)
        self.status_label.setText("Done!")

    def on_error(self, msg: str):
        self.status_label.setText(f"Error:\n{msg}")
        self.generate_btn.setEnabled(True)

    def save_preview(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preview", "mapart_preview.png", "PNG (*.png)"
        )
        if path:
            Image.fromarray(self.full_rendered).save(path)
            self.status_label.setText(f"Saved → {os.path.basename(path)}")

    def save_schematics(self):
        mw = self.maps_w.value()
        mh = self.maps_h.value()

        if mw * mh == 1:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Schematic", "mapart.schem", "Schematic (*.schem)"
            )
            if not path:
                return
            paths = {(0, 0): path}
        else:
            directory = QFileDialog.getExistingDirectory(
                self, "Select Output Directory"
            )
            if not directory:
                return
            paths = {
                (tc, tr): os.path.join(directory, f"mapart_{tc}_{tr}.schem")
                for tc in range(mw)
                for tr in range(mh)
            }

        for (tc, tr), path in paths.items():
            export_sponge(
                self.result_blocks[tc][tr],
                self.result_heights[tc][tr],
                output_path = path,
                name        = f"Map Art ({tc},{tr})",
            )

        self.status_label.setText(f"Saved {len(paths)} schematic(s)")


# ─────────────────────────────────────────────
#  Dark stylesheet
# ─────────────────────────────────────────────

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
QCheckBox::indicator:checked  { background: #2d5a8e; border-color: #2d5a8e; }
QCheckBox::indicator:hover    { border-color: #888; }
"""


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
