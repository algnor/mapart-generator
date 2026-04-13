import os
import numpy as np
import time
from PIL import Image
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QFileDialog, QProgressBar,
    QGroupBox, QSpinBox, QDoubleSpinBox, QSplitter, QGridLayout
)
from PyQt5.QtCore import Qt

import config
from export import export_sponge
from gui.widgets import ZoomableLabel, DropZone
from gui.worker import GenerateWorker
from gui.style import DARK_STYLE
from gui.utils import pil_to_qpixmap, ndarray_to_qpixmap, crop_to_tiles


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minecraft Map Art Generator")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(DARK_STYLE)

        self.source_image        = None
        self.full_rendered       = None
        self.result_blocks       = None
        self.result_heights      = None
        self.result_first_shades = None
        self.worker              = None
        self.total_cols          = 0
        self._done_cols          = 0

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_controls())

        view_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(view_splitter, stretch=1)

        in_group  = QGroupBox("Input  (scroll=zoom  drag=pan  dblclick=fit)")
        in_layout = QVBoxLayout(in_group)
        in_layout.setContentsMargins(4, 16, 4, 4)
        self.input_viewer = ZoomableLabel("Drop an image to begin")
        in_layout.addWidget(self.input_viewer)
        view_splitter.addWidget(in_group)

        out_group  = QGroupBox("Preview  (scroll=zoom  drag=pan  dblclick=fit)")
        out_layout = QVBoxLayout(out_group)
        out_layout.setContentsMargins(4, 16, 4, 4)
        self.output_viewer = ZoomableLabel("Preview will appear here")
        out_layout.addWidget(self.output_viewer)
        view_splitter.addWidget(out_group)

        view_splitter.setSizes([500, 500])

    def _build_controls(self) -> QWidget:
        controls = QWidget()
        controls.setFixedWidth(220)
        cl = QVBoxLayout(controls)
        cl.setSpacing(8)
        cl.setContentsMargins(0, 0, 0, 0)

        self.drop_zone = DropZone()
        self.drop_zone.image_dropped.connect(self.load_image)
        cl.addWidget(self.drop_zone)

        cl.addWidget(self._build_grid_group())
        cl.addWidget(self._build_solver_group())

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setEnabled(False)
        self.generate_btn.setMinimumHeight(36)
        self.generate_btn.clicked.connect(self.generate)
        cl.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMinimumHeight(28)
        self.cancel_btn.clicked.connect(self.cancel)
        cl.addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        cl.addWidget(self.progress_bar)

        self.status_label = QLabel("Load an image to begin")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #aaa; font-size: 13px; font-family: 'Courier New', monospace;"
        )
        cl.addWidget(self.status_label)

        cl.addWidget(self._build_export_group())
        cl.addStretch()

        return controls

    def _build_grid_group(self) -> QGroupBox:
        group  = QGroupBox("Map Grid")
        layout = QGridLayout(group)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Maps wide"), 0, 0)
        self.maps_w = QSpinBox()
        self.maps_w.setRange(1, 16)
        self.maps_w.setValue(1)
        self.maps_w.valueChanged.connect(self.update_input_preview)
        layout.addWidget(self.maps_w, 0, 1)

        layout.addWidget(QLabel("Maps tall"), 1, 0)
        self.maps_h = QSpinBox()
        self.maps_h.setRange(1, 16)
        self.maps_h.setValue(1)
        self.maps_h.valueChanged.connect(self.update_input_preview)
        layout.addWidget(self.maps_h, 1, 1)

        return group

    def _build_solver_group(self) -> QGroupBox:
        group  = QGroupBox("Solver")
        layout = QGridLayout(group)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Height Penalty"), 0, 0)
        self.hp_spin = QDoubleSpinBox()
        self.hp_spin.setRange(0.0, 20.0)
        self.hp_spin.setSingleStep(0.25)
        self.hp_spin.setValue(0.5)
        self.hp_spin.setToolTip("Higher = flatter, lower = more height variation")
        layout.addWidget(self.hp_spin, 0, 1)

        layout.addWidget(QLabel("Max Height"), 1, 0)
        self.mh_spin = QSpinBox()
        self.mh_spin.setRange(0, 16)
        self.mh_spin.setValue(config.MAX_HEIGHT)
        layout.addWidget(self.mh_spin, 1, 1)

        layout.addWidget(QLabel("Max Step"), 2, 0)
        self.ms_spin = QSpinBox()
        self.ms_spin.setRange(0, 16)
        self.ms_spin.setValue(config.MAX_STEP)
        layout.addWidget(self.ms_spin, 2, 1)

        layout.addWidget(QLabel("Beam Width"), 3, 0)
        self.bw_spin = QSpinBox()
        self.bw_spin.setRange(1, 256)
        self.bw_spin.setValue(config.BEAM_WIDTH)
        self.bw_spin.setToolTip("Higher = better quality, slower")
        layout.addWidget(self.bw_spin, 3, 1)

        layout.addWidget(QLabel("Dither Strength"), 4, 0)
        self.dither_spin = QDoubleSpinBox()
        self.dither_spin.setRange(0.0, 1.0)
        self.dither_spin.setSingleStep(0.05)
        self.dither_spin.setValue(0.2)
        self.dither_spin.setToolTip("0 = off, higher = more dithering (may cause artifacts)")
        layout.addWidget(self.dither_spin, 4, 1)

        self.per_beam_dither = QCheckBox("Per beam Dithering")
        layout.addWidget(self.per_beam_dither, 5, 0, 1, 2)

        self.flip_h = QCheckBox("Flip Horizontal")
        self.flip_v = QCheckBox("Flip Vertical")
        layout.addWidget(self.flip_h, 6, 0, 1, 2)
        layout.addWidget(self.flip_v, 7, 0, 1, 2)

        return group

    def _build_export_group(self) -> QGroupBox:
        group  = QGroupBox("Export")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        self.save_preview_btn = QPushButton("Save Preview PNG")
        self.save_preview_btn.setEnabled(False)
        self.save_preview_btn.clicked.connect(self.save_preview)
        layout.addWidget(self.save_preview_btn)

        self.save_schem_btn = QPushButton("Save Schematic(s)")
        self.save_schem_btn.setEnabled(False)
        self.save_schem_btn.clicked.connect(self.save_schematics)
        layout.addWidget(self.save_schem_btn)

        self.save_combined_btn = QPushButton("Save Combined Schematic")
        self.save_combined_btn.setEnabled(False)
        self.save_combined_btn.clicked.connect(self.save_combined_schematic)
        layout.addWidget(self.save_combined_btn)

        return group


    # ── slots ────────────────────────────────────────────────────────────────

    def load_image(self, path: str):
        self.source_image = Image.open(path).convert("RGB")
        self.drop_zone.setText(os.path.basename(path))
        self.update_input_preview()
        self.generate_btn.setEnabled(True)
        self.result_blocks       = None
        self.result_heights      = None
        self.result_first_shades = None
        self.save_preview_btn.setEnabled(False)
        self.save_schem_btn.setEnabled(False)
        self.save_combined_btn.setEnabled(False)
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

        config.MAX_STEP   = self.ms_spin.value()
        config.MAX_HEIGHT = self.mh_spin.value()
        config.BEAM_WIDTH = self.bw_spin.value()

        # recompute dh metadata in solver after config change
        import solver
        solver._DH_RANGE  = np.arange(-config.MAX_STEP, config.MAX_STEP + 1)
        solver._DH_SHADES = np.where(
            solver._DH_RANGE > 0, 2,
            np.where(solver._DH_RANGE < 0, 0, 1)
        ).astype(np.int8)
        solver._DH_COSTS = np.abs(solver._DH_RANGE).astype(np.float32)
        solver._N_DH     = len(solver._DH_RANGE)

        cropped = crop_to_tiles(self.source_image, mw, mh)
        if self.flip_h.isChecked():
            cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v.isChecked():
            cropped = cropped.transpose(Image.FLIP_TOP_BOTTOM)

        arr = np.array(cropped, dtype=np.float32)
        # tiles[tc][tr] -> (128, 128, 3), row=0 is top of image
        tiles = [
            [arr[tr*128:(tr+1)*128, tc*128:(tc+1)*128, :] for tr in range(mh)]
            for tc in range(mw)
        ]

        # progress is per map-column: mw * 128 total
        self.total_cols = mw * 128
        self._done_cols = 0

        self.full_rendered = np.zeros((mh * 128, mw * 128, 3), dtype=np.uint8)
        self._start_time   = time.perf_counter()

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_preview_btn.setEnabled(False)
        self.save_schem_btn.setEnabled(False)
        self.save_combined_btn.setEnabled(False)
        self.progress_bar.setRange(0, self.total_cols)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating...")

        self.worker = GenerateWorker(
            tiles,
            height_penalty  = self.hp_spin.value(),
            dither_strength = self.dither_spin.value(),
            per_beam_dither = self.per_beam_dither.isChecked(),
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.worker.wait()
        self.cancel_btn.setEnabled(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Cancelled")

    def on_progress(self, tc: int, col: int, cost: float, col_pixels: np.ndarray):
        self._done_cols += 1
        self.progress_bar.setValue(self._done_cols)
        elapsed = time.perf_counter() - self._start_time
        bps = (self._done_cols * 128) / elapsed if elapsed > 0 else 0
        self.status_label.setText(
            f"Tile col {tc*128 + col + 1}/{self.total_cols}\n"
            f"cost={cost:.1f} | {bps:.0f} blocks/s"
        )

        # col_pixels is (map_rows*128, 3), index 0 = top of full image
        x  = tc * 128 + col
        self.full_rendered[:, x, :] = col_pixels
        if self._done_cols % config.PREVIEW_INTERVAL == 0 or self._done_cols == self.total_cols:
            self.output_viewer.set_pixmap(ndarray_to_qpixmap(self.full_rendered))

    def on_finished(self, blocks, heights, first_shades):
        self.result_blocks       = blocks
        self.result_heights      = heights
        self.result_first_shades = first_shades

        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.save_preview_btn.setEnabled(True)
        self.save_schem_btn.setEnabled(True)
        self.save_combined_btn.setEnabled(True)
        self.progress_bar.setValue(self.total_cols)
        elapsed      = time.perf_counter() - self._start_time
        total_blocks = self.total_cols * 128
        bps          = total_blocks / elapsed if elapsed > 0 else 0
        self.status_label.setText(f"Done!  {elapsed:.1f}s\n{bps:.0f} blocks/s")

    def on_error(self, msg: str):
        self.status_label.setText(f"Error:\n{msg}")
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

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
            directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
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
                self.result_first_shades[tc][tr],
                output_path = path,
                name        = f"Map Art ({tc},{tr})",
            )
        self.status_label.setText(f"Saved {len(paths)} schematic(s)")

    def save_combined_schematic(self):
        mw = self.maps_w.value()
        mh = self.maps_h.value()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Combined Schematic", "mapart_full.schem", "Schematic (*.schem)"
        )
        if not path:
            return

        from export import export_sponge_combined
        export_sponge_combined(
            self.result_blocks,
            self.result_heights,
            self.result_first_shades,
            map_cols     = mw,
            map_rows     = mh,
            output_path  = path,
            name         = "Map Art (Full)",
        )
        self.status_label.setText(f"Saved → {os.path.basename(path)}")
