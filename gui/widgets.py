from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QPixmap


class ZoomableLabel(QWidget):
    def __init__(self, placeholder=""):
        super().__init__()
        self.pixmap             = None
        self.zoom               = 1.0
        self.offset             = QPointF(0, 0)
        self._drag_start        = None
        self._drag_offset_start = None
        self._placeholder       = placeholder
        self.setMinimumSize(256, 256)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        self.synced: list[ZoomableLabel] = []


    def set_pixmap(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.update()

    def fit(self):
        if not self.pixmap:
            return
        from PyQt5.QtCore import QRectF
        sw = self.width()  / self.pixmap.width()
        sh = self.height() / self.pixmap.height()
        self.zoom   = min(sw, sh)
        self.offset = QPointF(
            (self.width()  - self.pixmap.width()  * self.zoom) / 2,
            (self.height() - self.pixmap.height() * self.zoom) / 2,
        )
        self.sync_linked_viewers()

    def paintEvent(self, _):
        from PyQt5.QtCore import QRectF
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
        self.sync_linked_viewers()


    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_start        = QPointF(e.pos())
            self._drag_offset_start = QPointF(self.offset)


    def mouseMoveEvent(self, e):
        if self._drag_start is not None:
            self.offset = self._drag_offset_start + QPointF(e.pos()) - self._drag_start
            self.update()
            self.sync_linked_viewers()

    def mouseReleaseEvent(self, e):
        self._drag_start = None

    def mouseDoubleClickEvent(self, e):
        self.fit()
        self.update()

    def resizeEvent(self, e):
        self.fit()
    
    def sync_linked_viewers(self):
        for viewer in self.synced:
            viewer.offset = self.offset
            viewer.zoom = self.zoom
            viewer.update()




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
