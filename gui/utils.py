import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
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
