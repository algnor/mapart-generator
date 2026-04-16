import numpy as np
import config

BLOCK_NAMES  = list(config.BLOCKS.keys())
BLOCK_COLORS = np.array([config.BLOCKS[b] for b in BLOCK_NAMES], dtype=np.float32)
N_BLOCKS     = len(BLOCK_NAMES)

_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)

_M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)

def rgb_to_oklab(rgb_array: np.ndarray) -> np.ndarray:
    """(N,3) float32 0-255 -> (N,3) OKLab"""
    rgb  = rgb_array / 255.0
    mask = rgb > 0.04045
    lin  = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    lms  = lin @ _M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T

def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(N,3) OKLab -> (N,3) float32 0-255"""
    M2_inv = np.linalg.inv(_M2).astype(np.float32)
    M1_inv = np.linalg.inv(_M1).astype(np.float32)
    lms_   = lab @ M2_inv.T
    lms    = lms_ ** 3
    lin    = lms @ M1_inv.T
    lin    = np.clip(lin, 0, 1)
    mask   = lin > 0.0031308
    rgb    = np.where(mask, 1.055 * lin ** (1 / 2.4) - 0.055, lin * 12.92)
    return np.clip(rgb * 255, 0, 255).astype(np.float32)

def oklab_dist_batch(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    d = lab1 - lab2
    d[:, 1] *= config.CHROMA_WEIGHT
    d[:, 2] *= config.CHROMA_WEIGHT
    return np.sqrt(np.einsum('ij,ij->i', d, d))

def oklab_dist_batch_multi(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    d = lab1[None, :, :] - lab2[:, None, :]
    d[:, :, 1] *= config.CHROMA_WEIGHT
    d[:, :, 2] *= config.CHROMA_WEIGHT
    return np.sqrt(np.einsum('mnc,mnc->mn', d, d))


def precompute_shaded():
    shaded_rgb = np.zeros((N_BLOCKS, 3, 3), dtype=np.float32)
    shaded_lab = np.zeros((N_BLOCKS, 3, 3), dtype=np.float32)
    for si, m in enumerate(config.SHADE_MULTIPLIERS):
        rgb = np.clip(BLOCK_COLORS * m, 0, 255)
        shaded_rgb[:, si, :] = rgb
        shaded_lab[:, si, :] = rgb_to_oklab(rgb)
    return shaded_lab, shaded_rgb

def tonemap(image_rgb: np.ndarray, threshold: float = 220.0, strength: float = 0.5) -> np.ndarray:
    """
    Compress only highlights above threshold, leave shadows/midtones untouched.
    threshold: 0-255 luminance above which to compress
    strength: 0=no compression, 1=hard clip at threshold
    """
    img = image_rgb / 255.0
    t   = threshold / 255.0
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    # how far above threshold, 0 below
    excess = np.maximum(lum - t, 0.0) / (1.0 - t + 1e-6)   # 0..1+
    # compress excess
    compressed_excess = excess / (1.0 + excess) * strength
    # new luminance
    new_lum = lum - compressed_excess * (1.0 - t)

    scale = np.where(lum > 1e-6, new_lum / lum, 1.0)[..., None]
    return np.clip(img * scale * 255.0, 0, 255).astype(np.float32)



SHADED_LAB, SHADED_RGB = precompute_shaded()
