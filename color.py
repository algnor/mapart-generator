import numpy as np
from config import BLOCKS, SHADE_MULTIPLIERS

BLOCK_NAMES  = list(BLOCKS.keys())
BLOCK_COLORS = np.array([BLOCKS[b] for b in BLOCK_NAMES], dtype=np.float32)
N_BLOCKS     = len(BLOCK_NAMES)


def rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    """(N,3) float32 0-255 -> (N,3) CIE-LAB"""
    rgb   = rgb_array / 255.0
    mask  = rgb > 0.04045
    lin   = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M     = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz   = (lin @ M.T) / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    mask2 = xyz > 0.008856
    f     = np.where(mask2, xyz ** (1.0/3.0), (903.3 * xyz + 16.0) / 116.0)
    L     = 116.0 * f[:, 1] - 16.0
    a     = 500.0 * (f[:, 0] - f[:, 1])
    b     = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def cie94_batch(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """lab1: (N,3), lab2: (3,) -> (N,) CIE94 distance"""
    dL  = lab1[:, 0] - lab2[0]
    C1  = np.sqrt(lab1[:, 1]**2 + lab1[:, 2]**2)
    C2  = float(np.sqrt(lab2[1]**2 + lab2[2]**2))
    dC  = C1 - C2
    dH2 = np.maximum(
        (lab1[:, 1] - lab2[1])**2 + (lab1[:, 2] - lab2[2])**2 - dC**2, 0.0
    )
    SC = 1.0 + 0.045 * (C1 + C2) / 2
    SH = 1.0 + 0.015 * (C1 + C2) / 2
    return np.sqrt(dL**2 + (dC/SC)**2 + dH2/SH**2)

def cie94_batch_multi(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """lab1: (N, 3), lab2: (M, 3) -> (M, N) CIE94 distances"""
    # lab1 is the palette (N_BLOCKS), lab2 is the query points (beam paths)
    dL  = lab1[None, :, 0] - lab2[:, None, 0]          # (M, N)
    C1  = np.sqrt(lab1[None, :, 1]**2 + lab1[None, :, 2]**2)   # (1, N)
    C2  = np.sqrt(lab2[:, 1]**2 + lab2[:, 2]**2)[:, None]      # (M, 1)
    dC  = C1 - C2                                       # (M, N)
    dH2 = np.maximum(
        (lab1[None, :, 1] - lab2[:, None, 1])**2 +
        (lab1[None, :, 2] - lab2[:, None, 2])**2 - dC**2, 0.0
    )
    SC = 1.0 + 0.045 * (C1 + C2) / 2
    SH = 1.0 + 0.015 * (C1 + C2) / 2
    return np.sqrt(dL**2 + (dC / SC)**2 + dH2 / SH**3) # (M, N)


def precompute_shaded():
    shaded_rgb = np.zeros((N_BLOCKS, 3, 3), dtype=np.float32)
    shaded_lab = np.zeros((N_BLOCKS, 3, 3), dtype=np.float32)
    for si, m in enumerate(SHADE_MULTIPLIERS):
        rgb = np.clip(BLOCK_COLORS * m, 0, 255)
        shaded_rgb[:, si, :] = rgb
        shaded_lab[:, si, :] = rgb_to_lab(rgb)
    return shaded_lab, shaded_rgb


SHADED_LAB, SHADED_RGB = precompute_shaded()