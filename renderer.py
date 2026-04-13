import numpy as np
from color import SHADED_RGB


def render_strip(path: list) -> np.ndarray:
    """
    path     : [(block_idx, height), ...]
    returns  : (L, 3) uint8 RGB
    """
    L        = len(path)
    rendered = np.zeros((L, 3), dtype=np.uint8)
    for i, (b, h) in enumerate(path):
        if i == 0:
            shade = 1
        elif h > path[i-1][1]:
            shade = 2
        elif h < path[i-1][1]:
            shade = 0
        else:
            shade = 1
        rendered[i] = np.clip(SHADED_RGB[b, shade], 0, 255).astype(np.uint8)
    return rendered
