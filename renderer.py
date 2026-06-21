import numpy as np
from color import SHADED_RGB


def render_strip(path: list, first_shade: int = 1) -> np.ndarray:
    L        = len(path)
    rendered = np.zeros((L, 4), dtype=np.uint8)  # RGBA
    for i, (b, h) in enumerate(path):
        if b == -1:
            # transparent — leave as 0,0,0,0
            continue
        if i == 0:
            shade = first_shade
        elif h > path[i-1][1]:
            shade = 2
        elif h < path[i-1][1]:
            shade = 0
        else:
            shade = 1
        rendered[i, :3] = np.clip(SHADED_RGB[b, shade], 0, 255).astype(np.uint8)
        rendered[i, 3]  = 255
    return rendered

