import numpy as np
from color import oklab_dist_batch_multi, rgb_to_oklab, SHADED_LAB, oklab_to_rgb, oklab_dist_batch


def fs_predither(image_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    H, W = image_rgb.shape[:2]
    palette_lab = SHADED_LAB.reshape(-1, 3).astype(np.float32)  # (P, 3)
    lab = rgb_to_oklab(image_rgb.reshape(-1, 3)).reshape(H, W, 3).copy()

    out = np.zeros((H, W, 3), dtype=np.float32)

    for d in range(H + W - 1):
        # all (y, x) on this anti-diagonal
        y0 = max(0, d - W + 1)
        y1 = min(d + 1, H)
        ys = np.arange(y0, y1)
        xs = d - ys

        pixels = lab[ys, xs]                                     # (K, 3)
        dists  = oklab_dist_batch_multi(palette_lab, pixels)     # (K, P)
        best   = np.argmin(dists, axis=1)                        # (K,)
        new    = palette_lab[best]                               # (K, 3)
        out[ys, xs] = new


        err = (pixels - new) * strength                          # (K, 3)

        # diffuse to 4 neighbours
        mask = xs + 1 < W
        lab[ys[mask],     xs[mask] + 1] += err[mask] * (7 / 16)

        mask = (y + 1 < H for y in ys)  # rebuild per condition
        yn1  = ys + 1
        valid_y = yn1 < H

        m = valid_y & (xs - 1 >= 0)
        lab[yn1[m], xs[m] - 1] += err[m] * (3 / 16)

        m = valid_y
        lab[yn1[m], xs[m]    ] += err[m] * (5 / 16)

        m = valid_y & (xs + 1 < W)
        lab[yn1[m], xs[m] + 1] += err[m] * (1 / 16)

    return oklab_to_rgb(out.reshape(-1, 3)).reshape(H, W, 3)
