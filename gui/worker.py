import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from color import tonemap
import config
from fs_predither import fs_predither
from solver import solve_strip
from renderer import render_strip


class GenerateWorker(QThread):
    progress = pyqtSignal(int, int, float, np.ndarray)  # tc, col, cost, col_pixels (map_rows*128, 4)
    finished = pyqtSignal(object, object, object)        # blocks, heights, first_shades
    processed  = pyqtSignal(np.ndarray)                  # (H, W, 3) uint8
    error    = pyqtSignal(str)

    def __init__(self, image, size, height_penalty, dither_strength):
        super().__init__()
        self.image           = image
        self.size            = size
        self.height_penalty  = height_penalty
        self.dither_strength = dither_strength
        self._cancel         = False

    def cancel(self):
        self._cancel = True

    def run(self):
        has_alpha = self.image.shape[2] == 4
        rgb   = self.image[..., :3]
        alpha = self.image[..., 3:4] if has_alpha else None

        dithered_rgb = fs_predither(rgb, self.dither_strength)
        self.processed.emit(dithered_rgb.clip(0, 255).astype(np.uint8))

        if alpha is not None:
            dithered = np.concatenate([dithered_rgb, alpha], axis=-1)
        else:
            dithered = dithered_rgb

        self.tiles = [
            [dithered[tr*128:(tr+1)*128, tc*128:(tc+1)*128, :] for tr in range(self.size[1])]
            for tc in range(self.size[0])
        ]

        try:
            map_cols = len(self.tiles)
            map_rows = len(self.tiles[0])

            blocks       = [[None] * map_rows for _ in range(map_cols)]
            heights      = [[None] * map_rows for _ in range(map_cols)]
            first_shades = [[None] * map_rows for _ in range(map_cols)]

            for tc in range(map_cols):
                for col in range(128):
                    if self._cancel:
                        return

                    # concatenate all tile rows into one tall strip (map_rows*128, 3or4)
                    strip = np.concatenate([
                        self.tiles[tc][tr][:, col, :]
                        for tr in range(map_rows)
                    ], axis=0).astype(np.float32)

                    # only jitter RGB channels, not alpha
                    rng = np.random.default_rng(tc + col)
                    jitter = rng.uniform(-2.0, 2.0, (strip.shape[0], 3)).astype(np.float32)
                    strip[:, :3] = np.clip(strip[:, :3] + jitter, 0, 255)

                    path, cost, first_shade = solve_strip(
                        strip,
                        height_penalty=self.height_penalty,
                    )

                    for tr in range(map_rows):
                        chunk = path[tr * 128 : (tr + 1) * 128]

                        if blocks[tc][tr] is None:
                            blocks[tc][tr]       = []
                            heights[tc][tr]      = []
                            first_shades[tc][tr] = []

                        blocks[tc][tr].append([p[0] for p in chunk])
                        heights[tc][tr].append([p[1] for p in chunk])

                        if tr == 0:
                            first_shades[tc][tr].append(first_shade)
                        else:
                            prev_h = path[tr * 128 - 1][1]
                            cur_h  = path[tr * 128][1]
                            shade  = 2 if cur_h > prev_h else (0 if cur_h < prev_h else 1)
                            first_shades[tc][tr].append(shade)

                    col_pixels = render_strip(path, first_shade)  # (map_rows*128, 4)
                    self.progress.emit(tc, col, cost, col_pixels)

            self.finished.emit(blocks, heights, first_shades)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

