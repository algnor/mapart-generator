import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from color import tonemap
import config
from fs_predither import fs_predither
from solver import solve_strip
from renderer import render_strip


class GenerateWorker(QThread):
    # col_pixels is the full-height column across all tile rows
    progress = pyqtSignal(int, int, float, np.ndarray)  # tc, col, cost, col_pixels (map_rows*128, 3)
    finished = pyqtSignal(object, object, object)        # blocks, heights, first_shades
    processed  = pyqtSignal(np.ndarray)   # (H, W, 3) uint8
    error    = pyqtSignal(str)

    def __init__(self, image, size, height_penalty, dither_strength):
        super().__init__()
        
        self.image = image
        self.size = size
        self.height_penalty  = height_penalty
        self.dither_strength = dither_strength
        self._cancel         = False



    def cancel(self):
        self._cancel = True

    def run(self):
        self.dithered = fs_predither(self.image, self.dither_strength)
        self.processed.emit(self.dithered.clip(0, 255).astype(np.uint8))

        self.tiles = [
            [self.dithered[tr*128:(tr+1)*128, tc*128:(tc+1)*128, :] for tr in range(self.size[1])]
            for tc in range(self.size[0])
        ]
        try:
            map_cols = len(self.tiles)
            map_rows = len(self.tiles[0])

            # blocks[tc][tr] = list of 128 columns, each column = list of 128 block indices
            blocks       = [[None] * map_rows for _ in range(map_cols)]
            heights      = [[None] * map_rows for _ in range(map_cols)]
            first_shades = [[None] * map_rows for _ in range(map_cols)]

            for tc in range(map_cols):
                for col in range(128):
                    if self._cancel:
                        return

                    # concatenate all tile rows into one tall strip (map_rows*128, 3)
                    strip = np.concatenate([
                        self.tiles[tc][tr][:, col, :]
                        for tr in range(map_rows)
                    ], axis=0)

                    rng = np.random.default_rng(tc + col)
                    strip = strip + rng.uniform(-2.0, 2.0, strip.shape).astype(np.float32)
                    strip = np.clip(strip, 0, 255)


                    path, cost, first_shade = solve_strip(
                        strip,
                        height_penalty   = self.height_penalty
                    )

                    # split path into per-tile chunks and store
                    for tr in range(map_rows):
                        chunk = path[tr * 128 : (tr + 1) * 128]

                        if blocks[tc][tr] is None:
                            blocks[tc][tr]       = []
                            heights[tc][tr]      = []
                            first_shades[tc][tr] = []

                        blocks[tc][tr].append([p[0] for p in chunk])
                        heights[tc][tr].append([p[1] for p in chunk])

                        # first_shade only matters for tr=0 (top of the full strip)
                        if tr == 0:
                            first_shades[tc][tr].append(first_shade)
                        else:
                            # shade at chunk start is determined by the previous chunk's last height
                            prev_h = path[tr * 128 - 1][1]
                            cur_h  = path[tr * 128][1]
                            shade  = 2 if cur_h > prev_h else (0 if cur_h < prev_h else 1)
                            first_shades[tc][tr].append(shade)

                    col_pixels = render_strip(path, first_shade)  # (map_rows*128, 3)
                    self.progress.emit(tc, col, cost, col_pixels)

            self.finished.emit(blocks, heights, first_shades)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())