import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from solver import solve_strip
from renderer import render_strip


class GenerateWorker(QThread):
    progress = pyqtSignal(int, int, int, float)
    finished = pyqtSignal(object, object, object)
    error    = pyqtSignal(str)

    def __init__(self, tiles, height_penalty, dither_strength):
        super().__init__()
        self.tiles           = tiles
        self.height_penalty  = height_penalty
        self.dither_strength = dither_strength

    def run(self):
        try:
            map_cols = len(self.tiles)
            map_rows = len(self.tiles[0])

            rendered = [[None] * map_rows for _ in range(map_cols)]
            blocks   = [[None] * map_rows for _ in range(map_cols)]
            heights  = [[None] * map_rows for _ in range(map_cols)]

            for tc in range(map_cols):
                for tr in range(map_rows):
                    pixels       = self.tiles[tc][tr]
                    rimg         = np.zeros((128, 128, 3), dtype=np.uint8)
                    tile_blocks  = []
                    tile_heights = []

                    for col in range(128):
                        strip      = pixels[:, col, :]
                        path, cost = solve_strip(
                            strip,
                            height_penalty  = self.height_penalty,
                            dither_strength = self.dither_strength,
                        )
                        tile_blocks.append([p[0] for p in path])
                        tile_heights.append([p[1] for p in path])
                        rimg[:, col] = render_strip(path)
                        self.progress.emit(tc, tr, col + 1, cost)

                    rendered[tc][tr] = rimg
                    blocks[tc][tr]   = tile_blocks
                    heights[tc][tr]  = tile_heights

            self.finished.emit(rendered, blocks, heights)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())
