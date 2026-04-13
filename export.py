import time
import numpy as np
import nbtlib
from nbtlib import File, Compound, List, Int, Short, String, ByteArray

from color import BLOCK_NAMES


def _solver_name_to_mc(n: str) -> str:
    return f"minecraft:{n}"


def _encode_varint_array(indices: np.ndarray) -> np.ndarray:
    out = []
    for idx in indices:
        idx = int(idx)
        while True:
            b = idx & 0x7F
            idx >>= 7
            out.append(b | 0x80 if idx else b)
            if not idx:
                break
    return np.array(out, dtype=np.int8)


def export_sponge(
    all_blocks:    list,
    all_heights:   list,
    output_path:   str = "mapart.schem",
    name:          str = "Map Art",
    data_version:  int = 3700,
):
    """
    all_blocks  : [col][row] → block_idx
    all_heights : [col][row] → height

    Axes:
      X = image column  (WIDTH  = 128)
      Y = block height
      Z = image row     (LENGTH = 128)
      flat index = x + z*Width + y*Width*Length
    """
    n_cols = len(all_blocks)
    n_rows = len(all_blocks[0])
    min_h  = min(h for col in all_heights for h in col)
    max_h  = max(h for col in all_heights for h in col)

    WIDTH  = n_cols
    HEIGHT = max_h - min_h + 1
    LENGTH = n_rows

    mc_names    = [_solver_name_to_mc(n) for n in BLOCK_NAMES]
    palette_map = {"minecraft:air": 0}
    for mc in mc_names:
        if mc not in palette_map:
            palette_map[mc] = len(palette_map)

    flat = np.zeros(WIDTH * HEIGHT * LENGTH, dtype=np.int32)
    for x, (blk_col, h_col) in enumerate(zip(all_blocks, all_heights)):
        for z, (b, h) in enumerate(zip(blk_col, h_col)):
            y = h - min_h
            flat[x + z * WIDTH + y * WIDTH * LENGTH] = palette_map[mc_names[b]]

    nbt_palette = Compound({k: Int(v) for k, v in palette_map.items()})
    block_data  = _encode_varint_array(flat)

    nbt = File({
        "Schematic": Compound({
            "Version":       Int(2),
            "DataVersion":   Int(data_version),
            "Width":         Short(WIDTH),
            "Height":        Short(HEIGHT),
            "Length":        Short(LENGTH),
            "Offset":        nbtlib.IntArray(np.array([0, 0, 0], dtype=np.int32)),
            "PaletteMax":    Int(len(palette_map)),
            "Palette":       nbt_palette,
            "BlockData":     ByteArray(block_data),
            "BlockEntities": List[Compound]([]),
            "Metadata": Compound({
                "Name":   String(name),
                "Author": String("mapart_generator"),
                "Date":   nbtlib.Long(int(time.time() * 1000)),
            }),
        })
    }, gzipped=True)

    nbt.save(output_path)
    print(f"Saved {output_path}  ({WIDTH}x{HEIGHT}x{LENGTH}, "
          f"{len(palette_map)} palette entries)")
