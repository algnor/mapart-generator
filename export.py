import time
import numpy as np
import nbtlib
from nbtlib import File, Compound, List, Int, Short, String, ByteArray
from color import BLOCK_NAMES

_SACRIFICIAL_BLOCK = "minecraft:cobblestone"
_SUPPORT_BLOCK     = "minecraft:cobblestone"

# blocks that cannot be free-standing and need a solid block below them
_NEEDS_SUPPORT = {
    "minecraft:white_candle",
}


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


def _build_palette(mc_names):
    palette_map = {"minecraft:air": 0}
    if _SACRIFICIAL_BLOCK not in palette_map:
        palette_map[_SACRIFICIAL_BLOCK] = len(palette_map)
    if _SUPPORT_BLOCK not in palette_map:
        palette_map[_SUPPORT_BLOCK] = len(palette_map)
    for mc in mc_names:
        if mc not in palette_map:
            palette_map[mc] = len(palette_map)
    return palette_map


def _build_flat(all_blocks, all_heights, first_shades, n_x, n_z, palette_map, mc_names):
    """
    Shared logic for building the flat block array.
    Returns (flat, WIDTH, HEIGHT, LENGTH, min_h)
    Z=0 = sacrificial row, Z=1..n_z = map blocks.
    Support blocks are injected at Y-1 for any block in _NEEDS_SUPPORT.
    """
    sac_heights = []
    for x in range(n_x):
        h0    = all_heights[x][0]
        shade = first_shades[x]
        sac_h = h0 - 1 if shade == 2 else (h0 + 1 if shade == 0 else h0)
        sac_heights.append(sac_h)

    # collect all heights including potential support positions
    all_h_flat = [h for col in all_heights for h in col] + sac_heights
    for x, (blk_col, h_col) in enumerate(zip(all_blocks, all_heights)):
        for b, h in zip(blk_col, h_col):
            if mc_names[b] in _NEEDS_SUPPORT:
                all_h_flat.append(h - 1)

    min_h = min(all_h_flat)
    max_h = max(all_h_flat)

    WIDTH  = n_x
    HEIGHT = max_h - min_h + 1
    LENGTH = n_z + 1

    flat = np.zeros(WIDTH * HEIGHT * LENGTH, dtype=np.int32)

    # sacrificial row
    for x in range(n_x):
        y = sac_heights[x] - min_h
        flat[x + 0 * WIDTH + y * WIDTH * LENGTH] = palette_map[_SACRIFICIAL_BLOCK]

    # map blocks + supports
    for x, (blk_col, h_col) in enumerate(zip(all_blocks, all_heights)):
        for z, (b, h) in enumerate(zip(blk_col, h_col)):
            mc = mc_names[b]
            y  = h - min_h
            flat[x + (z + 1) * WIDTH + y * WIDTH * LENGTH] = palette_map[mc]
            if mc in _NEEDS_SUPPORT:
                y_sup = y - 1
                if y_sup >= 0:
                    idx = x + (z + 1) * WIDTH + y_sup * WIDTH * LENGTH
                    # only place support if that cell is air
                    if flat[idx] == palette_map["minecraft:air"]:
                        flat[idx] = palette_map[_SUPPORT_BLOCK]

    return flat, WIDTH, HEIGHT, LENGTH, min_h


def export_sponge(
    all_blocks:   list,
    all_heights:  list,
    first_shades: list,
    output_path:  str = "mapart.schem",
    name:         str = "Map Art",
    data_version: int = 3700,
):
    n_x = len(all_blocks)
    n_z = len(all_blocks[0])

    mc_names    = [_solver_name_to_mc(n) for n in BLOCK_NAMES]
    palette_map = _build_palette(mc_names)

    flat, WIDTH, HEIGHT, LENGTH, _ = _build_flat(
        all_blocks, all_heights, first_shades, n_x, n_z, palette_map, mc_names
    )

    _write_schem(flat, WIDTH, HEIGHT, LENGTH, palette_map, output_path, name, data_version)
    print(f"Saved {output_path}  ({WIDTH}x{HEIGHT}x{LENGTH}, "
          f"{len(palette_map)} palette entries)")


def export_sponge_combined(
    all_blocks:   list,
    all_heights:  list,
    first_shades: list,
    map_cols:     int,
    map_rows:     int,
    output_path:  str = "mapart_full.schem",
    name:         str = "Map Art (Full)",
    data_version: int = 3700,
):
    total_x = map_cols * 128
    total_z = map_rows * 128

    flat_blocks  = [[None] * total_z for _ in range(total_x)]
    flat_heights = [[None] * total_z for _ in range(total_x)]
    flat_shades  = [None] * total_x

    for tc in range(map_cols):
        for tr in range(map_rows):
            for col in range(128):
                x = tc * 128 + col
                for row in range(128):
                    z = tr * 128 + row
                    flat_blocks[x][z]  = all_blocks[tc][tr][col][row]
                    flat_heights[x][z] = all_heights[tc][tr][col][row]
                if tr == 0:
                    flat_shades[x] = first_shades[tc][tr][col]

    mc_names    = [_solver_name_to_mc(n) for n in BLOCK_NAMES]
    palette_map = _build_palette(mc_names)

    flat, WIDTH, HEIGHT, LENGTH, _ = _build_flat(
        flat_blocks, flat_heights, flat_shades, total_x, total_z, palette_map, mc_names
    )

    _write_schem(flat, WIDTH, HEIGHT, LENGTH, palette_map, output_path, name, data_version)
    print(f"Saved {output_path}  ({WIDTH}x{HEIGHT}x{LENGTH}, "
          f"{len(palette_map)} palette entries)")


def _write_schem(flat, WIDTH, HEIGHT, LENGTH, palette_map, output_path, name, data_version):
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
