import numpy as np
from PIL import Image
import nbtlib
import time
from nbtlib import File, Compound, List, Int, Short, String, ByteArray

SHADE_MULTIPLIERS = (0.71, 0.86, 1.0)

BLOCKS = {
    "wool_white":          (255, 255, 255),
    "wool_orange":         (216, 127,  51),
    "wool_magenta":        (178,  76, 216),
    "wool_light_blue":     (102, 153, 216),
    "wool_yellow":         (229, 229,  51),
    "wool_lime":           (127, 204,  25),
    "wool_pink":           (242, 127, 165),
    "wool_gray":           ( 76,  76,  76),
    "wool_light_gray":     (153, 153, 153),
    "wool_cyan":           ( 76, 127, 153),
    "wool_purple":         (127,  63, 178),
    "wool_blue":           ( 51,  76, 178),
    "wool_brown":          (102,  76,  51),
    "wool_green":          (102, 127,  51),
    "wool_red":            (153,  51,  51),
    "wool_black":          ( 25,  25,  25),
    "ice":                 (160, 160, 255),
    "clay":                (164, 168, 184),
    "dirt":                (151, 109,  77),
    "stone":               (112, 112, 112),
    "oak_log":             (143, 119,  72),
    "leaves":              (  0, 124,   0),
    "concrete_white":      (209, 209, 209),
    "concrete_orange":     (224,  97,   0),
    "concrete_magenta":    (169,  48, 159),
    "concrete_light_blue": ( 36, 137, 199),
    "concrete_yellow":     (240, 175,   0),
    "concrete_lime":       ( 94, 169,  24),
    "concrete_pink":       (214, 101, 143),
    "concrete_gray":       ( 55,  58,  62),
    "concrete_light_gray": (125, 125, 115),
    "concrete_cyan":       ( 21, 119, 136),
    "concrete_purple":     (100,  32, 156),
    "concrete_blue":       ( 45,  47, 143),
    "concrete_brown":      ( 96,  60,  32),
    "concrete_green":      ( 73,  91,  36),
    "concrete_red":        (142,  33,  33),
    "concrete_black":      (  8,  10,  15),
    "terracotta_white":    (209, 178, 161),
    "terracotta_orange":   (162,  84,  38),
    "terracotta_magenta":  (149,  88, 108),
    "terracotta_light_blue":(113, 108, 137),
    "terracotta_yellow":   (186, 133,  35),
    "terracotta_lime":     (103, 117,  52),
    "terracotta_pink":     (160,  77,  78),
    "terracotta_gray":     ( 57,  41,  35),
    "terracotta_light_gray":(135, 107,  98),
    "terracotta_cyan":     ( 87,  92,  92),
    "terracotta_purple":   (122,  42,  84),
    "terracotta_blue":     ( 74,  59,  91),
    "terracotta_brown":    ( 77,  51,  35),
    "terracotta_green":    ( 76,  83,  42),
    "terracotta_red":      (143,  61,  46),
    "terracotta_black":    ( 37,  22,  16),
    "grass":               (127, 178,  56),
    "sand":                (247, 233, 163),
    "snow":                (255, 255, 255),
}

BLOCK_NAMES  = list(BLOCKS.keys())
BLOCK_COLORS = np.array([BLOCKS[b] for b in BLOCK_NAMES], dtype=np.float32)
N_BLOCKS     = len(BLOCK_NAMES)

# ─────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────

def rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    rgb  = rgb_array / 255.0
    mask = rgb > 0.04045
    lin  = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M    = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz  = (lin @ M.T) / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    mask2 = xyz > 0.008856
    f    = np.where(mask2, xyz ** (1.0/3.0), (903.3 * xyz + 16.0) / 116.0)
    L    = 116.0 * f[:, 1] - 16.0
    a    = 500.0 * (f[:, 0] - f[:, 1])
    b    = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def cie94_batch(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    dL  = lab1[:, 0] - lab2[0]
    C1  = np.sqrt(lab1[:, 1]**2 + lab1[:, 2]**2)
    C2  = float(np.sqrt(lab2[1]**2 + lab2[2]**2))
    dC  = C1 - C2
    dH2 = np.maximum((lab1[:,1]-lab2[1])**2 + (lab1[:,2]-lab2[2])**2 - dC**2, 0.0)
    SC  = 1.0 + 0.045 * C1
    SH  = 1.0 + 0.015 * C1
    return np.sqrt(dL**2 + (dC/SC)**2 + dH2/SH**2)


def precompute_shaded():
    shaded_rgb = np.zeros((N_BLOCKS, 3, 3), dtype=np.float32)
    shaded_lab = np.zeros((N_BLOCKS, 3, 3), dtype=np.float32)
    for si, m in enumerate(SHADE_MULTIPLIERS):
        rgb = np.clip(BLOCK_COLORS * m, 0, 255)
        shaded_rgb[:, si, :] = rgb
        shaded_lab[:, si, :] = rgb_to_lab(rgb)
    return shaded_lab, shaded_rgb

SHADED_LAB, SHADED_RGB = precompute_shaded()

# ─────────────────────────────────────────────
#  Beam search - solves one strip along Z
# ─────────────────────────────────────────────

MAX_HEIGHT_DIFF = 4
BEAM_WIDTH      = 16


def solve_strip(target_pixels: np.ndarray,
                height_penalty: float = 2.0,
                dither: bool = True):
    """
    target_pixels : (L, 3) float32 — one strip along Z axis
    Shade rule    : current block HIGHER than previous (+Z) → brighter (shade 2)
                    current block LOWER                     → darker   (shade 0)
                    same                                    → normal   (shade 1)
    Returns       : [(block_idx, height), ...], cost
    """
    L       = target_pixels.shape[0]
    targets = target_pixels.astype(np.float32).copy()

    # first element: no predecessor → shade 1
    shade      = 1
    target_lab = rgb_to_lab(targets[0:1])[0]
    errors     = cie94_batch(SHADED_LAB[:, shade, :], target_lab)
    top_b      = np.argsort(errors)[:BEAM_WIDTH]
    n          = len(top_b)

    beam_costs    = errors[top_b].astype(np.float32)
    beam_heights  = np.zeros(n, dtype=np.int32)
    beam_paths    = np.zeros((n, L), dtype=np.int32)
    beam_hpaths   = np.zeros((n, L), dtype=np.int8)
    beam_rendered = SHADED_RGB[top_b, shade, :].astype(np.float32)
    beam_paths[:, 0] = top_b

    for i in range(1, L):
        if dither:
            residual      = targets[i-1] - beam_rendered[0]
            targets[i]    = np.clip(targets[i]   + residual * 0.50, 0, 255)
            if i + 1 < L:
                targets[i+1] = np.clip(targets[i+1] + residual * 0.25, 0, 255)

        target_lab   = rgb_to_lab(targets[i:i+1])[0]
        err_by_shade = np.stack([
            cie94_batch(SHADED_LAB[:, s, :], target_lab) for s in range(3)
        ])  # (3, N_BLOCKS)

        all_costs    = []
        all_heights  = []
        all_rendered = []
        all_pb       = []
        all_bk       = []

        for di, dh in enumerate((-1, 0, 1)):
            shade = di  # 0=dark,1=flat,2=bright
            new_h = beam_heights + dh
            valid = (new_h >= 0) & (new_h <= MAX_HEIGHT_DIFF)
            vi    = np.where(valid)[0]
            if not vi.size:
                continue

            e      = err_by_shade[shade]
            h_cost = height_penalty * abs(dh)
            cand   = beam_costs[vi, None] + e[None, :] + h_cost

            V      = len(vi)
            pb     = np.repeat(vi, N_BLOCKS)
            bk     = np.tile(np.arange(N_BLOCKS), V)

            all_costs.append(cand.ravel())
            all_heights.append(np.repeat(new_h[vi], N_BLOCKS))
            all_rendered.append(SHADED_RGB[bk, shade, :])
            all_pb.append(pb)
            all_bk.append(bk)

        merged_costs    = np.concatenate(all_costs)
        merged_heights  = np.concatenate(all_heights)
        merged_rendered = np.concatenate(all_rendered, axis=0)
        merged_pb       = np.concatenate(all_pb)
        merged_bk       = np.concatenate(all_bk)

        K    = min(BEAM_WIDTH, len(merged_costs))
        topk = np.argpartition(merged_costs, K-1)[:K]
        topk = topk[np.argsort(merged_costs[topk])]

        nb = len(topk)
        pb = merged_pb[topk]
        bk = merged_bk[topk]
        nh = merged_heights[topk].astype(np.int32)

        new_paths          = np.zeros((nb, L), dtype=np.int32)
        new_hpaths         = np.zeros((nb, L), dtype=np.int8)
        new_paths[:, :i]   = beam_paths[pb, :i]
        new_hpaths[:, :i]  = beam_hpaths[pb, :i]
        new_paths[:, i]    = bk
        new_hpaths[:, i]   = nh

        beam_costs    = merged_costs[topk]
        beam_heights  = nh
        beam_paths    = new_paths
        beam_hpaths   = new_hpaths
        beam_rendered = merged_rendered[topk]

    path = list(zip(beam_paths[0].tolist(),
                    beam_hpaths[0].astype(int).tolist()))
    return path, float(beam_costs[0])


# ─────────────────────────────────────────────
#  Preview renderer
# ─────────────────────────────────────────────

def render_strip(path):
    """Render a Z-strip back to RGB."""
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


# ─────────────────────────────────────────────
#  Sponge schematic v2 export
# ─────────────────────────────────────────────

def _solver_name_to_mc(n: str) -> str:
    for block_type in ("concrete", "wool", "terracotta"):
        if n.startswith(block_type + "_"):
            color = n[len(block_type)+1:]
            return f"minecraft:{color}_{block_type}"
    direct = {
        "leaves":  "minecraft:oak_leaves",
        "grass":   "minecraft:grass_block",
        "snow":    "minecraft:snow_block",
        "clay":    "minecraft:clay",
        "dirt":    "minecraft:dirt",
        "stone":   "minecraft:stone",
        "sand":    "minecraft:sand",
        "ice":     "minecraft:ice",
        "oak_log": "minecraft:oak_log",
    }
    return direct.get(n, f"minecraft:{n}")


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
    all_blocks,    # [col][row] → block_idx
    all_heights,   # [col][row] → height
    block_names,
    output_path: str = "mapart.schem",
    name:        str = "Map Art",
):
    """
    Axes:
      X = image column  (WIDTH  = 128)
      Y = block height
      Z = image row     (LENGTH = 128)
      flat index = x + z*Width + y*Width*Length
    """
    n_cols = len(all_blocks)       # 128
    n_rows = len(all_blocks[0])    # 128 (length along Z)

    min_h  = min(h for col in all_heights for h in col)
    max_h  = max(h for col in all_heights for h in col)

    WIDTH  = n_cols
    HEIGHT = max_h - min_h + 1
    LENGTH = n_rows

    palette_map = {"minecraft:air": 0}
    mc_names    = [_solver_name_to_mc(n) for n in block_names]
    for mc in mc_names:
        if mc not in palette_map:
            palette_map[mc] = len(palette_map)

    flat = np.zeros(WIDTH * HEIGHT * LENGTH, dtype=np.int32)
    for x, (blk_col, h_col) in enumerate(zip(all_blocks, all_heights)):
        for z, (b, h) in enumerate(zip(blk_col, h_col)):
            y = h - min_h
            flat[x + z * WIDTH + y * WIDTH * LENGTH] = palette_map[mc_names[b]]

    block_data  = _encode_varint_array(flat)
    nbt_palette = Compound({k: Int(v) for k, v in palette_map.items()})

    nbt = File({
        "Schematic": Compound({          # root tag MUST be "Schematic"
            "Version":       Int(2),
            "DataVersion":   Int(3700),  # 1.21 — adjust if needed
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


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def generate_mapart(
    image_path:     str,
    output_path:    str   = "mapart_preview.png",
    schematic_path: str   = "mapart.schem",
    height_penalty: float = 2.0,
    dither:         bool  = True,
):
    img    = Image.open(image_path).convert("RGB")
    img    = img.resize((128, 128), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)  # (H=row=Z, W=col=X, 3)

    rendered_img    = np.zeros((128, 128, 3), dtype=np.uint8)
    all_blocks_out  = []   # indexed [col][row]
    all_heights_out = []

    # solve each column (strip along Z axis)
    for col in range(128):
        strip        = pixels[:, col, :]          # (128, 3) along Z
        path, cost   = solve_strip(strip, height_penalty=height_penalty, dither=dither)
        all_blocks_out.append([p[0] for p in path])
        all_heights_out.append([p[1] for p in path])
        rendered_img[:, col] = render_strip(path)
        if col % 16 == 0:
            print(f"Col {col:3d}/128  cost={cost:.1f}")

    Image.fromarray(rendered_img).save(output_path)
    print(f"Preview → {output_path}")

    export_sponge(
        all_blocks_out,
        all_heights_out,
        BLOCK_NAMES,
        output_path=schematic_path,
        name="Map Art",
    )

    return all_blocks_out, all_heights_out


if __name__ == "__main__":
    generate_mapart(
        "input.png",
        output_path    = "mapart_preview.png",
        schematic_path = "mapart.schem",
        height_penalty = 2.0,
        dither         = True,
    )
