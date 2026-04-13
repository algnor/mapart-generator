import numpy as np
from PIL import Image
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

# --- precompute all shaded colors in LAB space ---

SHADE_MULTIPLIERS = (0.71, 0.86, 1.0)

BLOCKS = {
    "wool_white":    (255, 255, 255),
    "wool_orange":   (216, 127, 51),
    "wool_magenta":  (178, 76,  216),
    "wool_light_blue":(102,153, 216),
    "wool_yellow":   (229, 229, 51),
    "wool_lime":     (127, 204, 25),
    "wool_pink":     (242, 127, 165),
    "wool_gray":     (76,  76,  76),
    "wool_light_gray":(153,153,153),
    "wool_cyan":     (76,  127, 153),
    "wool_purple":   (127, 63,  178),
    "wool_blue":     (51,  76,  178),
    "wool_brown":    (102, 76,  51),
    "wool_green":    (102, 127, 51),
    "wool_red":      (153, 51,  51),
    "wool_black":    (25,  25,  25),
    "ice":           (160, 160, 255),
    "clay":          (164, 168, 184),
    "dirt":          (151, 109, 77),
    "stone":         (112, 112, 112),
    "oak_log":       (143, 119, 72),
    "leaves":        (0,   124, 0),
    "concrete_white":(209, 209, 209),
    "concrete_orange":(224,97,  0),
    "concrete_magenta":(169,48, 159),
    "concrete_light_blue":(36,137,199),
    "concrete_yellow":(240,175, 0),
    "concrete_lime": (94,  169, 24),
    "concrete_pink": (214, 101, 143),
    "concrete_gray": (55,  58,  62),
    "concrete_light_gray":(125,125,115),
    "concrete_cyan": (21,  119, 136),
    "concrete_purple":(100,32, 156),
    "concrete_blue": (45,  47,  143),
    "concrete_brown":(96,  60,  32),
    "concrete_green":(73,  91,  36),
    "concrete_red":  (142, 33,  33),
    "concrete_black":(8,   10,  15),
    "terracotta_white":(209,178,161),
    "terracotta_orange":(162,84, 38),
    "terracotta_magenta":(149,88,108),
    "terracotta_light_blue":(113,108,137),
    "terracotta_yellow":(186,133,35),
    "terracotta_lime":(103,117,52),
    "terracotta_pink":(160,77, 78),
    "terracotta_gray":(57, 41,  35),
    "terracotta_light_gray":(135,107,98),
    "terracotta_cyan":(87, 92,  92),
    "terracotta_purple":(122,42, 84),
    "terracotta_blue":(74, 59,  91),
    "terracotta_brown":(77, 51,  35),
    "terracotta_green":(76, 83,  42),
    "terracotta_red":(143,61,  46),
    "terracotta_black":(37, 22,  16),
    "grass":         (127, 178, 56),
    "sand":          (247, 233, 163),
    "snow":          (255, 255, 255),
}

BLOCK_NAMES = list(BLOCKS.keys())
BLOCK_COLORS = np.array([BLOCKS[b] for b in BLOCK_NAMES], dtype=np.float32)
N_BLOCKS = len(BLOCK_NAMES)

def rgb_to_lab(rgb_array):
    """Convert (N, 3) float32 RGB 0-255 to (N, 3) LAB"""
    rgb_norm = rgb_array / 255.0
    # linearize sRGB
    mask = rgb_norm > 0.04045
    rgb_lin = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)
    # to XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = rgb_lin @ M.T
    # normalize by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    # to LAB
    epsilon = 0.008856
    kappa = 903.3
    mask2 = xyz > epsilon
    f = np.where(mask2, xyz ** (1/3), (kappa * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)

def precompute_shaded_lab():
    """Returns (N_BLOCKS, 3_shades, 3_lab) and (N_BLOCKS, 3_shades, 3_rgb)"""
    shaded_rgb = np.zeros((N_BLOCKS, 3, 3), dtype=np.float32)
    shaded_lab = np.zeros((N_BLOCKS, 3, 3), dtype=np.float32)
    for si, m in enumerate(SHADE_MULTIPLIERS):
        rgb = np.clip(BLOCK_COLORS * m, 0, 255)
        shaded_rgb[:, si, :] = rgb
        shaded_lab[:, si, :] = rgb_to_lab(rgb)
    return shaded_lab, shaded_rgb

SHADED_LAB, SHADED_RGB = precompute_shaded_lab()

def ciede2000_batch(lab1_batch, lab2):
    """
    Approximate CIEDE2000 batch.
    lab1_batch: (N, 3)
    lab2: (3,)
    Returns (N,) distances
    """
    # For simplicity use CIE94 which is much faster and better than plain LAB
    # Full CIEDE2000 is complex; CIE94 gets most of the benefit
    L1, a1, b1 = lab1_batch[:, 0], lab1_batch[:, 1], lab1_batch[:, 2]
    L2, a2, b2 = lab2[0], lab2[1], lab2[2]
    
    dL = L1 - L2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    da = a1 - a2
    db = b1 - b2
    dH2 = da**2 + db**2 - dC**2
    dH2 = np.maximum(dH2, 0)
    
    kL, k1, k2 = 1.0, 0.045, 0.015
    SL = 1.0
    SC = 1 + k1 * C1
    SH = 1 + k2 * C1
    
    return np.sqrt((dL / (kL * SL))**2 + (dC / SC)**2 + dH2 / SH**2)

MAX_HEIGHT_DIFF = 4
BEAM_WIDTH = 16

def solve_row(target_pixels: np.ndarray, height_penalty: float = 2.0, dither: bool = True):
    """
    target_pixels: (W, 3) float32 RGB
    """
    W = target_pixels.shape[0]
    
    # working copy for dithering - we modify this as we go
    targets = target_pixels.astype(np.float32).copy()

    # beam: (cost, height, path, last_rendered_rgb)
    # we need last_rendered_rgb to compute dither residual
    beam = []

    # first column, shade=1
    shade = 1
    lab_rendered = SHADED_LAB[:, shade, :]       # (N, 3)
    target_lab = rgb_to_lab(targets[0:1])[0]     # (3,)
    errors = ciede2000_batch(lab_rendered, target_lab)

    for b in range(N_BLOCKS):
        beam.append((
            float(errors[b]),
            0,
            [(b, 0)],
            SHADED_RGB[b, shade].copy()  # last rendered rgb
        ))

    beam.sort(key=lambda x: x[0])
    beam = beam[:BEAM_WIDTH]

    for col in range(1, W):
        # Note: dithering in beam search is tricky because each beam state
        # would ideally have its own dither buffer. We approximate by using
        # a shared dither buffer from the BEST beam state only.
        # This is a known approximation - full per-state dither buffers are
        # memory expensive but more correct.
        best_rendered = beam[0][3]
        best_target = targets[col - 1]
        residual = best_target - best_rendered  # RGB error

        # propagate error forward (Floyd-Steinberg style, 1D)
        if dither:
            targets[col] = np.clip(targets[col] + residual * 0.5, 0, 255)
            if col + 1 < W:
                targets[col + 1] = np.clip(targets[col + 1] + residual * 0.25, 0, 255)

        target_lab = rgb_to_lab(targets[col:col+1])[0]
        candidates = []

        for (prev_cost, prev_h, path, _) in beam:
            for dh in (-1, 0, 1):
                new_h = prev_h + dh
                if new_h < 0 or new_h > MAX_HEIGHT_DIFF:
                    continue

                shade = 2 if dh > 0 else (0 if dh < 0 else 1)
                lab_rendered = SHADED_LAB[:, shade, :]
                errors = ciede2000_batch(lab_rendered, target_lab)
                h_cost = height_penalty * abs(dh)

                for b in range(N_BLOCKS):
                    candidates.append((
                        prev_cost + float(errors[b]) + h_cost,
                        new_h,
                        path + [(b, new_h)],
                        SHADED_RGB[b, shade].copy()
                    ))

        candidates.sort(key=lambda x: x[0])
        beam = candidates[:BEAM_WIDTH]

    return beam[0][2], beam[0][0]


def render_path(path):
    W = len(path)
    rendered = np.zeros((W, 3), dtype=np.uint8)
    for col, (b, h) in enumerate(path):
        shade = 1 if col == 0 else (2 if h > path[col-1][1] else (0 if h < path[col-1][1] else 1))
        rendered[col] = np.clip(SHADED_RGB[b, shade], 0, 255).astype(np.uint8)
    return rendered


def generate_mapart(image_path: str, output_path: str = "mapart_preview-v1.png",
                    height_penalty: float = 2.0, dither: bool = True):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)

    rendered_img = np.zeros((128, 128, 3), dtype=np.uint8)
    all_blocks, all_heights = [], []

    for row in range(128):
        path, cost = solve_row(pixels[row], height_penalty=height_penalty, dither=dither)
        all_blocks.append([p[0] for p in path])
        all_heights.append([p[1] for p in path])
        rendered_img[row] = render_path(path)
        if row % 16 == 0:
            print(f"Row {row}/128, cost={cost:.1f}")

    Image.fromarray(rendered_img).save(output_path)
    print(f"Saved to {output_path}")
    return all_blocks, all_heights

if __name__ == "__main__":
    blocks, heights = generate_mapart("input.png", height_penalty=2.0, dither=True)
