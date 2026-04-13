import argparse
import numpy as np
from PIL import Image

from solver   import solve_strip
from renderer import render_strip
from export   import export_sponge


def generate_mapart(
    image_path:     str,
    output_path:    str   = "mapart_preview.png",
    schematic_path: str   = "mapart.schem",
    height_penalty: float = 2.0,
    dither:         bool  = True,
    data_version:   int   = 3700,
):
    img    = Image.open(image_path).convert("RGB")
    img    = img.resize((128, 128), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)  # (Z, X, 3)

    rendered_img    = np.zeros((128, 128, 3), dtype=np.uint8)
    all_blocks_out  = []  # [col/X][row/Z]
    all_heights_out = []

    for col in range(128):
        strip      = pixels[:, col, :]   # (128, 3) along Z
        path, cost = solve_strip(strip, height_penalty=height_penalty, dither=dither)
        all_blocks_out.append([p[0] for p in path])
        all_heights_out.append([p[1] for p in path])
        rendered_img[:, col] = render_strip(path)
        if col % 8 == 0:
            print(f"Col {col:3d}/128  cost={cost:.1f}")

    Image.fromarray(rendered_img).save(output_path)
    print(f"Preview → {output_path}")

    export_sponge(
        all_blocks_out,
        all_heights_out,
        output_path  = schematic_path,
        name         = "Map Art",
        data_version = data_version,
    )

    return all_blocks_out, all_heights_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft map art generator")
    parser.add_argument("image",                               help="Input image path")
    parser.add_argument("-o", "--output",   default="mapart_preview.png")
    parser.add_argument("-s", "--schematic",default="mapart.schem")
    parser.add_argument("-p", "--penalty",  default=2.0,  type=float)
    parser.add_argument("-d", "--no-dither",action="store_true")
    parser.add_argument("-v", "--data-version", default=3700, type=int)
    args = parser.parse_args()

    generate_mapart(
        image_path     = args.image,
        output_path    = args.output,
        schematic_path = args.schematic,
        height_penalty = args.penalty,
        dither         = not args.no_dither,
        data_version   = args.data_version,
    )
