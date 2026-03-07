"""
Generate example images for the README. Run from repo root:
    python scripts/generate_readme_examples.py

Builds expressions directly using splitweave.symbolic (as in notebooks/test.ipynb)
and saves images to assets/ for inclusion in the README.

Expects a tile image at assets/tiles/tile_0.png (RGBA). Generates:
    assets/example_pattern.png   — brick-shifted RectRepeat + TileRecolor + background
"""
import os
import sys
import glob

import numpy as np
import torch as th
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import splitweave.symbolic as sws
import geolipi.symbolic as gls
import geolipi.symbolic.primitives_2d as prim2d
from geolipi.torch_compute import Sketcher
from splitweave.torch_compute import evaluate_pattern

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
RESOLUTION = 512
ASSETS_DIR = os.path.join(repo_root, "assets")
TILES_DIR = os.path.join(ASSETS_DIR, "tiles")


def load_tile(path: str, device: str = "cpu") -> th.Tensor:
    """Load an RGBA tile image, rotate 90° (matches splitweave convention), return float tensor."""
    pil_img = Image.open(path).convert("RGBA").rotate(90)
    arr = np.array(pil_img).astype("float32") / 255.0
    return th.from_numpy(arr).to(device)


def save_canvas(canvas: th.Tensor, path: str, resolution: int) -> None:
    """Reshape canvas [N,4] -> [R,R,3] and save as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = canvas.detach().cpu().reshape(resolution, resolution, 4).numpy()
    img_rgb = (np.clip(img[..., :3], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_rgb).save(path)
    print(f"Saved {os.path.relpath(path, repo_root)}")


def main():
    sketcher = Sketcher(device=DEVICE, resolution=RESOLUTION, n_dims=2)

    # Find a tile to use
    tile_candidates = sorted(
        glob.glob(os.path.join(TILES_DIR, "*.png"))
        + glob.glob(os.path.join(TILES_DIR, "*.jpg"))
    )
    if not tile_candidates:
        raise FileNotFoundError(
            f"No tile images found in {TILES_DIR}. "
            "Place at least one RGBA .png file there before running this script."
        )
    tile_path = tile_candidates[0]
    tile = load_tile(tile_path, device=DEVICE)
    print(f"Using tile: {os.path.relpath(tile_path, repo_root)}")

    # Build expression:
    #   Layout:    brick-shifted rectangle grid (shift = half cell width)
    #   Tile:      TileRecolor shifts hue by 0.3 for a visible colour change
    #   Background: warm off-white constant colour
    layout = sws.RectRepeatShiftedX(sws.CartesianGrid(), (0.2, 0.2), (0.1,))
    tile_expr = sws.TileRecolor(prim2d.TileUV2D(tile), 0.3)
    apply_tile = sws.ApplyTile(layout, tile_expr)
    bg_color = th.tensor([0.95, 0.92, 0.85, 1.0], device=DEVICE)
    bg = sws.ConstantBackground(bg_color)
    root = gls.SourceOver(apply_tile, bg)

    canvas, _ = evaluate_pattern(root, sketcher)
    save_canvas(canvas, os.path.join(ASSETS_DIR, "example_pattern.png"), RESOLUTION)


if __name__ == "__main__":
    main()
