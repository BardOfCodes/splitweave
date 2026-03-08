"""
Generate analogical quartet datasets (A, B, A*, B*) using splitweave.

Each quartet applies a single analogical transformation:
  - RSP: replace layout, coloring, foreground, or background
  - MTP: replace tileset, partial tile, add/remove tiles, replace layout,
         replace/add/remove cell effects, replace bg, replace border

Usage:
  python scripts/generate_analogical_quartets.py \
      --tile_dir assets/tiles/ \
      --out_dir /path/to/output \
      --n_pats 500 --run_id 0
"""

import argparse
import os
import time
import random
import string
import secrets
import pickle

import numpy as np
import torch as th
from PIL import Image
from pathlib import Path
from geolipi.torch_compute import Sketcher

from splitweave.utils.tiles import get_tile_content
from splitweave.cfg_to_sw.evaluate import evaluate_pattern_cfg
from splitweave.cfg_to_sw.parser import load_tile_tensors_from_config
from splitweave.pattern_gen.rsp_main import _collect_tile_cfg

from splitweave.analogy_gen.rsp_analogies import (
    rsp_replace_layout,
    rsp_replace_coloring,
    replace_foreground,
    replace_background,
)
from splitweave.analogy_gen.mtp_tile_analogies import (
    full_tileset_change,
    partial_tileset_change,
    remove_n_tiles,
    add_n_tiles,
)
from splitweave.analogy_gen.mtp_layout_analogies import full_layout_change
from splitweave.analogy_gen.mtp_cellfx_analogies import (
    full_cellfx_change,
    add_cellfx,
    remove_cellfx,
)
from splitweave.analogy_gen.mtp_bgx_analogies import bg_change, border_change

# ── Config ──────────────────────────────────────────────────────────────────

RESOLUTION = 1024
MTP_CROP = 256
MIN_FRAC = 0.15
MIN_FRAC_FX_BORDER = 0.10
MAX_RETRIES_PER_MODE = 15

MODE_TO_FREQ = {
    "mtp_replace_tileset":   3,
    "mtp_replace_tile":      2 / 3,
    "mtp_add_tile":          2 / 3,
    "mtp_remove_tile":       2 / 3,
    "mtp_replace_layout":    0.5,
    "mtp_replace_cellfx":    np.inf,
    "mtp_add_effect":        2 / 3,
    "mtp_remove_effect":     2 / 3,
    "mtp_replace_bg":        3,
    "mtp_replace_border":    3,
    "rsp_replace_layout":    2,
    "rsp_replace_coloring":  2,
    "rsp_replace_foreground": 3,
    "rsp_replace_background": 3,
}
MODE_TO_FREQ = {k: 1 / v for k, v in MODE_TO_FREQ.items()}
_total = sum(MODE_TO_FREQ.values())
MODE_TO_FREQ = {k: v / _total for k, v in MODE_TO_FREQ.items()}

MODE_TO_FUNC = {
    "mtp_replace_tileset":    full_tileset_change,
    "mtp_replace_tile":       partial_tileset_change,
    "mtp_add_tile":           add_n_tiles,
    "mtp_remove_tile":        remove_n_tiles,
    "mtp_replace_layout":     full_layout_change,
    "mtp_replace_cellfx":     full_cellfx_change,
    "mtp_add_effect":         add_cellfx,
    "mtp_remove_effect":      remove_cellfx,
    "mtp_replace_bg":         bg_change,
    "mtp_replace_border":     border_change,
    "rsp_replace_layout":     rsp_replace_layout,
    "rsp_replace_coloring":   rsp_replace_coloring,
    "rsp_replace_foreground": replace_foreground,
    "rsp_replace_background": replace_background,
}

MODES, MODE_PROBS = zip(*MODE_TO_FREQ.items())


# ── Helpers ─────────────────────────────────────────────────────────────────

def generate_random_name(length=8):
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(length))


def _ensure_tile_cfg(cfg):
    """Re-derive tile_cfg for RSP configs whose layout was modified by analogies."""
    if hasattr(cfg, "layout") and hasattr(cfg, "fill"):
        cfg.tile_cfg = _collect_tile_cfg(cfg.layout)


def execute_quartet(output_dict, sketcher, tile_dir, is_rsp):
    """Render the A/B/A*/B* quartet and return (images, diagnostics)."""
    keys = ["patterns_a", "patterns_b", "patterns_a_star", "patterns_b_star"]
    canvases, images = [], []

    for key in keys:
        cfg = output_dict[key][0]
        _ensure_tile_cfg(cfg)
        canvas, grid_ids = evaluate_pattern_cfg(cfg, sketcher, tile_dir=tile_dir)
        canvases.append(canvas)

        img = canvas.reshape(RESOLUTION, RESOLUTION, -1)
        if not is_rsp:
            img = img[MTP_CROP:-MTP_CROP, MTP_CROP:-MTP_CROP]
        img = img.cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(img))

    fracs, deltas = [], []
    for i in range(2):
        delta = canvases[i + 2] - canvases[i]
        delta_norm = th.norm(delta, dim=-1)
        deltas.append(delta_norm.mean().item())
        fracs.append((delta_norm > 0.001).float().mean().item())

    diagnostics = {
        "min_frac": min(fracs),
        "min_delta": min(deltas),
    }
    return images, diagnostics


# ── Main ────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if th.cuda.is_available() else "cpu"
    sketcher = Sketcher(device=device, resolution=RESOLUTION, n_dims=2)

    filenames, filename_to_indices = get_tile_content(args.tile_dir, mode=None)

    pat_dir = os.path.join(args.out_dir, "patterns")
    prog_dir = os.path.join(args.out_dir, "programs")
    meta_dir = os.path.join(args.out_dir, "metadata")

    for d in (pat_dir, prog_dir, meta_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        Path(os.path.join(pat_dir, mode)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(prog_dir, mode)).mkdir(parents=True, exist_ok=True)

    all_names = []
    failure_count = 0
    t0 = time.time()

    for i in range(args.n_pats):
        if i % 100 == 0:
            print(f"[{time.time() - t0:.1f}s] Generating quartet {i}/{args.n_pats}")

        variation_type = np.random.choice(MODES, p=MODE_PROBS)
        func = MODE_TO_FUNC[variation_type]
        is_rsp = variation_type.startswith("rsp_")
        retries = 0

        while True:
            harder_constraint = int(np.random.choice(5))
            try:
                output_dict = func(filenames, filename_to_indices, old_flag=True,
                                   harder_constraint=harder_constraint)
                images, diag = execute_quartet(
                    output_dict, sketcher, args.tile_dir, is_rsp)

                frac_lim = MIN_FRAC_FX_BORDER if (
                    "effect" in variation_type or "border" in variation_type
                ) else MIN_FRAC
                if diag["min_frac"] < frac_lim:
                    raise ValueError(
                        f"Fraction limit not met: {diag['min_frac']:.4f} < {frac_lim}")
                break
            except Exception as e:
                failure_count += 1
                retries += 1
                if retries > MAX_RETRIES_PER_MODE:
                    variation_type = np.random.choice(MODES, p=MODE_PROBS)
                    func = MODE_TO_FUNC[variation_type]
                    is_rsp = variation_type.startswith("rsp_")
                    retries = 0
                if args.verbose:
                    import traceback
                    print(traceback.print_exc())
                    print(f"  retry {retries} [{variation_type}]: {e}")

        name = generate_random_name()
        set_labels = ["A", "B", "A_star", "B_star"]
        cur_pat_dir = os.path.join(pat_dir, variation_type)
        for img, label in zip(images, set_labels):
            img.save(os.path.join(cur_pat_dir, f"{name}_{label}.png"))

        cur_prog_dir = os.path.join(prog_dir, variation_type)
        with open(os.path.join(cur_prog_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(output_dict, f)

        prefix = "RSP" if is_rsp else "MTP"
        all_names.append(f"{prefix}_{name}")

    meta_file = os.path.join(meta_dir, f"variations_{args.run_id}.txt")
    with open(meta_file, "w") as f:
        f.write("\n".join(all_names) + "\n")

    elapsed = time.time() - t0
    print(f"Done. {args.n_pats} quartets in {elapsed:.1f}s  "
          f"(failures: {failure_count})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate analogical quartet datasets with splitweave.")
    parser.add_argument("--tile_dir", type=str, required=True,
                        help="Directory containing tile images.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output root directory for patterns/programs/metadata.")
    parser.add_argument("--n_pats", type=int, default=500,
                        help="Number of quartets to generate.")
    parser.add_argument("--run_id", type=str, default="0",
                        help="Run identifier (used for seeding and metadata filename).")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", None],
                        help="Tile split to use (train/val/None for all).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print retry errors.")
    args = parser.parse_args()

    seed = int(args.run_id)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    main(args)
