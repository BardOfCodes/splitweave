"""
Patternator → Splitweave expression parsers.

These helpers take the existing patternator configs (yacs CfgNode)
and construct equivalent splitweave / geolipi expressions.

Supports:
- Layout: all patternator split types (Rect, Hex, Radial, Voronoi, Irregular, etc.)
  with pre_deform (t_x, t_y, rot, scale) and post_deform (rot, scale).
- Deformation: radial_deform, perlin_deform, decay_deform, strip_deform, swirl_deform (and no_deform).
- Tileset: tile tensors as TileUV2D, with optional per-tile effects (recolor, outline, shadow, rotate, scale, reflect_x/y, opacity).
- Background:
    plain → ConstantBackground(base_rgba)
    grid_n_color → ApplyColoring(layout_expr, base_rgba, recolor_params, signal_expr)
      where recolor_params = (recolor_type, recolor_seed, signal_mode) and
      signal_expr is a typed discrete signal (CheckerboardSignal, XStripeSignal, etc.).
  Root: SourceOver(ApplyTile, background_expr).
  NO CfgNode objects in the expression tree — everything is encoded as expressions/tensors/tuples.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
import torch as th
from yacs.config import CfgNode as CN
from PIL import Image


import geolipi.symbolic as gls
from geolipi.symbolic import primitives_2d as prim2d
from geolipi.torch_compute.colorspace_functions import hsl2rgb_torch

import splitweave.symbolic as sws


# ---- Discrete mode name → integer mapping (matches patternator signal_2d_func) ----
DISCRETE_MODE_TO_INT = {
    "checkerboard": 0,
    "x": 1,
    "y": 2,
    "xx": 3,
    "yy": 4,
    "xy": 5,
    "diag_1": 6,
    "diag_2": 7,
    "random": 8,
    "noisy_central_pivot": 9,
    "fully_random": 10,
    "count_diag_1": 11,
    "count_x": 12,
    "count_y": 13,
    "count_diag_2": 14,
}

INT_TO_DISCRETE_MODE = {v: k for k, v in DISCRETE_MODE_TO_INT.items()}


def load_tile_tensors_from_config(
    tile_cfg: CN,
    tile_dir: str,
    device: str = "cpu"
) -> Dict[str, th.Tensor]:
    """
    Load tile images referenced in tile_cfg.tileset from disk under tile_dir.
    Keys are the string form of tilefile (path or basename); values are [H, W, C] tensors.
    """
    tile_tensors: Dict[str, th.Tensor] = {}
    for t_cfg in tile_cfg.tileset:
        tilefile = getattr(t_cfg, "tilefile", None)
        if tilefile is None:
            continue
        key = str(tilefile)
        path = key if os.path.isabs(key) else os.path.join(tile_dir, os.path.basename(key))
        if not os.path.isfile(path):
            continue
        pil_img = Image.open(path).convert("RGBA")
        # rotate by 90
        pil_img = pil_img.rotate(90, expand=True)
        arr = np.array(pil_img).astype("float32") / 255.0
        tensor = th.from_numpy(arr).to(device)
        tile_tensors[key] = tensor
    return tile_tensors


def background_color_from_config(bg_cfg: CN, device: str = "cpu") -> th.Tensor:
    """
    Return RGBA tensor (4,) for the background. Plain and grid_n_color both use
    color_seed for the base color (HSL random, same as patternator execute_bg).
    """
    color_seed = int(getattr(bg_cfg, "color_seed", 0))
    st0 = np.random.get_state()
    np.random.seed(color_seed)
    hue = float(np.random.uniform(0, 1))
    sat = float(np.random.uniform(0.35, 1.0))
    val = float(np.random.uniform(0.35, 1.0))
    np.random.set_state(st0)
    hsl = th.tensor([hue, sat, val], device=device, dtype=th.float32)
    rgb = hsl2rgb_torch(hsl[None, :])[0]
    rgba = th.cat([rgb, th.tensor([1.0], device=device, dtype=th.float32)], dim=-1)
    return rgba


def border_color_from_config(border_cfg: CN, device: str = "cpu") -> th.Tensor:
    """Return RGBA tensor (4,) for the border from border_cfg.color_seed (same as bg color)."""
    color_seed = int(getattr(border_cfg, "color_seed", 0))
    st0 = np.random.get_state()
    np.random.seed(color_seed)
    hue = float(np.random.uniform(0, 1))
    sat = float(np.random.uniform(0.35, 1.0))
    val = float(np.random.uniform(0.35, 1.0))
    np.random.set_state(st0)
    hsl = th.tensor([hue, sat, val], device=device, dtype=th.float32)
    rgb = hsl2rgb_torch(hsl[None, :])[0]
    rgba = th.cat([rgb, th.tensor([1.0], device=device, dtype=th.float32)], dim=-1)
    return rgba


# Map discrete_mode to typed signal class for new-style expressions
_DISCRETE_MODE_TO_SIGNAL_CLASS = {
    "checkerboard": sws.CheckerboardSignal,
    "x": sws.XStripeSignal,
    "y": sws.YStripeSignal,
    "xx": sws.XXStripeSignal,
    "yy": sws.YYStripeSignal,
    "xy": sws.XYStripeSignal,
    "diag_1": lambda k, inv, ga, asym, dd: sws.DiagonalSignal(k, inv, ga, asym, dd, "diag_1"),
    "diag_2": lambda k, inv, ga, asym, dd: sws.DiagonalSignal(k, inv, ga, asym, dd, "diag_2"),
    "random": sws.RandomSignal,
    "noisy_central_pivot": sws.RandomSignal,  # patternator mode; no dedicated type, use RandomSignal
    "fully_random": sws.FullyRandomSignal,
    "count_diag_1": lambda k, inv, ga, asym, dd: sws.CountSignal(k, inv, ga, asym, dd, "count_diag_1"),
    "count_x": lambda k, inv, ga, asym, dd: sws.CountSignal(k, inv, ga, asym, dd, "count_x"),
    "count_y": lambda k, inv, ga, asym, dd: sws.CountSignal(k, inv, ga, asym, dd, "count_y"),
    "count_diag_2": lambda k, inv, ga, asym, dd: sws.CountSignal(k, inv, ga, asym, dd, "count_diag_2"),
}


def _signal_cfg_to_signal_expr(sig: CN):
    """Build a typed discrete signal expression from config."""
    mode = getattr(sig, "discrete_mode", "checkerboard")
    k = int(getattr(sig, "k", 2))
    inv = getattr(sig, "inverse", False)
    ga = getattr(sig, "group_alternate", False)
    asym = getattr(sig, "apply_sym", False)
    dd = getattr(sig, "double_dip", False)
    cls_or_fn = _DISCRETE_MODE_TO_SIGNAL_CLASS.get(mode, sws.CheckerboardSignal)
    if callable(cls_or_fn) and not isinstance(cls_or_fn, type):
        return cls_or_fn(k, inv, ga, asym, dd)
    return cls_or_fn(k, inv, ga, asym, dd)


def _recolor_cfg_to_expr(color_fx: CN) -> Tuple[tuple, "sws.DiscreteSignalBase"]:
    """
    Convert a patternator RecolorFx CfgNode to recolor_params and typed signal expression.

    Returns
    -------
    recolor_params : tuple of 3 floats
        (recolor_type, recolor_seed, signal_mode)
        recolor_type: 0.0 = smooth, 1.0 = discrete
        signal_mode: 0.0 = single, 1.0 = double
    signal_expr : typed discrete signal (CheckerboardSignal, etc.)
    """
    recolor_type = 0.0 if color_fx.recolor_type == "smooth" else 1.0
    recolor_seed = float(color_fx.recolor_seed)
    signal_mode = 0.0 if color_fx.mode == "single" else 1.0
    recolor_params = (recolor_type, recolor_seed, signal_mode)
    signal_expr = _signal_cfg_to_signal_expr(color_fx.signal)
    return recolor_params, signal_expr


def _get_split_type(layout_cfg: CN) -> str:
    if hasattr(layout_cfg, "split_type"):
        return layout_cfg.split_type
    if hasattr(layout_cfg, "split") and hasattr(layout_cfg.split, "_type"):
        return layout_cfg.split._type
    return "RectRepeat"


def _split_cfg(layout_cfg: CN) -> CN:
    """Return the split subconfig (layout_cfg.split or layout_cfg)."""
    if hasattr(layout_cfg, "split"):
        return layout_cfg.split
    return layout_cfg


def _get(s: CN, key: str, default):
    """Get attribute with default."""
    return getattr(s, key, default)


def _wrap_layout_with_cell_grid_effects(layout_expr: sws.GridFunction, cellfx_cfg: CN) -> sws.GridFunction:
    """
    Wrap layout_expr with LayoutCellTranslate, LayoutCellRotate, LayoutCellScale, LayoutCellReflect
    in patternator order (translate -> rotate -> scale -> reflect). Each effect uses typed signal expr.
    """
    effects = getattr(cellfx_cfg, "effects", [])
    if not effects:
        return layout_expr
    expr = layout_expr
    for fx in effects:
        fx_type = getattr(fx, "_type", None)
        if not hasattr(fx, "signal"):
            continue
        signal_expr = _signal_cfg_to_signal_expr(fx.signal)
        mode = str(getattr(fx, "mode", "single"))
        if fx_type == "TranslateFx":
            t_x = float(getattr(fx, "t_x", 0.1))
            t_y = float(getattr(fx, "t_y", 0.1))
            expr = sws.LayoutCellTranslate(expr, signal_expr, t_x, t_y, mode)
        elif fx_type == "RotateFx":
            rotation = float(getattr(fx, "rotation", 0.785))
            expr = sws.LayoutCellRotate(expr, signal_expr, rotation, mode)
        elif fx_type == "ScaleFx":
            scale = float(getattr(fx, "scale", 0.5))
            expr = sws.LayoutCellScale(expr, signal_expr, scale, mode)
        elif fx_type == "ReflectFx":
            reflect = str(getattr(fx, "reflect", "x"))
            expr = sws.LayoutCellReflect(expr, signal_expr, reflect, mode)
    return expr


def layout_config_to_expr(layout_cfg: CN) -> sws.GridFunction:
    """
    Map a layout_cfg (patternator) to a splitweave grid expression.

    Supports: pre_deform (t_x, t_y, rot, scale), deform (radial/perlin/decay/strip/swirl),
    split (all patternator types), post_deform (rot, scale).
    """
    expr: sws.GridFunction = sws.CartesianGrid()

    # ----- Pre-deform (from pre_deform subconfig or top-level) -----
    pre = getattr(layout_cfg, "pre_deform", layout_cfg)
    t_x = float(_get(pre, "t_x", 0.0))
    t_y = float(_get(pre, "t_y", 0.0))
    if t_x != 0.0 or t_y != 0.0:
        expr = sws.CartTranslate(expr, (t_x, t_y))
    rot = float(_get(pre, "rot", 0.0))
    if rot != 0.0:
        expr = sws.CartRotate(expr, (rot,))
    scale = float(_get(pre, "scale", 1.0))
    if scale != 1.0:
        expr = sws.CartScale(expr, (scale, scale))

    # ----- Deform: wrap with RadialDeform, PerlinDeform, DecayDeform, StripDeform, SwirlDeform -----
    deform = getattr(layout_cfg, "deform", None)
    if deform is not None:
        dtype = getattr(deform, "_type", None)
        if dtype == "radial_deform":
            center = _get(deform, "center", (0.0, 0.0))
            if hasattr(center, "__len__") and len(center) >= 2:
                center = (float(center[0]), float(center[1]))
            else:
                center = (float(_get(deform, "center_x", 0.0)), float(_get(deform, "center_y", 0.0)))
            signal_mode = str(_get(deform, "signal_mode", "linear"))
            dist_rate = float(_get(deform, "dist_rate", 0.2))
            sin_k = float(_get(deform, "sin_k", 6.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            expr = sws.RadialDeform(expr, center, signal_mode, dist_rate, sin_k, phase_shift)
        elif dtype == "perlin_deform":
            res_seq = getattr(deform, "resolution_seq", [4, 8])
            if hasattr(res_seq, "tolist"):
                res_seq = res_seq.tolist()
            res_seq = tuple(int(x) for x in res_seq) if res_seq else (4, 8)
            seed = int(_get(deform, "seed", 0))
            dist_mode = str(_get(deform, "dist_mode", "xy"))
            dist_rate = float(_get(deform, "dist_rate", 0.08))
            expr = sws.PerlinDeform(expr, res_seq, seed, dist_mode, dist_rate)
        elif dtype == "decay_deform":
            direction = _get(deform, "direction", (1.0, 0.0))
            if hasattr(direction, "__len__") and len(direction) >= 2:
                direction = (float(direction[0]), float(direction[1]))
            else:
                direction = (1.0, 0.0)
            axis = str(_get(deform, "axis", "x"))
            dist_rate = float(_get(deform, "dist_rate", 0.5))
            expr = sws.DecayDeform(expr, direction, axis, dist_rate)
        elif dtype == "strip_deform":
            angle = float(_get(deform, "angle", 0.0))
            axis = str(_get(deform, "axis", "x"))
            sin_k = float(_get(deform, "sin_k", 4.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            dist_rate = float(_get(deform, "dist_rate", 0.05))
            expr = sws.StripDeform(expr, angle, axis, sin_k, phase_shift, dist_rate)
        elif dtype == "swirl_deform":
            center = _get(deform, "center", (0.0, 0.0))
            if hasattr(center, "__len__") and len(center) >= 2:
                center = (float(center[0]), float(center[1]))
            else:
                center = (float(_get(deform, "center_x", 0.0)), float(_get(deform, "center_y", 0.0)))
            signal_mode = str(_get(deform, "signal_mode", "linear"))
            dist_rate = float(_get(deform, "dist_rate", 0.2))
            sin_k = float(_get(deform, "sin_k", 6.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            expr = sws.SwirlDeform(expr, center, signal_mode, dist_rate, sin_k, phase_shift)
        # no_deform or unknown: leave expr unchanged

    # ----- Split -----
    split_type = _get_split_type(layout_cfg)
    s = _split_cfg(layout_cfg)

    # Optional center for radial: translate so origin is at (center_x, center_y)
    center_x = float(_get(s, "center_x", 0.0))
    center_y = float(_get(s, "center_y", 0.0))
    if (center_x != 0.0 or center_y != 0.0) and split_type in (
        "RadialRepeat", "RadialRepeatBricked", "RadialRepeatFixedArc", "RadialRepeatFixedArcBricked"
    ):
        expr = sws.CartTranslate(expr, (center_x, center_y))

    if split_type == "RectRepeat":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        expr = sws.RectRepeat(expr, (x_size, y_size))
    elif split_type == "RectRepeatFitting":
        fx = float(_get(s, "fit_x_size", 0.25))
        fy = float(_get(s, "fit_y_size", 0.25))
        expr = sws.RectRepeatFitting(expr, (fx, fy))
    elif split_type == "RectRepeatShiftedX":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        x_shift = float(_get(s, "x_shift", x_size / 2))
        expr = sws.RectRepeatShiftedX(expr, (x_size, y_size), (x_shift,))
    elif split_type == "RectRepeatShiftedY":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        y_shift = float(_get(s, "y_shift", y_size / 2))
        expr = sws.RectRepeatShiftedY(expr, (x_size, y_size), (y_shift,))
    elif split_type == "HexRepeat":
        gs = float(_get(s, "grid_size", 0.2))
        expr = sws.HexRepeat(expr, (gs,))
    elif split_type == "HexRepeatY":
        gs = float(_get(s, "grid_size", 0.2))
        expr = sws.HexRepeatY(expr, (gs,))
    elif split_type == "RadialRepeat":
        ru = float(_get(s, "radial_unit", 0.25))
        au = float(_get(s, "angular_unit", 0.5236))
        expr = sws.RadialRepeatCentered(expr, (ru, au))
    elif split_type == "RadialRepeatBricked":
        ru = float(_get(s, "radial_unit", 0.25))
        au = float(_get(s, "angular_unit", 0.5236))
        expr = sws.RadialRepeatBricked(expr, (ru, au), (0.0,))
    elif split_type == "RadialRepeatFixedArc":
        ru = float(_get(s, "radial_unit", 0.25))
        arc = float(_get(s, "arc_size", 0.35))
        expr = sws.RadialRepeatFixedArc(expr, (ru, arc), (0.0,))
    elif split_type == "RadialRepeatFixedArcBricked":
        ru = float(_get(s, "radial_unit", 0.25))
        arc = float(_get(s, "arc_size", 0.35))
        expr = sws.RadialRepeatFixedArcBricked(expr, (ru, arc), (0.0,))
    elif split_type == "VoronoiRepeat":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        noise = float(_get(s, "noise_rate", 0.5))
        expr = sws.VoronoiRepeat(expr, (x_size, y_size), (noise,))
    elif split_type == "IrregularRepeat":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        noise = float(_get(s, "noise_rate", 0.5))
        expr = sws.IrregularRectRepeat(expr, (x_size, y_size), (noise,))
    elif split_type in ("RandomFillRectRepeat", "RandomFillVoronoi"):
        # Random fill: use underlying grid; we don't have random fill in splitweave yet -> RectRepeat fallback
        x_size = float(_get(s, "x_size", _get(s, "grid_size", 0.25)))
        y_size = float(_get(s, "y_size", x_size))
        expr = sws.RectRepeat(expr, (x_size, y_size))
    elif split_type in ("OverlapX", "OverlapY", "OverlapXY"):
        # Overlap layouts: fallback to RectRepeat
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        expr = sws.RectRepeat(expr, (x_size, y_size))
    else:
        expr = sws.RectRepeat(expr, (0.25, 0.25))

    # ----- Post-deform -----
    post = getattr(layout_cfg, "post_deform", None)
    if post is not None:
        rot = float(_get(post, "rot", 0.0))
        if rot != 0.0:
            expr = sws.CartRotate(expr, (rot,))
        scale = float(_get(post, "scale", 1.0))
        if scale != 1.0:
            expr = sws.CartScale(expr, (scale, scale))

    return expr


def layout_config_to_edge_expr(layout_cfg: CN) -> sws.GridFunction:
    """
    Build the edge (border distance) expression that matches layout_config_to_expr.
    Same pre_deform, deform, and post_deform; split part uses *Edge symbols with same params.
    Used for BorderEffect so we get edge distances for border masking.
    """
    expr = layout_config_to_expr(layout_cfg)
    split_type = _get_split_type(layout_cfg)
    s = _split_cfg(layout_cfg)

    # Map partition type to Edge type with same params (expr already has pre_deform + deform + split)
    # We need only the split part as Edge. So we build from scratch: base -> pre_deform -> deform -> Edge(split) -> post_deform
    base: sws.GridFunction = sws.CartesianGrid()
    pre = getattr(layout_cfg, "pre_deform", layout_cfg)
    t_x = float(_get(pre, "t_x", 0.0))
    t_y = float(_get(pre, "t_y", 0.0))
    if t_x != 0.0 or t_y != 0.0:
        base = sws.CartTranslate(base, (t_x, t_y))
    rot = float(_get(pre, "rot", 0.0))
    if rot != 0.0:
        base = sws.CartRotate(base, (rot,))
    scale = float(_get(pre, "scale", 1.0))
    if scale != 1.0:
        base = sws.CartScale(base, (scale, scale))

    deform = getattr(layout_cfg, "deform", None)
    if deform is not None:
        dtype = getattr(deform, "_type", None)
        if dtype == "radial_deform":
            center = _get(deform, "center", (0.0, 0.0))
            if hasattr(center, "__len__") and len(center) >= 2:
                center = (float(center[0]), float(center[1]))
            else:
                center = (float(_get(deform, "center_x", 0.0)), float(_get(deform, "center_y", 0.0)))
            signal_mode = str(_get(deform, "signal_mode", "linear"))
            dist_rate = float(_get(deform, "dist_rate", 0.2))
            sin_k = float(_get(deform, "sin_k", 6.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            base = sws.RadialDeform(base, center, signal_mode, dist_rate, sin_k, phase_shift)
        elif dtype == "perlin_deform":
            res_seq = getattr(deform, "resolution_seq", [4, 8])
            if hasattr(res_seq, "tolist"):
                res_seq = res_seq.tolist()
            res_seq = tuple(int(x) for x in res_seq) if res_seq else (4, 8)
            seed = int(_get(deform, "seed", 0))
            dist_mode = str(_get(deform, "dist_mode", "xy"))
            dist_rate = float(_get(deform, "dist_rate", 0.08))
            base = sws.PerlinDeform(base, res_seq, seed, dist_mode, dist_rate)
        elif dtype == "decay_deform":
            direction = _get(deform, "direction", (1.0, 0.0))
            if hasattr(direction, "__len__") and len(direction) >= 2:
                direction = (float(direction[0]), float(direction[1]))
            else:
                direction = (1.0, 0.0)
            axis = str(_get(deform, "axis", "x"))
            dist_rate = float(_get(deform, "dist_rate", 0.5))
            base = sws.DecayDeform(base, direction, axis, dist_rate)
        elif dtype == "strip_deform":
            angle = float(_get(deform, "angle", 0.0))
            axis = str(_get(deform, "axis", "x"))
            sin_k = float(_get(deform, "sin_k", 4.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            dist_rate = float(_get(deform, "dist_rate", 0.05))
            base = sws.StripDeform(base, angle, axis, sin_k, phase_shift, dist_rate)
        elif dtype == "swirl_deform":
            center = _get(deform, "center", (0.0, 0.0))
            if hasattr(center, "__len__") and len(center) >= 2:
                center = (float(center[0]), float(center[1]))
            else:
                center = (float(_get(deform, "center_x", 0.0)), float(_get(deform, "center_y", 0.0)))
            signal_mode = str(_get(deform, "signal_mode", "linear"))
            dist_rate = float(_get(deform, "dist_rate", 0.2))
            sin_k = float(_get(deform, "sin_k", 6.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            base = sws.SwirlDeform(base, center, signal_mode, dist_rate, sin_k, phase_shift)

    center_x = float(_get(s, "center_x", 0.0))
    center_y = float(_get(s, "center_y", 0.0))
    if (center_x != 0.0 or center_y != 0.0) and split_type in (
        "RadialRepeat", "RadialRepeatBricked", "RadialRepeatFixedArc", "RadialRepeatFixedArcBricked"
    ):
        base = sws.CartTranslate(base, (center_x, center_y))

    if split_type == "RectRepeat":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        base = sws.RectRepeatEdge(base, (x_size, y_size))
    elif split_type == "RectRepeatFitting":
        fx = float(_get(s, "fit_x_size", 0.25))
        fy = float(_get(s, "fit_y_size", 0.25))
        base = sws.RectRepeatEdge(base, (fx, fy))
    elif split_type == "RectRepeatShiftedX":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        x_shift = float(_get(s, "x_shift", x_size / 2))
        base = sws.RectRepeatShiftedXEdge(base, (x_size, y_size), (x_shift,))
    elif split_type == "RectRepeatShiftedY":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        y_shift = float(_get(s, "y_shift", y_size / 2))
        base = sws.RectRepeatShiftedYEdge(base, (x_size, y_size), (y_shift,))
    elif split_type == "HexRepeat":
        gs = float(_get(s, "grid_size", 0.2))
        base = sws.HexRepeatEdge(base, (gs,))
    elif split_type == "HexRepeatY":
        gs = float(_get(s, "grid_size", 0.2))
        base = sws.HexRepeatYEdge(base, (gs,))
    elif split_type == "RadialRepeat":
        ru = float(_get(s, "radial_unit", 0.25))
        au = float(_get(s, "angular_unit", 0.5236))
        base = sws.RadialRepeatEdge(base, (ru, au), 0.0)
    elif split_type == "RadialRepeatBricked":
        ru = float(_get(s, "radial_unit", 0.25))
        au = float(_get(s, "angular_unit", 0.5236))
        base = sws.RadialRepeatBrickedEdge(base, (ru, au), 0.0)
    elif split_type == "RadialRepeatFixedArc":
        ru = float(_get(s, "radial_unit", 0.25))
        arc = float(_get(s, "arc_size", 0.35))
        base = sws.RadialRepeatFixedArcEdge(base, (ru, arc), 0.0)
    elif split_type == "RadialRepeatFixedArcBricked":
        ru = float(_get(s, "radial_unit", 0.25))
        arc = float(_get(s, "arc_size", 0.35))
        base = sws.RadialRepeatFixedArcBrickedEdge(base, (ru, arc), 0.0)
    elif split_type == "VoronoiRepeat":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        noise = float(_get(s, "noise_rate", 0.5))
        base = sws.VoronoiRepeatEdge(base, (x_size, y_size), (noise,))
    elif split_type == "IrregularRepeat":
        x_size = float(_get(s, "x_size", 0.25))
        y_size = float(_get(s, "y_size", x_size))
        noise = float(_get(s, "noise_rate", 0.5))
        base = sws.IrregularRectRepeatEdge(base, (x_size, y_size), (noise,))
    elif split_type in ("RandomFillRectRepeat", "RandomFillVoronoi", "OverlapX", "OverlapY", "OverlapXY"):
        x_size = float(_get(s, "x_size", _get(s, "grid_size", 0.25)))
        y_size = float(_get(s, "y_size", x_size))
        base = sws.RectRepeatEdge(base, (x_size, y_size))
    else:
        base = sws.RectRepeatEdge(base, (0.25, 0.25))

    post = getattr(layout_cfg, "post_deform", None)
    if post is not None:
        rot = float(_get(post, "rot", 0.0))
        if rot != 0.0:
            base = sws.CartRotate(base, (rot,))
        scale = float(_get(post, "scale", 1.0))
        if scale != 1.0:
            base = sws.CartScale(base, (scale, scale))

    return base


def _apply_tile_effects_patternator(
    tile_expr: gls.GLFunction,
    te: CN,
    resolution: int,
    device: str,
) -> gls.GLFunction:
    """
    Apply tile effects by updating the tile expression (same pipeline as patternator
    execute_tile in mtp_tile.py). Uses GeoLIPI nodes only; no sws.Tile* for
    outline/shadow/rotate/scale/reflect/opacity. Recolor is applied last as
    sws.TileRecolor (requires major_hue at eval time) so the tree is
    TileRecolor(gls_stuff) and the evaluator can handle it before delegating to GeoLIPI.
    """
    # Outline: AlphaMask2D → AlphaToSDF2D → Dilate2D → ApplyColor2D → SourceOver(tile, outline)
    if getattr(te, "do_outline", False):
        outline = gls.AlphaMask2D(tile_expr)
        outline = gls.AlphaToSDF2D(outline, 1.0 / resolution)
        thick = float(getattr(te, "outline_thickness", 0.02))
        outline = gls.Dilate2D(outline, thick)
        oc = getattr(te, "outline_color", (0.0, 0.0, 0.0, 1.0))
        oc_tensor = th.tensor(oc, dtype=th.float32, device=device) if not isinstance(oc, th.Tensor) else oc.to(device)
        outline = gls.ApplyColor2D(outline, oc_tensor)
        tile_expr = gls.SourceOver(tile_expr, outline)

    # Shadow: AlphaMask2D → AlphaToSDF2D → Dilate2D → Translate2D(0.1,0.1) → ApplyColor2D(grey) → SourceOver(tile, shadow)
    if getattr(te, "do_shadow", False):
        shadow = gls.AlphaMask2D(tile_expr)
        shadow = gls.AlphaToSDF2D(shadow, 1.0 / resolution)
        thick = float(getattr(te, "shadow_thickness", 0.02))
        shadow = gls.Dilate2D(shadow, thick)
        shadow = gls.Translate2D(shadow, th.tensor([0.1, 0.1], dtype=th.float32, device=device))
        shadow = gls.ApplyColor2D(shadow, sp.Symbol("grey"))
        tile_expr = gls.SourceOver(tile_expr, shadow)

    # Rotate, scale, reflect, opacity: same as patternator (gls). Pass tensors so
    # GeoLIPI eval_mod / transform functions receive tensors, not Python floats.
    if getattr(te, "do_rotate", False):
        angle = float(getattr(te, "rot", 0.0))
        tile_expr = gls.EulerRotate2D(tile_expr, th.tensor(angle, dtype=th.float32, device=device))
    if getattr(te, "do_scale", False):
        scale = float(getattr(te, "scale", 1.0))
        tile_expr = gls.Scale2D(tile_expr, th.tensor([scale, scale], dtype=th.float32, device=device))
    if getattr(te, "ref_x", False):
        tile_expr = gls.ReflectCoords2D(tile_expr, th.tensor([1.0, 0.0], dtype=th.float32, device=device))
    if getattr(te, "ref_y", False):
        tile_expr = gls.ReflectCoords2D(tile_expr, th.tensor([0.0, 1.0], dtype=th.float32, device=device))
    if getattr(te, "do_opacity", False):
        op = float(getattr(te, "opacity", 1.0))
        # GeoLIPI eval_color_mod expects args[1:] to be Symbols (lookup_table or COLOR_MAP)
        op_tensor = th.tensor([op], dtype=th.float32, device=device)
        tile_expr = gls.ModifyOpacity2D(tile_expr, op_tensor)

    # Recolor last: sws.TileRecolor (needs major_hue at eval time) so tree is TileRecolor(gls_stuff)
    if getattr(te, "do_recolor", False):
        hue = float(getattr(te, "new_color_hue", 0.0))
        tile_expr = sws.TileRecolor(tile_expr, hue)

    return tile_expr


def tileset_config_to_expr(
    tile_cfg: CN,
    tile_tensors: Optional[Dict[str, th.Tensor]] = None,
    device: str = "cpu",
    resolution: Optional[int] = None,
) -> List[gls.GLFunction]:
    """
    Convert tile_cfg.tileset (patternator) into a list of geolipi tile expressions.
    Tile effects (outline, shadow, rotate, scale, reflect, opacity) are applied by
    updating the expression tree with GeoLIPI nodes, matching patternator's
    execute_tile in mtp_tile.py.

    Parameters
    ----------
    tile_cfg:
        Patternator tile config containing tileset list with tilefile paths.
    tile_tensors:
        Optional dict mapping tilefile path (str) to preloaded tensor [H, W, C].
        If None, only tiles with pre-attached tile_tensor attribute are used (legacy).
    device:
        Device string for tensor placement.
    resolution:
        Resolution used for AlphaToSDF2D in outline/shadow (default 256).
        Pass sketcher.resolution when available for exact match to runtime.

    Returns
    -------
    List of tile expressions (GeoLIPI GLFunction).
    """
    tile_exprs: List[gls.GLFunction] = []
    tile_tensors = tile_tensors or {}
    res = resolution if resolution is not None else 256

    for t_cfg in tile_cfg.tileset:
        tile_tensor = None

        # First, try to look up tensor by filename from tile_tensors dict
        tilefile = getattr(t_cfg, "tilefile", None)
        if tilefile is not None:
            tensor = tile_tensors.get(str(tilefile))
            if tensor is not None:
                tile_tensor = tensor.to(device)

        # Fallback: check for pre-attached tensor attribute (legacy support)
        if tile_tensor is None:
            tile_tensor = getattr(t_cfg, "tile_tensor", None)
            if tile_tensor is not None:
                tile_tensor = tile_tensor.to(device) if isinstance(tile_tensor, th.Tensor) else tile_tensor

        if tile_tensor is None:
            continue

        tile_expr = prim2d.TileUV2D(tile_tensor)
        te = getattr(t_cfg, "tile_effects", None)
        if te is not None:
            tile_expr = _apply_tile_effects_patternator(tile_expr, te, res, device)
        tile_exprs.append(tile_expr)

    return tile_exprs


def mtp_config_to_expr(
    cfg: CN,
    tile_dir: Optional[str] = None,
    tile_tensors: Optional[Dict[str, th.Tensor]] = None,
    device: str = "cpu",
    resolution: Optional[int] = None,
):
    """
    Map a full MTP config to a single expression combining background and tiling.

    For grid_n_color backgrounds, builds:
        SourceOver(
            ApplyTile(layout_expr, tile_expr),
            ApplyColoring(bg_layout_expr, base_rgba, recolor_params, signal_expr)
        )
    where recolor_params is (recolor_type, recolor_seed, signal_mode) and
    signal_expr is a typed discrete signal expression — NO CfgNode objects.

    For plain backgrounds:
        SourceOver(ApplyTile(...), ConstantBackground(base_rgba))

    If no bg_cfg:
        ApplyTile(layout_expr, tile_expr)
    """
    layout_expr = layout_config_to_expr(cfg.layout_cfg)
    cellfx_cfg = getattr(cfg, "cellfx_cfg", None)
    if cellfx_cfg is not None:
        layout_expr = _wrap_layout_with_cell_grid_effects(layout_expr, cellfx_cfg)

    if tile_tensors is None or len(tile_tensors) == 0:
        if tile_dir is None:
            raise ValueError(
                "Either tile_dir or tile_tensors must be provided. "
                "tile_dir: directory to load tile images from; tile_tensors: preloaded dict."
            )
        tile_tensors = load_tile_tensors_from_config(cfg.tile_cfg, tile_dir, device=device)

    tile_exprs = tileset_config_to_expr(
        cfg.tile_cfg, tile_tensors=tile_tensors, device=device, resolution=resolution
    )
    if not tile_exprs:
        raise ValueError(
            "No tile expressions could be constructed. "
            "Ensure tile_dir contains the tile files referenced in cfg.tile_cfg.tileset, "
            "or pass tile_tensors with an entry for each tilefile."
        )

    cellfx_cfg = getattr(cfg, "cellfx_cfg", None)
    tile_order_cfg = getattr(cellfx_cfg, "tile_order", None) if cellfx_cfg else None
    n_tiles = int(getattr(tile_order_cfg, "n_tiles", 1)) if tile_order_cfg else 1

    if n_tiles > 1 and tile_order_cfg is not None and hasattr(tile_order_cfg, "signal"):
        signal_expr = _signal_cfg_to_signal_expr(tile_order_cfg.signal)
        n_tiles = min(n_tiles, len(tile_exprs))
        args = [layout_expr, signal_expr, tile_exprs[0]]
        for i in range(1, n_tiles):
            args.append(tile_exprs[i])
        apply_tile = sws.ApplyMultiTile(*args)
    else:
        tile_expr = tile_exprs[0]
        apply_tile = sws.ApplyTile(layout_expr, tile_expr)

    # Wrap with canvas effects (RecolorFx, OutlineFx, OpacityFx) in patternator order
    effects = getattr(cellfx_cfg, "effects", []) if cellfx_cfg else []
    for fx in effects:
        fx_type = getattr(fx, "_type", None)
        if not hasattr(fx, "signal"):
            continue
        signal_expr = _signal_cfg_to_signal_expr(fx.signal)
        mode = str(getattr(fx, "mode", "single"))
        if fx_type == "RecolorFx":
            recolor_type = str(getattr(fx, "recolor_type", "smooth"))
            recolor_seed = int(getattr(fx, "recolor_seed", 0))
            apply_tile = sws.ApplyCellRecolor(
                apply_tile, signal_expr, recolor_type, recolor_seed, mode
            )
        elif fx_type == "OutlineFx":
            outline_color = getattr(fx, "outline_color", (0.0, 0.0, 0.0, 1.0))
            if not isinstance(outline_color, th.Tensor):
                outline_color = th.tensor(outline_color, dtype=th.float32, device=device)
            thickness = float(getattr(fx, "outline_thickness", 0.02))
            apply_tile = sws.ApplyCellOutline(
                apply_tile, signal_expr, outline_color, thickness, mode
            )
        elif fx_type == "OpacityFx":
            opacity = float(getattr(fx, "opacity", 0.5))
            apply_tile = sws.ApplyCellOpacity(apply_tile, signal_expr, opacity, mode)

    bg_cfg = getattr(cfg, "bg_cfg", None)
    if bg_cfg is not None:
        bg_mode = getattr(bg_cfg, "bg_mode", "plain")
        base_rgba = background_color_from_config(bg_cfg, device=device)

        if bg_mode == "grid_n_color":
            # Determine which layout the background grid uses
            grid_source = getattr(bg_cfg, "grid", "new")
            if grid_source == "new" and hasattr(bg_cfg, "layout_config"):
                bg_layout_expr = layout_config_to_expr(bg_cfg.layout_config)
            else:
                # "original" → reuse the same layout as the tile pattern
                bg_layout_expr = layout_expr

            recolor_params, signal_expr = _recolor_cfg_to_expr(bg_cfg.color_fx)

            bg_expr = sws.ApplyColoring(
                bg_layout_expr,
                base_rgba,
                recolor_params,
                signal_expr,
            )
        else:
            # plain background
            bg_expr = sws.ConstantBackground(base_rgba)

        # LayerOver: tile on top of background = SourceOver(apply_tile, background)
        root = gls.SourceOver(apply_tile, bg_expr)
    else:
        root = apply_tile

    # ----- Border: if border_cfg present, overlay BorderEffect on top -----
    border_cfg = getattr(cfg, "border_cfg", None)
    if border_cfg is not None and getattr(border_cfg, "_type", None) == "Border":
        edge_expr = layout_config_to_edge_expr(cfg.layout_cfg)
        border_mode = str(getattr(border_cfg, "border_mode", "simple"))
        border_size = float(getattr(border_cfg, "border_size", 0.01))
        border_color = border_color_from_config(border_cfg, device=device)
        if hasattr(border_cfg, "signal"):
            signal_expr = _signal_cfg_to_signal_expr(border_cfg.signal)
            border_effect = sws.BorderEffect(
                layout_expr, edge_expr, border_mode, border_size, border_color, signal_expr
            )
        else:
            border_effect = sws.BorderEffect(
                layout_expr, edge_expr, border_mode, border_size, border_color
            )
        root = gls.SourceOver(border_effect, root)

    return root




# ---- RSP config to expression ----

# RSP split type -> (id_reduce, mod_k or None)
# Use "reduce_sum" not "sum" to avoid sympify("sum") -> Python builtin
_RSP_SPLIT_ID_REDUCE = {
    "x_strip": ("x", None),
    "y_strip": ("y", None),
    "checkerboard": ("reduce_sum", None),
    "diamond_board": ("x", None),
    "polar_r": ("x", None),
    "polar_a": ("y", None),
    "hex_strip": ("x", None),
    "ring_hex": ("x", None),
    "ray_hex": ("sum_mod", "count"),
    "ray_diamond": ("sum_mod", "count"),
    "hex_scatter": ("y", None),
    "diamond_scatter": ("y", None),
    "polar_checkerboard": ("reduce_sum", None),
    "polar_fixed_arc": ("reduce_sum", None),
    "polar_fixed_arc_bricked": ("reduce_sum", None),
    "voronoi_checkerboard": ("reduce_sum", None),
    "irregular_rect_checkerboard": ("reduce_sum", None),
    "shape_sdf": ("none", None),
    "shape_overlay": ("none", None),
}


def _rsp_layout_to_expr(
    layout_cfg: CN,
    tile_dir: Optional[str] = None,
    tile_tensors: Optional[Dict[str, th.Tensor]] = None,
    device: str = "cpu",
) -> Tuple[sws.GridFunction, str, Optional[int]]:
    """
    Build RSP layout expression from cfg.layout.
    Returns (layout_expr, id_reduce, mod_k).
    """
    expr: sws.GridFunction = sws.CartesianGrid()
    pre = getattr(layout_cfg, "pre_deform", layout_cfg)
    t_x = float(_get(pre, "t_x", 0.0))
    t_y = float(_get(pre, "t_y", 0.0))
    if t_x != 0.0 or t_y != 0.0:
        expr = sws.CartTranslate(expr, (t_x, t_y))
    rot = float(_get(pre, "rot", 0.0))
    if rot != 0.0:
        expr = sws.CartRotate(expr, (rot,))
    scale = float(_get(pre, "scale", 1.0))
    if scale != 1.0:
        expr = sws.CartScale(expr, (scale, scale))

    deform = getattr(layout_cfg, "deform", None)
    if deform is not None:
        dtype = getattr(deform, "_type", None)
        if dtype == "radial_deform":
            center = _get(deform, "center", (0.0, 0.0))
            if hasattr(center, "__len__") and len(center) >= 2:
                center = (float(center[0]), float(center[1]))
            else:
                center = (float(_get(deform, "center_x", 0.0)), float(_get(deform, "center_y", 0.0)))
            signal_mode = str(_get(deform, "signal_mode", "linear"))
            dist_rate = float(_get(deform, "dist_rate", 0.2))
            sin_k = float(_get(deform, "sin_k", 6.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            expr = sws.RadialDeform(expr, center, signal_mode, dist_rate, sin_k, phase_shift)
        elif dtype == "perlin_deform":
            res_seq = getattr(deform, "resolution_seq", [4, 8])
            if hasattr(res_seq, "tolist"):
                res_seq = res_seq.tolist()
            res_seq = tuple(int(x) for x in res_seq) if res_seq else (4, 8)
            seed = int(_get(deform, "seed", 0))
            dist_mode = str(_get(deform, "dist_mode", "xy"))
            dist_rate = float(_get(deform, "dist_rate", 0.08))
            expr = sws.PerlinDeform(expr, res_seq, seed, dist_mode, dist_rate)
        elif dtype == "decay_deform":
            direction = _get(deform, "direction", (1.0, 0.0))
            if hasattr(direction, "__len__") and len(direction) >= 2:
                direction = (float(direction[0]), float(direction[1]))
            else:
                direction = (1.0, 0.0)
            axis = str(_get(deform, "axis", "x"))
            dist_rate = float(_get(deform, "dist_rate", 0.5))
            expr = sws.DecayDeform(expr, direction, axis, dist_rate)
        elif dtype == "strip_deform":
            angle = float(_get(deform, "angle", 0.0))
            axis = str(_get(deform, "axis", "x"))
            sin_k = float(_get(deform, "sin_k", 4.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            dist_rate = float(_get(deform, "dist_rate", 0.05))
            expr = sws.StripDeform(expr, angle, axis, sin_k, phase_shift, dist_rate)
        elif dtype == "swirl_deform":
            center = _get(deform, "center", (0.0, 0.0))
            if hasattr(center, "__len__") and len(center) >= 2:
                center = (float(center[0]), float(center[1]))
            else:
                center = (float(_get(deform, "center_x", 0.0)), float(_get(deform, "center_y", 0.0)))
            signal_mode = str(_get(deform, "signal_mode", "linear"))
            dist_rate = float(_get(deform, "dist_rate", 0.2))
            sin_k = float(_get(deform, "sin_k", 6.0))
            phase_shift = float(_get(deform, "phase_shift", 0.0))
            expr = sws.SwirlDeform(expr, center, signal_mode, dist_rate, sin_k, phase_shift)

    s = _split_cfg(layout_cfg)
    split_type = getattr(s, "_type", "x_strip")
    id_reduce, mod_k_key = _RSP_SPLIT_ID_REDUCE.get(split_type, ("none", None))
    mod_k = int(_get(s, mod_k_key, 5)) if mod_k_key == "count" else None

    if split_type == "x_strip":
        stripe_size = float(_get(s, "stripe_size", 0.15))
        expr = sws.RectRepeat(expr, (stripe_size, stripe_size))
    elif split_type == "y_strip":
        stripe_size = float(_get(s, "stripe_size", 0.15))
        expr = sws.RectRepeat(expr, (stripe_size, stripe_size))
    elif split_type == "checkerboard":
        x_size = float(_get(s, "x_size", 0.3))
        y_size = float(_get(s, "y_size", 0.3))
        expr = sws.RectRepeat(expr, (x_size, y_size))
    elif split_type == "diamond_board":
        stripe_size = float(_get(s, "stripe_size", 0.3))
        expr = sws.DiamondRepeat(expr, (stripe_size,))
    elif split_type == "polar_r":
        ru = float(_get(s, "radial_unit", 0.2))
        expr = sws.RadialRepeatEdge(sws.CartToPolar(expr), (ru, ru), 0.0)
    elif split_type == "polar_a":
        au = float(_get(s, "angular_unit", np.pi / 6))
        expr = sws.RadialRepeatEdge(sws.CartToPolar(expr), (au, au), 0.0)
    elif split_type == "hex_strip":
        stripe_size = float(_get(s, "stripe_size", 0.2))
        expr = sws.HexRepeat(expr, (stripe_size,))
    elif split_type == "ring_hex":
        stripe_size = float(_get(s, "stripe_size", 0.2))
        expr = sws.HexRepeat(sws.CartToPolar(expr), (stripe_size,))
    elif split_type == "ray_hex":
        stripe_size = float(_get(s, "stripe_size", 0.2))
        expr = sws.HexRepeat(sws.CartScale(sws.CartToPolar(expr), (1, np.pi)), (stripe_size,))
    elif split_type == "ray_diamond":
        stripe_size = float(_get(s, "stripe_size", 0.2))
        expr = sws.DiamondRepeat(sws.CartScale(sws.CartToPolar(expr), (1, np.pi)), (stripe_size,))
    elif split_type == "hex_scatter":
        stripe_size = float(_get(s, "stripe_size", 0.2))
        expr = sws.HexRepeat(sws.CartScale(sws.CartToPolar(expr), (1, np.pi)), (stripe_size,))
    elif split_type == "diamond_scatter":
        stripe_size = float(_get(s, "stripe_size", 0.2))
        expr = sws.DiamondRepeat(sws.CartScale(sws.CartToPolar(expr), (1, np.pi)), (stripe_size,))
    elif split_type == "polar_checkerboard":
        ru = float(_get(s, "radial_unit", 0.2))
        au = float(_get(s, "angular_unit", np.pi / 6))
        expr = sws.RadialRepeatInitRadial(sws.CartToPolar(expr), (ru, au), (0.0,))
    elif split_type == "polar_fixed_arc":
        ru = float(_get(s, "radial_unit", 0.2))
        au = float(_get(s, "angular_unit", np.pi / 6))
        expr = sws.RadialRepeatFixedArc(sws.CartToPolar(expr), (ru, au), (0.0,))
    elif split_type == "polar_fixed_arc_bricked":
        ru = float(_get(s, "radial_unit", 0.2))
        au = float(_get(s, "angular_unit", np.pi / 6))
        expr = sws.RadialRepeatFixedArcBricked(sws.CartToPolar(expr), (ru, au), (0.0,))
    elif split_type == "voronoi_checkerboard":
        x_size = float(_get(s, "x_size", 0.2))
        y_size = float(_get(s, "y_size", 0.2))
        noise = float(_get(s, "noise_rate", 0.5))
        if getattr(s, "radial", False):
            expr = sws.VoronoiRepeatRadialDeformed(sws.CartToPolar(expr), (x_size, y_size), (noise,))
        else:
            expr = sws.VoronoiRepeatRadialDeformed(expr, (x_size, y_size), (noise,))
    elif split_type == "irregular_rect_checkerboard":
        x_size = float(_get(s, "x_size", 0.2))
        y_size = float(_get(s, "y_size", 0.2))
        noise = float(_get(s, "noise_rate", 0.5))
        if getattr(s, "radial", False):
            expr = sws.IrregularRectRepeat(sws.CartToPolar(expr), (x_size, y_size), (noise,))
        else:
            expr = sws.IrregularRectRepeat(expr, (x_size, y_size), (noise,))
    elif split_type in ("shape_sdf", "shape_overlay"):
        tilespec = getattr(s, "tilespec", None)
        if tilespec is None:
            expr = sws.RectRepeat(expr, (0.25, 0.25))
        else:
            tilefile = getattr(tilespec, "tilefile", None)
            tile_tensor = None
            if tile_tensors and tilefile:
                tile_tensor = tile_tensors.get(str(tilefile))
            if tile_dir and tilefile and tile_tensor is None:
                path = str(tilefile) if os.path.isabs(str(tilefile)) else os.path.join(tile_dir, os.path.basename(str(tilefile)))
                if os.path.isfile(path):
                    from PIL import Image
                    pil_img = Image.open(path).convert("RGBA").rotate(90, expand=True)
                    arr = np.array(pil_img).astype("float32") / 255.0
                    tile_tensor = th.from_numpy(arr).to(device)
            if tile_tensor is not None:
                shape_scale = float(_get(s, "shape_scale", 1.0))
                tile_expr = prim2d.TileUV2D(tile_tensor)
                shape_expr = gls.Scale2D(tile_expr, (shape_scale, shape_scale))
                sdf_step = float(_get(s, "sdf_step_size", 0.01))
                ring_counts = int(_get(s, "ring_counts", 3))
                growth_mode = str(_get(s, "growth_mode", "linear"))
                expr = sws.SDFRingPartition(expr, shape_expr, sdf_step, ring_counts, growth_mode)
            else:
                expr = sws.RectRepeat(expr, (0.25, 0.25))
    else:
        expr = sws.RectRepeat(expr, (0.25, 0.25))

    return expr, id_reduce, mod_k


def _rsp_fill_to_expr(
    fill_cfg: CN,
    layout_expr: sws.GridFunction,
    id_reduce: str,
    mod_k: Optional[int],
    device: str,
) -> sws.ColorFill:
    """Convert RSP fill config to SolidColorFill / InterpColorFill / TriInterpColorFill."""
    coloring_mode = str(getattr(fill_cfg, "coloring_mode", "simple"))
    palette = getattr(fill_cfg, "color_palette", [])
    n_colors = int(getattr(fill_cfg, "n_colors", len(palette)))

    def _to_rgba(c):
        if isinstance(c, th.Tensor):
            c = c.to(device)
            if c.numel() == 3:
                c = th.cat([c.view(-1), th.ones(1, device=device)], dim=-1)
            return c
        return th.tensor(list(c) + [1.0], device=device, dtype=th.float32)

    palette_list = [_to_rgba(palette[i]) if i < len(palette) else _to_rgba(palette[0]) for i in range(n_colors)]
    # Stack to single tensor so GLFunction stores in lookup_table (lists can't be sympified)
    palette_tensor = th.stack(palette_list)

    if coloring_mode == "simple":
        args = [layout_expr, palette_tensor, n_colors, id_reduce]
        if mod_k is not None:
            args.append(mod_k)
        return sws.SolidColorFill(*args)
    elif coloring_mode == "discrete_interp":
        color_a = _to_rgba(palette[0]) if len(palette) > 0 else th.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        color_b = _to_rgba(palette[1]) if len(palette) > 1 else color_a
        interp_mode = str(getattr(fill_cfg, "interp_mode", "simple"))
        sin_k = int(getattr(fill_cfg, "sin_k", 1))
        mid_point = float(getattr(fill_cfg, "mid_point", 0.5))
        args = [layout_expr, color_a, color_b, interp_mode, sin_k, mid_point, id_reduce]
        if mod_k is not None:
            args.append(mod_k)
        return sws.InterpColorFill(*args)
    elif coloring_mode == "discrete_tri_interp":
        color_a = _to_rgba(palette[0]) if len(palette) > 0 else th.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        color_b = _to_rgba(palette[1]) if len(palette) > 1 else color_a
        color_c = _to_rgba(palette[2]) if len(palette) > 2 else color_b
        interp_mode = str(getattr(fill_cfg, "interp_mode", "simple"))
        sin_k = int(getattr(fill_cfg, "sin_k", 1))
        mid_point = float(getattr(fill_cfg, "mid_point", 0.5))
        args = [layout_expr, color_a, color_b, color_c, interp_mode, sin_k, mid_point, id_reduce]
        if mod_k is not None:
            args.append(mod_k)
        return sws.TriInterpColorFill(*args)
    else:
        args = [layout_expr, palette_tensor, n_colors, id_reduce]
        if mod_k is not None:
            args.append(mod_k)
        return sws.SolidColorFill(*args)


def rsp_config_to_expr(
    cfg: CN,
    tile_dir: Optional[str] = None,
    tile_tensors: Optional[Dict[str, th.Tensor]] = None,
    device: str = "cpu",
):
    """
    Convert an RSP config to a splitweave expression tree.

    RSP config has cfg.layout (pre_deform, deform, split, optional foreground)
    and cfg.fill (coloring_mode, color_palette, n_colors, etc.).
    """
    layout_cfg = cfg.layout
    layout_expr, id_reduce, mod_k = _rsp_layout_to_expr(
        layout_cfg, tile_dir=tile_dir, tile_tensors=tile_tensors, device=device
    )
    fill_expr = _rsp_fill_to_expr(cfg.fill, layout_expr, id_reduce, mod_k, device)

    foreground = getattr(layout_cfg, "foreground", None)
    if foreground is not None:
        fg_layout_expr, fg_id_reduce, _ = _rsp_layout_to_expr(
            CN({"pre_deform": layout_cfg.pre_deform, "deform": layout_cfg.deform, "split": foreground}),
            tile_dir=tile_dir, tile_tensors=tile_tensors, device=device,
        )
        ring_counts = int(getattr(foreground, "ring_counts", 3))

        def _to_rgba(c):
            if isinstance(c, th.Tensor):
                c = c.to(device)
                if c.numel() == 3:
                    c = th.cat([c.view(-1), th.ones(1, device=device)], dim=-1)
                return c
            return th.tensor(list(c) + [1.0], device=device, dtype=th.float32)

        fg_palette = [_to_rgba(c) for c in cfg.fill.color_palette[: ring_counts + 1]]
        while len(fg_palette) < ring_counts + 1:
            fg_palette.append(fg_palette[-1])
        fg_palette_tensor = th.stack(fg_palette)
        fg_fill = sws.SolidColorFill(fg_layout_expr, fg_palette_tensor, ring_counts + 1, "none")
        return gls.SourceOver(fg_fill, fill_expr)

    return fill_expr
