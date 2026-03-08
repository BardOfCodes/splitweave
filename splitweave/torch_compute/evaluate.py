"""
Pattern expression evaluator for splitweave.

Uses singledispatch on expression types to render (canvas, grid_ids)
from a pattern expression tree.

Top-level API
-------------
evaluate_pattern(expr, sketcher, aa=1)
    Public entry point.  ``aa`` controls super-sampling anti-aliasing:
    * aa=1  — single evaluation (no AA)
    * aa>=2 — evaluate at aa*resolution, then avg-pool down

rec_eval_pattern_expr
    Internal singledispatch function — one handler per expression type.
"""
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from typing import Tuple, Optional

import geolipi.symbolic as gls
from geolipi.torch_compute import Sketcher
from geolipi.torch_compute import recursive_evaluate
from geolipi.torch_compute.colorspace_functions import rgb2hsl_torch, hsl2rgb_torch
from geolipi.torch_compute.color_functions import source_over

import splitweave.symbolic as sws
from .eval_grid import grid_eval, rec_grid_eval
from .eval_discrete_signal import eval_discrete_signal
from .eval_tile import rec_eval_tile_expr
from .eval_cell_canvas import (
    process_signal as _process_signal,
    apply_recolor as _apply_recolor,
    apply_cell_outline,
    apply_cell_opacity,
)
from . import functions as fn

if sys.version_info >= (3, 11):
    from functools import singledispatch
else:
    from geolipi.torch_compute.patched_functools import singledispatch


# ============================================================
# Internal helpers for pattern evaluation
# ============================================================

def _layout_has_sdf_ring_partition(expr) -> bool:
    """Check whether the layout expression tree contains an SDFRingPartition node."""
    if isinstance(expr, sws.SDFRingPartition):
        return True
    if hasattr(expr, "args"):
        return any(
            _layout_has_sdf_ring_partition(a)
            for a in expr.args
            if isinstance(a, (gls.GLExpr, gls.GLFunction))
        )
    return False


def _reduce_grid_ids(grid_ids: th.Tensor, id_reduce: str, mod_k: Optional[int] = None) -> th.Tensor:
    """Collapse multi-dimensional grid_ids to 1D for RSP color assignment."""
    if id_reduce == "x":
        return grid_ids[..., 0:1]
    elif id_reduce == "y":
        return grid_ids[..., 1:2]
    elif id_reduce in ("sum", "reduce_sum"):  # "reduce_sum" avoids sympify("sum") -> builtin
        return grid_ids.sum(dim=-1, keepdim=True)
    elif id_reduce == "sum_mod" and mod_k is not None:
        return grid_ids.sum(dim=-1, keepdim=True) % mod_k
    return grid_ids  # "none" or default


def _eval_background_to_canvas(expr, sketcher: Sketcher,
                               grid_ids: Optional[th.Tensor] = None) -> th.Tensor:
    """
    Evaluate a background expression to a [N, 4] RGBA canvas.

    Handles:
    - ConstantBackground(color) -> solid color canvas
    - ApplyColoring(layout_expr, base_color, recolor_params, signal_expr)
      -> grid-based recolored canvas
    """
    base_grid = sketcher.get_base_coords()
    n = base_grid.shape[0]
    device = base_grid.device

    if isinstance(expr, sws.ConstantBackground):
        color = expr.get_arg(0)
        color = color.to(device) if isinstance(color, th.Tensor) else th.tensor(color, device=device, dtype=th.float32)
        return th.ones(n, 4, device=device, dtype=base_grid.dtype) * color.to(device)

    if isinstance(expr, sws.ApplyColoring):
        layout_val = expr.get_arg(0)
        base_color = expr.get_arg(1)
        if isinstance(base_color, th.Tensor):
            base_color = base_color.to(device)
        else:
            base_color = th.tensor(base_color, device=device, dtype=th.float32)

        rp = expr.get_arg(2)
        if isinstance(rp, th.Tensor):
            rp = rp.tolist()
        recolor_type = int(rp[0])
        recolor_seed = int(rp[1])
        signal_mode_double = bool(int(rp[2]))

        signal_expr = expr.get_arg(3)

        if isinstance(layout_val, (sws.GridFunction, gls.GLFunction)):
            _, bg_grid_ids = grid_eval(layout_val, sketcher)
        elif grid_ids is not None:
            bg_grid_ids = grid_ids
        else:
            raise ValueError("ApplyColoring: no layout expression and no grid_ids from parent")

        base_canvas = th.ones(n, 4, device=device, dtype=base_grid.dtype) * base_color

        signal = eval_discrete_signal(signal_expr, bg_grid_ids)

        new_canvas = _apply_recolor(
            recolor_type, recolor_seed, signal_mode_double,
            signal, base_canvas, bg_grid_ids, major_hues=0.0
        )
        return new_canvas

    raise TypeError(f"Unknown background expression type: {type(expr).__name__}")


def _apply_cell_canvas_effect_recolor(canvas: th.Tensor, grid_ids: th.Tensor,
                                      signal_expr, recolor_type_str: str, recolor_seed: int,
                                      mode_str: str) -> th.Tensor:
    """Apply per-cell recolor (hue shift) to canvas."""
    signal = eval_discrete_signal(signal_expr, grid_ids)
    mode_double = mode_str == "double"
    recolor_type = 0 if recolor_type_str == "smooth" else 1
    return _apply_recolor(
        recolor_type, recolor_seed, mode_double,
        signal, canvas, grid_ids, major_hues=0.0
    )


def _apply_cell_canvas_effect_opacity(canvas: th.Tensor, grid_ids: th.Tensor,
                                      signal_expr, opacity: float, mode_str: str) -> th.Tensor:
    """Apply per-cell opacity modulation."""
    signal = eval_discrete_signal(signal_expr, grid_ids)
    return apply_cell_opacity(canvas, grid_ids, signal, opacity)


def _apply_cell_canvas_effect_outline(canvas: th.Tensor, grid_ids: th.Tensor,
                                      signal_expr, outline_color, outline_thickness: float,
                                      mode_str: str, resolution: int) -> th.Tensor:
    """Apply per-cell outline."""
    signal = eval_discrete_signal(signal_expr, grid_ids)
    return apply_cell_outline(canvas, grid_ids, signal, outline_color, outline_thickness, resolution)


def _aa_eval_border(expr, sketcher: Sketcher, grid: th.Tensor, shift_amount: float = 5e-8):
    """Evaluate 2D primitive at grid with 8-shift average."""
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    evals = []
    for dy, dx in shifts:
        shift_tensor = th.tensor([dy, dx], device=sketcher.device, dtype=grid.dtype) * shift_amount
        new_grid = grid.clone()
        if new_grid.shape[-1] == 2:
            new_grid = new_grid + shift_tensor
            new_grid = sketcher.make_homogenous_coords(new_grid)
        else:
            new_grid[..., 0] = new_grid[..., 0] + shift_tensor[0]
            new_grid[..., 1] = new_grid[..., 1] + shift_tensor[1]
        out = recursive_evaluate(expr, sketcher, coords=new_grid)
        evals.append(out)
    return th.stack(evals, dim=0).mean(dim=0)


def _execute_border_like_patternator(
    edge_expr,
    border_mode: str,
    border_size: float,
    color: th.Tensor,
    grid_ids: th.Tensor,
    signal_expr,
    sketcher: Sketcher,
) -> th.Tensor:
    """Execute border rendering."""
    edge_grid, _ = grid_eval(edge_expr, sketcher)
    device = edge_grid.device
    border_grid = edge_grid.clone()

    def _vec2(*vals):
        return th.tensor(vals, device=device, dtype=th.float32)
    def _vec1(x):
        return th.tensor([x], device=device, dtype=th.float32)

    if border_mode == "dotted":
        angular_unit = np.pi / 8.0
        border_grid, _ = fn.polar_repeat_angular_grid(border_grid, angular_unit)
        border_expr = gls.ApplyColor2D(gls.Rectangle2D(_vec2(border_size, 0.2)), color)
    else:
        border_grid[:, 1] /= np.pi
        if border_mode == "simple":
            border_expr = gls.ApplyColor2D(gls.Rectangle2D(_vec2(border_size, 10.0)), color)
        elif border_mode == "siny":
            border_expr = gls.ApplyColor2D(
                gls.SinAlongAxisY2D(_vec1(10.0), _vec1(0.0), border_size), color
            )
        elif border_mode == "onion":
            inner = gls.Rectangle2D(_vec2(border_size, 10.0))
            border_expr = gls.ApplyColor2D(gls.Onion2D(inner, _vec1(0.001)), color)
        else:
            border_expr = gls.ApplyColor2D(gls.Rectangle2D(_vec2(border_size, 10.0)), color)

    if border_grid.shape[-1] == 2:
        border_grid_h = sketcher.make_homogenous_coords(border_grid)
    else:
        border_grid_h = border_grid
    border_out = _aa_eval_border(border_expr, sketcher, border_grid_h, shift_amount=5e-8)

    if signal_expr is not None:
        signal = eval_discrete_signal(signal_expr, grid_ids)
        processed = _process_signal(signal.float().unsqueeze(-1), False)
        mask = (processed > 0.5).float().squeeze(-1)
        border_out = border_out.clone()
        border_out[..., -1] = border_out[..., 0] * mask.to(device)
    return border_out


# ============================================================
# Single-dispatch pattern evaluator
# ============================================================

@singledispatch
def rec_eval_pattern_expr(expr:gls.GLExpr | gls.GLFunction, sketcher: Sketcher) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """Recursively evaluate a pattern expression to (canvas, grid_ids)."""
    raise TypeError(
        f"rec_eval_pattern_expr: unsupported expression type {type(expr).__name__}. "
        "Expected ApplyTile, ApplyMultiTile, ApplyCellRecolor/Outline/Opacity, BorderEffect, "
        "or SourceOver(...)."
    )


@rec_eval_pattern_expr.register
def _eval_source_over(expr:gls.SourceOver, sketcher: Sketcher):
    a0, a1 = expr.args[0], expr.args[1]

    # SourceOver(BorderEffect, pattern) — border drawn on top
    if isinstance(a0, sws.BorderEffect):
        border_canvas, grid_ids = rec_eval_pattern_expr(a0, sketcher)
        pattern_canvas, _ = rec_eval_pattern_expr(a1, sketcher)
        canvas = source_over(border_canvas, pattern_canvas)
        return canvas, grid_ids

    # SourceOver(fg_color_fill, bg_color_fill) — RSP foreground overlay
    _color_fill_types = (sws.SolidColorFill, sws.InterpColorFill, sws.TriInterpColorFill)
    if isinstance(a0, _color_fill_types) and isinstance(a1, _color_fill_types):
        fg_canvas, _ = rec_eval_pattern_expr(a0, sketcher)
        bg_canvas, grid_ids = rec_eval_pattern_expr(a1, sketcher)
        canvas = source_over(fg_canvas, bg_canvas)
        return canvas, grid_ids

    # SourceOver(tile_root, background) or SourceOver(background, tile_root)
    _tile_root_types = (
        sws.ApplyTile, sws.ApplyMultiTile,
        sws.ApplyCellRecolor, sws.ApplyCellOutline, sws.ApplyCellOpacity,
        sws.SolidColorFill, sws.InterpColorFill, sws.TriInterpColorFill,
    )
    apply_tile = None
    if isinstance(a0, _tile_root_types):
        apply_tile = a0
    elif isinstance(a1, _tile_root_types):
        apply_tile = a1
    bg_expr = a1 if apply_tile is a0 else a0

    if apply_tile is not None and isinstance(bg_expr, (sws.ConstantBackground, sws.ApplyColoring)):
        tile_canvas, grid_ids = rec_eval_pattern_expr(apply_tile, sketcher)
        bg_canvas = _eval_background_to_canvas(bg_expr, sketcher, grid_ids=grid_ids)
        canvas = source_over(tile_canvas, bg_canvas)
        return canvas, grid_ids

    raise TypeError(
        f"SourceOver: expected (tile_root, background) or (BorderEffect, pattern), "
        f"got ({type(a0).__name__}, {type(a1).__name__})"
    )


@rec_eval_pattern_expr.register(sws.ApplyTile)
def _eval_apply_tile(expr, sketcher: Sketcher):
    layout_expr, tile_expr = expr.args[0], expr.args[1]
    grid, grid_ids = grid_eval(layout_expr, sketcher)
    canvas = rec_eval_tile_expr(tile_expr, sketcher, grid)
    return canvas, grid_ids


@rec_eval_pattern_expr.register(sws.ApplyMultiTile)
def _eval_apply_multi_tile(expr, sketcher: Sketcher):
    layout_expr = expr.args[0]
    tile_exprs = [expr.args[i] for i in range(2, len(expr.args))]
    grid, grid_ids = grid_eval(layout_expr, sketcher)
    signal_expr = expr.get_arg(1)
    signal = eval_discrete_signal(signal_expr, grid_ids)
    n_tiles = len(tile_exprs)
    signal = signal.long().clamp(0, n_tiles - 1)
    if len(signal.shape) == 1:
        signal = signal.unsqueeze(-1)
    canvas = rec_eval_tile_expr(tile_exprs[0], sketcher, grid)
    for i in range(1, n_tiles):
        canvas_i = rec_eval_tile_expr(tile_exprs[i], sketcher, grid)
        canvas = th.where(signal == i, canvas_i, canvas)
    return canvas, grid_ids


@rec_eval_pattern_expr.register(sws.ApplyCellRecolor)
def _eval_apply_cell_recolor(expr: sws.ApplyCellRecolor, sketcher: Sketcher):
    pattern_expr = expr.args[0]
    canvas, grid_ids = rec_eval_pattern_expr(pattern_expr, sketcher)
    signal_expr = expr.get_arg(1)
    recolor_type = str(expr.args[2])
    recolor_seed = int(expr.get_arg(3))
    mode_str = str(expr.args[4])
    canvas = _apply_cell_canvas_effect_recolor(
        canvas, grid_ids, signal_expr, recolor_type, recolor_seed, mode_str
    )
    return canvas, grid_ids


@rec_eval_pattern_expr.register(sws.ApplyCellOutline)
def _eval_apply_cell_outline(expr: sws.ApplyCellOutline, sketcher: Sketcher):
    pattern_expr = expr.args[0]
    canvas, grid_ids = rec_eval_pattern_expr(pattern_expr, sketcher)
    signal_expr = expr.get_arg(1)
    outline_color = expr.get_arg(2)
    if isinstance(outline_color, th.Tensor):
        outline_color = outline_color.to(canvas.device)
    else:
        outline_color = th.tensor(outline_color, device=canvas.device, dtype=canvas.dtype)
    thickness = float(expr.get_arg(3))
    canvas = _apply_cell_canvas_effect_outline(
        canvas, grid_ids, signal_expr, outline_color, thickness, str(expr.args[4]), sketcher.resolution
    )
    return canvas, grid_ids


@rec_eval_pattern_expr.register(sws.ApplyCellOpacity)
def _eval_apply_cell_opacity(expr: sws.ApplyCellOpacity, sketcher: Sketcher):
    pattern_expr = expr.args[0]
    canvas, grid_ids = rec_eval_pattern_expr(pattern_expr, sketcher)
    signal_expr = expr.get_arg(1)
    opacity = float(expr.get_arg(2))
    canvas = _apply_cell_canvas_effect_opacity(canvas, grid_ids, signal_expr, opacity, str(expr.args[3]))
    return canvas, grid_ids


@rec_eval_pattern_expr.register(sws.BorderEffect)
def _eval_border_effect(expr, sketcher: Sketcher):
    layout_expr = expr.args[0]
    edge_expr = expr.args[1]
    grid, grid_ids = grid_eval(layout_expr, sketcher)
    border_mode = str(expr.get_arg(2))
    border_size = float(expr.get_arg(3))
    color = expr.get_arg(4)
    if not isinstance(color, th.Tensor):
        color = th.tensor(color, device=grid.device, dtype=th.float32)
    else:
        color = color.to(grid.device)
    signal_expr = expr.get_arg(5) if len(expr.args) > 5 else None
    border_canvas = _execute_border_like_patternator(
        edge_expr, border_mode, border_size, color, grid_ids,
        signal_expr, sketcher
    )
    return border_canvas, grid_ids


@rec_eval_pattern_expr.register(sws.SolidColorFill)
def _eval_solid_color_fill(expr: sws.SolidColorFill, sketcher: Sketcher):
    """Evaluate SolidColorFill: layout -> grid_ids -> palette colors."""
    layout_expr = expr.get_arg(0)
    grid, grid_ids = grid_eval(layout_expr, sketcher)
    id_reduce = str(expr.get_arg(3)) if len(expr.args) > 3 else "none"
    mod_k = int(expr.get_arg(4)) if len(expr.args) > 4 else None
    grid_ids = _reduce_grid_ids(grid_ids, id_reduce, mod_k)

    palette = expr.get_arg(1)
    n_colors = int(expr.get_arg(2))
    device = grid.device
    n = grid.shape[0]

    has_sdf_ring = _layout_has_sdf_ring_partition(layout_expr)
    if has_sdf_ring:
        colored_canvas = th.zeros(n, 4, device=device, dtype=th.float32)
    else:
        colored_canvas = th.ones(n, 4, device=device, dtype=th.float32)

    raw_ids = grid_ids[..., 0]
    valid = raw_ids >= 0 if has_sdf_ring else None
    condition = (raw_ids.long() % n_colors)
    for i in range(n_colors):
        c = palette[i] if i < len(palette) else palette[0]
        if not isinstance(c, th.Tensor):
            c = th.tensor(list(c) + [1.0], device=device, dtype=th.float32)
        else:
            c = c.to(device)
        if c.dim() == 0 or c.numel() == 3:
            c = th.cat([c.view(-1), th.ones(1, device=device)], dim=-1)
        mask = (condition == i)
        if valid is not None:
            mask = mask & valid
        colored_canvas = th.where(mask[..., None], c.expand_as(colored_canvas), colored_canvas)
    return colored_canvas, grid_ids


@rec_eval_pattern_expr.register(sws.InterpColorFill)
def _eval_interp_color_fill(expr: sws.InterpColorFill, sketcher: Sketcher):
    """Evaluate InterpColorFill: layout -> grid_ids -> 2-color interpolation.
    Arg order: layout(0), color_a(1), color_b(2), interp_mode(3), sin_k(4), mid_point(5), id_reduce(6), mod_k(7).
    """
    layout_expr = expr.get_arg(0)
    grid, grid_ids = grid_eval(layout_expr, sketcher)
    id_reduce = str(expr.get_arg(6)) if len(expr.args) > 6 else "none"
    mod_k = int(expr.get_arg(7)) if len(expr.args) > 7 else None
    grid_ids = _reduce_grid_ids(grid_ids, id_reduce, mod_k)

    color_a = expr.get_arg(1)
    color_b = expr.get_arg(2)
    device = grid.device
    if not isinstance(color_a, th.Tensor):
        color_a = th.tensor(list(color_a) + [1.0], device=device, dtype=th.float32)
    else:
        color_a = color_a.to(device)
    if not isinstance(color_b, th.Tensor):
        color_b = th.tensor(list(color_b) + [1.0], device=device, dtype=th.float32)
    else:
        color_b = color_b.to(device)

    grid_val = grid_ids[..., 0].float()
    upper_limit = 20
    grid_val = grid_val - grid_val.min()
    n_values = grid_val.max()
    if n_values > upper_limit:
        grid_val = th.round(grid_val * upper_limit / n_values)
    grid_val = grid_val / upper_limit

    interp_mode = str(expr.get_arg(3)) if len(expr.args) > 3 else "simple"
    if interp_mode == "symmetric":
        mid_point = float(expr.get_arg(5)) if len(expr.args) > 5 else 0.5
        grid_val = th.where(grid_val > mid_point, 1 - grid_val, grid_val)
    elif interp_mode == "siny":
        sin_k = float(expr.get_arg(4)) if len(expr.args) > 4 else 1
        grid_val = th.cos(grid_val * 2 * np.pi * sin_k) * 0.5 + 0.5

    colored_canvas = color_a * (1 - grid_val[..., None]) + color_b * grid_val[..., None]
    return colored_canvas, grid_ids


@rec_eval_pattern_expr.register(sws.TriInterpColorFill)
def _eval_tri_interp_color_fill(expr: sws.TriInterpColorFill, sketcher: Sketcher):
    """Evaluate TriInterpColorFill: layout -> grid_ids -> 3-color interpolation.
    Arg order: layout(0), color_a(1), color_b(2), color_c(3), interp_mode(4), sin_k(5), mid_point(6), id_reduce(7), mod_k(8).
    """
    layout_expr = expr.get_arg(0)
    grid, grid_ids = grid_eval(layout_expr, sketcher)
    id_reduce = str(expr.get_arg(7)) if len(expr.args) > 7 else "none"
    mod_k = int(expr.get_arg(8)) if len(expr.args) > 8 else None
    grid_ids = _reduce_grid_ids(grid_ids, id_reduce, mod_k)

    color_a = expr.get_arg(1)
    color_b = expr.get_arg(2)
    color_c = expr.get_arg(3)
    device = grid.device
    if not isinstance(color_a, th.Tensor):
        color_a = th.tensor(list(color_a) + [1.0], device=device, dtype=th.float32)
    else:
        color_a = color_a.to(device)
    if not isinstance(color_b, th.Tensor):
        color_b = th.tensor(list(color_b) + [1.0], device=device, dtype=th.float32)
    else:
        color_b = color_b.to(device)
    if not isinstance(color_c, th.Tensor):
        color_c = th.tensor(list(color_c) + [1.0], device=device, dtype=th.float32)
    else:
        color_c = color_c.to(device)

    grid_val = grid_ids[..., 0].float()
    upper_limit = 10
    grid_val = grid_val - grid_val.min()
    n_values = grid_val.max()
    if n_values > upper_limit:
        grid_val = th.round(grid_val * upper_limit / n_values)
    grid_val = grid_val / upper_limit

    interp_mode = str(expr.get_arg(4)) if len(expr.args) > 4 else "simple"
    mid_point = float(expr.get_arg(6)) if len(expr.args) > 6 else 0.5
    if interp_mode == "symmetric":
        grid_val = th.where(grid_val > mid_point, 1 - grid_val, grid_val)
    elif interp_mode == "siny":
        sin_k = float(expr.get_arg(5)) if len(expr.args) > 5 else 1
        grid_val = th.cos(grid_val * 2 * np.pi * sin_k) * 0.5 + 0.5

    factor_a = (grid_val - mid_point) / (1 - mid_point)
    factor_a = th.clamp(factor_a, 0, 1)
    color_hi = color_b + (color_c - color_b) * factor_a[..., None]
    factor_b = grid_val / mid_point
    factor_b = th.clamp(factor_b, 0, 1)
    color_lo = color_a + (color_b - color_a) * factor_b[..., None]
    colored_canvas = th.where(grid_val[..., None] >= mid_point, color_hi, color_lo)
    return colored_canvas, grid_ids


# ============================================================
# Anti-aliased evaluation (GeoLIPI primitives)
# ============================================================

def aa_eval(expr, sketcher, grid=None, mode=1, shift_amount=0.01,
            shift_mode="cartesian", relaxed_occupancy=True, relax_temperature=25.0):
    """Multi-sample anti-aliased evaluation for geolipi primitives."""
    if mode == 1:
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    elif mode == 0:
        shifts = [(0, 0)]
    elif mode == 2:
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    evals = []
    for shift in shifts:
        dy, dx = shift
        shift_tensor = th.tensor([dy, dx], device=sketcher.device) * shift_amount
        if grid is None:
            new_grid = sketcher.get_base_coords().clone() + shift_tensor
        else:
            new_grid = grid.clone() + shift_tensor
        output = recursive_evaluate(expr, sketcher,
                                    coords=new_grid,
                                    relaxed_occupancy=relaxed_occupancy,
                                    relax_temperature=relax_temperature)
        evals.append(output)
    evals = th.stack(evals, dim=0)
    output = evals.mean(dim=0)
    return output


# ============================================================
# Top-level public API
# ============================================================

def evaluate_pattern(expr, sketcher: Sketcher, aa: int = 1) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """
    Evaluate a pattern expression to (canvas, grid_ids).

    Parameters
    ----------
    expr : pattern expression
        The root of a pattern expression tree (e.g. SourceOver(ApplyTile(...), bg)).
    sketcher : Sketcher
        GeoLIPI Sketcher used for evaluation.
    aa : int
        Super-sampling anti-aliasing factor.
        * 1   — no AA, direct evaluation.
        * >=2 — evaluate at ``aa * resolution``, then average-pool down.

    Returns
    -------
    canvas : Tensor [R*R, C]
        The rendered pattern canvas (typically RGBA, C=4).
    grid_ids : Tensor or None
        Grid cell IDs from the layout expression.
    """
    if aa <= 1:
        return rec_eval_pattern_expr(expr, sketcher)

    R = sketcher.resolution
    R_aa = R * aa
    aa_sketcher = Sketcher(device=sketcher.device, resolution=R_aa, n_dims=2)
    canvas, grid_ids = rec_eval_pattern_expr(expr, aa_sketcher)

    C = canvas.shape[-1]
    canvas_2d = canvas.reshape(1, R_aa, R_aa, C).permute(0, 3, 1, 2)  # [1, C, R_aa, R_aa]
    canvas_2d = F.avg_pool2d(canvas_2d, kernel_size=aa, stride=aa)     # [1, C, R, R]
    canvas = canvas_2d.squeeze(0).permute(1, 2, 0).reshape(R * R, C)

    return canvas, grid_ids


# Backward-compatible alias
evaluate_pattern_expr = evaluate_pattern
