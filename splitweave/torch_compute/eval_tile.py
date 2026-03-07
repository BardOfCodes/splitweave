"""
Tile expression evaluation for splitweave.

Uses singledispatch on tile expression types. TileRecolor is handled
here (HSL hue shift); all other tile types delegate to GeoLIPI
recursive_evaluate.

API
---
rec_eval_tile_expr(expr, sketcher, grid)
    Evaluate a tile expression at grid coords. Returns [N, 4] RGBA tensor.
"""
import sys
import torch as th

from geolipi.torch_compute import Sketcher
from geolipi.torch_compute import recursive_evaluate
from geolipi.torch_compute.colorspace_functions import rgb2hsl_torch, hsl2rgb_torch

import splitweave.symbolic as sws

if sys.version_info >= (3, 11):
    from functools import singledispatch
else:
    from geolipi.torch_compute.patched_functools import singledispatch


@singledispatch
def rec_eval_tile_expr(expr, sketcher: Sketcher, grid: th.Tensor) -> th.Tensor:
    """
    Evaluate a tile expression at grid coords.
    Fallback: delegate to GeoLIPI recursive_evaluate.
    """
    return recursive_evaluate(expr, sketcher, coords=grid)


def _resolve_float(expr, arg_index: int, default=1.0):
    """Resolve a float param from expr via get_arg."""
    if arg_index >= len(expr.args):
        return default
    v = expr.get_arg(arg_index)
    if hasattr(v, "item"):
        return float(v.item()) if v.numel() == 1 else float(v.flatten()[0].item())
    return float(v) if v is not None else default


@rec_eval_tile_expr.register(sws.TileScale)
def _eval_tile_scale(expr: sws.TileScale, sketcher: Sketcher, grid: th.Tensor) -> th.Tensor:
    """TileScale: scale UV coords so tile appears larger (scale<1) or smaller (scale>1)."""
    inner_tile = expr.args[0]
    scale = _resolve_float(expr, 1, 1.0)
    scale = max(1e-6, scale)
    grid_scaled = grid.clone()
    if grid_scaled.shape[-1] >= 2:
        grid_scaled[..., 0] = grid_scaled[..., 0] / scale
        grid_scaled[..., 1] = grid_scaled[..., 1] / scale
    return rec_eval_tile_expr(inner_tile, sketcher, grid_scaled)


@rec_eval_tile_expr.register(sws.TileRecolor)
def _eval_tile_recolor(expr: sws.TileRecolor, sketcher: Sketcher, grid: th.Tensor) -> th.Tensor:
    """TileRecolor: HSL hue shift. Requires major_hue at eval time."""
    inner = rec_eval_tile_expr(expr.args[0], sketcher, grid)
    hue = _resolve_float(expr, 1, 0.0)
    alpha = inner[..., -1:].clone()
    hsl = rgb2hsl_torch(inner[..., :-1])
    hsl[..., 0:1] = (hsl[..., 0:1] + hue) % 1.0
    return th.cat([hsl2rgb_torch(hsl), alpha], dim=-1)
